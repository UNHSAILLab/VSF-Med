"""Shared scaffolding for HuggingFace transformers vision-language wrappers.

All open-weight models in the v2 lineup share the same overall pattern:
load processor + model → build chat messages → preprocess (text+image) →
generate → decode the trimmed continuation. Per-model subclasses override
the four hook methods that vary across model families.

Loading is lazy (deferred until first ``generate``) so wrapper modules can
be imported on CPU-only machines without pulling in CUDA.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, List, Optional

from .base import (
    ERROR_IMAGE_REJECTED,
    ERROR_OK,
    ERROR_OTHER,
    ModelRequest,
    ModelWrapper,
)


def _load_pil(image_path: Optional[str]):
    if not image_path:
        return None
    from PIL import Image  # local import — Pillow is light but keeps base.py provider-free

    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"image_path not found: {image_path}")
    return Image.open(p).convert("RGB")


class HFVLMWrapper(ModelWrapper):
    """Abstract base for HuggingFace transformers-backed VLM targets.

    Subclasses set the identity attributes and override:
      - ``_load_processor()``           — return processor / tokenizer object
      - ``_load_model()``               — return the generation model
      - ``_build_messages(prompt, image)`` — return chat messages list
      - ``_prepare_inputs(messages, image)`` — return tensor dict on device
      - ``_decode(inputs, out_ids)``    — return generated string
    """

    hf_repo: str = ""
    revision: str = "main"
    torch_dtype: str = "bfloat16"      # subclasses can override to "float16" or "auto"
    device_map: str = "auto"
    trust_remote_code: bool = False

    def __init__(self) -> None:
        self._loaded = False
        self.processor = None
        self.model = None
        self._device = "cuda"

    # ------------- lazy loading -------------

    def _lazy_load(self) -> None:
        if self._loaded:
            return
        import torch  # local import keeps top-level module CPU-safe

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = self._load_processor()
        self.model = self._load_model()
        self.model.eval()
        self._loaded = True

    @abc.abstractmethod
    def _load_processor(self): ...

    @abc.abstractmethod
    def _load_model(self): ...

    @abc.abstractmethod
    def _build_messages(self, prompt: str, image) -> List[dict]: ...

    @abc.abstractmethod
    def _prepare_inputs(self, messages: List[dict], image) -> Any: ...

    def _decode(self, inputs, out_ids) -> str:
        # Default: trim the prompt prefix from generated ids and decode
        in_len = inputs["input_ids"].shape[1]
        trimmed = out_ids[:, in_len:]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    # ------------- generation -------------

    def _call(self, request: ModelRequest) -> dict:
        try:
            image = _load_pil(request.image_path)
        except FileNotFoundError:
            return {"text": "", "error_status": ERROR_IMAGE_REJECTED}

        try:
            self._lazy_load()
        except Exception as exc:  # noqa: BLE001 — load failure should not crash the runner
            return {"text": "", "error_status": ERROR_OTHER, "load_error": repr(exc)}

        try:
            messages = self._build_messages(request.prompt, image)
            inputs = self._prepare_inputs(messages, image)

            with self._torch.inference_mode():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=request.decoding.max_output_tokens,
                    do_sample=False,                      # T=0 → greedy
                    pad_token_id=getattr(self.processor.tokenizer, "pad_token_id", None)
                    if hasattr(self.processor, "tokenizer")
                    else None,
                )

            text = self._decode(inputs, out_ids)
            in_tokens = int(inputs["input_ids"].shape[1])
            out_tokens = int(out_ids.shape[1] - in_tokens)
            return {
                "text": text,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "error_status": ERROR_OK,
            }
        except Exception as exc:  # noqa: BLE001 — normalize generation failures
            msg = str(exc)
            err = ERROR_OTHER
            if "out of memory" in msg.lower():
                err = ERROR_OTHER  # OOM is a config issue, not a model behavior
                self._torch.cuda.empty_cache() if self._device == "cuda" else None
            return {"text": "", "error_status": err, "exception": repr(exc)}
