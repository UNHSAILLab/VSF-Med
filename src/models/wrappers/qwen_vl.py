"""Qwen3.6-27B VL wrapper (Qwen/Qwen3.6-27B).

Disables ``<think>`` reasoning trace at chat-template time so output schema
matches non-reasoning targets in the lineup. Phase 5 inference can also be
served via vLLM with OpenAI-compatible API; this class uses HF transformers
directly to retain white-box access for adaptive attacks (Phase 8).
"""

from __future__ import annotations

from typing import List

from .hf_base import HFVLMWrapper


class Qwen36VLWrapper(HFVLMWrapper):
    hf_repo = "Qwen/Qwen3.6-27B"
    revision = "main"
    torch_dtype = "bfloat16"
    model_id = "qwen3_6_27b"
    model_provider = "alibaba"
    model_family = "open_generalist"
    input_image_format = "pil_image"

    def __init__(self) -> None:
        super().__init__()
        self.model_snapshot = f"{self.hf_repo}@{self.revision}"

    def _load_processor(self):
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(self.hf_repo, revision=self.revision)

    def _load_model(self):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            self.hf_repo,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )

    def _build_messages(self, prompt: str, image) -> List[dict]:
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _prepare_inputs(self, messages, image):
        try:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,           # disable <think> traces
            )
        except TypeError:
            # Older processor versions don't accept enable_thinking
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        images = [image] if image is not None else None
        inputs = self.processor(
            text=[text], images=images, padding=True, return_tensors="pt"
        )
        return inputs.to(self._device)
