"""MedGemma wrapper (google/medgemma-4b-it and google/medgemma-27b-it).

One class with size variant chosen at construction time. Both variants use
the Gemma 3 multimodal architecture with the standard transformers VLM API.
"""

from __future__ import annotations

from typing import List

from .hf_base import HFVLMWrapper


class MedGemmaWrapper(HFVLMWrapper):
    model_provider = "google"
    model_family = "medical_specialist"
    input_image_format = "pil_image"

    def __init__(self, size: str = "4b") -> None:
        super().__init__()
        if size not in ("4b", "27b"):
            raise ValueError(f"size must be '4b' or '27b', got {size!r}")
        self.size = size
        self.hf_repo = f"google/medgemma-{size}-it"
        self.revision = "main"
        self.model_id = f"medgemma_{size}_it"
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
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self._device)
