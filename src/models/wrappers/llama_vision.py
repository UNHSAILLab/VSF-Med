"""Llama 4 Scout wrapper (meta-llama/Llama-4-Scout-17B-16E-Instruct).

Uses the Llama 4 multimodal AutoModel/AutoProcessor pair from transformers.
"""

from __future__ import annotations

from typing import List

from .hf_base import HFVLMWrapper


class Llama4ScoutWrapper(HFVLMWrapper):
    hf_repo = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    revision = "main"
    torch_dtype = "bfloat16"
    model_id = "llama_4_scout"
    model_provider = "meta"
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
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [image] if image is not None else None
        inputs = self.processor(
            text=[text], images=images, padding=True, return_tensors="pt"
        )
        return inputs.to(self._device)
