"""LLaVA-Med v1.5 wrapper (microsoft/llava-med-v1.5-mistral-7b).

LLaVA-Med uses LLaVA's mistral-style chat format and the LlavaForConditionalGeneration
architecture. Image tokens are inserted via the processor's chat template.
"""

from __future__ import annotations

from typing import List

from .hf_base import HFVLMWrapper


class LlavaMedWrapper(HFVLMWrapper):
    hf_repo = "microsoft/llava-med-v1.5-mistral-7b"
    revision = "main"
    torch_dtype = "bfloat16"
    model_id = "llava_med_v1_5"
    model_provider = "microsoft"
    model_family = "medical_specialist"
    input_image_format = "pil_image"

    def __init__(self) -> None:
        super().__init__()
        self.model_snapshot = f"{self.hf_repo}@{self.revision}"

    def _load_processor(self):
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(self.hf_repo, revision=self.revision)

    def _load_model(self):
        from transformers import LlavaForConditionalGeneration

        return LlavaForConditionalGeneration.from_pretrained(
            self.hf_repo,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )

    def _build_messages(self, prompt: str, image) -> List[dict]:
        content = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _prepare_inputs(self, messages, image):
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        images = [image] if image is not None else None
        inputs = self.processor(
            text=text, images=images, return_tensors="pt", padding=True
        )
        return inputs.to(self._device)
