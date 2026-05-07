"""CheXOne 4B wrapper (StanfordAIMI/CheXOne).

Built on Qwen2.5-VL-3B-Instruct. Supports two inference modes:
  - reasoning: appends step-by-step reasoning suffix; produces \\boxed{} output
  - instruct:  no reasoning suffix; faster, no chain-of-thought
"""

from __future__ import annotations

from typing import List

from .hf_base import HFVLMWrapper

REASONING_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


class CheXOneWrapper(HFVLMWrapper):
    hf_repo = "StanfordAIMI/CheXOne"
    revision = "main"             # pin commit SHA at Phase 5 launch
    torch_dtype = "bfloat16"
    model_provider = "stanford_aimi"
    model_family = "medical_specialist"
    input_image_format = "pil_image"

    def __init__(self, mode: str = "reasoning") -> None:
        super().__init__()
        if mode not in ("reasoning", "instruct"):
            raise ValueError(f"mode must be 'reasoning' or 'instruct', got {mode!r}")
        self.mode = mode
        self.model_id = f"chexone_4b_{mode}"
        self.model_snapshot = f"{self.hf_repo}@{self.revision}#{mode}"

    def _load_processor(self):
        from transformers import AutoProcessor

        # Model card recommends min/max pixel limits aligned with training
        return AutoProcessor.from_pretrained(
            self.hf_repo,
            revision=self.revision,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 512,
        )

    def _load_model(self):
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.hf_repo,
            revision=self.revision,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )

    def _build_messages(self, prompt: str, image) -> List[dict]:
        text = prompt + (REASONING_SUFFIX if self.mode == "reasoning" else "")
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _prepare_inputs(self, messages, image):
        from qwen_vl_utils import process_vision_info

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self._device)
