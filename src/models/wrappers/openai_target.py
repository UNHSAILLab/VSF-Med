"""OpenAI GPT target wrapper (vision-capable)."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

from .anthropic_target import CLINICAL_SYSTEM_PROMPT
from .base import (
    ERROR_IMAGE_REJECTED,
    ERROR_OK,
    ERROR_OTHER,
    ERROR_RATE_LIMITED,
    ModelRequest,
    ModelWrapper,
)


def _image_data_url(path: str) -> str:
    p = Path(path)
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        p.suffix.lower().lstrip("."), "image/jpeg")
    return f"data:{mime};base64,{base64.standard_b64encode(p.read_bytes()).decode('ascii')}"


class OpenAIGPTTargetWrapper(ModelWrapper):
    model_provider = "openai"
    model_family = "frontier"
    input_image_format = "data_url_b64"

    def __init__(self, snapshot: str = "gpt-5.4-mini-2026-03-17",
                 max_output_tokens: int = 512,
                 api_key: Optional[str] = None) -> None:
        from openai import OpenAI

        self.model_snapshot = snapshot
        self.model_id = f"target_{snapshot}"
        self.max_output_tokens = max_output_tokens
        key = (api_key or os.environ.get("OPENAI_API_KEY")
               or os.environ.get("OPEN_AI_KEY"))
        self._client = OpenAI(api_key=key)

    def _call(self, request: ModelRequest) -> dict:
        try:
            data_url = _image_data_url(request.image_path) if request.image_path else None
        except Exception as exc:  # noqa: BLE001
            return {"text": "", "error_status": ERROR_IMAGE_REJECTED, "exc": repr(exc)}

        user_content = []
        if data_url:
            user_content.append({"type": "image_url", "image_url": {"url": data_url}})
        user_content.append({"type": "text", "text": request.prompt})

        try:
            # GPT-5.x uses max_completion_tokens; older models used max_tokens.
            resp = self._client.chat.completions.create(
                model=self.model_snapshot,
                messages=[
                    {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_completion_tokens=self.max_output_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            err = ERROR_RATE_LIMITED if "rate" in msg or "429" in msg else ERROR_OTHER
            return {"text": "", "error_status": err, "exc": repr(exc)}

        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return {
            "text": text,
            "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "error_status": ERROR_OK,
        }
