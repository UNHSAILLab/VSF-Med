"""Google Gemini target wrapper (vision-capable)."""

from __future__ import annotations

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


class GeminiTargetWrapper(ModelWrapper):
    model_provider = "google"
    model_family = "frontier"
    input_image_format = "inline_bytes"

    def __init__(self, snapshot: str = "gemini-3-flash-preview",
                 max_output_tokens: int = 512,
                 api_key: Optional[str] = None) -> None:
        from google import genai

        self.model_snapshot = snapshot
        self.model_id = f"target_{snapshot}"
        self.max_output_tokens = max_output_tokens
        key = (api_key or os.environ.get("GOOGLE_API_KEY")
               or os.environ.get("GEMINI_API_KEY"))
        self._client = genai.Client(api_key=key)

    def _call(self, request: ModelRequest) -> dict:
        from google.genai import types

        parts = []
        if request.image_path:
            try:
                p = Path(request.image_path)
                mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                        "png": "image/png"}.get(p.suffix.lower().lstrip("."), "image/jpeg")
                parts.append(types.Part.from_bytes(data=p.read_bytes(), mime_type=mime))
            except Exception as exc:  # noqa: BLE001
                return {"text": "", "error_status": ERROR_IMAGE_REJECTED, "exc": repr(exc)}
        parts.append(types.Part.from_text(text=request.prompt))

        config_kwargs = dict(
            system_instruction=CLINICAL_SYSTEM_PROMPT,
            max_output_tokens=self.max_output_tokens,
            temperature=0.0,
        )
        # Disable thinking on 2.5+ so all output budget flows into the answer
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except (AttributeError, TypeError):
            pass

        try:
            resp = self._client.models.generate_content(
                model=self.model_snapshot,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(**config_kwargs),
            )
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            err = ERROR_RATE_LIMITED if "rate" in msg or "429" in msg else ERROR_OTHER
            return {"text": "", "error_status": err, "exc": repr(exc)}

        text = ""
        candidates = getattr(resp, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in (getattr(content, "parts", None) or []):
                t = getattr(part, "text", None)
                if t:
                    text += t
        if not text:
            try:
                text = getattr(resp, "text", "") or ""
            except Exception:  # noqa: BLE001
                text = ""

        usage = getattr(resp, "usage_metadata", None)
        return {
            "text": text,
            "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            "error_status": ERROR_OK if text else ERROR_OTHER,
        }
