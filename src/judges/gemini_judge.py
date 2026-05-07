"""Google Gemini judge."""

from __future__ import annotations

import os
from typing import Optional

from .base import JudgeWrapper


class GeminiJudge(JudgeWrapper):
    judge_provider = "google"

    def __init__(
        self,
        snapshot: str = "gemini-3.0-pro",
        max_output_tokens: int = 1024,
        api_key: Optional[str] = None,
    ) -> None:
        from google import genai

        self.judge_snapshot = snapshot
        self.judge_model_id = f"judge_{snapshot}"
        self.max_output_tokens = max_output_tokens
        # GOOGLE_API_KEY first because GEMINI_API_KEY in some setups is the
        # legacy AI-Studio key and may not be valid for the v1 API.
        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self._client = genai.Client(api_key=key)

    def _call(self, system_prompt: str, user_prompt: str) -> dict:
        from google.genai import types

        # Gemini 2.5+ enables "thinking" by default which silently consumes
        # the output budget. Set thinking_budget=0 so all tokens flow into
        # the user-visible response.
        config_kwargs = dict(
            system_instruction=system_prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=0.0,
            response_mime_type="application/json",
        )
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except (AttributeError, TypeError):
            pass  # older SDK / model without thinking config

        resp = self._client.models.generate_content(
            model=self.judge_snapshot,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        # Gemini can return no candidate when safety filters trigger; pull text
        # defensively rather than relying on .text (which raises in that case).
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
        finish = candidates[0].finish_reason if candidates else None
        err = "ok"
        if not text:
            # Surface why we got nothing — usually safety or recitation filter
            err = f"empty_response:{finish or 'unknown'}"
        return {
            "text": text,
            "input_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
            "output_tokens": getattr(usage, "candidates_token_count", None) if usage else None,
            "error_status": err,
        }
