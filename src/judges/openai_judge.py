"""OpenAI GPT judge."""

from __future__ import annotations

import os
from typing import Optional

from .base import JudgeWrapper


class OpenAIJudge(JudgeWrapper):
    judge_provider = "openai"

    def __init__(
        self,
        snapshot: str = "gpt-5.2",       # strongest GPT, set by user at run time
        max_output_tokens: int = 1024,
        api_key: Optional[str] = None,
    ) -> None:
        from openai import OpenAI

        self.judge_snapshot = snapshot
        self.judge_model_id = f"judge_{snapshot}"
        self.max_output_tokens = max_output_tokens
        # SDK standard env var first, then user's project convention
        key = (
            api_key
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPEN_AI_KEY")
        )
        self._client = OpenAI(api_key=key)

    def _call(self, system_prompt: str, user_prompt: str) -> dict:
        resp = self._client.chat.completions.create(
            model=self.judge_snapshot,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=self.max_output_tokens,
            temperature=0.0,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return {
            "text": text,
            "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "error_status": "ok",
        }
