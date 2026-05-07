"""Anthropic Claude judge."""

from __future__ import annotations

import os
from typing import Optional

from .base import JudgeWrapper


class AnthropicJudge(JudgeWrapper):
    judge_provider = "anthropic"

    def __init__(
        self,
        snapshot: str = "claude-opus-4-7",        # protocol/model_list.yaml#judges
        max_output_tokens: int = 1024,
        api_key: Optional[str] = None,
    ) -> None:
        from anthropic import Anthropic

        self.judge_snapshot = snapshot
        self.judge_model_id = f"judge_{snapshot}"
        self.max_output_tokens = max_output_tokens
        key = (
            api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("CLAUDE_KEY")
        )
        self._client = Anthropic(api_key=key)

    def _call(self, system_prompt: str, user_prompt: str) -> dict:
        resp = self._client.messages.create(
            model=self.judge_snapshot,
            max_tokens=self.max_output_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_parts = [
            blk.text for blk in resp.content
            if getattr(blk, "type", None) == "text"
        ]
        usage = getattr(resp, "usage", None)
        return {
            "text": "".join(text_parts),
            "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
            "error_status": "ok",
        }
