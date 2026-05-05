"""
LLM-based extraction of A/B/C/D from think-style MC outputs (OpenAI-compatible API, e.g. lmdeploy).
"""

from __future__ import annotations

import re
from typing import Optional

from utils.prompts import prompt_template


def parse_mc_letter(text: str) -> str:
    """Normalize model reply to a single letter A–D, or '' if invalid."""
    if not text:
        return ""
    t = text.strip().upper()
    if t in ("A", "B", "C", "D"):
        return t
    m = re.search(r"\b([ABCD])\b", t)
    if m:
        return m.group(1)
    if len(t) == 1 and t in "ABCD":
        return t
    return ""


class ThinkMCAnswerExtractor:
    """Calls a chat model with system+user prompts from prompt_template['mc_think_extract']."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:23333/v1",
        model_name: str = "Qwen/Qwen2.5-72B-Instruct-AWQ",
        api_key: str = "EMPTY",
        timeout: int = 120,
    ):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout)
        self.model_name = model_name
        pt = prompt_template["mc_think_extract"]
        self._system = pt["system"]
        self._user_template = pt["user_template"]

    def extract(self, model_output: str) -> str:
        raw = (model_output or "").strip()
        if not raw:
            return ""
        user = self._user_template.format(model_output=raw)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=16,
            top_p=1.0,
        )
        out = (resp.choices[0].message.content or "").strip()
        letter = parse_mc_letter(out)
        return letter
