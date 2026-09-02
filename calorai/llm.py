"""Model clients.

Three distinct models, three distinct jobs. See config.py for the split and the
README for why Gemini at all.

The single most important line in this file is `thinking_budget=0` on the text
model. Gemini 2.5 runs an internal reasoning pass by default; for "had 2 rotis"
that pass costs more wall-clock than the entire rest of the turn and buys
nothing. Turning it off took the text p50 from multi-second to sub-second.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI

from .config import (
    EXTRACTOR_MODEL,
    GEMINI_API_KEY,
    TEXT_MODEL,
    TEXT_THINKING_BUDGET,
    VISION_MODEL,
    VISION_THINKING_BUDGET,
)


class MissingKey(RuntimeError):
    pass


def _require_key() -> str:
    if not GEMINI_API_KEY:
        raise MissingKey(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/apikey"
        )
    return GEMINI_API_KEY


@lru_cache(maxsize=None)
def text_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """Conversation + tool calling. On the critical path for every message."""
    return ChatGoogleGenerativeAI(
        model=TEXT_MODEL,
        google_api_key=_require_key(),
        temperature=temperature,
        thinking_budget=TEXT_THINKING_BUDGET,
        max_output_tokens=512,
        timeout=20,
    )


@lru_cache(maxsize=None)
def vision_llm() -> ChatGoogleGenerativeAI:
    """Images only. Never gets tools -- it returns one structured payload."""
    return ChatGoogleGenerativeAI(
        model=VISION_MODEL,
        google_api_key=_require_key(),
        temperature=0.1,
        thinking_budget=VISION_THINKING_BUDGET,
        max_output_tokens=768,
        timeout=30,
    )


@lru_cache(maxsize=None)
def extractor_llm() -> ChatGoogleGenerativeAI:
    """Memory extraction and nutrition fallback. Always off the reply path."""
    return ChatGoogleGenerativeAI(
        model=EXTRACTOR_MODEL,
        google_api_key=_require_key(),
        temperature=0.0,
        thinking_budget=0,
        max_output_tokens=512,
        timeout=15,
    )


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_json(text: str):
    """Models add markdown fences and prose no matter how firmly you ask them not to."""
    cleaned = _FENCE.sub("", text or "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost balanced object or array in the string.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def quick_json(prompt: str):
    """One-shot structured call on the cheap model."""
    resp = extractor_llm().invoke(prompt)
    return parse_json(resp.content if isinstance(resp.content, str) else str(resp.content))
