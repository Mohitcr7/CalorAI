"""Model clients, with Claude primary and Gemini as failover.

WHAT FAILS OVER, AND WHAT DELIBERATELY DOES NOT
-----------------------------------------------
Failover is for TRANSIENT provider problems only: rate limits, overload (529),
timeouts, connection errors, 5xx. Those are the failures where a second
provider genuinely helps.

Authentication errors, permission errors and malformed requests do NOT fail
over. They are configuration or code bugs, and silently succeeding on the
backup provider would hide them forever -- the app would look healthy while
running permanently on the failover model and nobody would find out until the
Gemini key also expired. Those raise loudly instead.

WHY max_retries=0 ON THE TEXT PATH
----------------------------------
The Anthropic SDK retries twice by default with exponential backoff. On a 429
that means several seconds burned *before* the fallback is even reached, on top
of the fallback's own latency. On a messaging product that is the wrong trade:
a fast answer from the backup model beats a slow answer from the primary one.
So the text path retries zero times and fails over immediately. The extractor,
which runs off the critical path where nobody is waiting, keeps its retries.

STREAMING SAFETY
----------------
LangChain's RunnableWithFallbacks.stream() pulls the first chunk inside the try
block and only falls back if that first chunk fails. Once tokens have been
emitted, a mid-stream error propagates rather than restarting on the backup
model. That is the behaviour we want: a user must never see half a reply from
Claude followed by a fresh, different reply from Gemini.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import (
    ANTHROPIC_API_KEY,
    EXTRACTOR_MODEL,
    EXTRACTOR_TIMEOUT,
    FALLBACK_EXTRACTOR_MODEL,
    FALLBACK_TEXT_MODEL,
    FALLBACK_VISION_MODEL,
    GEMINI_API_KEY,
    PROVIDER_MODE,
    TEXT_FALLBACK_TIMEOUT,
    TEXT_MODEL,
    TEXT_THINKING_BUDGET,
    TEXT_TIMEOUT,
    VISION_FALLBACK_TIMEOUT,
    VISION_MODEL,
    VISION_THINKING_BUDGET,
    VISION_TIMEOUT,
)


class MissingKey(RuntimeError):
    pass


# --------------------------------------------------------------------------
# which exceptions are worth a second provider
# --------------------------------------------------------------------------

def _transient_exceptions() -> tuple[type[BaseException], ...]:
    """Transient provider failures. Auth and bad-request are excluded on purpose."""
    types: list[type[BaseException]] = [TimeoutError, ConnectionError]
    try:
        import anthropic

        types += [
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,   # 5xx, includes 529 overloaded
        ]
    except ImportError:
        pass
    try:
        from google.api_core import exceptions as gexc

        types += [
            gexc.ResourceExhausted,
            gexc.ServiceUnavailable,
            gexc.DeadlineExceeded,
            gexc.InternalServerError,
        ]
    except ImportError:
        pass
    return tuple(types)


TRANSIENT = _transient_exceptions()


def have_anthropic() -> bool:
    return bool(ANTHROPIC_API_KEY) and PROVIDER_MODE in ("auto", "anthropic")


def have_google() -> bool:
    return bool(GEMINI_API_KEY) and PROVIDER_MODE in ("auto", "google")


# --------------------------------------------------------------------------
# raw per-provider clients
# --------------------------------------------------------------------------

def _claude(model: str, *, temperature: float, timeout: float,
            max_tokens: int, max_retries: int) -> ChatAnthropic:
    return ChatAnthropic(
        model=model,
        api_key=ANTHROPIC_API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        default_request_timeout=timeout,
        max_retries=max_retries,
    )


def _gemini(model: str, *, temperature: float, timeout: float,
            max_tokens: int, thinking_budget: int | None) -> ChatGoogleGenerativeAI:
    kwargs = dict(
        model=model,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
        max_output_tokens=max_tokens,
        timeout=timeout,
    )
    if thinking_budget is not None:
        kwargs["thinking_budget"] = thinking_budget
    return ChatGoogleGenerativeAI(**kwargs)


def _compose(primary: BaseChatModel | None, fallback: BaseChatModel | None,
             role: str) -> BaseChatModel:
    """Attach the fallback if both providers are configured.

    With only one provider configured this returns that provider's client
    unwrapped -- no fallback machinery, no behaviour change. Running on a single
    key is a supported configuration, not a degraded one.
    """
    if primary is None and fallback is None:
        raise MissingKey(
            f"No provider configured for the {role} model. Set ANTHROPIC_API_KEY "
            "(primary) and/or GEMINI_API_KEY (failover) in .env. "
            "Keys: https://console.anthropic.com/settings/keys and "
            "https://aistudio.google.com/apikey"
        )
    if primary is None:
        return fallback
    if fallback is None:
        return primary
    return primary.with_fallbacks([fallback], exceptions_to_handle=TRANSIENT)


# --------------------------------------------------------------------------
# the three roles
# --------------------------------------------------------------------------

def _build_text(temperature: float, tools: tuple | None):
    """Build the text runnable, optionally with tools bound.

    Tools are bound to EACH provider before the fallback is composed. This is
    not a stylistic choice: with_fallbacks() returns a plain Runnable, which has
    no .bind_tools(), so binding after composing would fail outright -- and
    binding only to the primary would silently give a toolless agent the moment
    a failover happened, which is worse. Both providers get the same tools or
    neither runs.
    """
    primary = _claude(
        TEXT_MODEL, temperature=temperature, timeout=TEXT_TIMEOUT,
        max_tokens=512, max_retries=0,
    ) if have_anthropic() else None
    fallback = _gemini(
        FALLBACK_TEXT_MODEL, temperature=temperature, timeout=TEXT_FALLBACK_TIMEOUT,
        max_tokens=512, thinking_budget=TEXT_THINKING_BUDGET,
    ) if have_google() else None

    if tools:
        if primary is not None:
            primary = primary.bind_tools(list(tools))
        if fallback is not None:
            fallback = fallback.bind_tools(list(tools))
    return _compose(primary, fallback, "text")


@lru_cache(maxsize=None)
def text_llm(temperature: float = 0.2):
    """Conversation without tools. Used by the bench warm-up and simple calls.

    max_retries=0: fail over immediately rather than burning seconds on the
    SDK's exponential backoff while the user watches a typing indicator.
    """
    return _build_text(temperature, None)


_TOOL_LLM_CACHE: dict[tuple, object] = {}


def text_llm_with_tools(tools, temperature: float = 0.2):
    """Tool-bound text runnable. Cached, because rebuilding two clients and
    re-serialising ten tool schemas on every turn is pure latency."""
    key = (temperature, tuple(getattr(t, "name", str(t)) for t in tools))
    if key not in _TOOL_LLM_CACHE:
        _TOOL_LLM_CACHE[key] = _build_text(temperature, tuple(tools))
    return _TOOL_LLM_CACHE[key]


@lru_cache(maxsize=None)
def vision_llm() -> BaseChatModel:
    """Images only. Never gets tools -- returns one structured payload.

    One retry here, unlike the text path: re-encoding and re-sending an image is
    expensive, so it is worth one attempt before switching providers.
    """
    primary = _claude(
        VISION_MODEL, temperature=0.1, timeout=VISION_TIMEOUT,
        max_tokens=768, max_retries=1,
    ) if have_anthropic() else None
    fallback = _gemini(
        FALLBACK_VISION_MODEL, temperature=0.1, timeout=VISION_FALLBACK_TIMEOUT,
        max_tokens=768, thinking_budget=VISION_THINKING_BUDGET,
    ) if have_google() else None
    return _compose(primary, fallback, "vision")


@lru_cache(maxsize=None)
def extractor_llm() -> BaseChatModel:
    """Memory extraction and nutrition fallback. Always off the reply path, so
    retries are free here -- nobody is waiting."""
    primary = _claude(
        EXTRACTOR_MODEL, temperature=0.0, timeout=EXTRACTOR_TIMEOUT,
        max_tokens=512, max_retries=2,
    ) if have_anthropic() else None
    fallback = _gemini(
        FALLBACK_EXTRACTOR_MODEL, temperature=0.0, timeout=EXTRACTOR_TIMEOUT,
        max_tokens=512, thinking_budget=0,
    ) if have_google() else None
    return _compose(primary, fallback, "extractor")


def active_providers() -> str:
    """Human-readable summary, printed by the CLI banner so it is always obvious
    which models a given run is actually using."""
    if have_anthropic() and have_google():
        return f"{TEXT_MODEL} / {VISION_MODEL}  (failover: {FALLBACK_TEXT_MODEL} / {FALLBACK_VISION_MODEL})"
    if have_anthropic():
        return f"{TEXT_MODEL} / {VISION_MODEL}  (no failover configured)"
    if have_google():
        return f"{FALLBACK_TEXT_MODEL} / {FALLBACK_VISION_MODEL}  (Gemini only)"
    return "no provider configured"


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


def parse_json(text: str):
    """Models add markdown fences and prose no matter how firmly you ask them not to."""
    cleaned = _FENCE.sub("", text or "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
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
