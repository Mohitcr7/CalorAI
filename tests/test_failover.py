"""Failover behaviour, proven without touching either provider.

These use stub runnables rather than real clients, so the routing policy is
tested deterministically and for free. What is asserted:

  1. a transient failure on the primary silently falls through to the backup
  2. an auth failure does NOT fall through -- it raises, loudly
  3. streaming falls over cleanly when the primary dies before any token
  4. streaming does NOT restart on the backup once tokens are already out,
     which would show the user two different replies to one message

    python -m tests.test_failover
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.runnables import RunnableLambda  # noqa: E402

from calorai.llm import TRANSIENT  # noqa: E402

GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"{mark}  {name}" + (f"  {RED}{detail}{RESET}" if not ok else ""))


class FakeRateLimit(Exception):
    """Stands in for anthropic.RateLimitError, which needs a real HTTP response
    object to construct."""


def raises(exc: Exception):
    def _fn(_):
        raise exc
    return RunnableLambda(_fn)


def returns(value: str):
    return RunnableLambda(lambda _: value)


def main() -> None:
    # 1. transient failure -> backup answers
    chain = raises(FakeRateLimit("429")).with_fallbacks(
        [returns("from-gemini")], exceptions_to_handle=(FakeRateLimit,)
    )
    check("transient failure falls over", chain.invoke("x") == "from-gemini")

    # 2. auth failure is NOT in the transient set -> must raise
    class FakeAuthError(Exception):
        pass

    chain2 = raises(FakeAuthError("401")).with_fallbacks(
        [returns("from-gemini")], exceptions_to_handle=TRANSIENT
    )
    try:
        chain2.invoke("x")
        check("auth failure raises instead of hiding", False, "it fell over silently")
    except FakeAuthError:
        check("auth failure raises instead of hiding", True)
    except Exception as exc:
        check("auth failure raises instead of hiding", False, f"wrong error: {exc!r}")

    # 3. the real transient list covers what we claim it does
    names = {t.__name__ for t in TRANSIENT}
    expected = {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}
    check("anthropic transient errors registered", expected <= names,
          f"missing {expected - names}")
    check("auth errors deliberately absent",
          not {"AuthenticationError", "BadRequestError", "PermissionDeniedError"} & names)

    # 4. streaming: primary dies before first token -> backup streams instead
    def gen_ok(_):
        yield "he"
        yield "llo"

    def gen_dead(_):
        raise FakeRateLimit("429")
        yield  # unreachable, makes this a generator

    s_chain = RunnableLambda(gen_dead).with_fallbacks(
        [RunnableLambda(gen_ok)], exceptions_to_handle=(FakeRateLimit,)
    )
    check("stream fails over before first token",
          "".join(s_chain.stream("x")) == "hello")

    # 5. streaming: primary dies AFTER emitting -> must NOT restart on backup
    def gen_half_dead(_):
        yield "par"
        raise FakeRateLimit("429")

    s_chain2 = RunnableLambda(gen_half_dead).with_fallbacks(
        [RunnableLambda(gen_ok)], exceptions_to_handle=(FakeRateLimit,)
    )
    got, raised = [], False
    try:
        for tok in s_chain2.stream("x"):
            got.append(tok)
    except FakeRateLimit:
        raised = True
    check("mid-stream failure does not replay on backup",
          raised and "".join(got) == "par",
          f"got {got!r} raised={raised}")

    passed = sum(1 for _, ok, _ in _results if ok)
    total = len(_results)
    colour = GREEN if passed == total else RED
    print(f"\n{colour}{passed}/{total} passed{RESET}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
