"""Latency benchmark: p50 / p95 for the text path and the image path.

Measures wall clock for a full turn as the user experiences it -- message in,
complete reply out, including every tool call the agent decided to make. Two
numbers are reported per path:

  total        the whole turn
  first_token  when the user starts seeing words (streaming)

first_token is arguably the number that matters on WhatsApp, but total is the
honest one, so both are reported.

    python -m bench.latency --runs 12
    python -m bench.latency --runs 6 --image path/to/plate.jpg
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calorai import db, memory  # noqa: E402

# Spread across the shapes of message the agent actually sees: a plain log, a
# correction, a query answered from the prompt, a memory recall, a fractional
# portion. Averaging only over easy logs would flatter the numbers.
TEXT_MESSAGES = [
    "had 2 parathas and chai for breakfast",
    "actually that was 3 parathas not 2",
    "how am I doing on calories?",
    "leftover biryani, maybe two thirds of the box",
    "3 rotis and dal for dinner",
    "how much protein have I had today?",
]


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def report(label: str, totals: list[float], firsts: list[float]) -> None:
    if not totals:
        return
    print(f"\n{label}  (n={len(totals)})")
    print(f"  total        p50 {pct(totals, .5):5.2f}s   p95 {pct(totals, .95):5.2f}s   "
          f"mean {statistics.mean(totals):5.2f}s   min {min(totals):5.2f}s   max {max(totals):5.2f}s")
    if firsts:
        print(f"  first token  p50 {pct(firsts, .5):5.2f}s   p95 {pct(firsts, .95):5.2f}s")


def bench_text(runs: int, verbose: bool) -> tuple[list[float], list[float]]:
    from calorai.agent import chat_stream

    totals, firsts = [], []
    for i in range(runs):
        uid = f"bench_text_{i % 3}"          # a few users, so history is realistic
        msg = TEXT_MESSAGES[i % len(TEXT_MESSAGES)]
        start = time.perf_counter()
        first = None
        out = []
        for tok in chat_stream(uid, msg):
            if first is None:
                first = time.perf_counter() - start
            out.append(tok)
        total = time.perf_counter() - start
        totals.append(total)
        if first is not None:
            firsts.append(first)
        if verbose:
            print(f"  [{total:5.2f}s] {msg}  ->  {''.join(out)[:70]}")
    return totals, firsts


def bench_image(runs: int, image: str, verbose: bool) -> tuple[list[float], list[float]]:
    from calorai.agent import chat_stream

    totals, firsts = [], []
    for i in range(runs):
        uid = f"bench_img_{i % 3}"
        caption = "" if i % 2 == 0 else "half of this was my brother's"
        start = time.perf_counter()
        first = None
        out = []
        for tok in chat_stream(uid, caption, image):
            if first is None:
                first = time.perf_counter() - start
            out.append(tok)
        total = time.perf_counter() - start
        totals.append(total)
        if first is not None:
            firsts.append(first)
        if verbose:
            print(f"  [{total:5.2f}s] [photo] {caption}  ->  {''.join(out)[:70]}")
    return totals, firsts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--image", help="food photo; enables the image path benchmark")
    ap.add_argument("--image-runs", type=int, default=6)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    print(f"warming up (first call pays connection setup, excluded from stats)...")
    from calorai.agent import chat
    chat("bench_warmup", "hi")

    t_totals, t_firsts = bench_text(args.runs, args.verbose)
    report("TEXT PATH", t_totals, t_firsts)

    if args.image:
        i_totals, i_firsts = bench_image(args.image_runs, args.image, args.verbose)
        report("IMAGE PATH", i_totals, i_firsts)

    memory.drain(timeout=10)
    print()


if __name__ == "__main__":
    main()
