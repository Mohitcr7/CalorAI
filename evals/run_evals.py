"""Run the eval set.

Each case gets its own user id and a clean slate, so cases cannot contaminate
each other. Assertions run against the database after every turn.

    python -m evals.run_evals
    python -m evals.run_evals --image path/to/plate.jpg
    python -m evals.run_evals --only correction_no_double_count
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calorai import db, memory  # noqa: E402
from evals.cases import CASES  # noqa: E402

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def run_case(case, image_path: str | None, verbose: bool) -> tuple[bool, list[str]]:
    from calorai.agent import chat

    uid = f"eval_{case.name}"
    # Clean slate for this user only.
    with db.tx() as c:
        c.execute("DELETE FROM meals WHERE user_id=?", (uid,))
        c.execute("DELETE FROM memories WHERE user_id=?", (uid,))
        c.execute("DELETE FROM messages WHERE user_id=?", (uid,))
    if case.setup:
        case.setup(uid)

    failures: list[str] = []
    for n, turn in enumerate(case.turns, 1):
        img = image_path if turn.image == "__IMAGE__" else turn.image
        try:
            reply = chat(uid, turn.text, img)
        except Exception as exc:
            failures.append(f"turn {n} raised: {exc}")
            if verbose:
                traceback.print_exc()
            break

        if verbose:
            shown = turn.text or f"[image {img}]"
            print(f"    {DIM}> {shown}{RESET}")
            print(f"    {DIM}< {reply}{RESET}")

        low = reply.lower()
        for frag in turn.expect_reply_none:
            if frag.lower() in low:
                failures.append(f"turn {n}: reply should not contain '{frag}'")
        if turn.expect_reply_any and not any(f.lower() in low for f in turn.expect_reply_any):
            failures.append(f"turn {n}: reply matched none of {turn.expect_reply_any}")
        for label, check in turn.expect_db:
            try:
                if not check(uid):
                    failures.append(f"turn {n}: {label}")
            except Exception as exc:
                failures.append(f"turn {n}: check '{label}' errored: {exc}")

    return not failures, failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="food photo for the image cases")
    ap.add_argument("--only", help="run one case by name")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each exchange")
    args = ap.parse_args()

    cases = [c for c in CASES if not args.only or c.name == args.only]
    if not args.image:
        skipped = [c.name for c in cases if c.needs_image]
        cases = [c for c in cases if not c.needs_image]
        if skipped:
            print(f"{YELLOW}skipping image cases (pass --image): {', '.join(skipped)}{RESET}\n")

    passed, results = 0, []
    t0 = time.perf_counter()
    for case in cases:
        start = time.perf_counter()
        ok, failures = run_case(case, args.image, args.verbose)
        dur = time.perf_counter() - start
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"{mark}  {case.name}  {DIM}({dur:.1f}s){RESET}")
        if not ok:
            print(f"      {DIM}{case.why}{RESET}")
            for f in failures:
                print(f"      {RED}- {f}{RESET}")
        passed += ok
        results.append((case.name, ok))

    memory.drain(timeout=10)
    total = len(cases)
    colour = GREEN if passed == total else RED
    print(f"\n{colour}{passed}/{total} passed{RESET} in {time.perf_counter() - t0:.1f}s")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
