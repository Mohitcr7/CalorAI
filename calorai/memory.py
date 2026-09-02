"""Long-term memory: what gets stored, when it is written, how it comes back.

This is the part of the system I spent the most time on, so the reasoning is
written down here rather than in the README alone.

THREE KINDS OF MEMORY, deliberately separated
---------------------------------------------
1. FACTS      - durable, small, about the person. "vegetarian", "targets 140g
                protein", "trains in the evening". Dozens at most, ever.
2. ALIASES    - a phrase the user uses to mean a specific meal. "my usual" ->
                2 parathas + chai. Each carries a JSON meal template.
3. EPISODIC   - what they actually ate. This is NOT in the memory store at all;
                it is the `meals` table. "same as yesterday" is a SQL query, not
                a recall problem, and treating it as one is both faster and
                exactly correct.

Conflating these is the usual failure mode. Dumping every meal into a vector
store gives you a memory system that is enormous, slow, and still gets "how
many calories today" wrong.

WHEN MEMORY IS WRITTEN
----------------------
Off the critical path. After the reply has been streamed to the user, a
background thread runs a cheap extraction call over the turn. The user never
waits on it. The cost of that choice is that a fact stated in message N is
occasionally not available until message N+1; in a messaging product that is
invisible, and it buys roughly 300-600 ms on every single turn.

Writes are upserts on (user_id, key) against a BOUNDED key namespace. This is
the mechanism that stops memory from growing without limit: the tenth time a
user mentions being vegetarian it overwrites one row rather than adding a
tenth. Overwrites are recorded in memory_history so a change is explainable.

HOW MEMORY IS RETRIEVED (the anti-bloat design)
-----------------------------------------------
Tier 1 - always injected. Facts only, rendered one short line each, hard-capped
         at MAX_TIER1_FACTS and ordered by a fixed priority so the cap drops the
         least useful first. Costs ~150-250 tokens, bounded regardless of how
         long the user has been around.
Tier 2 - on demand. Alias *names* are listed in tier 1 (a few tokens each) so
         the model knows what exists, but their payloads are only fetched when
         the model calls the recall tool. A user with 40 saved meal templates
         still pays a fixed, tiny prompt cost.

That split is the whole trick: names are cheap and go in the prompt, bodies are
expensive and stay in the database until asked for.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from . import db

# --------------------------------------------------------------------------
# Bounded key namespace
# --------------------------------------------------------------------------
# A new observation must land on one of these keys (or an alias.* key). An
# unbounded namespace is what turns a memory store into a junk drawer.

FACT_KEYS: dict[str, str] = {
    "diet.restriction":  "dietary restriction (vegetarian, vegan, halal, jain, none)",
    "diet.allergy":      "food allergies or intolerances",
    "diet.dislikes":     "foods they avoid by preference",
    "goal.calories":     "daily calorie target",
    "goal.protein_g":    "daily protein target in grams",
    "goal.weight":       "weight goal (lose/gain/maintain, target weight)",
    "profile.name":      "what to call them",
    "profile.household": "who they usually eat with",
    "habit.breakfast":   "typical breakfast pattern",
    "habit.schedule":    "meal timing habits, fasting windows, training times",
    "habit.cooking":     "cooks at home vs eats out",
    "pref.detail":       "how much nutritional detail they want in replies",
}

# Lower number = kept first when the tier-1 cap bites.
_PRIORITY = {
    "diet.restriction": 0, "diet.allergy": 0, "goal.protein_g": 1,
    "goal.calories": 1, "goal.weight": 2, "profile.name": 2,
    "habit.breakfast": 3, "habit.schedule": 3, "pref.detail": 4,
    "diet.dislikes": 4, "profile.household": 5, "habit.cooking": 5,
}


def _priority(key: str) -> int:
    return _PRIORITY.get(key, 6)


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def tier1_context(user_id: str, max_facts: int | None = None) -> str:
    """The block injected into every system prompt. Bounded by construction."""
    from .config import MAX_TIER1_FACTS

    cap = max_facts if max_facts is not None else MAX_TIER1_FACTS
    rows = db.get_memories(user_id)
    facts = sorted(
        (r for r in rows if r["kind"] == "fact"),
        key=lambda r: (_priority(r["key"]), r["updated_at"]),
    )[:cap]
    aliases = [r for r in rows if r["kind"] == "alias"]

    lines: list[str] = []
    if facts:
        lines.append("What you know about this user:")
        lines += [f"- {f['value']}" for f in facts]
    if aliases:
        # Names only. Payloads stay in the DB until recall_memory is called.
        names = ", ".join(f'"{a["key"].removeprefix("alias.")}"' for a in aliases)
        lines.append(
            f"Saved meal shortcuts (call recall_memory to expand one): {names}"
        )
    return "\n".join(lines)


def recall(user_id: str, query: str) -> str:
    """Tier 2. Keyword search over the full memory store, payloads included."""
    q = query.lower().strip()
    rows = db.get_memories(user_id)
    if not rows:
        return "No memories stored for this user yet."

    scored = []
    for r in rows:
        hay = f"{r['key']} {r['value']} {r['payload'] or ''}".lower()
        score = sum(1 for tok in q.split() if tok and tok in hay)
        if r["key"].removeprefix("alias.") in q or q in r["key"]:
            score += 5
        if score:
            scored.append((score, r))
    if not scored:
        return "Nothing stored matches that."

    scored.sort(key=lambda t: -t[0])
    out = []
    for _, r in scored[:5]:
        line = f"{r['key']}: {r['value']}"
        if r["payload"]:
            line += f"\n  template: {r['payload']}"
        out.append(line)
    return "\n".join(out)


# Words that carry no meaning in a shortcut phrase. Stripping these is what
# lets "the usual" find the alias saved as "my usual".
_STOP = {"my", "the", "a", "an", "your", "our", "usual's"}


def _alias_tokens(phrase: str) -> set[str]:
    toks = re.sub(r"[^a-z0-9 ]", " ", phrase.lower()).split()
    return {t for t in toks if t not in _STOP} or set(toks)


def get_alias(user_id: str, phrase: str) -> dict | None:
    """Expand a saved shortcut into its meal template."""
    key = f"alias.{phrase.lower().strip()}"
    row = db.get_memory(user_id, key)
    if not row:
        # Tolerate near-misses: "the usual" vs "my usual", "usual breakfast".
        want = _alias_tokens(phrase)
        best, best_score = None, 0.0
        for r in db.get_memories(user_id, kind="alias"):
            have = _alias_tokens(r["key"].removeprefix("alias."))
            if not have:
                continue
            overlap = len(want & have) / len(have)
            if overlap > best_score:
                best, best_score = r, overlap
        if best is not None and best_score >= 0.5:
            row = best
    if not row:
        return None
    return {
        "phrase": row["key"].removeprefix("alias."),
        "description": row["value"],
        "items": json.loads(row["payload"])["items"] if row["payload"] else [],
    }


def save_alias(user_id: str, phrase: str, description: str, items: list[dict]) -> None:
    db.upsert_memory(
        user_id,
        f"alias.{phrase.lower().strip()}",
        description,
        kind="alias",
        payload={"items": items},
        confidence=0.9,
    )


def save_fact(user_id: str, key: str, value: str, confidence: float = 0.85) -> bool:
    if key not in FACT_KEYS:
        key = "habit.schedule" if "time" in key or "when" in key else key
    return db.upsert_memory(user_id, key, value, kind="fact", confidence=confidence)


# --------------------------------------------------------------------------
# Extraction (write path)
# --------------------------------------------------------------------------

_EXTRACT_PROMPT = """You maintain long-term memory for a meal-logging assistant.

Read the exchange below and decide whether it revealed anything DURABLE about
this user -- something still true next month.

Allowed keys and what they mean:
{keys}

STORE: standing preferences, restrictions, goals, habits, what to call them.
DO NOT STORE:
- anything about a specific meal on a specific day (that lives in the meals
  database already) -- "had 2 rotis", "skipped lunch today"
- one-off states: "I'm full", "feeling hungry"
- anything the assistant said about itself
- restatements of a fact already in "Already known" below, unless the new
  message CHANGES it

Already known:
{known}

--- exchange ---
User: {user_msg}
Assistant: {assistant_msg}
--- end ---

Reply with ONLY a JSON array, empty if nothing durable was revealed:
[{{"key": "<one of the allowed keys>", "value": "<short third-person statement>",
   "confidence": 0.0-1.0}}]

"value" must read as a standalone note, e.g. "Is vegetarian", "Targets 140g
protein per day". Keep it under 12 words."""


def extract_sync(user_id: str, user_msg: str, assistant_msg: str) -> list[dict]:
    """Run the extractor and write what it finds. Returns what changed."""
    from .llm import quick_json

    known = db.get_memories(user_id, kind="fact")
    known_str = "\n".join(f"- {k['key']}: {k['value']}" for k in known) or "(nothing yet)"
    keys_str = "\n".join(f"- {k}: {v}" for k, v in FACT_KEYS.items())

    try:
        data = quick_json(
            _EXTRACT_PROMPT.format(
                keys=keys_str, known=known_str,
                user_msg=user_msg[:1500], assistant_msg=assistant_msg[:800],
            )
        )
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    written = []
    for cand in data:
        if not isinstance(cand, dict):
            continue
        key, value = cand.get("key"), cand.get("value")
        conf = float(cand.get("confidence", 0.7))
        # Namespace enforcement plus a confidence floor: a guess is not a memory.
        if key not in FACT_KEYS or not value or conf < 0.55:
            continue
        if db.upsert_memory(user_id, key, str(value)[:120], kind="fact", confidence=conf):
            written.append({"key": key, "value": value})
    return written


_threads: list[threading.Thread] = []


def extract_async(user_id: str, user_msg: str, assistant_msg: str) -> None:
    """Fire and forget. The user has already been answered by the time this runs."""
    t = threading.Thread(
        target=extract_sync, args=(user_id, user_msg, assistant_msg), daemon=True
    )
    t.start()
    _threads.append(t)


def drain(timeout: float = 12.0) -> None:
    """Wait for outstanding writes. Used by evals and at clean shutdown only."""
    for t in list(_threads):
        t.join(timeout=timeout)
    _threads.clear()
