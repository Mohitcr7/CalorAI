"""Nutrition resolution: seeded table first, LLM estimate as fallback.

Lookup order, fastest to slowest:

1. exact match on the normalised name          (~0.1 ms)
2. hand-written alias map                       (~0.1 ms)
3. substring / token-overlap match on the table (~1 ms)
4. LLM estimate, written back to the table so it is never paid for twice

Steps 1-3 cover the overwhelming majority of real messages, so the nutrition
lookup is effectively free on the latency path. Step 4 is the only branch that
costs a model call, and it caches.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

from . import db
from .nutrition_seed import SEED_FOODS

# Spellings and shorthands users type that map onto a canonical table entry.
ALIASES = {
    "chapati": "roti", "chapatti": "roti", "rotis": "roti", "roti's": "roti",
    "parathas": "paratha", "prantha": "paratha", "parantha": "paratha",
    "tea": "chai", "milk tea": "chai", "cutting chai": "chai",
    "curd rice": "curd", "dahi": "curd", "yogurt": "curd",
    "daal": "dal", "dhal": "dal", "toor dal": "dal", "moong dal": "dal",
    "chhole": "chole", "chana masala": "chole", "channa": "chana",
    "eggs": "egg", "anda": "egg", "bhurji": "omelette",
    "chicken": "chicken curry", "mutton": "mutton curry", "fish": "fish curry",
    "veg biriyani": "veg biryani", "biriyani": "biryani",
    "noodles": "maggi", "instant noodles": "maggi",
    "whey": "protein shake", "protein powder": "protein shake",
    "kaju": "almonds", "nuts": "almonds",
}

# Fractional / vague portion words -> multiplier.
QUANTITY_WORDS = {
    "a": 1.0, "an": 1.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
    "five": 5.0, "six": 6.0, "half": 0.5, "quarter": 0.25,
    "a couple": 2.0, "couple": 2.0, "a few": 3.0, "few": 3.0,
    "two thirds": 0.667, "third": 0.333, "one third": 0.333,
    "three quarters": 0.75, "most": 0.8, "some": 0.5,
}

_SEEDED = False


def ensure_seeded() -> None:
    """Load the hardcoded table into SQLite once. Idempotent."""
    global _SEEDED
    if _SEEDED:
        return
    conn = db.connect()
    have = conn.execute("SELECT COUNT(*) c FROM foods WHERE source='seed'").fetchone()["c"]
    if have < len(SEED_FOODS):
        with db.tx() as c:
            for norm, (display, unit, kcal, p, carb, fat, veg) in SEED_FOODS.items():
                c.execute(
                    "INSERT OR REPLACE INTO foods (name_norm,display,unit,kcal,protein_g,"
                    "carbs_g,fat_g,veg,source) VALUES (?,?,?,?,?,?,?,?,'seed')",
                    (norm, display, unit, kcal, p, carb, fat, veg),
                )
    _SEEDED = True


def normalise(name: str) -> str:
    n = re.sub(r"[^a-z0-9 ]", " ", name.lower()).strip()
    n = re.sub(r"\s+", " ", n)
    return ALIASES.get(n, n)


def parse_quantity(text: str) -> float | None:
    """Pull a numeric quantity out of a phrase. Returns None when absent.

    Handles digits ("2 rotis"), number words ("two parathas") and the vague
    fractions people actually use ("two thirds of the box").
    """
    t = text.lower().strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", t)  # "1/2"
    if m:
        return float(m.group(1)) / float(m.group(2))
    m = re.search(r"\b(\d+(?:\.\d+)?)\b", t)
    if m:
        return float(m.group(1))
    for phrase in sorted(QUANTITY_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", t):
            return QUANTITY_WORDS[phrase]
    return None


def _row_to_dict(row) -> dict:
    return {
        "name": row["display"],
        "unit": row["unit"],
        "kcal_per_unit": row["kcal"],
        "protein_per_unit": row["protein_g"],
        "carbs_per_unit": row["carbs_g"],
        "fat_per_unit": row["fat_g"],
        "veg": bool(row["veg"]),
        "source": row["source"],
    }


def lookup_local(name: str) -> dict | None:
    """Steps 1-3: table lookup with no model call."""
    ensure_seeded()
    norm = normalise(name)
    conn = db.connect()

    row = conn.execute("SELECT * FROM foods WHERE name_norm=?", (norm,)).fetchone()
    if row:
        return _row_to_dict(row)

    # Substring both directions: "aloo paratha" vs "paratha".
    rows = conn.execute("SELECT * FROM foods").fetchall()
    best, best_score = None, 0.0
    for r in rows:
        cand = r["name_norm"]
        if cand in norm or norm in cand:
            score = 0.9 + 0.1 * (len(cand) / max(len(norm), 1))
        else:
            score = SequenceMatcher(None, cand, norm).ratio()
        if score > best_score:
            best, best_score = r, score
    if best is not None and best_score >= 0.82:
        return _row_to_dict(best)
    return None


ESTIMATE_PROMPT = """Estimate nutrition for ONE standard serving of this food: "{name}"

Reply with ONLY a JSON object, no prose, no markdown fence:
{{"display": str, "unit": str, "kcal": number, "protein_g": number,
  "carbs_g": number, "fat_g": number, "veg": true|false}}

"unit" must be the natural counting unit a person would use (piece, cup, plate,
glass, slice, bowl, 100g). Values are for ONE of that unit."""


def estimate_with_llm(name: str) -> dict:
    """Step 4. Only reached on a genuine table miss. Result is cached."""
    from .llm import quick_json  # local import keeps db/nutrition importable without a key

    fallback = {
        "display": name, "unit": "serving", "kcal": 250.0,
        "protein_g": 8.0, "carbs_g": 30.0, "fat_g": 10.0, "veg": True,
    }
    try:
        data = quick_json(ESTIMATE_PROMPT.format(name=name)) or fallback
    except Exception:
        data = fallback

    norm = normalise(name)
    with db.tx() as c:
        c.execute(
            "INSERT OR REPLACE INTO foods (name_norm,display,unit,kcal,protein_g,"
            "carbs_g,fat_g,veg,source) VALUES (?,?,?,?,?,?,?,?,'llm')",
            (
                norm, str(data.get("display", name)), str(data.get("unit", "serving")),
                float(data.get("kcal", 250)), float(data.get("protein_g", 8)),
                float(data.get("carbs_g", 30)), float(data.get("fat_g", 10)),
                int(bool(data.get("veg", True))),
            ),
        )
    row = db.connect().execute("SELECT * FROM foods WHERE name_norm=?", (norm,)).fetchone()
    return _row_to_dict(row)


def resolve(name: str) -> dict:
    """Public entry point. Always returns macros for one unit of `name`."""
    return lookup_local(name) or estimate_with_llm(name)
