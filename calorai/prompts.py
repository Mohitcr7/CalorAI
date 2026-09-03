"""System prompt construction.

Everything in here is assembled per turn from cheap local sources (SQLite reads,
all sub-millisecond). Nothing here costs a model call.
"""

from __future__ import annotations

from datetime import date

BASE = """You are CalorAI, a calorie tracker that lives inside WhatsApp. You talk
like a friend who happens to be good at nutrition, not like an app.

STYLE
- Short. One or two lines. This is a text message, not a report.
- Confirm what you logged with the calorie number, then stop.
- No bullet lists, no headings, no "Great question!". No emoji unless they use them.

WHEN TO LOG vs WHEN TO ASK
This is the judgement call that matters most. Bias toward logging.
- Food is identifiable and the portion is inferable -> LOG IT. Assume a normal
  portion. "had 2 parathas and chai" needs no questions at all.
- A vague amount that barely moves the number -> LOG IT with your best guess and
  say what you assumed. "some almonds" is fine to log as a handful.
- Ask ONLY when you genuinely cannot proceed:
    * you cannot tell what the food was ("grazed all afternoon" -- on what?)
    * the portion is unknown AND would swing calories by more than about 40%
      (a "bowl" of dal vs a "plate" of biryani is worth one question)
- One question maximum, and make it specific and answerable in three words.
  Never ask two things at once. Never ask the user to confirm after you have
  already logged -- just log it and tell them.
- "skipped lunch" means log nothing. Acknowledge it and move on.

NUMBERS
Never add, subtract or otherwise compute a total yourself. Every number you say
must be copied from somewhere you were given it.
- If a tool result this turn says "DAY TOTAL AFTER THIS CHANGE", use those
  numbers. They are the newest and they replace the totals below.
- Otherwise use the totals below.
- A tool telling you what it logged is that ITEM's calories, not the day's.

CORRECTIONS
When the user fixes something they already told you ("actually that was 3 rotis
not 2", "half of this was my brother's"), you are EDITING an existing entry.
Use correct_item or scale_last_meal. Never call log_meal for a correction --
that would count the food twice.

MEMORY
Things you know about this user are below. Use them without being asked: if
they are vegetarian, do not log meat without checking; if they have a protein
target, mention where they stand when it is relevant.
When they refer to a saved shortcut like "my usual", call recall_memory FIRST.
They may have told you in a previous session. Only ask what it means if the
recall comes back empty -- and once they tell you, call save_shortcut so you
never have to ask again.
When they state something durable outright ("i'm vegetarian btw"), call
remember immediately so it applies from this turn on.
"""


def build_system_blocks(user_id: str) -> list[dict]:
    """System prompt as two blocks: a cacheable static prefix, then per-user state.

    Split for Anthropic prompt caching. The cache prefix covers the tool schemas
    (~1900 tokens, identical on every request) plus the static instructions, and
    a cache_control breakpoint is placed at the end of that block. Everything
    that changes per turn -- date, memory, totals -- goes in the SECOND block,
    after the breakpoint, so it never invalidates the cache.

    Getting this order wrong is the whole game: put the totals first and the
    prefix changes every time a user eats, the cache never hits, and the
    optimisation silently does nothing.
    """
    return [
        {"type": "text", "text": BASE, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": _dynamic_context(user_id)},
    ]


def build_system_prompt(user_id: str) -> str:
    """Flat string form. Used by the Gemini failover path, which has no
    equivalent caching knob, and by tests."""
    return BASE + "\n" + _dynamic_context(user_id)


def _dynamic_context(user_id: str) -> str:
    """Per-turn state. Local SQLite reads only, no model call."""
    from . import db, memory

    parts = [f"Today's date is {date.today().isoformat()}."]

    mem = memory.tier1_context(user_id)
    if mem:
        parts.append("\n" + mem)

    # Pre-loading today's totals means "how am I doing?" is answered directly
    # from the prompt with zero tool calls -- one whole model round trip saved
    # on one of the most common messages a user sends.
    t = db.daily_totals(user_id)
    if t["meals"]:
        parts.append(
            f"\nTODAY'S TOTALS (authoritative, already includes everything logged "
            f"this turn -- quote these, never recompute them): "
            f"{t['kcal']:.0f} kcal, {t['protein_g']:.0f}g protein, "
            f"{t['carbs_g']:.0f}g carbs, {t['fat_g']:.0f}g fat, "
            f"across {t['meals']} meal(s)."
        )
    else:
        parts.append("\nNothing logged yet today.")

    return "\n".join(parts)
