"""The agent's tool surface.

How the surface is split, and why
---------------------------------
Tools are divided by SIDE EFFECT, not by topic. The model's hardest job is not
picking between "log" and "lookup"; it is not corrupting the day's totals. So
every operation that mutates a logged meal is its own narrow tool with a name
that says exactly what it does to the data:

    log_meal            creates a new meal
    correct_item        changes a quantity on an existing item   <- never creates
    scale_last_meal     multiplies an existing meal
    delete_last_meal    soft-deletes an existing meal

That separation is the whole reason "actually that was 3 rotis not 2" cannot
double-count. A single fat `update_meal` tool with an ambiguous signature would
be one bad model decision away from logging five rotis, and no amount of prompt
wording reliably prevents that. Making the wrong action *unrepresentable* is
cheaper than making it unlikely.

Reads are separate and cheap (`today_summary`, `meals_on_day`,
`lookup_nutrition`) so the model can check state before mutating it.

Memory gets two tools, matching the two tiers in memory.py: `recall_memory`
pulls a stored shortcut on demand, `remember` writes a fact immediately when
the user states one outright. The background extractor still runs, but the
explicit tool exists because "i'm vegetarian btw" should be acknowledged in the
*same* turn, not the next one.

`user_id` is not a tool argument. It comes from a contextvar set per turn, so
the model cannot accidentally read or write another user's data.
"""

from __future__ import annotations

from contextvars import ContextVar

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from . import db, memory, nutrition

CURRENT_USER: ContextVar[str] = ContextVar("current_user", default="default")


def _uid() -> str:
    return CURRENT_USER.get()


def _authoritative_total(uid: str) -> str:
    """Post-mutation totals, worded to win against the copy in the system prompt.

    Both the system prompt and this string carry the day's totals, and they must
    never disagree. The prompt's copy is built before the tool runs, so after a
    mutation it is stale by one change. Rather than remove it -- it is what lets
    "how am I doing?" answer with zero tool calls -- this string states
    precedence explicitly and sits immediately before generation, where a small
    model actually reads it. Observed failure without this: the model invents a
    plausible number instead of quoting either one.
    """
    t = db.daily_totals(uid)
    return (
        f" DAY TOTAL AFTER THIS CHANGE (use exactly these numbers, they replace "
        f"any totals given earlier in your instructions): {t['kcal']:.0f} kcal, "
        f"{t['protein_g']:.0f}g protein, {t['carbs_g']:.0f}g carbs, {t['fat_g']:.0f}g fat."
    )


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------

class FoodItem(BaseModel):
    name: str = Field(description="Plain food name, e.g. 'paratha', 'chai', 'dal'")
    quantity: float = Field(default=1.0, description="How many units. Fractions are fine (0.5, 0.67)")
    unit: str = Field(default="", description="piece/cup/plate/glass/bowl/slice. Leave blank to use the standard unit")


class LogMealArgs(BaseModel):
    items: list[FoodItem]
    meal_type: str = Field(default="unknown", description="breakfast, lunch, dinner, snack, or unknown")
    note: str = Field(default="", description="Anything worth keeping about this meal, e.g. 'shared, ate half'")


# --------------------------------------------------------------------------
# write tools
# --------------------------------------------------------------------------

@tool(args_schema=LogMealArgs)
def log_meal(items: list[FoodItem], meal_type: str = "unknown", note: str = "") -> str:
    """Log a NEW meal the user just ate. Do not use this to fix a meal already
    logged -- use correct_item, scale_last_meal or delete_last_meal for that."""
    uid = _uid()
    resolved, lines = [], []
    for it in items:
        info = nutrition.resolve(it.name)
        qty = float(it.quantity or 1)
        resolved.append(
            {
                "name": info["name"],
                "quantity": qty,
                "unit": it.unit or info["unit"],
                "kcal_per_unit": info["kcal_per_unit"],
                "protein_per_unit": info["protein_per_unit"],
                "carbs_per_unit": info["carbs_per_unit"],
                "fat_per_unit": info["fat_per_unit"],
            }
        )
        lines.append(f"{qty:g} {it.unit or info['unit']} {info['name']} = {qty * info['kcal_per_unit']:.0f} kcal")

    if not resolved:
        return "Nothing to log -- no foods were given."

    db.insert_meal(uid, resolved, meal_type=meal_type, source="text", note=note)
    return "Logged: " + "; ".join(lines) + "." + _authoritative_total(uid)


@tool
def correct_item(food: str, new_quantity: float) -> str:
    """Fix the quantity of something ALREADY logged, e.g. 'that was 3 rotis not 2'.
    Finds the most recent matching item and changes its quantity in place. This
    replaces the old number -- it does not add to it."""
    uid = _uid()
    item = db.find_recent_item(uid, food)
    if not item:
        return f"Couldn't find anything matching '{food}' logged recently."
    old = item["quantity"]
    db.update_item_quantity(item["id"], float(new_quantity))
    return (
        f"Updated {item['name']}: was {old:g}, now {new_quantity:g} {item['unit']}."
        + _authoritative_total(uid)
    )


@tool
def scale_last_meal(factor: float, reason: str = "") -> str:
    """Multiply every item in the most recent meal by a factor. Use for
    'half of that was my brother's' (0.5) or 'I only ate two thirds' (0.67)."""
    uid = _uid()
    meals = db.get_meals(uid)
    if not meals:
        return "No meal logged today to adjust."
    last = meals[-1]
    db.scale_meal(last["meal_id"], float(factor))
    return (
        f"Scaled that meal to {factor:g}x{(' (' + reason + ')') if reason else ''}."
        + _authoritative_total(uid)
    )


@tool
def delete_last_meal() -> str:
    """Remove the most recently logged meal entirely."""
    uid = _uid()
    meals = db.get_meals(uid)
    if not meals:
        return "Nothing logged today to remove."
    db.soft_delete_meal(meals[-1]["meal_id"])
    return "Removed that meal." + _authoritative_total(uid)


# --------------------------------------------------------------------------
# read tools
# --------------------------------------------------------------------------

@tool
def today_summary() -> str:
    """Current running totals for today, plus progress against any stored goal.
    Use this for 'how am I doing', 'how much protein have I had', 'calories today'."""
    uid = _uid()
    t = db.daily_totals(uid)
    out = (
        f"Today: {t['kcal']:.0f} kcal, {t['protein_g']:.0f}g protein, "
        f"{t['carbs_g']:.0f}g carbs, {t['fat_g']:.0f}g fat across {t['meals']} meal(s)."
    )
    for key, label, unit in (
        ("goal.protein_g", "protein", "g"),
        ("goal.calories", "calorie", " kcal"),
    ):
        goal = db.get_memory(uid, key)
        if goal:
            out += f" ({label} target on file: {goal['value']})"
    return out


@tool
def meals_on_day(days_ago: int = 0) -> str:
    """List what the user ate on a given day. days_ago=0 is today, 1 is
    yesterday. Use this for 'same as yesterday' and 'what did I have Monday'."""
    uid = _uid()
    day = db.today_str(-abs(int(days_ago)))
    meals = db.get_meals(uid, day)
    if not meals:
        return f"Nothing logged for {day}."
    parts = []
    for m in meals:
        items = ", ".join(f"{i['quantity']:g} {i['unit']} {i['name']}" for i in m["items"])
        parts.append(f"{m['meal_type']}: {items}")
    t = db.daily_totals(uid, day)
    return f"{day} -- " + " | ".join(parts) + f" (total {t['kcal']:.0f} kcal)"


@tool
def lookup_nutrition(food: str, quantity: float = 1.0) -> str:
    """Check calories and macros for a food WITHOUT logging it. Use when the
    user asks 'how many calories in X' or when you need the number to decide
    whether a portion guess matters."""
    info = nutrition.resolve(food)
    return (
        f"{quantity:g} {info['unit']} {info['name']}: "
        f"{quantity * info['kcal_per_unit']:.0f} kcal, "
        f"{quantity * info['protein_per_unit']:.1f}g protein, "
        f"{quantity * info['carbs_per_unit']:.1f}g carbs, "
        f"{quantity * info['fat_per_unit']:.1f}g fat"
        + ("" if info["veg"] else " (non-vegetarian)")
    )


# --------------------------------------------------------------------------
# memory tools
# --------------------------------------------------------------------------

@tool
def recall_memory(query: str) -> str:
    """Look up something stored about this user -- especially a saved meal
    shortcut like 'my usual'. Call this BEFORE asking the user what a shortcut
    means; they have probably told you already in an earlier session."""
    return memory.recall(_uid(), query)


@tool
def remember(key: str, value: str) -> str:
    """Store a durable fact the user just stated outright, e.g. they are
    vegetarian or targeting 140g of protein. Only for things still true next
    month -- never for what they ate today.

    key must be one of: diet.restriction, diet.allergy, diet.dislikes,
    goal.calories, goal.protein_g, goal.weight, profile.name,
    profile.household, habit.breakfast, habit.schedule, habit.cooking,
    pref.detail"""
    if key not in memory.FACT_KEYS:
        return f"'{key}' is not a valid memory key. Valid keys: {', '.join(memory.FACT_KEYS)}"
    memory.save_fact(_uid(), key, value, confidence=0.95)
    return f"Noted: {value}"


class SaveShortcutArgs(BaseModel):
    phrase: str = Field(description="What the user calls it, e.g. 'my usual'")
    description: str = Field(description="Short human description, e.g. '2 parathas and chai'")
    items: list[FoodItem]


@tool(args_schema=SaveShortcutArgs)
def save_shortcut(phrase: str, description: str, items: list[FoodItem]) -> str:
    """Teach a named meal shortcut so the user can say it again later, e.g.
    after they explain what 'my usual' is. Call this once you know the contents."""
    memory.save_alias(
        _uid(), phrase, description,
        [{"name": i.name, "quantity": i.quantity, "unit": i.unit} for i in items],
    )
    return f"Saved '{phrase}' as {description}."


ALL_TOOLS = [
    log_meal, correct_item, scale_last_meal, delete_last_meal,
    today_summary, meals_on_day, lookup_nutrition,
    recall_memory, remember, save_shortcut,
]
