"""What "correct" means for this agent.

Deliberately NOT judged on the wording of the reply. A meal logger is correct
when the DATABASE is right -- the right foods, the right quantities, the right
running total, and no phantom extra meals. Two graders:

  * db assertions  -- the state after the turn. This is the real bar.
  * reply contains -- used only where the product behaviour IS the reply, e.g.
                      the agent must ASK rather than log.

Each case is a script of turns run against a clean database for one user, so
multi-turn behaviour (corrections, teaching a shortcut, then using it) is
tested as a sequence rather than in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from calorai import db, memory


@dataclass
class Turn:
    text: str = ""
    image: str | None = None
    # (name, fn(user_id) -> bool) assertions on database state after this turn
    expect_db: list[tuple[str, Callable[[str], bool]]] = field(default_factory=list)
    # substrings, any one of which must appear in the reply (lowercased)
    expect_reply_any: list[str] = field(default_factory=list)
    # the reply must NOT contain these
    expect_reply_none: list[str] = field(default_factory=list)


@dataclass
class Case:
    name: str
    why: str                       # what this case is actually testing
    turns: list[Turn]
    setup: Callable[[str], None] | None = None
    needs_image: bool = False


# ---------------------------------------------------------------- helpers

def kcal_between(lo: float, hi: float) -> Callable[[str], bool]:
    return lambda uid: lo <= db.daily_totals(uid)["kcal"] <= hi


def protein_between(lo: float, hi: float) -> Callable[[str], bool]:
    return lambda uid: lo <= db.daily_totals(uid)["protein_g"] <= hi


def meal_count(n: int) -> Callable[[str], bool]:
    return lambda uid: db.daily_totals(uid)["meals"] == n


def logged_nothing(uid: str) -> bool:
    return db.daily_totals(uid)["meals"] == 0


def has_item(fragment: str, qty: float | None = None, tol: float = 0.2) -> Callable[[str], bool]:
    def check(uid: str) -> bool:
        for m in db.get_meals(uid):
            for i in m["items"]:
                if fragment.lower() in i["name"].lower():
                    if qty is None:
                        return True
                    return abs(i["quantity"] - qty) <= tol * max(qty, 1)
        return False
    return check


def memory_has(key: str, fragment: str = "") -> Callable[[str], bool]:
    def check(uid: str) -> bool:
        memory.drain(timeout=10)
        row = db.get_memory(uid, key)
        return bool(row) and fragment.lower() in row["value"].lower()
    return check


def alias_saved(phrase: str) -> Callable[[str], bool]:
    def check(uid: str) -> bool:
        memory.drain(timeout=10)
        return memory.get_alias(uid, phrase) is not None
    return check


def seed_yesterday(uid: str) -> None:
    """Give the user a yesterday, for 'same as yesterday'."""
    db.insert_meal(
        uid,
        [
            {"name": "roti / chapati", "quantity": 3, "unit": "piece",
             "kcal_per_unit": 104, "protein_per_unit": 3.1, "carbs_per_unit": 20, "fat_per_unit": 1.7},
            {"name": "dal (tadka)", "quantity": 1, "unit": "cup",
             "kcal_per_unit": 180, "protein_per_unit": 9, "carbs_per_unit": 24, "fat_per_unit": 5},
        ],
        meal_type="dinner",
        meal_date=db.today_str(-1),
    )


def seed_usual(uid: str) -> None:
    memory.save_alias(
        uid, "my usual", "2 parathas and a chai",
        [{"name": "paratha", "quantity": 2, "unit": "piece"},
         {"name": "chai", "quantity": 1, "unit": "cup"}],
    )


# ---------------------------------------------------------------- cases

CASES: list[Case] = [
    Case(
        name="basic_log",
        why="The bread-and-butter message. Must log both foods, ask nothing.",
        turns=[Turn(
            text="had 2 parathas and chai for breakfast",
            expect_db=[
                ("two items logged as one meal", meal_count(1)),
                ("paratha qty 2", has_item("paratha", 2)),
                ("chai logged", has_item("chai")),
                ("total in range", kcal_between(400, 700)),
            ],
            expect_reply_none=["?"],
        )],
    ),

    Case(
        name="correction_no_double_count",
        why=(
            "The headline case. After 'actually 3 not 2' the total must reflect "
            "THREE parathas, not five, and there must still be exactly one meal."
        ),
        turns=[
            Turn(text="had 2 parathas and chai for breakfast",
                 expect_db=[("logged", kcal_between(400, 700))]),
            Turn(text="actually that was 3 parathas not 2",
                 expect_db=[
                     ("still one meal, not a second", meal_count(1)),
                     ("quantity is 3", has_item("paratha", 3)),
                     ("total updated not doubled", kcal_between(620, 850)),
                 ]),
        ],
    ),

    Case(
        name="fractional_portion",
        why="'two thirds of the box' must become a fractional quantity, not 2 or 3.",
        turns=[Turn(
            text="leftover biryani, maybe two thirds of the box",
            expect_db=[("biryani logged", has_item("biryani")),
                       ("plausible single-portion total", kcal_between(120, 600))],
        )],
    ),

    Case(
        name="underspecified_asks",
        why=(
            "'grazed all afternoon' names no food. The agent must ask rather than "
            "invent a snack. Under-asking here produces garbage data."
        ),
        turns=[Turn(
            text="skipped lunch but grazed all afternoon",
            expect_db=[("nothing invented", logged_nothing)],
            expect_reply_any=["?"],
        )],
    ),

    Case(
        name="totals_query",
        why="Must answer with the real number from the database, not a guess.",
        turns=[
            Turn(text="had 3 rotis and a cup of dal",
                 expect_db=[("logged", kcal_between(400, 600))]),
            Turn(text="how am I doing on calories?",
                 expect_reply_any=["49", "50", "48", "kcal", "calor"],
                 expect_db=[("query did not log anything new", meal_count(1))]),
        ],
    ),

    Case(
        name="protein_query",
        why="Macro-specific question, answered from stored data.",
        turns=[
            Turn(text="2 eggs and a protein shake"),
            Turn(text="how much protein have I had today?",
                 expect_reply_any=["3", "protein"],
                 expect_db=[("no phantom meal", meal_count(1))]),
        ],
    ),

    Case(
        name="memory_write_fact",
        why="A durable fact stated in passing must reach the memory store.",
        turns=[Turn(
            text="i'm vegetarian btw",
            expect_db=[("stored as diet.restriction", memory_has("diet.restriction", "veg"))],
        )],
    ),

    Case(
        name="memory_recall_alias",
        why=(
            "'my usual' set in a PREVIOUS session must resolve without asking. "
            "This is memory, not parsing -- nothing in the message says paratha."
        ),
        setup=seed_usual,
        turns=[Turn(
            text="my usual",
            expect_db=[("expanded the shortcut", has_item("paratha")),
                       ("logged as one meal", meal_count(1))],
            expect_reply_none=["what do you mean", "what is your usual"],
        )],
    ),

    Case(
        name="memory_teach_alias",
        why="When the shortcut is unknown the agent should ask once, then save it.",
        turns=[
            Turn(text="my usual", expect_reply_any=["?"]),
            Turn(text="2 parathas and a chai, same every morning",
                 expect_db=[("logged it", has_item("paratha")),
                            ("and remembered it for next time", alias_saved("my usual"))]),
        ],
    ),

    Case(
        name="same_as_yesterday",
        why=(
            "Episodic recall. Answered by querying the meals table, not by the "
            "model remembering -- must reproduce yesterday's actual foods."
        ),
        setup=seed_yesterday,
        turns=[Turn(
            text="same as yesterday",
            expect_db=[("rotis copied", has_item("roti", 3)),
                       ("dal copied", has_item("dal")),
                       ("today has one meal", meal_count(1))],
        )],
    ),

    # The image cases assume samples/plate.jpg: an Indian thali holding a
    # chapati, a bowl of curd, a dark curry, a dry sabzi and two laddus.
    #
    # Asserting only meal_count(1) here was a mistake worth recording: it passed
    # while the vision model was reporting "idli, dosa, sambar, chutney" for that
    # plate. One meal was logged, so the eval went green on a completely wrong
    # read. These now check that something bread-like actually came back, which
    # is the one item every correct read of this photo contains.
    Case(
        name="image_only",
        why="Photo path end to end: vision identifies the real plate, text model logs once.",
        needs_image=True,
        turns=[Turn(image="__IMAGE__",
                    expect_db=[
                        ("logged exactly one meal", meal_count(1)),
                        ("recognised the flatbread on the plate",
                         lambda uid: any(
                             any(k in i["name"].lower()
                                 for k in ("roti", "chapati", "paratha", "flatbread", "bread"))
                             for m in db.get_meals(uid) for i in m["items"])),
                        ("plausible thali total", kcal_between(300, 1600)),
                    ])],
    ),

    Case(
        name="image_with_caption",
        why=(
            "Both models must resolve to ONE meal. The caption halves the portion, "
            "so the result must be a single halved meal -- not two meals, not the "
            "full plate, and not a meal followed by a correction."
        ),
        needs_image=True,
        turns=[Turn(text="half of this was my brother's", image="__IMAGE__",
                    expect_db=[
                        ("exactly one meal", meal_count(1)),
                        # A full read of this plate lands around 800-900 kcal, so
                        # anything at or above that means the caption was ignored.
                        ("caption actually halved the portions", kcal_between(80, 700)),
                    ])],
    ),

    Case(
        name="veg_awareness",
        why="A stored restriction should visibly change behaviour, not just sit in a table.",
        setup=lambda uid: memory.save_fact(uid, "diet.restriction", "Is vegetarian"),
        turns=[Turn(text="had chicken biryani for lunch",
                    expect_reply_any=["veg", "chicken", "?"])],
    ),

    Case(
        name="session_isolation",
        why=(
            "Two users share one database. One user's meals and shortcuts must "
            "never leak into another's totals or answers. Tested here rather than "
            "by hand because it is the kind of thing that breaks silently."
        ),
        setup=lambda uid: (
            db.insert_meal(
                "other_user_dont_leak",
                [{"name": "chicken biryani", "quantity": 4, "unit": "cup",
                  "kcal_per_unit": 350, "protein_per_unit": 17,
                  "carbs_per_unit": 42, "fat_per_unit": 13}],
                meal_type="lunch",
            ),
            memory.save_alias(
                "other_user_dont_leak", "my usual", "4 cups of chicken biryani",
                [{"name": "chicken biryani", "quantity": 4, "unit": "cup"}],
            ),
        ),
        turns=[
            Turn(text="had 2 rotis",
                 expect_db=[
                     ("only this user's meal counted", meal_count(1)),
                     ("the other user's 1400 kcal did not leak in",
                      kcal_between(150, 300)),
                 ]),
            Turn(text="my usual",
                 expect_db=[("did not inherit the other user's shortcut",
                             lambda uid: not any(
                                 "biryani" in i["name"].lower()
                                 for m in db.get_meals(uid) for i in m["items"]))],
                 expect_reply_any=["?"]),
        ],
    ),

    Case(
        name="delete_meal",
        why="Deletions must pull the totals back down, not leave orphaned calories.",
        turns=[
            Turn(text="had 2 samosas", expect_db=[("logged", kcal_between(400, 600))]),
            Turn(text="actually ignore that, I didn't eat them",
                 expect_db=[("back to zero", kcal_between(0, 1))]),
        ],
    ),
]
