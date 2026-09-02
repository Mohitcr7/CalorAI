"""SQLite persistence layer.

Design note -- the one thing this schema is built around:

    Daily totals are NEVER stored. They are always a SUM() over meal_items at
    read time.

Every other design here follows from that. A meal item stores macros *per unit*
plus a quantity, so "actually that was 3 rotis not 2" is a single UPDATE of
`quantity` and the day's totals are correct on the next read with no
recomputation, no reconciliation, and no possibility of double-counting.
Deletion is a soft delete on the meal row, which the totals query filters out.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable

from .config import DB_PATH

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meals (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    meal_date   TEXT NOT NULL,           -- YYYY-MM-DD, the day this counts toward
    logged_at   TEXT NOT NULL,           -- ISO8601 UTC, when we recorded it
    meal_type   TEXT,                    -- breakfast|lunch|dinner|snack|unknown
    source      TEXT NOT NULL,           -- text|image|image+text
    raw_input   TEXT,                    -- what the user actually said
    note        TEXT,                    -- e.g. "skipped lunch", "grazed"
    deleted     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_meals_user_date ON meals(user_id, meal_date, deleted);

CREATE TABLE IF NOT EXISTS meal_items (
    id           TEXT PRIMARY KEY,
    meal_id      TEXT NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
    name         TEXT NOT NULL,
    quantity     REAL NOT NULL DEFAULT 1,
    unit         TEXT NOT NULL DEFAULT 'serving',
    -- macros are PER ONE UNIT; multiply by quantity at read time
    kcal_per_unit    REAL NOT NULL DEFAULT 0,
    protein_per_unit REAL NOT NULL DEFAULT 0,
    carbs_per_unit   REAL NOT NULL DEFAULT 0,
    fat_per_unit     REAL NOT NULL DEFAULT 0,
    confidence   REAL NOT NULL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_items_meal ON meal_items(meal_id);

-- Nutrition cache. Seeded with common foods; misses are filled by the LLM and
-- written back so the same food is never estimated twice.
CREATE TABLE IF NOT EXISTS foods (
    name_norm   TEXT PRIMARY KEY,
    display     TEXT NOT NULL,
    unit        TEXT NOT NULL,
    kcal        REAL NOT NULL,
    protein_g   REAL NOT NULL,
    carbs_g     REAL NOT NULL,
    fat_g       REAL NOT NULL,
    veg         INTEGER NOT NULL DEFAULT 1,
    source      TEXT NOT NULL DEFAULT 'seed'   -- seed|llm
);

-- Long-term memory. UNIQUE(user_id, key) is what stops memory from growing
-- without bound: a new value for a known key is an upsert, not a new row.
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    key         TEXT NOT NULL,
    kind        TEXT NOT NULL,           -- fact|alias
    value       TEXT NOT NULL,
    payload     TEXT,                    -- JSON, used by aliases to store a meal template
    confidence  REAL NOT NULL DEFAULT 0.8,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE(user_id, key)
);
CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id, kind);

-- Audit trail so an overwritten memory is recoverable and explainable.
CREATE TABLE IF NOT EXISTS memory_history (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    key         TEXT NOT NULL,
    old_value   TEXT,
    new_value   TEXT NOT NULL,
    changed_at  TEXT NOT NULL
);

-- Conversation transcript, so a new process resumes mid-thread.
CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_user ON messages(user_id, created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


_conn: sqlite3.Connection | None = None


def connect() -> sqlite3.Connection:
    """Process-wide connection. SQLite handles our concurrency fine at this size."""
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(SCHEMA)
        _conn.commit()
    return _conn


@contextmanager
def tx():
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def reset_db() -> None:
    """Drop everything. Used by the eval harness, never at runtime."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
    if DB_PATH.exists():
        DB_PATH.unlink()
    connect()


# --------------------------------------------------------------------------
# meals
# --------------------------------------------------------------------------

def today_str(offset_days: int = 0) -> str:
    return (date.today() + timedelta(days=offset_days)).isoformat()


def insert_meal(
    user_id: str,
    items: Iterable[dict[str, Any]],
    *,
    meal_type: str = "unknown",
    source: str = "text",
    raw_input: str = "",
    note: str = "",
    meal_date: str | None = None,
) -> str:
    meal_id = new_id()
    with tx() as conn:
        conn.execute(
            "INSERT INTO meals (id,user_id,meal_date,logged_at,meal_type,source,raw_input,note,deleted)"
            " VALUES (?,?,?,?,?,?,?,?,0)",
            (meal_id, user_id, meal_date or today_str(), _now(), meal_type, source, raw_input, note),
        )
        for it in items:
            conn.execute(
                "INSERT INTO meal_items (id,meal_id,name,quantity,unit,kcal_per_unit,"
                "protein_per_unit,carbs_per_unit,fat_per_unit,confidence)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    new_id(),
                    meal_id,
                    it["name"],
                    float(it.get("quantity", 1)),
                    it.get("unit", "serving"),
                    float(it.get("kcal_per_unit", 0)),
                    float(it.get("protein_per_unit", 0)),
                    float(it.get("carbs_per_unit", 0)),
                    float(it.get("fat_per_unit", 0)),
                    float(it.get("confidence", 1.0)),
                ),
            )
    return meal_id


def daily_totals(user_id: str, day: str | None = None) -> dict[str, float]:
    """The single source of truth for 'how am I doing today'."""
    day = day or today_str()
    row = connect().execute(
        """
        SELECT COALESCE(SUM(i.quantity * i.kcal_per_unit), 0)    AS kcal,
               COALESCE(SUM(i.quantity * i.protein_per_unit), 0) AS protein_g,
               COALESCE(SUM(i.quantity * i.carbs_per_unit), 0)   AS carbs_g,
               COALESCE(SUM(i.quantity * i.fat_per_unit), 0)     AS fat_g,
               COUNT(DISTINCT m.id)                              AS meals
        FROM meals m JOIN meal_items i ON i.meal_id = m.id
        WHERE m.user_id = ? AND m.meal_date = ? AND m.deleted = 0
        """,
        (user_id, day),
    ).fetchone()
    return {k: (round(row[k], 1) if k != "meals" else row[k]) for k in row.keys()}


def get_meals(user_id: str, day: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    day = day or today_str()
    meals = connect().execute(
        "SELECT * FROM meals WHERE user_id=? AND meal_date=? AND deleted=0"
        " ORDER BY logged_at LIMIT ?",
        (user_id, day, limit),
    ).fetchall()
    out = []
    for m in meals:
        items = connect().execute(
            "SELECT * FROM meal_items WHERE meal_id=?", (m["id"],)
        ).fetchall()
        out.append(
            {
                "meal_id": m["id"],
                "meal_type": m["meal_type"],
                "logged_at": m["logged_at"],
                "source": m["source"],
                "note": m["note"],
                "items": [
                    {
                        "item_id": i["id"],
                        "name": i["name"],
                        "quantity": i["quantity"],
                        "unit": i["unit"],
                        "kcal": round(i["quantity"] * i["kcal_per_unit"], 1),
                        "protein_g": round(i["quantity"] * i["protein_per_unit"], 1),
                    }
                    for i in items
                ],
            }
        )
    return out


def find_recent_item(user_id: str, name_fragment: str, days_back: int = 1) -> dict | None:
    """Locate the item a correction refers to: most recent fuzzy name match."""
    since = today_str(-days_back)
    row = connect().execute(
        """
        SELECT i.*, m.meal_date FROM meal_items i JOIN meals m ON m.id = i.meal_id
        WHERE m.user_id = ? AND m.deleted = 0 AND m.meal_date >= ?
          AND LOWER(i.name) LIKE ?
        ORDER BY m.logged_at DESC LIMIT 1
        """,
        (user_id, since, f"%{name_fragment.lower().strip()}%"),
    ).fetchone()
    return dict(row) if row else None


def update_item_quantity(item_id: str, quantity: float) -> None:
    with tx() as conn:
        conn.execute("UPDATE meal_items SET quantity=? WHERE id=?", (quantity, item_id))


def scale_meal(meal_id: str, factor: float) -> None:
    """Used by 'half of this was my brother's'."""
    with tx() as conn:
        conn.execute("UPDATE meal_items SET quantity = quantity * ? WHERE meal_id=?", (factor, meal_id))


def soft_delete_meal(meal_id: str) -> None:
    with tx() as conn:
        conn.execute("UPDATE meals SET deleted=1 WHERE id=?", (meal_id,))


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------

def upsert_memory(
    user_id: str, key: str, value: str, *, kind: str = "fact",
    payload: dict | None = None, confidence: float = 0.8,
) -> bool:
    """Returns True if anything actually changed (used to avoid noisy history)."""
    prev = connect().execute(
        "SELECT value FROM memories WHERE user_id=? AND key=?", (user_id, key)
    ).fetchone()
    if prev and prev["value"] == value:
        return False
    now = _now()
    with tx() as conn:
        conn.execute(
            """
            INSERT INTO memories (id,user_id,key,kind,value,payload,confidence,created_at,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id,key) DO UPDATE SET
                value=excluded.value, payload=excluded.payload,
                confidence=excluded.confidence, updated_at=excluded.updated_at
            """,
            (new_id(), user_id, key, kind, value,
             json.dumps(payload) if payload else None, confidence, now, now),
        )
        conn.execute(
            "INSERT INTO memory_history (id,user_id,key,old_value,new_value,changed_at)"
            " VALUES (?,?,?,?,?,?)",
            (new_id(), user_id, key, prev["value"] if prev else None, value, now),
        )
    return True


def get_memories(user_id: str, kind: str | None = None, limit: int = 100) -> list[dict]:
    q = "SELECT * FROM memories WHERE user_id=?"
    args: list[Any] = [user_id]
    if kind:
        q += " AND kind=?"
        args.append(kind)
    q += " ORDER BY updated_at DESC LIMIT ?"
    args.append(limit)
    return [dict(r) for r in connect().execute(q, args).fetchall()]


def get_memory(user_id: str, key: str) -> dict | None:
    r = connect().execute(
        "SELECT * FROM memories WHERE user_id=? AND key=?", (user_id, key)
    ).fetchone()
    return dict(r) if r else None


def delete_memory(user_id: str, key: str) -> None:
    with tx() as conn:
        conn.execute("DELETE FROM memories WHERE user_id=? AND key=?", (user_id, key))


# --------------------------------------------------------------------------
# transcript
# --------------------------------------------------------------------------

def append_message(user_id: str, role: str, content: str) -> None:
    with tx() as conn:
        conn.execute(
            "INSERT INTO messages (id,user_id,role,content,created_at) VALUES (?,?,?,?,?)",
            (new_id(), user_id, role, content, _now()),
        )


def recent_messages(user_id: str, limit: int = 12) -> list[dict]:
    rows = connect().execute(
        "SELECT role, content FROM messages WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    ).fetchall()
    return [dict(r) for r in reversed(rows)]
