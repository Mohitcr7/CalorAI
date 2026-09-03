"""Minimal web UI for demoing and recording the agent.

Deliberately thin -- the brief says not to spend time on frontend, and this is
one HTML file plus four endpoints. It exists because a terminal recording hides
the interesting part: the agent's *state*. Totals, stored memories and which
tools fired are what make the design legible, and none of them are visible in a
chat transcript alone.

    python -m calorai.web          # then open http://127.0.0.1:8000

Endpoints:
    GET  /                 the page
    POST /api/chat         SSE stream of {tool|token|done} events
    POST /api/upload       stash an image, returns a path for /api/chat
    GET  /api/state        totals, meals and memories for the sidebar
    POST /api/reset        clear one user's data (demo convenience)
"""

from __future__ import annotations

import json
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import db, memory
from .llm import active_providers

app = FastAPI(title="CalorAI")

STATIC = Path(__file__).parent / "static"
UPLOADS = Path(tempfile.gettempdir()) / "calorai_uploads"
UPLOADS.mkdir(exist_ok=True)

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text()


@app.get("/api/state")
def state(user: str = "demo") -> JSONResponse:
    """Everything the sidebar shows. Pure SQLite reads, no model calls."""
    totals = db.daily_totals(user)
    meals = db.get_meals(user)
    memories = db.get_memories(user)
    return JSONResponse(
        {
            "providers": active_providers(),
            "totals": totals,
            "meals": [
                {
                    "type": m["meal_type"],
                    "items": [
                        {
                            "name": i["name"],
                            "quantity": i["quantity"],
                            "unit": i["unit"],
                            "kcal": i["kcal"],
                        }
                        for i in m["items"]
                    ],
                }
                for m in meals
            ],
            "memories": [
                {
                    "key": r["key"],
                    "value": r["value"],
                    "kind": r["kind"],
                    "payload": json.loads(r["payload"]) if r["payload"] else None,
                }
                for r in memories
            ],
        }
    )


@app.post("/api/upload")
async def upload(file: UploadFile) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_SUFFIXES:
        return JSONResponse({"error": f"unsupported file type: {suffix or 'none'}"}, status_code=400)
    dest = UPLOADS / f"{int(time.time() * 1000)}{suffix}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    return JSONResponse({"path": str(dest), "name": file.filename})


# How many events may queue up before the producer thread blocks. Small on
# purpose: back-pressure is preferable to buffering a whole reply in memory.
_QUEUE_MAX = 256
_SENTINEL = object()


@app.post("/api/chat")
async def chat(user: str = Form("demo"), text: str = Form(""), image: str = Form("")):
    """Server-sent events so the UI can render tokens and tool calls as they happen.

    The agent turn runs on ONE dedicated thread and pushes events through a
    queue, rather than being iterated directly as a streaming generator.

    That is not incidental. Tool calls read the current user from a ContextVar,
    and FastAPI iterates a sync generator across its threadpool -- successive
    next() calls can land on different worker threads, so the tools execute in a
    context that never saw the value set at the start of the turn and silently
    fall back to the "default" user. Every user's meals then land in one shared
    bucket while the UI still shows a perfectly correct reply. It looks fine
    right up until two people use it.

    Pinning the turn to a single thread keeps the context intact for its whole
    lifetime, which is the property the ContextVar design actually depends on.
    """
    from .agent import chat_stream_events

    def produce(q: queue.Queue) -> None:
        try:
            for event in chat_stream_events(user, text, image or None):
                q.put(event)
        except Exception as exc:  # surfaced in the UI rather than a dead stream
            q.put({"type": "error", "message": str(exc)})
        finally:
            q.put(_SENTINEL)

    def gen():
        start = time.perf_counter()
        q: queue.Queue = queue.Queue(maxsize=_QUEUE_MAX)
        worker = threading.Thread(target=produce, args=(q,), daemon=True)
        worker.start()
        while True:
            event = q.get()
            if event is _SENTINEL:
                break
            if event.get("type") == "done":
                event["elapsed"] = round(time.perf_counter() - start, 2)
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/reset")
def reset(user: str = Form("demo")) -> JSONResponse:
    """Wipe one user so a demo can be re-run from a clean slate."""
    with db.tx() as conn:
        conn.execute("DELETE FROM meals WHERE user_id=?", (user,))
        conn.execute("DELETE FROM memories WHERE user_id=?", (user,))
        conn.execute("DELETE FROM memory_history WHERE user_id=?", (user,))
        conn.execute("DELETE FROM messages WHERE user_id=?", (user,))
    return JSONResponse({"ok": True})


def main() -> None:
    import uvicorn

    db.connect()
    memory  # noqa: B018 -- imported for its side-effect-free availability in state()
    print("CalorAI web UI  →  http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
