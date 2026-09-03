"""Terminal chat. Deliberately thin -- the interesting code is elsewhere.

An image is attached by putting its path in the message. A bare path logs the
photo; a path plus text is one combined turn:

    /Users/me/plate.jpg half of this was my brother's
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rich.console import Console

from . import db, memory

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".gif", ".bmp"}
console = Console()

BANNER = """[bold green]CalorAI[/] -- text what you ate.
Attach a photo by including its path in the message.
Commands: [cyan]/today[/] [cyan]/memory[/] [cyan]/meals[/] [cyan]/user <id>[/] [cyan]/quit[/]
"""


def split_image(text: str) -> tuple[str, str | None]:
    """Pull an image path out of a message, leaving the caption behind."""
    for raw in text.split():
        token = raw.strip("\"'")
        if Path(token).suffix.lower() in IMAGE_EXT:
            p = Path(token).expanduser()
            if p.exists():
                return text.replace(raw, "").strip(), str(p)
    return text, None


def handle_command(cmd: str, user_id: str) -> tuple[bool, str]:
    """Returns (handled, possibly-new-user-id)."""
    parts = cmd.split()
    head = parts[0]

    if head in ("/quit", "/exit"):
        memory.drain(timeout=3)
        console.print("[dim]bye[/]")
        sys.exit(0)

    if head == "/today":
        t = db.daily_totals(user_id)
        console.print(
            f"[bold]{t['kcal']:.0f} kcal[/] | {t['protein_g']:.0f}g protein | "
            f"{t['carbs_g']:.0f}g carbs | {t['fat_g']:.0f}g fat | {t['meals']} meals"
        )
        return True, user_id

    if head == "/memory":
        memory.drain(timeout=5)
        rows = db.get_memories(user_id)
        if not rows:
            console.print("[dim]nothing remembered yet[/]")
        for r in rows:
            console.print(f"[cyan]{r['key']}[/]: {r['value']}"
                          + (f"  [dim]{r['payload']}[/]" if r["payload"] else ""))
        return True, user_id

    if head == "/meals":
        for m in db.get_meals(user_id):
            items = ", ".join(f"{i['quantity']:g} {i['unit']} {i['name']} ({i['kcal']:.0f} kcal)"
                              for i in m["items"])
            console.print(f"[dim]{m['meal_type']}[/] {items}")
        return True, user_id

    if head == "/user" and len(parts) > 1:
        console.print(f"[dim]switched to user {parts[1]}[/]")
        return True, parts[1]

    console.print("[red]unknown command[/]")
    return True, user_id


def main() -> None:
    ap = argparse.ArgumentParser(description="CalorAI conversational meal logger")
    ap.add_argument("--user", default="default", help="user id (session isolation)")
    ap.add_argument("--no-stream", action="store_true", help="wait for the full reply")
    ap.add_argument("--timing", action="store_true", help="print per-turn latency")
    args = ap.parse_args()

    from .agent import chat, chat_stream  # imported late so --help works without a key

    user_id = args.user
    console.print(BANNER)
    # Always show which provider is actually serving, so a silent failover or a
    # single-key run is never mistaken for the intended configuration.
    from .llm import active_providers
    console.print(f"[dim]models: {active_providers()}[/]\n")

    while True:
        try:
            raw = console.input("[bold blue]you[/] > ").strip()
        except (EOFError, KeyboardInterrupt):
            memory.drain(timeout=3)
            console.print("\n[dim]bye[/]")
            return
        if not raw:
            continue
        if raw.startswith("/"):
            _, user_id = handle_command(raw, user_id)
            continue

        text, image = split_image(raw)
        if image:
            console.print(f"[dim]analysing {Path(image).name}...[/]")

        start = time.perf_counter()
        try:
            if args.no_stream:
                console.print(f"[bold green]calorai[/] > {chat(user_id, text, image)}")
            else:
                console.print("[bold green]calorai[/] > ", end="")
                for tok in chat_stream(user_id, text, image):
                    console.print(tok, end="")
                console.print()
        except Exception as exc:
            console.print(f"[red]error:[/] {exc}")
            continue

        if args.timing:
            console.print(f"[dim]{time.perf_counter() - start:.2f}s[/]")


if __name__ == "__main__":
    main()
