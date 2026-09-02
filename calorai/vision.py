"""Image path: plate photo -> structured food items.

Handoff contract (the important bit)
------------------------------------
The vision model identifies FOOD AND PORTION ONLY. It is never asked for
calories or macros. Those come from the same nutrition table the text path
uses, keyed on the food name the vision model returns.

Two reasons. First, a photo logged as "2 rotis" and a text message logged as
"2 rotis" then produce byte-identical numbers, so a user's daily total does not
depend on how they happened to report the meal. Second, vision models are
noticeably better at "that is a paratha" than at "that is 210 kcal", and asking
them for the second invites confident, unverifiable numbers into the database.

Ambiguity and error handling
----------------------------
The model returns a per-item confidence and an overall confidence. Three outcomes:

* confident, items found        -> hand to the agent to log
* low confidence, or a plate it
  cannot read, or no food at all -> `needs_clarification` with a specific
                                    question; the agent asks instead of logging
* nothing parseable / API error -> a clean failure the agent turns into "I
                                   couldn't read that photo, what was it?"

The agent never silently logs something the vision model was unsure about,
because a wrong meal in the database is worse than one extra question.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import HumanMessage

from .config import VISION_MIN_CONFIDENCE
from .llm import parse_json, vision_llm

VISION_PROMPT = """You are looking at a photo of food. Identify what is on the plate.

Rules:
- Name foods in plain, common terms a person would use ("paratha", "dal",
  "rice", "chicken curry"). Prefer the specific name when you are sure of it
  ("aloo paratha"), the general one when you are not ("paratha").
- Estimate quantity in natural counting units: piece, cup, bowl, plate, slice,
  glass, serving.
- Do NOT estimate calories or macros. Only identify food and amount.
- If the image is blurry, not food, or you genuinely cannot tell what a dish
  is, say so rather than guessing.

Reply with ONLY this JSON, no prose and no markdown fence:
{
  "items": [
    {"name": "...", "quantity": 1.0, "unit": "piece", "confidence": 0.0-1.0}
  ],
  "overall_confidence": 0.0-1.0,
  "is_food": true|false,
  "question": "one short clarifying question, or empty string if none needed"
}

Set "question" when a reasonable person would need to ask -- an unidentifiable
dish, a portion you cannot judge, or a plate that might be shared."""


@dataclass
class VisionResult:
    items: list[dict] = field(default_factory=list)
    overall_confidence: float = 0.0
    is_food: bool = True
    question: str = ""
    error: str = ""

    @property
    def needs_clarification(self) -> bool:
        return bool(
            self.error
            or not self.is_food
            or not self.items
            or self.overall_confidence < VISION_MIN_CONFIDENCE
        )

    def summary(self) -> str:
        if not self.items:
            return "nothing identifiable"
        return ", ".join(
            f"{it.get('quantity', 1):g} {it.get('unit', 'serving')} {it['name']}"
            for it in self.items
        )


def _encode(path: str | Path) -> tuple[str, str]:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"No such image: {p}")
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return mime, base64.b64encode(p.read_bytes()).decode()


def analyse_image(image_path: str | Path, caption: str = "") -> VisionResult:
    """Single vision call. No tools, no retries -- this is on the user's clock.

    `caption` is passed through as context only. The caption is NOT applied
    here; modifiers like "half of this was my brother's" are resolved by the
    text model after this returns, so that a photo plus a caption produces one
    meal rather than two independent interpretations of the same plate.
    """
    try:
        mime, b64 = _encode(image_path)
    except Exception as exc:
        return VisionResult(error=str(exc))

    prompt = VISION_PROMPT
    if caption:
        prompt += (
            f'\n\nThe user sent this photo with the caption: "{caption}".\n'
            "Use it only to help identify the food. Do not apply portion "
            "adjustments from the caption -- report what is physically on the plate."
        )

    msg = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
        ]
    )
    try:
        resp = vision_llm().invoke([msg])
    except Exception as exc:
        return VisionResult(error=f"vision model call failed: {exc}")

    data = parse_json(resp.content if isinstance(resp.content, str) else str(resp.content))
    if not isinstance(data, dict):
        return VisionResult(error="vision model returned unparseable output")

    items = []
    for it in data.get("items") or []:
        if not isinstance(it, dict) or not it.get("name"):
            continue
        try:
            qty = float(it.get("quantity", 1) or 1)
        except (TypeError, ValueError):
            qty = 1.0
        items.append(
            {
                "name": str(it["name"]),
                "quantity": qty,
                "unit": str(it.get("unit", "serving")),
                "confidence": float(it.get("confidence", 0.6) or 0.6),
            }
        )

    return VisionResult(
        items=items,
        overall_confidence=float(data.get("overall_confidence", 0.0) or 0.0),
        is_food=bool(data.get("is_food", True)),
        question=str(data.get("question", "") or ""),
    )
