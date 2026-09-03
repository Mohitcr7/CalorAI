"""The LangGraph agent.

Graph shape is deliberately small:

    perceive -> assistant -> tools -> assistant -> ...

`perceive` is the image branch. It runs the vision model when the turn carries a
photo and rewrites the user's message into something the text model can act on
in one shot. Everything after that is the standard tool-calling loop.

Why the vision result is folded into the text turn rather than being its own
agent: a photo with the caption "half of this was my brother's" has to become
ONE meal. If the image and the caption are handled by two independent passes you
get two meals, or one meal plus a correction, and the day's totals are wrong or
the user gets two replies. Vision produces a description; the text model does
all the reasoning and makes exactly one tool call.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from . import db, memory, tools, vision
from .config import PORTION_AMBIGUITY_THRESHOLD
from .llm import have_anthropic, text_llm_with_tools
from .prompts import build_system_blocks, build_system_prompt


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_id: str
    image_path: str | None
    vision_note: str


MAX_TOOL_LOOPS = 4


def _perceive(state: AgentState) -> dict:
    """Image branch. Skipped entirely -- zero cost -- for text-only turns."""
    path = state.get("image_path")
    if not path:
        return {"vision_note": ""}

    caption = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage) and isinstance(m.content, str):
            caption = m.content
            break

    result = vision.analyse_image(path, caption=caption)

    if result.error:
        note = (
            f"[VISION] Could not read the image ({result.error}). Ask the user "
            "what was in the photo. Do not log anything."
        )
    elif not result.is_food:
        note = "[VISION] That image does not appear to be food. Say so briefly. Do not log."
    elif result.needs_clarification:
        q = result.question or "what the dish was"
        note = (
            f"[VISION] Uncertain read (confidence {result.overall_confidence:.2f}): "
            f"{result.summary() or 'nothing identifiable'}. Ask one short question about {q}. "
            "Do not log until they answer."
        )
    else:
        note = (
            f"[VISION] The photo shows: {result.summary()} "
            f"(confidence {result.overall_confidence:.2f}). "
            "Log this as ONE meal with a single log_meal call. "
            "If the user's message adjusts the portion (e.g. 'half of this was my "
            "brother's', 'only ate two thirds'), apply that adjustment to the "
            "quantities INSIDE that same log_meal call -- do not log first and "
            "correct afterwards."
        )
    return {"vision_note": note}


def _assistant(state: AgentState) -> dict:
    uid = state["user_id"]

    extra = ""
    if state.get("vision_note"):
        extra = "\n\n" + state["vision_note"] + (
            f"\n(Ambiguity rule: only ask if a portion guess would change the "
            f"meal's calories by more than {int(PORTION_AMBIGUITY_THRESHOLD * 100)}%.)"
        )

    # Block form on Anthropic so the static prefix and tool schemas are cached;
    # flat string on the Gemini failover, which has no equivalent knob. The
    # content is identical either way.
    if have_anthropic():
        blocks = build_system_blocks(uid)
        if extra:
            blocks[-1]["text"] += extra
        system: object = blocks
    else:
        system = build_system_prompt(uid) + extra

    llm = text_llm_with_tools(tools.ALL_TOOLS)
    resp = llm.invoke([SystemMessage(content=system), *state["messages"]])
    return {"messages": [resp]}


def _route(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        # Hard stop on runaway loops; a stuck agent is worse than a short answer.
        if sum(1 for m in state["messages"] if isinstance(m, AIMessage) and m.tool_calls) > MAX_TOOL_LOOPS:
            return END
        return "tools"
    return END


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("perceive", _perceive)
    g.add_node("assistant", _assistant)
    g.add_node("tools", ToolNode(tools.ALL_TOOLS))
    g.add_edge(START, "perceive")
    g.add_edge("perceive", "assistant")
    g.add_conditional_edges("assistant", _route, {"tools": "tools", END: END})
    g.add_edge("tools", "assistant")
    return g.compile()


_GRAPH = None


def graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def _text_of(content) -> str:
    """Pull plain text out of a message's content.

    Anthropic returns a LIST of typed blocks (text blocks interleaved with
    tool_use blocks); Gemini returns a plain string. Assuming the string form --
    which is what a Gemini-only implementation naturally does -- silently yields
    nothing on Anthropic, so streaming looks like it works and prints an empty
    reply. Handle both shapes.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for block in content:
            if isinstance(block, str):
                out.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                out.append(block.get("text", ""))
        return "".join(out)
    return ""


def _history(user_id: str, limit: int = 8) -> list[BaseMessage]:
    """Rehydrate the thread from SQLite so a fresh process continues mid-conversation."""
    out: list[BaseMessage] = []
    for m in db.recent_messages(user_id, limit=limit):
        out.append(HumanMessage(content=m["content"]) if m["role"] == "user"
                   else AIMessage(content=m["content"]))
    return out


def _turn_state(user_id: str, text: str, image_path: str | None) -> AgentState:
    tools.CURRENT_USER.set(user_id)
    user_text = text or ("[sent a photo of their food]" if image_path else "")
    return {
        "messages": [*_history(user_id), HumanMessage(content=user_text)],
        "user_id": user_id,
        "image_path": image_path,
        "vision_note": "",
    }


def _finish(user_id: str, user_text: str, reply: str, image_path: str | None) -> None:
    """Persist the turn, then kick memory extraction off the critical path."""
    logged = user_text or f"[photo: {image_path}]"
    db.append_message(user_id, "user", logged)
    db.append_message(user_id, "assistant", reply)
    memory.extract_async(user_id, logged, reply)


def chat(user_id: str, text: str, image_path: str | None = None) -> str:
    """Blocking single turn. Used by the eval harness and the latency benchmark."""
    state = _turn_state(user_id, text, image_path)
    result = graph().invoke(state)
    reply = ""
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            text = _text_of(m.content).strip()
            if text:
                reply = text
                break
    _finish(user_id, text, reply, image_path)
    return reply


def chat_stream(user_id: str, text: str, image_path: str | None = None):
    """Token-by-token generator. The CLI uses this so the user sees words as
    soon as the model produces them rather than after the whole turn."""
    state = _turn_state(user_id, text, image_path)
    emitted: list[str] = []
    for chunk, meta in graph().stream(state, stream_mode="messages"):
        if meta.get("langgraph_node") != "assistant":
            continue
        if not isinstance(chunk, AIMessage):
            continue
        text = _text_of(chunk.content)
        if text:
            emitted.append(text)
            yield text
    _finish(user_id, text, "".join(emitted).strip(), image_path)
