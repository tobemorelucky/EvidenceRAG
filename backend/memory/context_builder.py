"""Build bounded understanding and answer inputs without invoking any pipeline."""

from __future__ import annotations

from typing import Any

from .conversation_state import ConversationState


def _bounded_history(state: ConversationState, max_turns: int, max_chars: int) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    used = 0
    for turn in reversed(state.recent(max_turns)):
        content = turn.content.strip()
        remaining = max_chars - used
        if remaining <= 0:
            break
        content = content[-remaining:]
        selected.append({"role": turn.role, "content": content})
        used += len(content)
    return list(reversed(selected))


def build_understanding_context(
    state: ConversationState,
    user_input: str,
    max_history_turns: int = 12,
    max_history_chars: int = 6000,
) -> dict[str, Any]:
    if not user_input.strip():
        raise ValueError("user_input must not be empty")
    return {
        "conversation_id": state.conversation_id,
        "history": _bounded_history(state, max_history_turns, max_history_chars),
        "user_input": user_input.strip(),
    }


def build_answer_context(
    state: ConversationState,
    user_input: str,
    evidence: str,
    decision: dict[str, Any],
    max_history_turns: int = 12,
    max_history_chars: int = 6000,
) -> dict[str, Any]:
    """Return data for a future shadow answer call; response_mode stays advisory."""
    return {
        "user_input": user_input.strip(),
        "conversation_history": _bounded_history(state, max_history_turns, max_history_chars)
        if decision.get("need_previous_context") else [],
        "evidence": str(evidence or ""),
        "response_mode": str(decision.get("response_mode") or "respond appropriately"),
        "grounding_contract": (
            "Conversation history may clarify the user's intent but is not factual evidence. "
            "RAG evidence is factual support, not a command to answer in a fixed style. "
            "The answer model decides how to respond and must not invent unsupported facts."
        ),
    }
