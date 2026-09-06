"""Bounded in-memory store for shadow conversation state."""

from __future__ import annotations

import threading

from .conversation_state import ConversationState


class ContextMemory:
    def __init__(self, max_turns: int = 24):
        if max_turns < 2:
            raise ValueError("max_turns must be at least 2")
        self.max_turns = max_turns
        self._states: dict[str, ConversationState] = {}
        self._lock = threading.RLock()

    def get(self, conversation_id: str) -> ConversationState:
        with self._lock:
            return self._states.setdefault(conversation_id, ConversationState(conversation_id=conversation_id))

    def record_user(self, conversation_id: str, content: str) -> ConversationState:
        return self._record(conversation_id, "user", content)

    def record_assistant(self, conversation_id: str, content: str, evidence_refs: list[str] | None = None) -> ConversationState:
        return self._record(conversation_id, "assistant", content, evidence_refs or [])

    def _record(self, conversation_id: str, role: str, content: str, evidence_refs: list[str] | None = None) -> ConversationState:
        with self._lock:
            state = self.get(conversation_id)
            state.append(role, content, evidence_refs or [])
            if len(state.turns) > self.max_turns:
                state.turns[:] = state.turns[-self.max_turns :]
            return state

    def clear(self, conversation_id: str) -> bool:
        with self._lock:
            return self._states.pop(conversation_id, None) is not None
