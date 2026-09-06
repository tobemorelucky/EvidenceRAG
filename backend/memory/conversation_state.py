"""Serializable state for the Conversation-aware RAG v1 shadow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


VALID_ROLES = {"user", "assistant"}


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"Unsupported conversation role: {self.role}")
        if not self.content.strip():
            raise ValueError("Conversation content must not be empty")


@dataclass
class ConversationState:
    conversation_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.conversation_id.strip():
            raise ValueError("conversation_id must not be empty")

    @property
    def is_first_user_turn(self) -> bool:
        return not any(turn.role == "user" for turn in self.turns)

    def append(self, role: str, content: str, evidence_refs: list[str] | tuple[str, ...] = ()) -> ConversationTurn:
        turn = ConversationTurn(role=role, content=content, evidence_refs=tuple(evidence_refs))
        self.turns.append(turn)
        return turn

    def recent(self, limit: int = 12) -> list[ConversationTurn]:
        if limit <= 0:
            return []
        return self.turns[-limit:]

    def to_dict(self) -> dict[str, Any]:
        return {"conversation_id": self.conversation_id, "turns": [asdict(turn) for turn in self.turns], "metadata": dict(self.metadata)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConversationState":
        return cls(
            conversation_id=str(payload["conversation_id"]),
            turns=[ConversationTurn(**turn) for turn in payload.get("turns", [])],
            metadata=dict(payload.get("metadata") or {}),
        )
