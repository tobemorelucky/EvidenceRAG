"""Write-through coordination for Conversation-aware RAG v5 memory."""

from __future__ import annotations

from typing import Any

from .conversation_state import ConversationState
from .persistent_memory_store import PersistentMemoryStore
from .redis_memory_store import RedisMemoryStore


class MemoryUpdateService:
    """Keep PostgreSQL authoritative and Redis disposable."""

    def __init__(
        self,
        persistent: PersistentMemoryStore | None = None,
        redis_store: RedisMemoryStore | None = None,
        *,
        recent_message_limit: int = 24,
    ):
        self.persistent = persistent or PersistentMemoryStore()
        self.redis = redis_store or RedisMemoryStore()
        self.recent_message_limit = max(2, recent_message_limit)

    def _cache(self, operation, *args) -> bool:
        try:
            operation(*args)
            return True
        except Exception:
            return False

    def create_conversation(
        self, user_id: str, conversation_id: str | None = None, metadata: dict | None = None,
    ) -> tuple[dict, dict]:
        conversation = self.persistent.create_conversation(user_id, conversation_id, metadata)
        state = ConversationState(conversation_id=conversation["conversation_id"], metadata=conversation["metadata"])
        cache_written = self._cache(
            self.redis.save_state, user_id, conversation["conversation_id"], state.to_dict(),
        )
        self._cache(self.redis.save_recent_messages, user_id, conversation["conversation_id"], [])
        return conversation, {"postgres_written": True, "redis_written": cache_written}

    def load_state(self, user_id: str, conversation_id: str) -> tuple[ConversationState | None, dict]:
        try:
            cached = self.redis.load_state(user_id, conversation_id)
        except Exception:
            cached = None
        if cached:
            return ConversationState.from_dict(cached), {"source": "redis", "cache_hit": True}
        state = self.persistent.load_state(user_id, conversation_id)
        if state is None:
            return None, {"source": "none", "cache_hit": False}
        cache_written = self._cache(self.redis.save_state, user_id, conversation_id, state.to_dict())
        self._cache(
            self.redis.save_recent_messages,
            user_id,
            conversation_id,
            self.persistent.list_messages(user_id, conversation_id, limit=self.recent_message_limit),
        )
        return state, {"source": "postgres", "cache_hit": False, "redis_written": cache_written}

    def append_message(
        self, user_id: str, conversation_id: str, role: str, content: str,
        *, evidence_refs: list[str] | None = None, trace: dict | None = None,
    ) -> tuple[dict, dict]:
        message = self.persistent.append_message(
            user_id, conversation_id, role, content, evidence_refs=evidence_refs, trace=trace,
        )
        state = self.persistent.load_state(user_id, conversation_id)
        recent = self.persistent.list_messages(user_id, conversation_id, limit=self.recent_message_limit)
        state_cached = bool(state) and self._cache(self.redis.save_state, user_id, conversation_id, state.to_dict())
        messages_cached = self._cache(self.redis.save_recent_messages, user_id, conversation_id, recent)
        return message, {
            "postgres_written": True, "redis_state_written": state_cached,
            "redis_messages_written": messages_cached,
        }

    def save_evidence_state(self, user_id: str, conversation_id: str, evidence: dict) -> dict:
        cached = self._cache(self.redis.save_evidence_state, user_id, conversation_id, evidence)
        return {"redis_written": cached, "evidence_chars": len(str(evidence.get("evidence") or ""))}

    def load_evidence_state(self, user_id: str, conversation_id: str) -> tuple[dict, dict]:
        try:
            value = self.redis.load_evidence_state(user_id, conversation_id) or {}
            return value, {"cache_hit": bool(value)}
        except Exception:
            return {}, {"cache_hit": False, "redis_error": True}

    def save_summary(
        self, user_id: str, conversation_id: str, summary: str,
        *, through_message_id: int | None = None, metadata: dict | None = None,
    ) -> tuple[dict, dict]:
        value = self.persistent.save_summary(
            user_id, conversation_id, summary,
            through_message_id=through_message_id, metadata=metadata,
        )
        state, _ = self.load_state(user_id, conversation_id)
        cache_written = False
        if state:
            state.metadata["latest_summary"] = summary
            cache_written = self._cache(self.redis.save_state, user_id, conversation_id, state.to_dict())
        return value, {"postgres_written": True, "redis_written": cache_written}

    def latest_summary(self, user_id: str, conversation_id: str) -> dict | None:
        return self.persistent.latest_summary(user_id, conversation_id)

    def list_conversations(self, user_id: str) -> list[dict]:
        return self.persistent.list_conversations(user_id)

    def list_messages(self, user_id: str, conversation_id: str) -> list[dict]:
        if self.persistent.get_conversation(user_id, conversation_id) is None:
            raise KeyError("conversation not found")
        return self.persistent.list_messages(user_id, conversation_id)

    def delete_conversation(self, user_id: str, conversation_id: str) -> tuple[bool, dict]:
        deleted = self.persistent.delete_conversation(user_id, conversation_id)
        if not deleted:
            return False, {"postgres_deleted": False, "redis_keys_deleted": 0}
        try:
            redis_keys_deleted = self.redis.delete_conversation(user_id, conversation_id)
            return True, {"postgres_deleted": True, "redis_keys_deleted": redis_keys_deleted}
        except Exception:
            return True, {
                "postgres_deleted": True,
                "redis_keys_deleted": 0,
                "redis_error": True,
            }
