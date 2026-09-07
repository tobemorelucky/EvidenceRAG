"""Redis cache for Conversation-aware RAG v5 memory snapshots."""

from __future__ import annotations

import json
import os
from typing import Any

import redis


class RedisMemoryStore:
    def __init__(self, client=None, *, key_prefix: str | None = None, ttl_seconds: int | None = None):
        self._client = client
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        base_prefix = os.getenv("REDIS_KEY_PREFIX", "evidencerag")
        self.key_prefix = (key_prefix or f"{base_prefix}:conversation_memory:v5").rstrip(":")
        self.ttl_seconds = ttl_seconds or int(os.getenv("CONVERSATION_MEMORY_REDIS_TTL_SECONDS", "86400"))

    def _get_client(self):
        if self._client is None:
            self._client = redis.Redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _key(self, user_id: str, conversation_id: str, kind: str) -> str:
        if not user_id.strip() or not conversation_id.strip():
            raise ValueError("user_id and conversation_id must not be empty")
        return f"{self.key_prefix}:{user_id}:{conversation_id}:{kind}"

    def _set(self, user_id: str, conversation_id: str, kind: str, value: Any) -> None:
        self._get_client().setex(
            self._key(user_id, conversation_id, kind),
            self.ttl_seconds,
            json.dumps(value, ensure_ascii=False),
        )

    def _get(self, user_id: str, conversation_id: str, kind: str) -> Any | None:
        value = self._get_client().get(self._key(user_id, conversation_id, kind))
        return json.loads(value) if value else None

    def save_state(self, user_id: str, conversation_id: str, state: dict) -> None:
        self._set(user_id, conversation_id, "state", state)

    def load_state(self, user_id: str, conversation_id: str) -> dict | None:
        return self._get(user_id, conversation_id, "state")

    def save_recent_messages(self, user_id: str, conversation_id: str, messages: list[dict]) -> None:
        self._set(user_id, conversation_id, "messages", messages)

    def load_recent_messages(self, user_id: str, conversation_id: str) -> list[dict] | None:
        return self._get(user_id, conversation_id, "messages")

    def save_evidence_state(self, user_id: str, conversation_id: str, evidence: dict) -> None:
        self._set(user_id, conversation_id, "evidence", evidence)

    def load_evidence_state(self, user_id: str, conversation_id: str) -> dict | None:
        return self._get(user_id, conversation_id, "evidence")

    def delete_conversation(self, user_id: str, conversation_id: str) -> int:
        keys = [self._key(user_id, conversation_id, kind) for kind in ("state", "messages", "evidence")]
        return int(self._get_client().delete(*keys))
