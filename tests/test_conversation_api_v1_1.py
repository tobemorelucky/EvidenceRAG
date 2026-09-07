import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import api
from backend.memory.memory_update import MemoryUpdateService
from backend.memory.persistent_memory_store import (
    MemoryConversation,
    MemoryMessage,
    MemorySummary,
    PersistentMemoryStore,
)
from backend.memory.redis_memory_store import RedisMemoryStore
from backend.rag_trace_service import RagTraceRecord, RagTraceService


class FakeRedis:
    def __init__(self):
        self.data = {}

    def setex(self, key, _ttl, value):
        self.data[key] = value

    def get(self, key):
        return self.data.get(key)

    def delete(self, *keys):
        removed = 0
        for key in keys:
            removed += key in self.data
            self.data.pop(key, None)
        return removed


class FakeConversationService:
    def __init__(self):
        self.deleted = False
        self.items = [{
            "conversation_id": "db-conversation-1",
            "title": "Revenue?",
            "created_at": "2026-09-07T00:00:00",
            "updated_at": "2026-09-07T00:00:00",
            "message_count": 2,
            "metadata": {"topic": "finance"},
        }]

    def create(self, user_id, metadata):
        assert user_id == "alice"
        return {
            "conversation_id": "db-conversation-1",
            "created_at": "2026-09-07T00:00:00",
            "memory_trace": {"postgres_written": True},
        }

    def list_conversations(self, user_id):
        assert user_id == "alice"
        return [] if self.deleted else self.items

    def chat(self, user_id, conversation_id, message, **_kwargs):
        assert (user_id, conversation_id) == ("alice", "db-conversation-1")
        return {
            "conversation_id": conversation_id,
            "response": f"Answer: {message}",
            "citations": [{"id": "e1", "filename": "report.pdf", "page_number": 4}],
            "usage": {"total_tokens": 12},
            "trace": {"retrieval_decision": {"called": True}},
        }

    async def stream_chat(self, user_id, conversation_id, message, **_kwargs):
        result = self.chat(user_id, conversation_id, message)
        yield {"type": "status", "stage": "understanding", "label": "正在分析问题...", "detail": ""}
        yield {"type": "status", "stage": "retrieval", "label": "正在检索证据...", "detail": ""}
        yield {"type": "content", "content": "Answer: "}
        yield {"type": "content", "content": message}
        for citation in result["citations"]:
            yield {"type": "citation", "citation": citation}
        yield {"type": "trace", "rag_trace": result["trace"], "citations": result["citations"]}
        yield {"type": "done", "conversation_id": conversation_id, "usage": result["usage"]}

    def messages(self, user_id, conversation_id):
        assert (user_id, conversation_id) == ("alice", "db-conversation-1")
        if self.deleted:
            raise KeyError("conversation not found")
        return [{
            "id": 1,
            "role": "user",
            "content": "Revenue?",
            "evidence_refs": [],
            "trace": None,
            "created_at": "2026-09-07T00:00:01",
        }]

    def delete(self, user_id, conversation_id):
        assert (user_id, conversation_id) == ("alice", "db-conversation-1")
        if self.deleted:
            return False, {"postgres_deleted": False, "redis_keys_deleted": 0}
        self.deleted = True
        return True, {"postgres_deleted": True, "redis_keys_deleted": 3}

    def traces(self, *_args, **_kwargs):
        return []


@pytest.fixture()
def client(monkeypatch):
    service = FakeConversationService()
    monkeypatch.setattr(api, "conversation_memory_v5_enabled", lambda: True)
    monkeypatch.setattr(api, "conversation_memory_v5_service", service)
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(username="alice", role="user")
    with TestClient(app) as test_client:
        yield test_client, service


def test_create_conversation_uses_server_generated_id(client):
    test_client, _ = client
    response = test_client.post(
        "/conversation/create",
        json={"metadata": {"topic": "finance"}, "conversation_id": "client-id"},
    )
    assert response.status_code == 200
    assert response.json()["conversation_id"] == "db-conversation-1"


def test_send_and_read_conversation_messages(client):
    test_client, _ = client
    sent = test_client.post(
        "/conversation/chat",
        json={"conversation_id": "db-conversation-1", "message": "Revenue?"},
    )
    assert sent.status_code == 200
    assert sent.json()["response"] == "Answer: Revenue?"
    history = test_client.get("/conversation/db-conversation-1/messages")
    assert history.status_code == 200
    assert history.json()["messages"][0]["role"] == "user"


def test_list_and_delete_conversation(client):
    test_client, _ = client
    listed = test_client.get("/conversation")
    assert listed.status_code == 200
    assert listed.json()["conversations"][0]["message_count"] == 2
    deleted = test_client.delete("/conversation/db-conversation-1")
    assert deleted.status_code == 200
    assert deleted.json()["deletion_trace"]["redis_keys_deleted"] == 3
    assert test_client.get("/conversation").json()["conversations"] == []


def test_conversation_stream_uses_legacy_compatible_sse_events(client):
    test_client, _ = client
    response = test_client.post(
        "/conversation/chat/stream",
        json={"conversation_id": "db-conversation-1", "message": "Revenue?"},
    )
    assert response.status_code == 200
    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [event["type"] for event in events] == [
        "status", "status", "content", "content", "citation", "trace", "done",
    ]
    assert events[0]["stage"] == "understanding"
    assert events[1]["stage"] == "retrieval"
    assert "".join(event.get("content", "") for event in events if event["type"] == "content") == "Answer: Revenue?"
    assert events[-2]["rag_trace"]["retrieval_decision"]["called"]
    assert events[-1]["conversation_id"] == "db-conversation-1"


def test_delete_removes_postgres_memory_trace_and_redis_cache():
    engine = create_engine("sqlite:///:memory:")
    MemoryConversation.__table__.create(engine)
    MemoryMessage.__table__.create(engine)
    MemorySummary.__table__.create(engine)
    RagTraceRecord.__table__.create(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    persistent = PersistentMemoryStore(sessions)
    redis_store = RedisMemoryStore(FakeRedis(), key_prefix="test:v1.1", ttl_seconds=60)
    memory = MemoryUpdateService(persistent, redis_store)
    memory.create_conversation("alice", "c1")
    memory.append_message("alice", "c1", "user", "Revenue?")
    memory.save_summary("alice", "c1", "Revenue discussion")
    memory.save_evidence_state("alice", "c1", {"evidence": "Revenue was 10."})
    RagTraceService(sessions).save("alice", {
        "conversation_id": "c1",
        "user_query": "Revenue?",
        "conversation_understanding": {},
        "policy_decision": {},
    })

    deleted, trace = memory.delete_conversation("alice", "c1")
    assert deleted and trace["postgres_deleted"]
    assert persistent.get_conversation("alice", "c1") is None
    db = sessions()
    try:
        assert db.query(MemoryMessage).count() == 0
        assert db.query(MemorySummary).count() == 0
        assert db.query(RagTraceRecord).count() == 0
    finally:
        db.close()
    assert redis_store.load_state("alice", "c1") is None


def test_persistent_list_counts_messages_and_delete_is_user_scoped():
    engine = create_engine("sqlite:///:memory:")
    MemoryConversation.__table__.create(engine)
    MemoryMessage.__table__.create(engine)
    MemorySummary.__table__.create(engine)
    RagTraceRecord.__table__.create(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    store = PersistentMemoryStore(sessions)
    store.create_conversation("alice", "alice-conversation")
    store.append_message("alice", "alice-conversation", "user", "First")
    store.append_message("alice", "alice-conversation", "assistant", "Second")
    store.create_conversation("bob", "bob-conversation")
    store.create_conversation("alice", "empty-conversation")

    rows = store.list_conversations("alice")
    assert len(rows) == 1 and rows[0]["message_count"] == 2
    assert rows[0]["title"] == "First"
    assert not store.delete_conversation("alice", "bob-conversation")
    assert store.get_conversation("bob", "bob-conversation") is not None
