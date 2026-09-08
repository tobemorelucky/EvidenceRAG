import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.memory.memory_update import MemoryUpdateService
from backend.memory.persistent_memory_store import (
    MemoryConversation,
    MemoryMessage,
    MemorySummary,
    PersistentMemoryStore,
)
from backend.memory.redis_memory_store import RedisMemoryStore
from backend.memory.token_budget_manager import MemoryBudget, TokenBudgetManager


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.ttls = {}

    def setex(self, key, ttl, value):
        self.data[key] = value
        self.ttls[key] = ttl

    def get(self, key):
        return self.data.get(key)

    def delete(self, *keys):
        removed = 0
        for key in keys:
            removed += key in self.data
            self.data.pop(key, None)
        return removed


class BrokenRedis(FakeRedis):
    def setex(self, key, ttl, value):
        raise ConnectionError("redis unavailable")

    def get(self, key):
        raise ConnectionError("redis unavailable")


def stores(redis_client=None):
    engine = create_engine("sqlite:///:memory:")
    MemoryConversation.__table__.create(engine)
    MemoryMessage.__table__.create(engine)
    MemorySummary.__table__.create(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    persistent = PersistentMemoryStore(sessions)
    redis_store = RedisMemoryStore(redis_client or FakeRedis(), key_prefix="test:v5", ttl_seconds=60)
    return persistent, redis_store


def test_01_redis_state_roundtrip():
    _, store = stores()
    store.save_state("u", "c", {"conversation_id": "c", "turns": [], "metadata": {}})
    assert store.load_state("u", "c")["conversation_id"] == "c"


def test_02_redis_recent_messages_roundtrip():
    _, store = stores()
    store.save_recent_messages("u", "c", [{"role": "user", "content": "hello"}])
    assert store.load_recent_messages("u", "c")[0]["content"] == "hello"


def test_03_redis_evidence_roundtrip():
    _, store = stores()
    store.save_evidence_state("u", "c", {"evidence": "fact", "citations": [1]})
    assert store.load_evidence_state("u", "c")["evidence"] == "fact"


def test_04_redis_delete_removes_all_memory_kinds():
    _, store = stores()
    store.save_state("u", "c", {"x": 1})
    store.save_recent_messages("u", "c", [])
    store.save_evidence_state("u", "c", {})
    assert store.delete_conversation("u", "c") == 3
    assert store.load_state("u", "c") is None


def test_05_redis_uses_configured_ttl():
    client = FakeRedis()
    store = RedisMemoryStore(client, key_prefix="test:v5", ttl_seconds=75)
    store.save_state("u", "c", {})
    assert set(client.ttls.values()) == {75}


def test_06_redis_rejects_empty_identity():
    _, store = stores()
    with pytest.raises(ValueError):
        store.load_state("", "c")


def test_07_persistent_create_conversation():
    store, _ = stores()
    row = store.create_conversation("alice", "c1", {"topic": "finance"})
    assert row["conversation_id"] == "c1" and row["metadata"]["topic"] == "finance"


def test_08_persistent_generates_conversation_id():
    store, _ = stores()
    assert store.create_conversation("alice")["conversation_id"]


def test_09_persistent_rejects_duplicate_for_same_user():
    store, _ = stores()
    store.create_conversation("alice", "c1")
    with pytest.raises(ValueError):
        store.create_conversation("alice", "c1")


def test_10_persistent_isolates_same_id_by_user():
    store, _ = stores()
    store.create_conversation("alice", "shared")
    store.create_conversation("bob", "shared")
    assert store.get_conversation("alice", "shared")["user_id"] == "alice"
    assert store.get_conversation("bob", "shared")["user_id"] == "bob"


def test_11_persistent_appends_message_with_trace_and_evidence_refs():
    store, _ = stores()
    store.create_conversation("alice", "c1")
    row = store.append_message("alice", "c1", "assistant", "answer", evidence_refs=["e1"], trace={"retrieval": True})
    assert row["evidence_refs"] == ["e1"] and row["trace"]["retrieval"]


def test_12_persistent_validates_message():
    store, _ = stores()
    store.create_conversation("alice", "c1")
    with pytest.raises(ValueError):
        store.append_message("alice", "c1", "tool", "x")


def test_13_persistent_recent_message_limit_preserves_order():
    store, _ = stores()
    store.create_conversation("alice", "c1")
    for value in ("one", "two", "three"):
        store.append_message("alice", "c1", "user", value)
    assert [item["content"] for item in store.list_messages("alice", "c1", limit=2)] == ["two", "three"]


def test_14_persistent_saves_and_loads_latest_summary():
    store, _ = stores()
    store.create_conversation("alice", "c1")
    store.save_summary("alice", "c1", "first")
    store.save_summary("alice", "c1", "latest", metadata={"version": 2})
    assert store.latest_summary("alice", "c1")["summary"] == "latest"


def test_15_persistent_hydrates_conversation_state():
    store, _ = stores()
    store.create_conversation("alice", "c1", {"topic": "x"})
    store.append_message("alice", "c1", "user", "question")
    state = store.load_state("alice", "c1")
    assert state.metadata["topic"] == "x" and state.turns[0].content == "question"


def test_16_update_service_writes_postgres_then_cache():
    persistent, redis_store = stores()
    service = MemoryUpdateService(persistent, redis_store)
    conversation, trace = service.create_conversation("alice", "c1")
    assert conversation["conversation_id"] == "c1"
    assert trace == {"postgres_written": True, "redis_written": True}


def test_17_update_service_survives_redis_write_failure():
    persistent, _ = stores()
    service = MemoryUpdateService(persistent, RedisMemoryStore(BrokenRedis(), key_prefix="test:v5"))
    conversation, trace = service.create_conversation("alice", "c1")
    assert conversation["conversation_id"] == "c1" and not trace["redis_written"]
    assert persistent.get_conversation("alice", "c1") is not None


def test_18_update_service_reads_redis_state_first():
    persistent, redis_store = stores()
    service = MemoryUpdateService(persistent, redis_store)
    service.create_conversation("alice", "c1")
    state, trace = service.load_state("alice", "c1")
    assert state.conversation_id == "c1" and trace == {"source": "redis", "cache_hit": True}


def test_19_update_service_hydrates_cache_after_miss_and_syncs_messages():
    persistent, redis_store = stores()
    persistent.create_conversation("alice", "c1")
    persistent.append_message("alice", "c1", "user", "question")
    service = MemoryUpdateService(persistent, redis_store)
    state, trace = service.load_state("alice", "c1")
    assert state.turns[0].content == "question" and trace["source"] == "postgres"
    service.append_message("alice", "c1", "assistant", "answer")
    assert len(redis_store.load_recent_messages("alice", "c1")) == 2


def test_20_token_budget_preserves_recent_messages_and_bounds_context():
    manager = TokenBudgetManager(MemoryBudget(max_tokens=20, message_tokens=8, summary_tokens=2))
    result = manager.build_context(
        [{"role": "user", "content": "old" * 20}, {"role": "assistant", "content": "newest"}],
        summary="summary text", evidence="evidence" * 50,
    )
    assert result["messages"][-1]["content"] == "newest"
    assert result["estimated_tokens"] <= 20 and result["evidence_truncated"]


def test_21_explicit_evidence_budget_is_independent_of_history_and_summary():
    manager = TokenBudgetManager(MemoryBudget(max_tokens=20, message_tokens=8, summary_tokens=2))
    evidence = "e" * 100
    result = manager.build_context(
        [{"role": "user", "content": "history" * 20}],
        summary="summary" * 20,
        evidence=evidence,
        evidence_char_budget=100,
    )
    assert result["evidence"] == evidence
    assert result["evidence_char_budget"] == 100
    assert not result["evidence_truncated"]
