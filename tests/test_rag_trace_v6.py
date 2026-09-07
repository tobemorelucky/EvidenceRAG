import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.conversation_memory_v5 import ConversationMemoryV5Service
from backend.conversation_understanding_v3 import ConversationDecisionV3
from backend.memory.conversation_state import ConversationState
from backend.memory.persistent_memory_store import MemoryConversation
from backend.memory.token_budget_manager import MemoryBudget, TokenBudgetManager
from backend.rag_trace_service import RagTraceRecord, RagTraceService, build_rag_trace_payload


def payload(**changes):
    value = {
        "conversation_id": "c1", "user_query": "What was revenue?",
        "conversation_understanding": {"standalone_query": "Revenue FY2023"},
        "policy_decision": {"retrieval": True}, "standalone_query": "Revenue FY2023",
        "retrieval_executed": True, "dense_count": 120, "bm25_count": 30, "rrf_count": 100,
        "rerank_parameters": {"model": "jina"}, "evidence_items": [{"id": "e1"}],
        "answer_model": "flash", "token_usage": {"total_tokens": 10},
        "latency_ms": {"total": 100},
    }
    value.update(changes)
    return value


def trace_store():
    engine = create_engine("sqlite:///:memory:")
    MemoryConversation.__table__.create(engine)
    RagTraceRecord.__table__.create(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    db = sessions()
    db.add(MemoryConversation(user_id="alice", conversation_id="c1", metadata_json={}))
    db.add(MemoryConversation(user_id="bob", conversation_id="c1", metadata_json={}))
    db.commit()
    db.close()
    return RagTraceService(sessions)


def build(**changes):
    args = {
        "conversation_id": "c1", "user_query": "question",
        "decision": {"standalone_query": "resolved"}, "policy": {"retrieval": True},
        "rag_result": {"rag_trace": {}}, "citations": [], "answer_model": "flash",
        "understanding_usage": {}, "answer_usage": {}, "latency_ms": {"total": 1},
    }
    args.update(changes)
    return build_rag_trace_payload(**args)


def test_01_payload_records_conversation_and_query():
    row = build()
    assert row["conversation_id"] == "c1" and row["user_query"] == "question"


def test_02_payload_records_understanding_and_policy():
    row = build(decision={"standalone_query": "x", "need_rag": True}, policy={"retrieval": False})
    assert row["conversation_understanding"]["need_rag"] and not row["retrieval_executed"]


def test_03_payload_extracts_primary_retrieval_counts():
    row = build(rag_result={"rag_trace": {"dense_candidate_count": 12, "bm25_candidate_count": 7, "rrf_fused_candidate_count": 9}})
    assert (row["dense_count"], row["bm25_count"], row["rrf_count"]) == (12, 7, 9)


def test_04_payload_extracts_fallback_retrieval_counts():
    row = build(rag_result={"rag_trace": {"initial_dense_candidates": 10, "initial_bm25_candidates": 4, "rrf_candidates": 8}})
    assert (row["dense_count"], row["bm25_count"], row["rrf_count"]) == (10, 4, 8)


def test_05_payload_records_rerank_parameters():
    row = build(rag_result={"rag_trace": {"rerank_enabled": True, "rerank_applied": True, "rerank_model": "jina", "candidate_k": 80, "final_top_k": 12}})
    assert row["rerank_parameters"]["model"] == "jina" and row["rerank_parameters"]["final_top_k"] == 12


def test_05b_payload_records_runtime_and_depth_parameters(monkeypatch):
    monkeypatch.setenv("DENSE_TOP_K", "240")
    monkeypatch.setenv("BM25_TOP_K", "240")
    monkeypatch.setenv("RRF_TOP_K", "120")
    monkeypatch.setenv("JINA_INPUT_K", "80")
    monkeypatch.setenv("JINA_OUTPUT_K", "12")
    row = build(profile="finance", execution_mode="auto", answer_prompt_mode="baseline")
    assert row["policy_decision"]["profile"] == "finance"
    assert row["policy_decision"]["execution_mode"] == "auto"
    assert row["policy_decision"]["answer_prompt_mode"] == "baseline"
    assert row["rerank_parameters"]["dense_top_k"] == "240"
    assert row["rerank_parameters"]["input_k"] == "80"


def test_06_payload_prefers_citation_evidence():
    row = build(citations=[{"id": "e1", "filename": "x.pdf", "page_number": 2, "text": "fact"}])
    assert row["evidence_items"][0]["filename"] == "x.pdf"


def test_07_payload_falls_back_to_docs():
    row = build(rag_result={"docs": [{"chunk_id": "k", "filename": "x.pdf", "text": "fact"}], "rag_trace": {}})
    assert row["evidence_items"][0]["chunk_id"] == "k"


def test_08_payload_bounds_evidence_preview():
    row = build(citations=[{"id": "e1", "text": "x" * 2000}])
    assert len(row["evidence_items"][0]["text"]) == 1200


def test_09_payload_aggregates_model_tokens():
    row = build(understanding_usage={"total_tokens": 4}, answer_usage={"total_tokens": 9})
    assert row["token_usage"]["total_tokens"] == 13


def test_10_payload_normalizes_invalid_counts():
    row = build(rag_result={"rag_trace": {"dense_candidate_count": "bad", "bm25_candidate_count": -2}})
    assert row["dense_count"] == 0 and row["bm25_count"] == 0


def test_11_store_saves_complete_trace():
    saved = trace_store().save("alice", payload())
    assert saved["conversation_id"] == "c1" and saved["retrieval_counts"]["dense"] == 120


def test_12_store_generates_unique_trace_id():
    store = trace_store()
    first = store.save("alice", payload())["trace_id"]
    second = store.save("alice", payload())["trace_id"]
    assert first != second


def test_13_store_preserves_provided_trace_id():
    assert trace_store().save("alice", payload(trace_id="fixed"))["trace_id"] == "fixed"


def test_14_store_lists_in_chronological_order():
    store = trace_store()
    store.save("alice", payload(trace_id="a"))
    store.save("alice", payload(trace_id="b"))
    assert [item["trace_id"] for item in store.list_for_conversation("alice", "c1")] == ["a", "b"]


def test_15_store_applies_limit_to_most_recent_records():
    store = trace_store()
    for value in ("a", "b", "c"):
        store.save("alice", payload(trace_id=value))
    assert [item["trace_id"] for item in store.list_for_conversation("alice", "c1", limit=2)] == ["b", "c"]


def test_16_store_isolates_users_with_same_conversation_id():
    store = trace_store()
    store.save("alice", payload(trace_id="alice-trace"))
    assert store.list_for_conversation("bob", "c1") == []


def test_17_store_rejects_missing_conversation_on_save():
    with pytest.raises(KeyError):
        trace_store().save("alice", payload(conversation_id="missing"))


def test_18_store_rejects_missing_conversation_on_read():
    with pytest.raises(KeyError):
        trace_store().list_for_conversation("alice", "missing")


class FakeMemory:
    def __init__(self):
        self.state = ConversationState("c1")
        self.saved = []

    def load_state(self, *_): return self.state, {"source": "fake"}
    def load_evidence_state(self, *_): return {}, {"cache_hit": False}
    def latest_summary(self, *_): return None
    def append_message(self, *args, **kwargs): self.saved.append(args); return {}, {}
    def save_evidence_state(self, *args): return {"redis_written": True}


class CapturingTraceService:
    def __init__(self, fail=False): self.payloads, self.fail = [], fail
    def save(self, user_id, value):
        if self.fail: raise RuntimeError("trace db unavailable")
        self.payloads.append(value)
        return {"trace_id": "trace-1"}
    def list_for_conversation(self, *_args, **_kwargs): return []


def service(trace_service):
    return ConversationMemoryV5Service(
        FakeMemory(), TokenBudgetManager(MemoryBudget(100, 20, 5)),
        understanding_fn=lambda *_: (ConversationDecisionV3(True, True, False, False, "none", "query", "answer"), {"usage": {"total_tokens": 3}}),
        retrieval_fn=lambda *_args, **_kwargs: {"evidence": "fact", "rag_trace": {"dense_candidate_count": 5, "bm25_candidate_count": 3, "rrf_fused_candidate_count": 4}},
        answer_fn=lambda *_args, **_kwargs: ("answer", {"total_tokens": 7}), trace_service=trace_service,
    )


def test_19_v5_chat_persists_v6_observability_trace():
    traces = CapturingTraceService()
    result = service(traces).chat("alice", "c1", "question")
    assert result["trace"]["observability"] == {"persisted": True, "trace_id": "trace-1"}
    assert traces.payloads[0]["token_usage"]["total_tokens"] == 10


def test_20_trace_failure_does_not_change_answer():
    result = service(CapturingTraceService(fail=True)).chat("alice", "c1", "question")
    assert result["response"] == "answer"
    assert not result["trace"]["observability"]["persisted"]
