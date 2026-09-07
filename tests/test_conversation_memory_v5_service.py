import asyncio

from backend.conversation_memory_v5 import ConversationMemoryV5Service, conversation_memory_v5_enabled
from backend.conversation_understanding_v3 import ConversationDecisionV3
from backend.memory.conversation_state import ConversationState
from backend.memory.token_budget_manager import MemoryBudget, TokenBudgetManager


class FakeMemory:
    def __init__(self, state, evidence=None):
        self.state = state
        self.evidence = evidence or {}
        self.messages = []

    def load_state(self, user_id, conversation_id):
        return self.state, {"source": "fake", "cache_hit": True}

    def load_evidence_state(self, user_id, conversation_id):
        return self.evidence, {"cache_hit": bool(self.evidence)}

    def latest_summary(self, user_id, conversation_id):
        return None

    def append_message(self, user_id, conversation_id, role, content, **kwargs):
        self.messages.append((role, content, kwargs))
        return {"role": role}, {"postgres_written": True}

    def save_evidence_state(self, user_id, conversation_id, evidence):
        self.evidence = evidence
        return {"redis_written": True}


class FakeTraceService:
    def __init__(self):
        self.payloads = []

    def save(self, user_id, payload):
        self.payloads.append((user_id, payload))
        return {"trace_id": "trace-1"}

    def list_for_conversation(self, user_id, conversation_id, limit=100):
        return []


def test_v5_feature_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("ENABLE_CONVERSATION_MEMORY", raising=False)
    assert not conversation_memory_v5_enabled()


def test_v5_first_turn_runs_existing_retrieval_and_answer_with_trace():
    state = ConversationState("c1")
    memory = FakeMemory(state)
    retrieval_calls = []
    service = ConversationMemoryV5Service(
        memory,
        TokenBudgetManager(MemoryBudget(100, 20, 5)),
        understanding_fn=lambda *_: (
            ConversationDecisionV3(True, True, False, False, "none", "Revenue FY2023", "answer"),
            {"model_called": False},
        ),
        retrieval_fn=lambda query, **kwargs: retrieval_calls.append(query) or {
            "evidence": "Revenue was 10.", "citations": [{"id": "e1"}], "route_reason": "test",
        },
        answer_fn=lambda *args, **kwargs: ("Revenue was 10.", {"total_tokens": 10}),
        trace_service=FakeTraceService(),
    )
    result = service.chat("alice", "c1", "What was revenue?")
    assert retrieval_calls == ["What was revenue?"]
    assert result["trace"]["retrieval_decision"]["called"]
    assert result["trace"]["evidence_reuse"]["source"] == "fresh"
    assert [item[0] for item in memory.messages] == ["user", "assistant"]


def test_v5_citation_followup_reuses_evidence_without_retrieval():
    state = ConversationState("c1")
    state.append("user", "What was revenue?")
    state.append("assistant", "Revenue was 10.")
    memory = FakeMemory(state, {"evidence": "Source: report.pdf | Page: 4\nRevenue was 10.", "citations": [{"id": "e1"}]})
    service = ConversationMemoryV5Service(
        memory,
        TokenBudgetManager(MemoryBudget(100, 20, 5)),
        understanding_fn=lambda *_: (
            ConversationDecisionV3(True, False, True, True, "last exchange", "Cite revenue source", "cite"),
            {"model_called": True},
        ),
        retrieval_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("retrieval must not run")),
        answer_fn=lambda question, evidence, **kwargs: ("report.pdf, page 4.", {}),
        trace_service=FakeTraceService(),
    )
    result = service.chat("alice", "c1", "Which source page supports that?")
    assert not result["trace"]["retrieval_decision"]["called"]
    assert result["trace"]["evidence_reuse"]["previous_reused"]
    assert result["citations"] == [{"id": "e1"}]


def test_v5_stream_chat_emits_lifecycle_and_unbuffered_content():
    state = ConversationState("c1")
    memory = FakeMemory(state)

    async def fake_stream(*_args, **_kwargs):
        yield "Revenue ", {}
        yield "was 10.", {"total_tokens": 10}

    service = ConversationMemoryV5Service(
        memory,
        TokenBudgetManager(MemoryBudget(100, 20, 5)),
        understanding_fn=lambda *_: (
            ConversationDecisionV3(True, True, False, False, "none", "Revenue FY2023", "answer"),
            {"model_called": False},
        ),
        retrieval_fn=lambda *_args, **_kwargs: {
            "evidence": "Revenue was 10.",
            "citations": [{"id": "e1", "filename": "report.pdf", "page_number": 4}],
            "rag_trace": {"rerank_enabled": True, "rerank_applied": True},
        },
        stream_answer_fn=fake_stream,
        trace_service=FakeTraceService(),
    )

    async def collect():
        return [event async for event in service.stream_chat("alice", "c1", "What was revenue?", profile="finance", execution_mode="auto")]

    events = asyncio.run(collect())
    stages = [event["stage"] for event in events if event["type"] == "status"]
    assert stages == ["understanding", "retrieval", "rerank", "evidence_build", "answering"]
    assert [event["content"] for event in events if event["type"] == "content"] == ["Revenue ", "was 10."]
    assert events[-2]["type"] == "trace"
    assert events[-1] == {"type": "done", "conversation_id": "c1", "usage": {"total_tokens": 10}}
    assert [item[0] for item in memory.messages] == ["user", "assistant"]
