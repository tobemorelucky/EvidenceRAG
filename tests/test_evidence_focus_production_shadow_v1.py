from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

from backend.answer_generator import resolve_answer_prompt_route
from backend.conversation_memory_v5 import ConversationMemoryV5Service
from backend.conversation_understanding_v3 import ConversationDecisionV3
from backend.memory.conversation_state import ConversationState
from backend.memory.token_budget_manager import MemoryBudget, TokenBudgetManager
from backend.rag_trace_service import RagTraceService, build_rag_trace_payload


ROOT = Path(__file__).resolve().parents[1]


class MemoryStub:
    def __init__(self):
        self.state = ConversationState("c1")

    def load_state(self, *_args):
        return self.state, {"source": "stub"}

    def load_evidence_state(self, *_args):
        return {}, {"cache_hit": False}

    def latest_summary(self, *_args):
        return None

    def append_message(self, *_args, **_kwargs):
        return {}, {}

    def save_evidence_state(self, *_args, **_kwargs):
        return {}


class TraceStub:
    def __init__(self):
        self.payload = None

    def save(self, _user_id, payload):
        self.payload = payload
        return {"trace_id": "trace-router"}


def route(monkeypatch, question, profile="finance", enabled="true", mode="baseline"):
    monkeypatch.setenv("FINANCE_EVIDENCE_FOCUS_ROUTER_ENABLED", enabled)
    return resolve_answer_prompt_route(question, profile, mode)


def test_calculation_question_routes_to_focus(monkeypatch):
    result = route(monkeypatch, "Calculate the current ratio for FY2022")
    assert result["selected_prompt"] == "finance_evidence_focus_v1"
    assert "calculation_or_reasoning_indicator" in result["router_reason"]


def test_trend_question_routes_to_focus(monkeypatch):
    result = route(monkeypatch, "Did operating income improve from FY2021 to FY2022?")
    assert result["selected_prompt"] == "finance_evidence_focus_v1"


def test_lookup_question_routes_to_clean_baseline(monkeypatch):
    result = route(monkeypatch, "What was revenue in FY2022?")
    assert result["selected_prompt"] == "clean_baseline_v1"
    assert result["router_reason"] == ["ordinary_lookup"]


def test_non_finance_always_stays_baseline(monkeypatch):
    result = route(monkeypatch, "Calculate the ratio", profile="general")
    assert result["selected_prompt"] == "baseline"
    assert result["prompt_router_applied"] is False


def test_disabled_router_always_uses_finance_clean_baseline(monkeypatch):
    result = route(monkeypatch, "Calculate the margin trend", enabled="false")
    assert result["selected_prompt"] == "clean_baseline_v1"
    assert result["prompt_router_enabled"] is False


def test_explicit_modes_remain_switchable(monkeypatch):
    clean = route(monkeypatch, "Calculate ratio", mode="clean_baseline_v1")
    focus = route(monkeypatch, "What was revenue?", enabled="false", mode="finance_evidence_focus_v1")
    assert clean["selected_prompt"] == "clean_baseline_v1"
    assert focus["selected_prompt"] == "finance_evidence_focus_v1"


def test_trace_payload_persists_prompt_router_fields():
    prompt_trace = {
        "prompt_router_enabled": True,
        "prompt_router_applied": True,
        "selected_prompt": "finance_evidence_focus_v1",
        "router_reason": ["calculation_or_reasoning_indicator"],
        "router_version": "evidence_focus_semantic_router_v2",
    }
    payload = build_rag_trace_payload(
        conversation_id="c1",
        user_query="Calculate ratio",
        decision={},
        policy={"retrieval": True, **prompt_trace},
        rag_result={"rag_trace": {}},
        citations=[],
        answer_model="flash",
        understanding_usage={},
        answer_usage={},
        latency_ms={},
        profile="finance",
        answer_prompt_mode="finance_evidence_focus_v1",
    )
    for key, value in prompt_trace.items():
        assert payload["policy_decision"][key] == value


def test_persisted_trace_exposes_router_fields_at_top_level():
    policy = {
        "prompt_router_enabled": True,
        "prompt_router_applied": True,
        "selected_prompt": "finance_evidence_focus_v1",
        "router_reason": ["calculation_or_reasoning_indicator"],
        "router_version": "evidence_focus_semantic_router_v2",
    }
    row = SimpleNamespace(
        trace_id="t1", conversation_id="c1", user_query="Calculate ratio",
        conversation_understanding={}, policy_decision=policy, standalone_query="Calculate ratio",
        retrieval_executed=True, dense_count=1, bm25_count=1, rrf_count=1,
        rerank_parameters={}, evidence_items=[], answer_model="flash", token_usage={},
        latency_ms={}, created_at=datetime(2026, 1, 1),
    )
    serialized = RagTraceService._serialize(row)
    for key, value in policy.items():
        assert serialized[key] == value


def test_default_environment_value_is_disabled(monkeypatch):
    monkeypatch.delenv("FINANCE_EVIDENCE_FOCUS_ROUTER_ENABLED", raising=False)
    result = resolve_answer_prompt_route("Calculate ratio", "finance", "baseline")
    assert result["selected_prompt"] == "clean_baseline_v1"
    assert "FINANCE_EVIDENCE_FOCUS_ROUTER_ENABLED=false" in (ROOT / ".env.example").read_text(encoding="utf-8")


def test_conversation_finance_path_passes_selected_prompt_and_trace(monkeypatch):
    monkeypatch.setenv("FINANCE_EVIDENCE_FOCUS_ROUTER_ENABLED", "true")
    captured = {}
    trace_service = TraceStub()
    service = ConversationMemoryV5Service(
        MemoryStub(),
        TokenBudgetManager(MemoryBudget(100, 20, 5)),
        understanding_fn=lambda *_: (
            ConversationDecisionV3(True, True, False, False, "none", "Calculate ratio", "answer"),
            {"model_called": False},
        ),
        retrieval_fn=lambda *_args, **_kwargs: {
            "evidence": "Revenue 10; liabilities 5.",
            "citations": [{"id": "e1"}],
            "rag_trace": {},
        },
        answer_fn=lambda *_args, **kwargs: (captured.update(kwargs) or "2.0", {}),
        trace_service=trace_service,
    )
    result = service.chat("u1", "c1", "Calculate the current ratio", profile="finance")
    assert captured["prompt_mode"] == "finance_evidence_focus_v1"
    assert result["trace"]["selected_prompt"] == "finance_evidence_focus_v1"
    assert result["trace"]["router_version"] == "evidence_focus_semantic_router_v2"
    assert trace_service.payload["policy_decision"]["prompt_router_enabled"] is True
