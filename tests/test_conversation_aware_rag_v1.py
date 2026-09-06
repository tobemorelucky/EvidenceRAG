import json
from pathlib import Path

import pytest

from backend.conversation_understanding import (
    conversation_memory_enabled,
    parse_conversation_decision,
    understand_conversation,
)
from backend.memory.context_builder import build_answer_context
from backend.memory.context_memory import ContextMemory
from backend.memory.conversation_state import ConversationState
from scripts.evaluate_conversation_aware_rag_shadow_v1 import evaluate_cases


ROOT = Path(__file__).resolve().parents[1]


def _decision(**overrides):
    value = {
        "need_rag": True, "need_retrieval": True, "need_rewrite": False,
        "need_previous_context": False, "standalone_query": "original question",
        "response_mode": "answer with evidence",
    }
    value.update(overrides)
    return value


def test_feature_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("ENABLE_CONVERSATION_MEMORY", raising=False)
    assert conversation_memory_enabled() is False
    assert conversation_memory_enabled("true") is True


def test_state_round_trip_and_bounded_memory():
    memory = ContextMemory(max_turns=2)
    memory.record_user("s1", "one")
    memory.record_assistant("s1", "two", ["doc:p1"])
    state = memory.record_user("s1", "three")
    assert [turn.content for turn in state.turns] == ["two", "three"]
    restored = ConversationState.from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()


def test_first_turn_preserves_original_retrieval_without_model_call():
    class FailIfCalled:
        def invoke(self, _messages):
            raise AssertionError("First turn must not call conversation understanding")

    decision, trace = understand_conversation("What was revenue?", ConversationState("new"), FailIfCalled())
    assert decision.standalone_query == "What was revenue?"
    assert decision.need_retrieval and not decision.need_rewrite
    assert trace["model_called"] is False


def test_strict_json_schema_and_invariants():
    assert parse_conversation_decision(json.dumps(_decision())).response_mode == "answer with evidence"
    with pytest.raises(ValueError):
        parse_conversation_decision(json.dumps({**_decision(), "intent": "lookup"}))
    with pytest.raises(ValueError):
        parse_conversation_decision(json.dumps(_decision(need_rewrite=True, need_previous_context=False)))
    with pytest.raises(ValueError):
        parse_conversation_decision(json.dumps(_decision(need_rag=False, need_retrieval=True)))


def test_answer_context_uses_history_only_when_requested_and_preserves_evidence():
    state = ConversationState("s")
    state.append("user", "Earlier question")
    state.append("assistant", "Earlier answer")
    without = build_answer_context(state, "Current", "EVIDENCE", _decision())
    with_history = build_answer_context(state, "Current", "EVIDENCE", _decision(need_previous_context=True))
    assert without["conversation_history"] == []
    assert len(with_history["conversation_history"]) == 2
    assert with_history["evidence"] == "EVIDENCE"
    assert "not factual evidence" in with_history["grounding_contract"]


def test_twenty_human_authored_multi_turn_contracts_pass_offline():
    cases = json.loads((ROOT / "tests/fixtures/conversation_aware_rag_v1_cases.json").read_text(encoding="utf-8"))
    records = evaluate_cases(cases)
    assert len(records) == 20
    assert all(record["passed"] and record["evidence_preserved"] for record in records)


def test_shadow_evaluator_has_no_external_or_production_pipeline_calls():
    import inspect
    import scripts.evaluate_conversation_aware_rag_shadow_v1 as module

    source = inspect.getsource(module)
    forbidden = ("JinaReranker", "prepare_rag_response", "generate_answer", "init_chat_model", "milvus_client")
    assert all(token not in source for token in forbidden)
