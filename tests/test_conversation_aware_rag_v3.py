import json

import pytest

from backend.conversation_understanding_v3 import parse_conversation_decision_v3
from scripts.run_conversation_aware_rag_v3_schema_shadow import load_cases, summarize


def _decision(**changes):
    value = {
        "need_rag": True, "need_retrieval": True, "depends_on_history": True,
        "query_resolution_needed": True, "required_memory_scope": "last exchange",
        "standalone_query": "Acme revenue FY2023", "response_mode": "answer with evidence",
    }
    value.update(changes)
    return value


def test_v3_schema_removes_ambiguous_v2_fields():
    parsed = parse_conversation_decision_v3(json.dumps(_decision()))
    assert parsed.depends_on_history and parsed.query_resolution_needed
    assert not hasattr(parsed, "need_rewrite") and not hasattr(parsed, "need_previous_context")


def test_v3_schema_enforces_memory_contract():
    with pytest.raises(ValueError):
        parse_conversation_decision_v3(json.dumps(_decision(depends_on_history=False)))
    with pytest.raises(ValueError):
        parse_conversation_decision_v3(json.dumps(_decision(depends_on_history=False, required_memory_scope="none")))
    parsed = parse_conversation_decision_v3(json.dumps(_decision(depends_on_history=False, query_resolution_needed=False, required_memory_scope="none")))
    assert parsed.required_memory_scope == "none"


def test_v3_reuses_exactly_the_frozen_fifty_inputs():
    cases = load_cases()
    assert len(cases) == len({case["id"] for case in cases}) == 50
    assert all("expected_v3" in case for case in cases)


def test_summary_compares_redefined_fields_without_models():
    cases = [{"id": "x", "category": "new_question", "expected_v3": {"need_retrieval": True}}]
    record = {"id": "x", "status": "ok", "depends_on_history_correct": True, "query_resolution_correct": True,
        "need_rag_correct": True, "need_retrieval_correct": True, "standalone_quality": {"quality_pass": True, "term_retention": 1},
        "trace": {"usage": {}}, "latency_ms": 1}
    v2 = {"json_legal_rate": 1, "need_rag_accuracy": 1, "need_retrieval_accuracy": 1,
        "need_previous_context_accuracy": 0.5, "need_rewrite_accuracy": 0.5,
        "standalone_query_quality_pass_rate": 0.5}
    result = summarize(cases, [record], v2)
    assert result["depends_on_history_accuracy"] == 1
    assert result["v2_comparison"]["query_resolution"]["delta"] == 0.5


def test_v3_runner_has_no_forbidden_pipeline_calls():
    import inspect
    import scripts.run_conversation_aware_rag_v3_schema_shadow as module

    source = inspect.getsource(module)
    forbidden = ("prepare_rag_response", "generate_answer", "judge_answer", "JinaReranker", "milvus_client", "hybrid_retrieve")
    assert all(token not in source for token in forbidden)
