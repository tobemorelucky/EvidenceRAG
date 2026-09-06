import json

import pytest

from backend.financial_operation_planner_v2 import attach_plan, parse_plan, planner_enabled, planner_trigger
from scripts.run_financial_operation_planner_shadow_v2 import added_calculation, answer_signature, build_summary


PLAN = {
    "task_type": "calculation", "entity": "company", "period": "FY2023", "metric": "ratio",
    "required_facts": ["numerator", "denominator"], "operation": "divide", "formula": "a/b", "direction": None,
}


def test_feature_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("FINANCIAL_OPERATION_PLANNER_ENABLED", raising=False)
    assert planner_enabled() is False
    assert planner_enabled("true") is True


@pytest.mark.parametrize("question", [
    "Calculate the current ratio.", "Did margin increase?", "Which segment was lowest?", "Is liquidity healthy?",
])
def test_triggered_question_classes(question):
    assert planner_trigger(question)[0] is True


def test_plain_lookup_does_not_trigger():
    assert planner_trigger("What was revenue in FY2023?") == (False, "lookup")


def test_strict_plan_and_guidance_preserve_evidence():
    parsed = parse_plan(json.dumps(PLAN))
    guidance = attach_plan("ORIGINAL EVIDENCE", parsed)
    assert "ORIGINAL EVIDENCE" in guidance and "not factual evidence" in guidance
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_plan("```json\n{}\n```")


def test_material_signature_and_added_calculation():
    assert answer_signature("It increased to 12%") != answer_signature("It decreased to 10%")
    assert added_calculation("The result is 2%.", "Calculation: 10 / 5 = 2%") is True


def test_summary_labels_candidates_not_measured_gain():
    records = [
        {"id": "a", "status": "ok", "planner_triggered": True, "material_changed": True,
         "text_changed": True, "planner_added_calculation": False, "baseline_judge_score": 0,
         "planner_usage": {}, "answer_usage": {}, "planner_latency_ms": 1, "answer_latency_ms": 1},
        {"id": "b", "status": "ok", "planner_triggered": False, "material_changed": False,
         "text_changed": False, "planner_added_calculation": False, "baseline_judge_score": 1,
         "planner_usage": {}, "answer_usage": {}, "planner_latency_ms": 0, "answer_latency_ms": 0},
    ]
    summary = build_summary(records)
    assert summary["estimated_gain"] == 1 and "not measured" in summary["estimated_gain_note"].lower()


def test_shadow_has_no_judge_retrieval_or_jina_call():
    import inspect
    import scripts.run_financial_operation_planner_shadow_v2 as module

    source = inspect.getsource(module)
    assert "judge_answer(" not in source and "JinaReranker" not in source and "milvus_client" not in source
