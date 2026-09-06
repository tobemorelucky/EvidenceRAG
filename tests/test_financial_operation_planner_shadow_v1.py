import json

import pytest

from scripts.run_financial_operation_planner_shadow_v1 import PLANNER_PROMPT, parse_plan, select_cohorts, summarize


VALID_PLAN = {
    "task_type": "calculation", "entity": "company", "period": "FY2023", "metric": "ratio",
    "required_facts": ["numerator", "denominator"], "operation": "divide", "formula": "a / b", "direction": None,
}


def test_parse_plan_accepts_strict_schema():
    assert parse_plan(json.dumps(VALID_PLAN))["task_type"] == "calculation"


def test_planner_prompt_formats_literal_json_schema():
    formatted = PLANNER_PROMPT.format(question="What changed?")
    assert '"task_type"' in formatted and "What changed?" in formatted


@pytest.mark.parametrize("text", ["```json\n{}\n```", '{"task_type":"unknown"}', '{"task_type":"lookup","extra":1}'])
def test_parse_plan_rejects_non_strict_output(text):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        parse_plan(text)


def test_cohort_selection_is_deterministic():
    answers = {}
    judges = {}
    for index in range(40):
        key = str(index)
        answers[key] = {"status": "ok", "context_metrics": {"candidate_gold_page_hit": True}}
        judges[key] = {"status": "ok", "judge": {"score": int(index >= 22)}}
    first = select_cohorts(answers, judges)
    second = select_cohorts(answers, judges)
    assert first == second and len(first[0]) == 22 and len(first[1]) == 15


def test_summary_counts_recovery_and_regression():
    plans = [{"id": str(i), "status": "ok", "usage": {}, "latency_ms": 1} for i in range(2)]
    answers = [{"id": str(i), "status": "ok", "usage": {}, "latency_ms": 1} for i in range(2)]
    judges = [
        {"id": "0", "cohort": "failure", "status": "ok", "judge": {"score": 1, "usage": {}}, "latency_ms": 1},
        {"id": "1", "cohort": "control", "status": "ok", "judge": {"score": 0, "usage": {}}, "latency_ms": 1},
    ]
    summary = summarize(plans, answers, judges)
    assert summary["outcomes"]["recovered"] == summary["outcomes"]["regressed"] == 1


def test_shadow_has_no_retrieval_or_jina_path():
    import inspect
    import scripts.run_financial_operation_planner_shadow_v1 as module

    source = inspect.getsource(module)
    assert "JinaReranker" not in source and "rewrite_queries(" not in source and "milvus_client" not in source
