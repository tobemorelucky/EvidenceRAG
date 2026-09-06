from scripts.evaluate_financial_operation_planner_gating_shadow_v1 import evaluate_strategy, gate_task, strategy_enabled


def _record(question, task_type="calculation", formula=None, triggered=True):
    return {"question": question, "planner_triggered": triggered, "planner": {"task_type": task_type, "metric": "", "formula": formula}}


def test_specific_gate_task_precedes_general_calculation():
    assert gate_task(_record("Calculate the gross margin")) == "margin"
    assert gate_task(_record("Calculate the current ratio")) == "ratio"
    assert gate_task(_record("Calculate the difference")) == "calculation"


def test_gating_strategies_are_distinct():
    growth = _record("What was revenue growth?")
    assert strategy_enabled("B_calculation_ratio_margin_turnover_growth", growth)
    assert not strategy_enabled("C_calculation_ratio", growth)
    assert not strategy_enabled("D_explicit_formula", growth)
    growth["planner"]["formula"] = "(new-old)/old"
    assert strategy_enabled("D_explicit_formula", growth)


def test_offline_replay_falls_back_to_baseline_when_gate_is_off(monkeypatch):
    import scripts.evaluate_financial_operation_planner_gating_shadow_v1 as module

    results = {"a": _record("Revenue growth?"), "b": _record("Current ratio?")}
    baseline = {"a": {"judge": {"score": 0}}, "b": {"judge": {"score": 1}}}
    planner = {"a": {"judge": {"score": 1}}, "b": {"judge": {"score": 0}}}
    summary, records = evaluate_strategy("C_calculation_ratio", results, baseline, planner)
    assert {record["id"]: record["strategy_score"] for record in records} == {"a": 0, "b": 0}
    assert summary["newly_correct"] == 0 and summary["regressed"] == 1


def test_script_contains_no_model_or_pipeline_call():
    import inspect
    import scripts.evaluate_financial_operation_planner_gating_shadow_v1 as module

    source = inspect.getsource(module)
    forbidden = ("judge_answer(", "generate_answer(", "plan_question(", "JinaReranker", "milvus_client", "init_chat_model")
    assert all(token not in source for token in forbidden)
