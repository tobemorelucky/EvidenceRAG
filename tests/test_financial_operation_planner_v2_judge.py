from scripts.run_financial_operation_planner_shadow_v2_judge import classify_change, summarize


def _baseline(score):
    return {"judge": {"score": score}}


def _new(question_id, score, reason=""):
    return {"id": question_id, "status": "ok", "judge": {"score": score, "reason": reason, "usage": {}}, "latency_ms": 1}


def test_classification_uses_generic_plan_and_reason():
    record = {"question": "Did it increase?", "planner_answer": "It decreased.", "planner": {"task_type": "trend"}}
    assert classify_change(record, "wrong direction", regression=False) == "trend/direction"


def test_summary_reuses_unselected_baseline(monkeypatch):
    import scripts.run_financial_operation_planner_shadow_v2_judge as module

    monkeypatch.setattr(module, "EXPECTED_SELECTION", 2)
    selected = [
        {"id": "a", "question": "Calculate ratio", "planner_answer": "2", "planner": {"task_type": "calculation"}},
        {"id": "b", "question": "Trend?", "planner_answer": "down", "planner": {"task_type": "trend"}},
    ]
    all_results = {"a": {}, "b": {}, "c": {}}
    baseline = {"a": _baseline(0), "b": _baseline(1), "c": _baseline(1)}
    summary = summarize(selected, all_results, baseline, [_new("a", 1, "correct calculation"), _new("b", 0, "wrong direction")])
    assert summary["baseline_strict_judge"]["correct"] == 2
    assert summary["planner_strict_judge"]["correct"] == 2
    assert summary["outcomes"]["baseline_incorrect_planner_correct"] == 1
    assert summary["outcomes"]["baseline_correct_planner_incorrect"] == 1
    assert summary["net_gain_questions"] == 0


def test_judge_script_has_no_planner_answer_retrieval_or_jina_calls():
    import inspect
    import scripts.run_financial_operation_planner_shadow_v2_judge as module

    source = inspect.getsource(module)
    assert "plan_question(" not in source
    assert "generate_answer(" not in source
    assert "JinaReranker" not in source
    assert "milvus_client" not in source
