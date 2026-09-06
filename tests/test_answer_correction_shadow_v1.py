from scripts.run_answer_correction_shadow_v1 import CRITIC_SYSTEM_PROMPT, build_summary, select_failures


def test_failure_selection_requires_wrong_and_context_hit():
    answers = {
        "a": {"status": "ok", "context_metrics": {"candidate_gold_page_hit": True}},
        "b": {"status": "ok", "context_metrics": {"candidate_gold_page_hit": False}},
    }
    judges = {
        "a": {"status": "ok", "judge": {"score": 0}},
        "b": {"status": "ok", "judge": {"score": 0}},
    }
    assert select_failures(answers, judges) == ["a"]


def test_critic_is_limited_and_has_no_reference():
    lowered = CRITIC_SYSTEM_PROMPT.lower()
    assert all(term in lowered for term in ("calculation", "direction", "metric", "scope"))
    assert "reference answer" in lowered and "do not mention" in lowered
    assert "financebench" not in lowered


def test_summary_acceptance_threshold():
    revisions = [{"id": str(index), "status": "ok", "usage": {}, "latency_ms": 1} for index in range(6)]
    judges = [
        {"id": str(index), "status": "ok", "judge": {"score": int(index < 5), "usage": {}}, "latency_ms": 1}
        for index in range(6)
    ]
    summary = build_summary(revisions, judges)
    assert summary["recovered"] == 5
    assert summary["acceptance"]["passed"] is True
    assert summary["newly_incorrect"] == 0


def test_shadow_script_does_not_import_retrieval_or_jina():
    import inspect
    import scripts.run_answer_correction_shadow_v1 as module

    source = inspect.getsource(module)
    assert "JinaReranker" not in source
    assert "rewrite_queries(" not in source
    assert "milvus_client" not in source
