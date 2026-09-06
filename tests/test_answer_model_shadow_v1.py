from scripts.run_answer_model_shadow_v1 import ANSWER_MODEL, comparison_counts, digest


def test_shadow_uses_pro_answer_model():
    assert ANSWER_MODEL == "deepseek-v4-pro-ga-260813"


def test_context_digest_is_stable():
    assert digest("frozen context") == digest("frozen context")
    assert digest("frozen context") != digest("changed context")


def test_comparison_counts_all_four_outcomes():
    rows = [{"financebench_id": key} for key in ("a", "b", "c", "d")]
    baseline = {
        "a": {"judge": {"score": 0}}, "b": {"judge": {"score": 1}},
        "c": {"judge": {"score": 0}}, "d": {"judge": {"score": 1}},
    }
    pro = {
        "a": {"judge_status": "ok", "judge": {"score": 1}},
        "b": {"judge_status": "ok", "judge": {"score": 0}},
        "c": {"judge_status": "ok", "judge": {"score": 0}},
        "d": {"judge_status": "ok", "judge": {"score": 1}},
    }
    counts, groups = comparison_counts(rows, baseline, pro)
    assert counts == {"newly_correct": 1, "regressed": 1, "still_incorrect": 1, "still_correct": 1}
    assert groups["newly_correct"] == ["a"] and groups["regressed"] == ["b"]


def test_shadow_has_no_retrieval_or_jina_call_path():
    import inspect
    import scripts.run_answer_model_shadow_v1 as module

    source = inspect.getsource(module)
    assert "JinaReranker" not in source
    assert "rewrite_queries(" not in source
    assert "milvus_client" not in source
