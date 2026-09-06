from scripts.run_query_rewrite_jina_answer_shadow_v1 import _classification, summarize


def _route(score, context_hit, answer_tokens=10, judge_tokens=5):
    return {
        "answer_status": "ok", "judge_status": "ok",
        "judge": {"score": score, "verdict": "correct" if score else "incorrect", "usage": {"total_tokens": judge_tokens}},
        "usage": {"total_tokens": answer_tokens}, "latency_ms": {"answer": 2, "judge": 1},
        "evidence_metrics": {"evidence_context_hit": context_hit, "required_number_hit": True, "required_period_hit": True},
    }


def test_classifies_converted_and_unused_context_gain():
    converted = {"baseline": _route(0, False), "rewrite_jina": _route(1, True)}
    unused = {"baseline": _route(0, False), "rewrite_jina": _route(0, True)}
    assert _classification(converted) == "retrieval_gain_converted"
    assert _classification(unused) == "retrieval_gain_unused"


def test_classifies_context_improved_answer_regression():
    record = {"baseline": _route(1, False), "rewrite_jina": _route(0, True)}
    assert _classification(record) == "retrieval_regression"


def test_summary_counts_tokens_repairs_and_regressions():
    records = [
        {"question_id": "a", "baseline": _route(0, False), "rewrite_jina": _route(1, True), "classification": "retrieval_gain_converted"},
        {"question_id": "b", "baseline": _route(1, True), "rewrite_jina": _route(0, True), "classification": "no_change"},
    ]
    result = summarize(records)
    assert result["baseline"]["correct"] == result["rewrite_jina"]["correct"] == 1
    assert result["repaired"] == result["regressions"] == 1
    assert result["rewrite_jina"]["answer_token"] == 20
    assert result["jina_calls"] == 0
