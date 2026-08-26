import json

from scripts.evaluate_financebench_oracle import build_summary, classify_failure, parse_gold_pages


def _row(records):
    return {
        "doc_name": "GENERIC_REPORT",
        "evidence": json.dumps(records),
    }


def test_parse_gold_pages_prefers_full_page_and_deduplicates_page():
    row = _row(
        [
            {
                "doc_name": "GENERIC_REPORT",
                "evidence_page_num": 7,
                "evidence_text": "short evidence",
                "evidence_text_full_page": "complete page text with table headers",
            },
            {
                "doc_name": "GENERIC_REPORT",
                "evidence_page_num": 7,
                "evidence_text": "another snippet",
                "evidence_text_full_page": "complete page text with table headers",
            },
        ]
    )

    documents = parse_gold_pages(row, "gold_page")

    assert len(documents) == 1
    assert documents[0]["filename"] == "GENERIC_REPORT.pdf"
    assert documents[0]["page_number"] == 7
    assert documents[0]["text"] == "complete page text with table headers"
    assert documents[0]["full_page_available"] is True


def test_parse_gold_evidence_uses_snippet_and_accepts_numeric_page_string():
    row = _row(
        [
            {
                "evidence_page_num": "3.0",
                "evidence_text": "reported value was 42",
                "evidence_text_full_page": "full page",
            }
        ]
    )

    documents = parse_gold_pages(row, "gold_evidence")

    assert documents[0]["page_number"] == 3
    assert documents[0]["text"] == "reported value was 42"
    assert documents[0]["type"] == "oracle_gold_evidence"


def test_oracle_failure_classifier_distinguishes_unjudged_and_execution_gap():
    base = {
        "documents": [{"text": "evidence"}],
        "evidence": "evidence",
        "task_spec": {"task_type": "calculation"},
        "coverage": {"status": "complete"},
        "calculation": None,
        "error": "",
    }

    assert classify_failure(**base, judge={}) == "not_judged"
    assert classify_failure(**base, judge={"score": 0, "verdict": "incorrect"}) == "gold_page_calculation_not_executed"


def test_oracle_summary_marks_fixed_seen_regression():
    summary = build_summary(
        [
            {
                "task_type": "lookup",
                "is_calculation": False,
                "calculation": None,
                "failure_type": "none",
                "judge_result": {"score": 1},
                "answer_input_tokens": 100,
                "usage": {"total_tokens": 110},
                "latency_ms": 20,
            },
            {
                "task_type": "calculation",
                "is_calculation": True,
                "calculation": {"result": "1"},
                "failure_type": "gold_page_answer_incorrect",
                "judge_result": {"score": 0},
                "answer_input_tokens": 120,
                "usage": {"total_tokens": 130},
                "latency_ms": 30,
            },
        ]
    )

    assert summary["accuracy"] == 0.5
    assert summary["answer_input_tokens"] == 220
    assert summary["benchmark_status"] == "fixed_seen_regression"
