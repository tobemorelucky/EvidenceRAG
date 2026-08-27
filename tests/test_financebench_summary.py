import json

from scripts.summarize_financebench_experiment import _summarize_split


def _write_jsonl(path, records):
    path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")


def test_summary_reports_page_loss_structured_execution_and_supplement(tmp_path):
    answers = tmp_path / "answers.jsonl"
    judges = tmp_path / "judges.jsonl"
    _write_jsonl(
        answers,
        [
            {
                "financebench_id": "q1",
                "langsmith_trace_id": "run-1",
                "citations": [{"filename": "report.pdf", "page_number": 8}],
                "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                "evaluation_latency": {"total_ms": 500},
                "rag_trace": {
                    "task_type": "calculation",
                    "page_first_selected_pages": [{"filename": "report.pdf", "page_number": 7}],
                    "answer_context_pages": [{"filename": "report.pdf", "page_number": 8}],
                    "evidence_frame_count": 4,
                    "frames_used_for_execution": 2,
                    "supplemental_triggered": True,
                    "coverage_before": {"answerable": False},
                    "coverage_after": {"answerable": True},
                    "remote_attempt_count": 1,
                    "remote_success": True,
                    "remote_rerank_input_chars": 900,
                },
            },
            {
                "financebench_id": "q2",
                "langsmith_trace_id": "run-2",
                "citations": [{"filename": "other.pdf", "page_number": 2}],
                "usage": {"total_tokens": 80},
                "evaluation_latency": {"total_ms": 300},
                "rag_trace": {
                    "task_type": "lookup",
                    "page_first_selected_pages": [{"filename": "report.pdf", "page_number": 9}],
                    "answer_context_pages": [{"filename": "other.pdf", "page_number": 2}],
                },
            },
        ],
    )
    _write_jsonl(
        judges,
        [
            {"run_id": "run-1", "score": 1, "verdict": "correct"},
            {"run_id": "run-2", "score": 0, "verdict": "incorrect"},
        ],
    )
    gold = {"q1": {("report", 8)}, "q2": {("report", 9)}}

    summary, identifiers = _summarize_split("fixed", answers, judges, gold)

    assert identifiers == {"q1", "q2"}
    assert summary["accuracy"] == 0.5
    assert summary["candidate_gold_page_hits"] == 1
    assert summary["context_gold_page_hits"] == 1
    assert summary["candidate_to_context_losses"] == 1
    assert summary["structured_executions"] == 1
    assert summary["supplemental_recovered"] == 1
    assert summary["task_types"]["calculation"]["accuracy"] == 1.0
    assert summary["average_answer_tokens"] == 100.0
    assert summary["average_latency_ms"] == 400.0
