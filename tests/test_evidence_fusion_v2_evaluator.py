from scripts.evaluate_evidence_fusion_v2_ab import render_markdown, summarize


def _record():
    return {
        "financebench_id": "generic_1",
        "question": "What was revenue?",
        "question_types": ["lookup", "table_likely"],
        "gold_evidence_pages": [{"filename": "generic.pdf", "page_number": 2}],
        "frozen_retrieval": {
            "selected_pages": [{"filename": "generic.pdf", "page_number": 2}],
            "context_pages": [{"filename": "generic.pdf", "page_number": 2}],
            "selected_pages_hash": "frozen",
        },
        "average_inputs": {"selected_page_count": 1, "loaded_table_count": 1},
        "evidence_chars": {"baseline_page_text": 100, "fusion_v2": 100},
        "gold_row_hit": {"baseline_page_text": True, "fusion_v2": True},
        "required_numbers": ["120"],
        "required_number_hit": {"baseline_page_text": True, "fusion_v2": True},
        "required_periods": ["2024"],
        "required_period_hit": {"baseline_page_text": True, "fusion_v2": True},
        "table_contribution": {
            "trusted_table_count": 1,
            "trusted_table_ids": ["t1"],
            "rejected_table_count": 0,
            "chars": 25,
            "ratio": 0.25,
            "gold_row_hit_in_table_layer": True,
            "required_number_hit_in_table_layer": True,
            "required_number_recovered_over_baseline": False,
        },
    }


def test_fusion_summary_reports_requested_metrics_and_table_contribution():
    summary = summarize([_record()])

    assert summary["average_baseline_chars"] == 100
    assert summary["average_fusion_chars"] == 100
    assert summary["fusion_gold_row_hit"] == 1.0
    assert summary["fusion_required_number_hit"] == 1.0
    assert summary["fusion_required_period_hit"] == 1.0
    assert summary["table_contribution_coverage"] == 1.0
    assert summary["average_table_contribution_chars"] == 25


def test_fusion_markdown_declares_frozen_and_zero_external_calls():
    record = _record()
    payload = {
        "frozen_report": "frozen.json",
        "variant": "C_global_local_merge",
        "summary": summarize([record]),
        "records": [record],
    }

    report = render_markdown(payload)

    assert "Retrieval=0, Jina=0, LLM=0, Judge=0" in report
    assert "Frozen selected/context pages" in report
    assert "Table contribution" in report
