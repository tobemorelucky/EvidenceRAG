from scripts.evaluate_evidence_fusion_v3_ab import render_markdown, summarize


def _version(*, row_hit: bool, number_hit: bool, chars: int):
    return {
        "evidence_chars": chars,
        "gold_row_hit": row_hit,
        "required_number_hit": number_hit,
        "required_period_hit": True,
        "table_contribution": {
            "trusted_table_count": 1,
            "trusted_table_ids": ["t1"],
            "rejected_table_count": 0,
            "chars": 100,
            "ratio": 0.1,
            "gold_row_hit": row_hit,
            "required_number_hit": number_hit,
        },
        "row_selection": [],
    }


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
        "required_numbers": ["120"],
        "required_periods": ["2024"],
        "fusion_v2": _version(row_hit=False, number_hit=False, chars=900),
        "fusion_v3": _version(row_hit=True, number_hit=True, chars=950),
        "raw_trusted_table_ceiling": {"required_number_hit": True, "gold_row_hit": True},
    }


def test_v3_summary_reports_gains_and_version_metrics():
    summary = summarize([_record()])

    assert summary["fusion_v2"]["gold_row_hit"] == 0.0
    assert summary["fusion_v3"]["gold_row_hit"] == 1.0
    assert summary["gold_row_gains"] == 1
    assert summary["gold_row_regressions"] == 0
    assert summary["required_number_gains"] == 1
    assert summary["raw_trusted_table_required_number_ceiling"] == 1.0


def test_v3_markdown_declares_frozen_pages_and_no_external_calls():
    record = _record()
    payload = {
        "frozen_report": "frozen.json",
        "variant": "C_global_local_merge",
        "summary": summarize([record]),
        "records": [record],
    }

    report = render_markdown(payload)

    assert "Retrieval=0, Jina=0, LLM=0, Judge=0" in report
    assert "Fusion v2; B: Fusion v3" in report
    assert "Frozen selected/context pages" in report
