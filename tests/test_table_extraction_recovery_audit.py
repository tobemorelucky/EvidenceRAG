from scripts.audit_table_extraction_recovery import classify_recovery, select_failure_pages


def test_select_failure_pages_uses_only_post_contract_missing_pages():
    payload = {"cases": [{
        "financebench_id": "q1",
        "question": "question",
        "gold_document": "report.pdf",
        "gold_page_audits": [
            {"benchmark_evidence_page_number": 8, "post_alignment": {"status": "resolved_by_page_boundary"}},
            {"benchmark_evidence_page_number": 9, "post_alignment": {"status": "still_missing_after_page_boundary_fix", "classification": "A"}},
        ],
    }]}
    assert select_failure_pages(payload) == [{
        "financebench_id": "q1",
        "question": "question",
        "filename": "report.pdf",
        "page_number": 9,
        "previous_classification": "A",
    }]


def test_recovery_classification_is_deterministic():
    assert classify_recovery({"image_count": 1, "raw_text_chars": 20})[0] == "C"
    assert classify_recovery({"evidence_prose_like": True, "evidence_text_recall_in_page": 0.9, "rejected_word_candidates": 2})[0] == "D"
    assert classify_recovery({"accepted_word_candidates": 1, "raw_text_chars": 1000})[0] == "A"
    assert classify_recovery({"rejected_word_candidates": 1, "raw_text_chars": 1000})[0] == "B"
    assert classify_recovery({"table_like_text_lines": 5, "raw_text_chars": 1000})[0] == "D"
