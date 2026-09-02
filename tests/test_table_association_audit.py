from scripts.audit_table_association import _text_similarity, classify_page, select_cases, table_like_line_count


def test_select_cases_focuses_missing_records_with_adjacent_tables():
    records = [
        {"financebench_id": "direct", "gold_table_ids": ["t"], "gold_table_diagnostics": {"adjacent_table_count": 2}},
        {"financebench_id": "focused", "gold_table_ids": [], "gold_table_diagnostics": {"adjacent_table_count": 1}},
        {"financebench_id": "other-missing", "gold_table_ids": [], "gold_table_diagnostics": {"adjacent_table_count": 0}},
    ]
    missing, focused = select_cases(records)
    assert [item["financebench_id"] for item in missing] == ["focused", "other-missing"]
    assert [item["financebench_id"] for item in focused] == ["focused"]


def test_classification_precedence_and_labels():
    assert classify_page({"benchmark_boundary_mismatch": True})[0] == "B"
    assert classify_page({"identity_mismatch": True})[0] == "B"
    assert classify_page({"pdf_available": True, "pdf_table_count": 1})[0] == "A"
    assert classify_page({"pdf_available": True, "stored_table_text": True})[0] == "C"
    assert classify_page({"pdf_available": True, "nearby_table_count": 2})[0] == "D"
    assert classify_page({"pdf_available": False})[0] == "E"


def test_table_like_line_count_requires_multiple_numeric_cells_per_row():
    text = "Revenue 2023 100 2022 90\nMargin 2023 20% 2022 18%\nNarrative 42\nCash 100 80"
    assert table_like_line_count(text) == 3


def test_evidence_text_similarity_detects_matching_page_content():
    evidence = "Consolidated Balance Sheets Cash 4,835 Current liabilities 6,369"
    matching = "Table of Contents Consolidated Balance Sheets Cash $ 4,835 Current liabilities 6,369"
    unrelated = "Consolidated Statements of Comprehensive Income Net income 1,320"
    assert _text_similarity(evidence, matching) > _text_similarity(evidence, unrelated) + 0.3
