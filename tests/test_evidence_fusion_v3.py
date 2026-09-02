from backend.evidence_fusion_v3 import build_evidence_fusion_v3
from backend.evidence_identity import build_document_id, build_page_id, build_table_id


def _fixture():
    document_id = build_document_id(filename="generic-report.pdf", content_digest="c" * 64)
    page_id = build_page_id(document_id, 5)
    page = {
        "document_id": document_id,
        "page_id": page_id,
        "filename": "generic-report.pdf",
        "page_number": 5,
        "page_text": (
            "Consolidated Balance Sheets. Cash and cash equivalents 120. Accounts receivable 80. "
            "Inventories 30. Total current liabilities 250. Goodwill 900."
        ),
    }
    table = {
        "document_id": document_id,
        "page_id": page_id,
        "table_id": build_table_id(page_id, 1),
        "page_number": 5,
        "start_page": 5,
        "end_page": 5,
        "table_index": 1,
        "parser_backend": "pdfplumber_words",
        "quality_score": 0.9,
        "title": "Consolidated Balance Sheets",
        "columns": ["Metric", "2024"],
        "rows": [
            {"Metric": "Goodwill", "2024": "900"},
            {"Metric": "Cash and cash equivalents", "2024": "120"},
            {"Metric": "Accounts receivable", "2024": "80"},
            {"Metric": "Inventories", "2024": "30"},
            {"Metric": "Total current liabilities", "2024": "250"},
        ],
        "unit": "USD",
        "scale": "millions",
        "before_context": "At December 31",
        "after_context": "See notes.",
    }
    return page, table


def test_fusion_v3_adds_relevance_selected_rows_and_trace():
    page, table = _fixture()

    evidence, trace = build_evidence_fusion_v3(
        "What was the quick ratio in FY2024?",
        [page],
        [table],
        max_context_chars=6000,
    )

    assert "[Page Text Evidence P1]" in evidence
    assert "Relevant rows:" in evidence
    assert "Cash and cash equivalents" in evidence
    assert "Total current liabilities" in evidence
    assert trace["evidence_assembly_version"] == "fusion_v3"
    assert trace["row_selection"][0]["method"] == "bm25_lexical_finance_synonyms"


def test_fusion_v3_preserves_quality_gate_and_context_budget():
    page, table = _fixture()
    table["quality_score"] = 0.2

    evidence, trace = build_evidence_fusion_v3(
        "What was the quick ratio?",
        [page],
        [table],
        max_context_chars=800,
    )

    assert len(evidence) <= 800
    assert "[Trusted Table Evidence]" not in evidence
    assert trace["trusted_table_count"] == 0
    assert trace["rejected_tables"][0]["reason"] == "quality_below_threshold"
    assert trace["quality_threshold"] == 0.65
    assert trace["page_match_threshold"] == 0.35
