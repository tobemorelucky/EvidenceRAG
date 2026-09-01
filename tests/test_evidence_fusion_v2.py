from backend.evidence_fusion_v2 import build_evidence_fusion_v2
from backend.evidence_identity import build_document_id, build_page_id, build_table_id


def _page_and_table():
    document_id = build_document_id(filename="generic-report.pdf", content_digest="b" * 64)
    page_id = build_page_id(document_id, 7)
    page = {
        "document_id": document_id,
        "page_id": page_id,
        "filename": "generic-report.pdf",
        "page_number": 7,
        "page_text": "Statements of Operations. Revenue was 120 in 2024 and 100 in 2023. " * 50,
    }
    table = {
        "document_id": document_id,
        "page_id": page_id,
        "table_id": build_table_id(page_id, 1),
        "page_number": 7,
        "start_page": 7,
        "end_page": 7,
        "table_index": 1,
        "parser_backend": "pdfplumber_words",
        "quality_score": 0.92,
        "title": "Statements of Operations",
        "columns": ["Metric", "2024", "2023"],
        "rows": [
            {"Metric": "Revenue", "2024": "120", "2023": "100"},
            {"Metric": "Operating income", "2024": "24", "2023": "18"},
        ],
        "unit": "USD",
        "scale": "millions",
        "before_context": "For the years ended December 31",
        "after_context": "See accompanying notes.",
    }
    return page, table


def test_fusion_keeps_page_text_and_adds_trusted_table():
    page, table = _page_and_table()

    evidence, trace = build_evidence_fusion_v2(
        "What was operating income in 2024?",
        [page],
        [table],
        max_context_chars=6000,
    )

    assert "[Page Text Evidence P1]" in evidence
    assert "[Trusted Table Evidence]" in evidence
    assert "Table title: Statements of Operations" in evidence
    assert "Header/columns: Metric | 2024 | 2023" in evidence
    assert "Operating income" in evidence
    assert "Unit: USD" in evidence
    assert "Scale: millions" in evidence
    assert "Nearby text before:" in evidence
    assert trace["evidence_assembly_version"] == "fusion_v2"
    assert trace["page_text_included_count"] == 1
    assert trace["trusted_table_count"] == 1


def test_fusion_keeps_page_text_when_table_is_rejected_without_lowering_gate():
    page, table = _page_and_table()
    table["quality_score"] = 0.2

    evidence, trace = build_evidence_fusion_v2(
        "What was revenue?",
        [page],
        [table],
        max_context_chars=6000,
    )

    assert "[Page Text Evidence P1]" in evidence
    assert "[Trusted Table Evidence]" not in evidence
    assert trace["trusted_table_count"] == 0
    assert trace["rejected_tables"][0]["reason"] == "quality_below_threshold"
    assert trace["quality_threshold"] == 0.65
    assert trace["page_match_threshold"] == 0.35


def test_fusion_respects_70_25_5_caps_and_total_budget():
    page, table = _page_and_table()

    evidence, trace = build_evidence_fusion_v2(
        "What was revenue?",
        [page],
        [table],
        max_context_chars=1000,
    )
    budget = trace["budget"]

    assert len(evidence) <= 1000
    assert budget["metadata_cap_chars"] == 50
    assert budget["table_cap_chars"] == 250
    assert budget["metadata_used_chars"] <= 50
    assert budget["table_used_chars"] <= 250
    assert budget["page_text_budget_chars"] >= 690
    assert trace["table_contribution_chars"] == budget["table_used_chars"]


def test_fusion_uses_unused_table_budget_for_page_text():
    page, _ = _page_and_table()

    evidence, trace = build_evidence_fusion_v2(
        "What was revenue?",
        [page],
        [],
        max_context_chars=1000,
    )

    assert len(evidence) <= 1000
    assert trace["table_contribution_chars"] == 0
    assert trace["budget"]["page_text_budget_chars"] > 900


def test_fusion_preserves_query_relevant_numeric_row_near_page_end():
    page, _ = _page_and_table()
    page["page_text"] = "\n".join(
        ["General risk disclosure without relevant values."] * 100
        + ["Metric 2024 2023", "Operating income 24 18", "Footnote for operating income."]
    )

    evidence, _ = build_evidence_fusion_v2(
        "What was operating income in 2024?",
        [page],
        [],
        max_context_chars=1000,
    )

    assert "Operating income 24 18" in evidence
