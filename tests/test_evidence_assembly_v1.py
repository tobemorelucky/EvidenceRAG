import random

from backend.evidence_assembly_v1 import build_evidence_assembly_v1
from backend.evidence_identity import build_document_id, build_page_id, build_table_id
from backend.table_quality import table_is_eligible


def _trusted_page_and_table():
    document_id = build_document_id(filename="generic-report.pdf", content_digest="a" * 64)
    page_id = build_page_id(document_id, 4)
    page = {
        "document_id": document_id,
        "page_id": page_id,
        "filename": "generic-report.pdf",
        "page_number": 4,
        "page_text": (
            "Consolidated Statements of Operations. USD millions. "
            "Metric 2024 2023 Revenue 120 100 Operating income 24 18."
        ),
    }
    table = {
        "document_id": document_id,
        "page_id": page_id,
        "table_id": build_table_id(page_id, 1),
        "page_number": 4,
        "start_page": 4,
        "end_page": 4,
        "table_index": 1,
        "parser_backend": "pdfplumber_words",
        "quality_score": 0.92,
        "title": "Consolidated Statements of Operations",
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


def test_random_100_table_page_associations_use_page_id():
    rng = random.Random(20260901)
    pairs = []
    for document_index in range(10):
        digest = f"{document_index:064x}"
        document_id = build_document_id(filename="ignored.pdf", content_digest=digest)
        for page_number in range(10):
            page_id = build_page_id(document_id, page_number)
            pairs.append((
                {"document_id": document_id, "page_id": page_id, "page_number": page_number},
                {
                    "document_id": document_id,
                    "page_id": page_id,
                    "table_id": build_table_id(page_id, 1),
                    "page_number": page_number,
                    "start_page": page_number,
                    "end_page": page_number,
                },
            ))
    rng.shuffle(pairs)

    assert len(pairs) == 100
    assert all(page["page_id"] == table["page_id"] for page, table in pairs)
    assert len({table["table_id"] for _, table in pairs}) == 100


def test_evidence_builder_uses_trusted_table_and_target_rows():
    page, table = _trusted_page_and_table()

    evidence, trace = build_evidence_assembly_v1(
        "What was operating income in 2024?",
        [page],
        [table],
        max_context_chars=4000,
    )

    assert "[Trusted Table Evidence]" in evidence
    assert "Operating income" in evidence
    assert "Unit: USD" in evidence
    assert "Scale: millions" in evidence
    assert trace["trusted_table_count"] == 1
    assert trace["page_text_fallback_count"] == 0


def test_evidence_builder_falls_back_to_original_page_text_for_untrusted_table():
    page, table = _trusted_page_and_table()
    table["quality_score"] = 0.2

    evidence, trace = build_evidence_assembly_v1(
        "What was revenue?",
        [page],
        [table],
        max_context_chars=4000,
    )

    assert "[Page Text Evidence]" in evidence
    assert page["page_text"] in evidence
    assert "[Trusted Table Evidence]" not in evidence
    assert trace["trusted_table_count"] == 0
    assert trace["page_text_fallback_count"] == 1
    assert trace["rejected_tables"][0]["reason"] == "quality_below_threshold"


def test_evidence_builder_rejects_wrong_page_id_even_when_table_is_structurally_valid():
    page, table = _trusted_page_and_table()
    table["page_id"] = build_page_id(page["document_id"], 3)

    eligible, reason, _ = table_is_eligible(table, page)

    assert eligible is False
    assert reason == "page_id_mismatch"


def test_evidence_builder_preserves_existing_context_budget():
    page, table = _trusted_page_and_table()
    page["page_text"] += " extra" * 2000

    evidence, trace = build_evidence_assembly_v1(
        "What was revenue?",
        [page],
        [table],
        max_context_chars=700,
    )

    assert len(evidence) <= 700
    assert trace["answer_context_max_chars"] == 700

