from backend.evidence_coverage import build_document_scoped_supplemental_query
from backend import rag_orchestrator, rag_utils


def test_supplemental_query_uses_missing_fields_periods_and_statement_anchors():
    query = build_document_scoped_supplemental_query(
        "Calculate the ratio.",
        {
            "required_fields": ["operating_income", "revenue"],
            "required_periods": ["2024"],
            "statement_types": ["income_statement"],
        },
        {"missing_fields": ["revenue"], "structured_missing": ["row_supported"]},
    )

    assert "revenue" in query.lower()
    assert "2024" in query
    assert "income statement" in query.lower()


def test_document_scoped_retrieval_applies_filename_filter_once(monkeypatch):
    captured = {}

    def fake_retrieve(query, *, top_k, filter_expr, retrieval_scope):
        captured.update(query=query, top_k=top_k, filter_expr=filter_expr, retrieval_scope=retrieval_scope)
        return {"docs": [{"filename": "selected.pdf", "page_number": 8}]}

    monkeypatch.setattr(rag_utils, "_retrieve_leaf_chunks", fake_retrieve)
    result = rag_utils.retrieve_document_scoped_candidates(
        "missing evidence",
        ["selected.pdf", "selected.pdf"],
        top_k=7,
    )

    assert result == [{"filename": "selected.pdf", "page_number": 8}]
    assert 'filename == "selected.pdf"' in captured["filter_expr"]
    assert captured["top_k"] == 7
    assert captured["retrieval_scope"] == "supplemental:document_scoped_once"


def test_partial_supplement_opens_hit_and_adjacent_pages_within_document(monkeypatch):
    monkeypatch.setenv("SUPPLEMENTAL_FIND_ENABLED", "true")
    calls = {"retrieval": 0}

    def fake_retrieve(query, filenames, *, top_k):
        calls["retrieval"] += 1
        assert filenames == ["selected.pdf"]
        return [
            {"filename": "selected.pdf", "page_number": 8},
            {"filename": "other.pdf", "page_number": 40},
        ]

    def fake_open(pages, limit):
        assert pages == [
            {"filename": "selected.pdf", "page_number": 7},
            {"filename": "selected.pdf", "page_number": 8},
            {"filename": "selected.pdf", "page_number": 9},
        ]
        return [{**page, "text": f"page {page['page_number']}"} for page in pages]

    monkeypatch.setattr(rag_orchestrator, "retrieve_document_scoped_candidates", fake_retrieve)
    monkeypatch.setattr(rag_orchestrator, "open_pages", fake_open)
    documents, trace = rag_orchestrator._supplement_partial_evidence(
        "Calculate the ratio for 2024.",
        {
            "company": "Example Co",
            "required_fields": ["revenue"],
            "required_periods": ["2024"],
            "statement_types": ["income_statement"],
        },
        [{"filename": "selected.pdf", "page_number": 6, "text": "initial"}],
        {"status": "partial", "missing_fields": ["revenue"]},
    )

    assert calls["retrieval"] == 1
    assert trace["supplemental_triggered"] is True
    assert trace["searched_documents"] == ["selected.pdf"]
    assert [item["page_number"] for item in trace["new_pages"]] == [7, 8, 9]
    assert len(documents) == 4


def test_supplement_does_not_run_for_complete_coverage(monkeypatch):
    monkeypatch.setenv("SUPPLEMENTAL_FIND_ENABLED", "true")
    monkeypatch.setattr(
        rag_orchestrator,
        "retrieve_document_scoped_candidates",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not retrieve")),
    )

    documents, trace = rag_orchestrator._supplement_partial_evidence(
        "What was revenue?",
        {"required_fields": ["revenue"]},
        [{"filename": "selected.pdf", "page_number": 2}],
        {"status": "complete"},
    )

    assert len(documents) == 1
    assert trace["supplemental_triggered"] is False
