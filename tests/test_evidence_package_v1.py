from copy import deepcopy

import pytest

from backend.evidence_package_v1 import build_evidence_packages_v1, render_evidence_packages_v1


def _chunk(chunk_id, rank, document_id="doc-1", page_id="doc-1:page:000010", page_number=10):
    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "page_id": page_id,
        "filename": "ACME_2023_10K.pdf",
        "page_number": page_number,
        "text": f"Evidence text for {chunk_id} with 2023 values.",
        "jina_rank": rank,
        "company": "ACME",
        "report_year": 2023,
        "table_title": "Income Statement",
    }


def _table(table_id="table-1", document_id="doc-1", page_id="doc-1:page:000010", page_number=10):
    return {
        "table_id": table_id,
        "document_id": document_id,
        "page_id": page_id,
        "filename": "ACME_2023_10K.pdf",
        "page_number": page_number,
        "title": "Income Statement",
        "columns": ["Metric", "2023", "2022"],
        "rows": [{"_raw_line": "Revenue 120 100"}],
        "unit": "USD",
        "scale": "millions",
    }


def test_packages_use_best_jina_chunk_as_anchor_and_group_exact_page():
    chunks = [_chunk("c2", 2), _chunk("c1", 1), _chunk("c3", 3, page_id="doc-1:page:000011", page_number=11)]
    original = deepcopy(chunks)

    packages = build_evidence_packages_v1(chunks, [_table()])

    assert chunks == original
    assert len(packages) == 2
    assert packages[0]["anchor_chunk_id"] == "c1"
    assert [chunk["chunk_id"] for chunk in packages[0]["text_chunks"]] == ["c1", "c2"]
    assert [table["table_id"] for table in packages[0]["related_tables"]] == ["table-1"]
    assert packages[0]["metadata"] == {"entity": "ACME", "period": "2023", "metric": "Income Statement"}


def test_tables_must_match_both_document_and_page_and_never_use_adjacent_page():
    chunks = [_chunk("c1", 1)]
    tables = [
        _table("correct"),
        _table("wrong-doc", document_id="doc-2"),
        _table("adjacent", page_id="doc-1:page:000011", page_number=11),
    ]

    packages = build_evidence_packages_v1(chunks, tables)

    assert [table["table_id"] for table in packages[0]["related_tables"]] == ["correct"]


def test_every_input_must_be_an_identified_ranked_jina_chunk():
    invalid = _chunk("c1", 0)
    with pytest.raises(ValueError, match="ranked Jina chunk"):
        build_evidence_packages_v1([invalid], [])


def test_render_respects_budget_and_reports_dropped_chunks_and_table_contribution():
    chunks = [_chunk("c1", 1), _chunk("c2", 2), _chunk("c3", 3, page_id="doc-1:page:000011", page_number=11)]
    chunks[0]["text"] = "A" * 180
    chunks[1]["text"] = "B" * 180
    chunks[2]["text"] = "C" * 180
    packages = build_evidence_packages_v1(chunks, [_table()])

    context, trace = render_evidence_packages_v1(packages, max_chars=500)

    assert len(context) <= 500
    assert "c1" in trace["included_chunk_ids"]
    assert trace["dropped_chunks"]
    assert set(trace["included_chunk_ids"]).isdisjoint(item["chunk_id"] for item in trace["dropped_chunks"])


def test_package_ids_are_deterministic():
    chunks = [_chunk("c1", 1)]
    first = build_evidence_packages_v1(chunks, [_table()])
    second = build_evidence_packages_v1(chunks, [_table()])
    assert first == second


def test_table_is_not_rendered_when_its_package_anchor_does_not_fit():
    chunk = _chunk("c1", 1)
    chunk["text"] = "A" * 500
    packages = build_evidence_packages_v1([chunk], [_table()])

    context, trace = render_evidence_packages_v1(packages, max_chars=100)

    assert context == ""
    assert trace["included_table_ids"] == []
    assert trace["dropped_chunks"][0]["reason"] == "anchor_context_budget"
