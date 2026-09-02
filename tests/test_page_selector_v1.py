from backend.page_selector_v1 import select_pages_v1


def _page(number: int, text: str, *, page_id: str | None = None) -> dict:
    return {
        "document_id": "doc-generic",
        "page_id": page_id or f"doc-generic:page:{number:06d}",
        "filename": "generic.pdf",
        "page_number": number,
        "page_text": text,
        "page_candidate_rank": number,
        "expanded_from": [{"chunk_id": f"c{number}", "merged_rank": number, "distance": 0}],
    }


def test_selector_returns_weighted_trace_and_stable_ids():
    chunks = [
        {"chunk_id": "c1", "filename": "generic.pdf", "page_number": 1, "score": 1.0},
        {"chunk_id": "c2", "filename": "generic.pdf", "page_number": 2, "score": 0.5},
    ]
    pages = [_page(1, "Overview and background"), _page(2, "Revenue summary for 2024")]

    selected, trace = select_pages_v1("What was revenue in 2024?", chunks, page_records=pages, top_k=2)

    assert len(selected) == 2
    assert trace["weights"] == {
        "best_chunk": 0.5,
        "multi_chunk_support": 0.2,
        "title_section_lexical": 0.15,
        "table_structure": 0.1,
        "period_year": 0.05,
    }
    assert trace["page_scores"][0]["page_id"]
    assert abs(sum(trace["page_scores"][0]["contributions"].values()) - trace["page_scores"][0]["score"]) < 1e-7


def test_relevant_table_structure_can_break_equal_chunk_support():
    chunks = [
        {"chunk_id": "c1", "filename": "generic.pdf", "page_number": 1, "score": 1.0},
        {"chunk_id": "c2", "filename": "generic.pdf", "page_number": 2, "score": 1.0},
    ]
    pages = [_page(1, "Annual report"), _page(2, "Annual report")]
    pages[1]["expanded_from"][0]["merged_rank"] = 1
    tables = [{
        "document_id": "doc-generic",
        "page_id": "doc-generic:page:000002",
        "filename": "generic.pdf",
        "page_number": 2,
        "title": "Operating results",
        "columns": ["2024", "2023"],
        "rows": [{"row_label": "Net revenue"}],
    }]

    selected, trace = select_pages_v1(
        "What was net revenue in 2024?",
        chunks,
        page_records=pages,
        table_metadata=tables,
        top_k=1,
    )

    assert selected[0]["page_number"] == 2
    assert trace["selected_pages"][0]["components"]["table_structure"] > 0


def test_empty_input_is_deterministic():
    selected, trace = select_pages_v1("generic question", [], page_records=[], top_k=8)
    assert selected == []
    assert trace["page_scores"] == []


def test_neighbor_sources_are_not_counted_as_direct_multi_chunk_support():
    chunks = [
        {"chunk_id": "c1", "filename": "generic.pdf", "page_number": 1, "score": 1.0, "merged_rank": 1},
        {"chunk_id": "c2", "filename": "generic.pdf", "page_number": 3, "score": 1.0, "merged_rank": 2},
    ]
    page = _page(2, "Relevant heading")
    page["expanded_from"] = [
        {"chunk_id": "c1", "merged_rank": 1, "distance": 1},
        {"chunk_id": "c2", "merged_rank": 2, "distance": 1},
    ]

    _, trace = select_pages_v1("Relevant heading", chunks, page_records=[page], top_k=1)

    assert trace["selected_pages"][0]["supporting_chunks"] == 0
    assert trace["selected_pages"][0]["components"]["multi_chunk_support"] == 0.0
