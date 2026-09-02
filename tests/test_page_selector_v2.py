from backend.page_selector_v2 import build_evidence_groups, select_page_groups_v2


def _page(document: str, number: int, text: str, chunk: str, *, distance: int = 0) -> dict:
    return {
        "document_id": document,
        "page_id": f"{document}:page:{number:06d}",
        "filename": f"{document}.pdf",
        "page_number": number,
        "page_text": text,
        "page_candidate_rank": number + 1,
        "expanded_from": [{"chunk_id": chunk, "merged_rank": number + 1, "distance": distance}],
    }


def test_groups_only_related_adjacent_pages_in_same_document():
    pages = [
        _page("doc-a", 1, "Revenue 2024", "shared"),
        _page("doc-a", 2, "Revenue table", "shared", distance=1),
        _page("doc-a", 4, "Revenue note", "other"),
        _page("doc-b", 2, "Revenue table", "shared"),
    ]
    chunks = [{"chunk_id": "shared", "merged_rank": 1}, {"chunk_id": "other", "merged_rank": 2}]

    groups, _ = build_evidence_groups("Revenue 2024", chunks, page_records=pages)
    page_sets = [{(item["record"]["document_id"], item["page_number"]) for item in group["pages"]} for group in groups]

    assert {("doc-a", 1), ("doc-a", 2)} in page_sets
    assert not any({("doc-a", 2), ("doc-a", 4)} <= item for item in page_sets)
    assert not any({("doc-a", 2), ("doc-b", 2)} <= item for item in page_sets)


def test_coverage_greedy_selects_group_with_new_table_and_year_evidence():
    pages = [
        _page("doc-a", 1, "General revenue discussion", "c1"),
        _page("doc-a", 2, "Results table 2024", "c1", distance=1),
        _page("doc-a", 7, "General revenue discussion", "c2"),
    ]
    chunks = [{"chunk_id": "c1", "merged_rank": 1}, {"chunk_id": "c2", "merged_rank": 2}]
    tables = [{
        "document_id": "doc-a",
        "page_id": "doc-a:page:000002",
        "filename": "doc-a.pdf",
        "page_number": 2,
        "title": "Results",
        "columns": ["2024", "2023"],
        "rows": [{"row_label": "Revenue"}],
    }]

    selected, trace = select_page_groups_v2(
        "What was revenue in 2024?", chunks, page_records=pages, table_metadata=tables, page_budget=2,
    )

    assert {item["page_number"] for item in selected} == {1, 2}
    assert len(trace["selected_groups"][0]["pages"]) == 2
    assert trace["selected_groups"][0]["new_coverage"]["table_header"]
    assert trace["selected_groups"][0]["new_coverage"]["year"] == ["2024"]


def test_empty_page_budget_is_deterministic():
    selected, trace = select_page_groups_v2("question", [], page_records=[], page_budget=0)
    assert selected == []
    assert trace["evidence_groups"] == []
