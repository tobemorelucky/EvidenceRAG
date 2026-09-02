from backend.evidence_block_v1 import build_evidence_blocks_v1, select_evidence_blocks_v1


def _page(number: int, text: str) -> dict:
    return {
        "document_id": "doc-generic",
        "page_id": f"doc-generic:page:{number:06d}",
        "filename": "generic.pdf",
        "page_number": number,
        "page_text": text,
    }


def test_same_page_section_similar_chunks_merge_but_other_page_does_not():
    pages = [_page(1, "Results. Revenue increased in 2024."), _page(2, "Other discussion.")]
    chunks = [
        {"chunk_id": "c1", "document_id": "doc-generic", "filename": "generic.pdf", "page_number": 1, "section": "Results", "text": "Revenue increased in 2024 due to volume", "merged_rank": 1},
        {"chunk_id": "c2", "document_id": "doc-generic", "filename": "generic.pdf", "page_number": 1, "section": "Results", "text": "Revenue increased in 2024 due to higher volume", "merged_rank": 2},
        {"chunk_id": "c3", "document_id": "doc-generic", "filename": "generic.pdf", "page_number": 2, "section": "Results", "text": "Revenue increased in 2024 due to higher volume", "merged_rank": 3},
    ]

    blocks = build_evidence_blocks_v1("Why did revenue increase in 2024?", chunks, page_metadata=pages)
    text_blocks = [block for block in blocks if block["block_type"] in {"text", "chunk_merge"}]

    assert any(block["block_type"] == "chunk_merge" and set(block["source_chunk_ids"]) == {"c1", "c2"} for block in text_blocks)
    assert any(block["source_chunk_ids"] == ["c3"] for block in text_blocks)


def test_table_block_contains_title_header_target_rows_unit_and_source_page():
    pages = [_page(3, "Operating results table")]
    chunks = [{"chunk_id": "c1", "document_id": "doc-generic", "filename": "generic.pdf", "page_number": 3, "text": "Operating results", "merged_rank": 1}]
    tables = [{
        "table_id": "t1", "document_id": "doc-generic", "page_id": pages[0]["page_id"],
        "filename": "generic.pdf", "page_number": 3, "title": "Operating Results",
        "columns": ["Metric", "2024", "2023"],
        "rows": [["Revenue", "120", "100"], ["Employees", "10", "9"]], "unit": "USD millions",
    }]

    blocks = build_evidence_blocks_v1("What was revenue in 2024?", chunks, page_metadata=pages, table_metadata=tables)
    table = next(block for block in blocks if block["block_type"] == "table")

    assert "Operating Results" in table["content"]
    assert "Metric | 2024 | 2023" in table["content"]
    assert "Revenue | 120 | 100" in table["content"]
    assert "USD millions" in table["content"]
    assert table["source_pages"][0]["page_number"] == 3


def test_selection_respects_block_and_character_budget_and_emits_trace():
    pages = [_page(1, "Revenue 2024. Revenue was 120."), _page(2, "Costs 2024. Costs were 80.")]
    chunks = [
        {"chunk_id": "c1", "document_id": "doc-generic", "filename": "generic.pdf", "page_number": 1, "text": "Revenue was 120 in 2024.", "merged_rank": 1},
        {"chunk_id": "c2", "document_id": "doc-generic", "filename": "generic.pdf", "page_number": 2, "text": "Costs were 80 in 2024.", "merged_rank": 2},
    ]

    selected, context, trace = select_evidence_blocks_v1(
        "What was revenue in 2024?", chunks, page_metadata=pages, max_blocks=1, max_context_chars=1000,
    )

    assert len(selected) == 1
    assert len(context) <= 1000
    assert trace["selected_block_count"] == 1
    assert trace["block_scores"][0]["score_components"]
