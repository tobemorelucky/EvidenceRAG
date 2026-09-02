from backend.table_retrieval import (
    build_table_document,
    fuse_table_routes,
    merge_text_and_table_pages,
)


def _table():
    return {
        "document_id": "doc_1",
        "page_id": "doc_1:page:000005",
        "page_number": 5,
        "table_id": "doc_1:page:000005:table:0001",
        "filename": "generic.pdf",
        "title": "Consolidated Balance Sheets",
        "columns": ["Metric", "2024", "2023"],
        "rows": [
            {"Metric": "Cash and cash equivalents", "2024": "120", "2023": "100"},
            {"Metric": "Current liabilities", "2024": "250", "2023": "230"},
        ],
        "unit": "USD",
        "scale": "millions",
        "before_context": "At December 31, 2024 and 2023",
        "after_context": "See note 12 for details",
        "quality_score": 0.9,
    }


def test_table_document_contains_required_metadata_without_value_matrix():
    document = build_table_document(_table())

    assert document["document_id"] == "doc_1"
    assert document["page_id"] == "doc_1:page:000005"
    assert document["page_number"] == 5
    assert document["table_id"].endswith("table:0001")
    assert "Consolidated Balance Sheets" in document["search_text"]
    assert "Cash and cash equivalents" in document["search_text"]
    assert "Current liabilities" in document["search_text"]
    assert "120" not in document["search_text"]
    assert "250" not in document["search_text"]
    assert "2024" in document["headers"]


def test_table_route_rrf_keeps_dense_and_bm25_ranks():
    dense = [
        {"table_id": "t1", "filename": "a.pdf", "page_number": 1},
        {"table_id": "t2", "filename": "a.pdf", "page_number": 2},
    ]
    bm25 = [
        {"table_id": "t2", "filename": "a.pdf", "page_number": 2},
        {"table_id": "t3", "filename": "a.pdf", "page_number": 3},
    ]

    fused = fuse_table_routes(dense, bm25, top_k=3)

    assert fused[0]["table_id"] == "t2"
    assert fused[0]["dense_rank"] == 2
    assert fused[0]["bm25_rank"] == 1
    assert {item["table_id"] for item in fused} == {"t1", "t2", "t3"}


def test_shadow_page_fusion_does_not_mutate_text_ranking():
    text = [
        {"filename": "a.pdf", "page_number": 1},
        {"filename": "a.pdf", "page_number": 2},
    ]
    tables = [
        {"filename": "a.pdf", "page_number": 2, "table_id": "t2"},
        {"filename": "b.pdf", "page_number": 8, "table_id": "t8"},
    ]

    combined = merge_text_and_table_pages(text, tables)

    assert text == [
        {"filename": "a.pdf", "page_number": 1},
        {"filename": "a.pdf", "page_number": 2},
    ]
    assert combined[0]["filename"] == "a.pdf"
    assert combined[0]["page_number"] == 2
    assert {item["filename"] for item in combined} == {"a.pdf", "b.pdf"}
