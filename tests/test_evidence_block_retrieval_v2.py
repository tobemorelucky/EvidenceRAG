from backend.evidence_block_retrieval_v2 import (
    build_evidence_blocks_v2,
    fuse_block_routes,
    index_document,
)
from scripts.evaluate_evidence_block_retrieval_v2 import _render_ranked_units, summarize


def _page():
    return {
        "document_id": "doc_generic",
        "page_id": "doc_generic:page:000005",
        "filename": "generic.pdf",
        "page_number": 5,
    }


def _table():
    return {
        **_page(),
        "table_id": "doc_generic:page:000005:table:0001",
        "title": "Consolidated Balance Sheets",
        "columns": ["Metric", "2024", "2023"],
        "rows": [
            {"Metric": "Cash", "2024": "120", "2023": "100"},
            {"Metric": "Liabilities", "2024": "250", "2023": "230"},
        ],
        "unit": "USD",
        "scale": "millions",
        "before_context": "At December 31, 2024 and 2023",
        "after_context": "See accompanying notes.",
        "quality_score": 0.9,
    }


def test_builder_emits_uniform_text_table_and_mixed_blocks():
    chunks = [
        {"filename": "generic.pdf", "page_number": 5, "chunk_idx": 2, "chunk_id": "c2", "text": "Cash was 120."},
        {"filename": "generic.pdf", "page_number": 5, "chunk_idx": 3, "chunk_id": "c3", "text": "Liabilities were 250."},
    ]

    blocks = build_evidence_blocks_v2(chunks, pages=[_page()], tables=[_table()])

    assert {block["source_type"] for block in blocks} == {"text", "table", "mixed"}
    assert all({"block_id", "document_id", "page_id", "source_type", "text", "metadata"} <= set(block) for block in blocks)
    text = next(block for block in blocks if block["source_type"] == "text")
    assert text["metadata"]["chunk_ids"] == ["c2", "c3"]
    table = next(block for block in blocks if block["source_type"] == "table")
    assert "Consolidated Balance Sheets" in table["text"]
    assert "Cash | 120 | 100" in table["text"]
    mixed = next(block for block in blocks if block["source_type"] == "mixed")
    assert "At December 31" in mixed["text"]


def test_text_builder_does_not_merge_non_adjacent_chunks():
    chunks = [
        {"filename": "generic.pdf", "page_number": 5, "chunk_idx": 1, "chunk_id": "c1", "text": "First."},
        {"filename": "generic.pdf", "page_number": 5, "chunk_idx": 4, "chunk_id": "c4", "text": "Fourth."},
    ]

    blocks = build_evidence_blocks_v2(chunks, pages=[_page()], tables=[])

    assert len(blocks) == 2
    assert all(block["metadata"]["chunk_count"] == 1 for block in blocks)


def test_index_document_preserves_uniform_identity_and_metadata():
    block = build_evidence_blocks_v2(
        [{"filename": "generic.pdf", "page_number": 5, "chunk_idx": 1, "chunk_id": "c1", "text": "First."}],
        pages=[_page()], tables=[],
    )[0]

    document = index_document(block, [0.1, 0.2])

    assert document["block_id"] == block["block_id"]
    assert document["document_id"] == "doc_generic"
    assert document["page_id"] == "doc_generic:page:000005"
    assert document["source_type"] == "text"
    assert document["metadata"]["chunk_ids"] == ["c1"]
    assert document["dense_embedding"] == [0.1, 0.2]


def test_block_rrf_keeps_route_ranks():
    dense = [{"block_id": "a"}, {"block_id": "b"}]
    bm25 = [{"block_id": "b"}, {"block_id": "c"}]

    fused = fuse_block_routes(dense, bm25, top_k=3)

    assert fused[0]["block_id"] == "b"
    assert fused[0]["dense_rank"] == 2
    assert fused[0]["bm25_rank"] == 1
    assert {item["block_id"] for item in fused} == {"a", "b", "c"}


def test_context_renderer_honors_unit_and_character_budgets():
    items = [
        {"block_id": "a", "source_type": "text", "filename": "a.pdf", "page_number": 1, "text": "A" * 20},
        {"block_id": "b", "source_type": "table", "filename": "a.pdf", "page_number": 2, "text": "B" * 20},
    ]

    context, units = _render_ranked_units(items, max_units=1, max_context_chars=200)

    assert len(units) == 1
    assert units[0]["block_id"] == "a"
    assert len(context) <= 200


def test_summary_does_not_fabricate_strict_judge_without_calls():
    metrics = {
        "answer_evidence_coverage": {"ratio": 0.5},
        "required_number_hit": True,
        "required_period_hit": True,
        "gold_page_hit": False,
        "all_gold_pages_hit": False,
        "context_chars": 100,
        "block_count": 2,
    }
    record = {
        "financebench_id": "generic_1",
        "group": "candidate_miss10",
        "routes": {
            "current_chunk_retrieval": {"metrics": metrics, "retrieval_latency_ms": 10},
            "evidence_block_retrieval_v2": {"metrics": {**metrics, "gold_page_hit": True}, "retrieval_latency_ms": 5},
        },
    }

    result = summarize([record])

    assert result["current_chunk_retrieval"]["strict_judge"] is None
    assert result["evidence_block_retrieval_v2"]["strict_judge"] is None
    assert result["external_calls"]["strict_judge"] == 0
