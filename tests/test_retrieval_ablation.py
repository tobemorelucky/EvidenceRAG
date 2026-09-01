from backend import retrieval_ablation
from backend.retrieval_ablation import build_page_candidates, rank_pages, rrf_fuse
from backend.rag_core_v4 import (
    expand_and_rank_pages,
    merge_global_local_chunks,
    merge_dense_primary,
    retrieve_document_local_chunks,
    score_candidate_documents,
    select_document_first_pages,
)
from backend.runtime_profile import (
    RETRIEVAL_ABLATION_FIELD_AWARE_PROFILE,
    RETRIEVAL_ABLATION_STRUCTURAL_PROFILE,
    RETRIEVAL_DENSE_PRIMARY_PROFILE,
    RETRIEVAL_DENSE_PRIMARY_NEIGHBORS_PROFILE,
    RETRIEVAL_DOCUMENT_LOCAL_PROFILE,
    apply_runtime_profile,
    feature_state,
)


def _chunk(chunk_id, filename, page, text="text"):
    return {
        "chunk_id": chunk_id,
        "filename": filename,
        "page_number": page,
        "chunk_idx": 0,
        "text": text,
    }


def test_rrf_preserves_independent_route_ranks():
    dense = [_chunk("a", "one.pdf", 1), _chunk("b", "two.pdf", 2)]
    bm25 = [_chunk("b", "two.pdf", 2), _chunk("c", "three.pdf", 3)]

    fused = rrf_fuse(dense, bm25)

    assert fused[0]["chunk_id"] == "b"
    assert fused[0]["dense_rank"] == 2
    assert fused[0]["bm25_rank"] == 1
    assert [page["filename"] for page in rank_pages(fused)] == ["two.pdf", "one.pdf", "three.pdf"]


def test_dense_primary_merge_preserves_dense_order_and_appends_only_new_bm25():
    dense = [_chunk("a", "one.pdf", 1), _chunk("b", "one.pdf", 2), _chunk("c", "one.pdf", 3)]
    bm25 = [_chunk("c", "one.pdf", 3), _chunk("d", "one.pdf", 4), _chunk("e", "one.pdf", 5)]

    result = merge_dense_primary(dense, bm25)

    assert [item["chunk_id"] for item in result] == ["a", "b", "c", "d", "e"]
    assert result[2]["dense_rank"] == 3
    assert result[2]["bm25_rank"] == 1
    assert result[2]["candidate_source"] == "both"
    assert result[3]["dense_rank"] is None
    assert result[3]["bm25_rank"] == 2
    assert [item["merged_rank"] for item in result] == [1, 2, 3, 4, 5]


class _NeighborPageStore:
    def get_pages_by_keys(self, keys):
        pages = {
            ("one.pdf", 3): {"filename": "one.pdf", "page_number": 3, "page_text": "footnote", "page_dense_embedding": [0.2, 0.8]},
            ("one.pdf", 4): {"filename": "one.pdf", "page_number": 4, "page_text": "unrelated", "page_dense_embedding": [0.0, 1.0]},
            ("one.pdf", 5): {"filename": "one.pdf", "page_number": 5, "page_text": "target revenue 2023", "page_dense_embedding": [1.0, 0.0]},
        }
        return [pages[key] for key in keys if key in pages]


def test_page_expansion_stays_in_document_and_ranks_relevant_neighbor():
    chunks = [{**_chunk("a", "one.pdf", 4, "target"), "merged_rank": 1}]

    pages, trace = expand_and_rank_pages(
        "target revenue 2023", chunks, [1.0, 0.0], page_store=_NeighborPageStore(),
    )

    assert [(item["filename"], item["page_number"]) for item in pages] == [
        ("one.pdf", 5), ("one.pdf", 3), ("one.pdf", 4),
    ]
    assert trace["requested_page_count"] == 3
    assert pages[0]["neighbor_distance"] == 1


def test_document_first_selection_keeps_one_global_escape():
    pages = [
        {"filename": "primary.pdf", "page_number": page, "page_score": 1.0 - page / 100}
        for page in range(1, 10)
    ] + [
        {"filename": "other.pdf", "page_number": 1, "page_score": 0.7},
    ]

    selected, trace = select_document_first_pages(pages, final_page_k=8, global_escape_pages=1)

    assert len(selected) == 8
    assert sum(item["filename"] == "primary.pdf" for item in selected) == 7
    assert selected[-1]["filename"] == "other.pdf"
    assert trace["primary_document"] == "primary.pdf"


def test_document_shortlist_uses_rank_and_structural_support_only():
    chunks = [
        {**_chunk("a1", "a.pdf", 1), "dense_rank": 1, "bm25_rank": None},
        {**_chunk("a2", "a.pdf", 2), "dense_rank": 8, "bm25_rank": 3},
        {**_chunk("b1", "b.pdf", 1), "dense_rank": 2, "bm25_rank": None},
    ]

    shortlist, trace = score_candidate_documents(chunks, shortlist_k=2)

    assert shortlist == ["a.pdf", "b.pdf"]
    assert trace[0]["document_rank"] == 1
    assert trace[0]["chunk_count"] == 2
    assert trace[0]["page_count"] == 2


class _LocalManager:
    def __init__(self):
        self.filters = []

    def dense_retrieve(self, embedding, top_k, filter_expr):
        self.filters.append(("dense", filter_expr, top_k))
        filename = "a.pdf" if 'a.pdf' in filter_expr else "b.pdf"
        return [_chunk(f"{filename}-dense", filename, 4)]

    def bm25_retrieve(self, question, top_k, filter_expr):
        self.filters.append(("bm25", filter_expr, top_k))
        filename = "a.pdf" if 'a.pdf' in filter_expr else "b.pdf"
        return [_chunk(f"{filename}-bm25", filename, 5)]


def test_document_local_search_scopes_each_dense_and_bm25_call():
    manager = _LocalManager()

    result = retrieve_document_local_chunks(
        "revenue", ["a.pdf", "b.pdf"], [1.0, 0.0], local_k=20, manager=manager,
    )

    assert result["dense_calls"] == 2
    assert result["bm25_calls"] == 2
    assert len(result["chunks"]) == 4
    assert all('filename == "' in item[1] for item in manager.filters)
    assert all(item[2] == 20 for item in manager.filters)


def test_document_local_search_reserves_bm25_supplement_slots():
    class ManyResults:
        def dense_retrieve(self, embedding, top_k, filter_expr):
            return [_chunk(f"d{index}", "a.pdf", index) for index in range(30)]

        def bm25_retrieve(self, question, top_k, filter_expr):
            return [_chunk(f"b{index}", "a.pdf", 100 + index) for index in range(30)]

    result = retrieve_document_local_chunks(
        "revenue", ["a.pdf"], [1.0], local_k=30, dense_slots=20, manager=ManyResults(),
    )

    assert len(result["chunks"]) == 30
    assert sum(item["local_dense_rank"] is not None for item in result["chunks"]) == 20
    assert sum(item["local_bm25_rank"] is not None for item in result["chunks"]) == 10
    assert result["routes"][0]["bm25_supplement_slots"] == 10


def test_global_local_merge_prefers_local_and_records_both_ranks():
    shared = _chunk("shared", "a.pdf", 1)
    local = [shared, _chunk("local", "a.pdf", 2)]
    global_chunks = [_chunk("global", "b.pdf", 3), shared]

    merged = merge_global_local_chunks(global_chunks, local)

    assert [item["chunk_id"] for item in merged] == ["shared", "local", "global"]
    assert merged[0]["candidate_source"] == "both"
    assert merged[0]["local_rank"] == 1
    assert merged[0]["global_rank"] == 2


class _PageStore:
    def get_pages_by_keys(self, keys):
        return [{"filename": "one.pdf", "page_number": 4, "page_text": "Revenue table heading"}]


class _TableStore:
    def get_tables_by_page_keys(self, keys):
        return [{
            "filename": "one.pdf",
            "page_number": 4,
            "title": "Revenue",
            "columns": ["2023", "2022"],
            "rows": [{"label": "Net revenue", "2023": "100", "2022": "90"}],
        }]


def test_page_representation_is_compact_and_keeps_table_header():
    chunks = [
        {**_chunk("a", "one.pdf", 4, "Net revenue increased in 2023."), "rrf_rank": 1},
        {**_chunk("b", "one.pdf", 4, "The prior-year amount was 90."), "rrf_rank": 2},
    ]

    pages = build_page_candidates(
        "What was net revenue in 2023?",
        chunks,
        representation_chars=800,
        page_store=_PageStore(),
        table_store=_TableStore(),
    )

    assert len(pages) == 1
    assert len(pages[0]["text"]) <= 800
    assert "Columns:" in pages[0]["text"]
    assert "2023" in pages[0]["text"]
    assert "Net revenue" in pages[0]["text"]
    assert pages[0]["table_representation_chars"] > 0


def test_retrieval_ablation_profiles_are_isolated(monkeypatch):
    monkeypatch.setenv("FINANCE_POLICY_ENABLED", "true")
    apply_runtime_profile(RETRIEVAL_ABLATION_STRUCTURAL_PROFILE)
    structural = feature_state(RETRIEVAL_ABLATION_STRUCTURAL_PROFILE)
    assert structural["page_first"] is True
    assert structural["field_aware"] is False
    assert structural["modules"]["Finance Policy"] is False

    apply_runtime_profile(RETRIEVAL_ABLATION_FIELD_AWARE_PROFILE)
    field_aware = feature_state(RETRIEVAL_ABLATION_FIELD_AWARE_PROFILE)
    assert field_aware["page_first"] is True
    assert field_aware["field_aware"] is True
    assert field_aware["modules"]["Agent/Planner"] is False
    assert field_aware["modules"]["Finance Policy"] is False

    apply_runtime_profile(RETRIEVAL_DENSE_PRIMARY_PROFILE)
    dense_primary = feature_state(RETRIEVAL_DENSE_PRIMARY_PROFILE)
    assert dense_primary["profile"] == RETRIEVAL_DENSE_PRIMARY_PROFILE
    assert dense_primary["modules"]["Explicit Formula Skill"] is True
    assert dense_primary["modules"]["Canonical Finance Metric Skill"] is True
    assert dense_primary["page_first"] is False

    apply_runtime_profile(RETRIEVAL_DENSE_PRIMARY_NEIGHBORS_PROFILE)
    neighbors = feature_state(RETRIEVAL_DENSE_PRIMARY_NEIGHBORS_PROFILE)
    assert neighbors["profile"] == RETRIEVAL_DENSE_PRIMARY_NEIGHBORS_PROFILE
    assert neighbors["modules"]["Explicit Formula Skill"] is True

    apply_runtime_profile(RETRIEVAL_DOCUMENT_LOCAL_PROFILE)
    document_local = feature_state(RETRIEVAL_DOCUMENT_LOCAL_PROFILE)
    assert document_local["profile"] == RETRIEVAL_DOCUMENT_LOCAL_PROFILE
    assert document_local["modules"]["Explicit Formula Skill"] is True
    assert document_local["modules"]["Canonical Finance Metric Skill"] is True
    assert document_local["modules"]["Agent/Planner"] is False
