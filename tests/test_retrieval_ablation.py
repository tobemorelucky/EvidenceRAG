from backend import retrieval_ablation
from backend.retrieval_ablation import build_page_candidates, rank_pages, rrf_fuse
from backend.runtime_profile import (
    RETRIEVAL_ABLATION_FIELD_AWARE_PROFILE,
    RETRIEVAL_ABLATION_STRUCTURAL_PROFILE,
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
