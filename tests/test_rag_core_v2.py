import os

from backend.rag_core_v2 import (
    build_core_v2_evidence,
    choose_core_v2_context_pages,
    merge_opened_pages,
    select_core_v2_pages,
)
from backend.runtime_profile import (
    RAG_CORE_V2_OVERRIDES,
    apply_runtime_profile,
    feature_state,
    uses_rag_core_v2_path,
)


def _chunk(filename, page, chunk_id, score=0.1, text="revenue 2023"):
    return {
        "filename": filename,
        "page_number": page,
        "chunk_id": chunk_id,
        "score": score,
        "text": text,
    }


def test_core_v2_profiles_are_isolated_and_skills_are_explicit(monkeypatch):
    for name in RAG_CORE_V2_OVERRIDES:
        monkeypatch.setenv(name, os.getenv(name, ""))
    assert apply_runtime_profile("rag_core_v2") == "rag_core_v2"
    state = feature_state("rag_core_v2")
    assert uses_rag_core_v2_path("rag_core_v2")
    assert state["retrieval_mode"] == "baseline"
    assert not any(state["modules"].values())

    assert apply_runtime_profile("rag_core_v2_skills") == "rag_core_v2_skills"
    modules = feature_state("rag_core_v2_skills")["modules"]
    assert modules["Explicit Formula Skill"]
    assert modules["Canonical Finance Metric Skill"]
    assert sum(bool(value) for value in modules.values()) == 2


def test_page_selection_is_soft_and_keeps_global_escape_page():
    candidates = [
        _chunk("a.pdf", 1, "a1", text="revenue fiscal 2023"),
        _chunk("a.pdf", 2, "a2", text="revenue details"),
        _chunk("b.pdf", 4, "b4", text="other data"),
        _chunk("c.pdf", 9, "c9", text="revenue fiscal 2023 exact table"),
    ]
    reranked = [
        {**candidates[0], "rerank_score": 0.9},
        {**candidates[3], "rerank_score": 0.8},
        {**candidates[1], "rerank_score": 0.2},
    ]
    selected, trace = select_core_v2_pages(
        "What was revenue in fiscal 2023?",
        candidates,
        reranked,
        document_top_k=1,
        page_top_k=3,
        global_escape_pages=1,
    )
    assert len(selected) == 3
    assert trace["selected_documents"] == ["a.pdf"]
    assert trace["global_escape_pages"] == [{"filename": "c.pdf", "page_number": 9}]
    assert {item["filename"] for item in selected} == {"a.pdf", "c.pdf"}
    context = choose_core_v2_context_pages(
        selected, trace["global_escape_pages"], final_page_k=2,
    )
    assert [_chunk["filename"] for _chunk in context] == ["a.pdf", "c.pdf"]


def test_context_keeps_contiguous_text_and_attaches_same_page_table():
    selected = [{
        "filename": "report.pdf",
        "page_number": 7,
        "page_score": 1.0,
        "best_chunk": {"text": "anchor row 2023 120"},
    }]
    page_text = "header\n" + ("narrative line\n" * 100) + "anchor row 2023 120\nfollowing row 2022 100\nfooter"
    opened = [{"filename": "report.pdf", "page_number": 7, "page_text": page_text, "text": page_text}]
    pages = merge_opened_pages(selected, opened)
    tables = [{
        "table_id": "table-7",
        "filename": "report.pdf",
        "page_number": 7,
        "title": "Revenue",
        "before_context": "USD millions",
        "columns": ["Metric", "2023", "2022"],
        "rows": [{"Metric": "Revenue", "2023": "120", "2022": "100"}],
    }]
    evidence, trace = build_core_v2_evidence(
        "What was revenue in 2023?", pages, tables, max_context_chars=8000, max_table_chars=2000,
    )
    assert "anchor row 2023 120\nfollowing row 2022 100" in evidence
    assert "Columns: Metric | 2023 | 2022" in evidence
    assert trace["tables_available_on_selected_pages"] == 1
    assert trace["tables_attached"] == 1
    assert trace["table_ids"] == ["table-7"]
    assert trace["answer_context_strategy"] == "rag_core_v2_contiguous_pages"


def test_page_store_rescore_only_reorders_existing_candidate_pages():
    candidates = [
        _chunk("a.pdf", 1, "a1", text="general annual report"),
        _chunk("a.pdf", 9, "a9", text="target measure details"),
        _chunk("b.pdf", 3, "b3", text="unrelated"),
    ]
    page_records = [
        {"filename": "a.pdf", "page_number": 1, "page_text": "general annual report"},
        {"filename": "a.pdf", "page_number": 9, "page_text": "target measure target measure 2023"},
        {"filename": "a.pdf", "page_number": 99, "page_text": "target measure 2023"},
    ]
    selected, trace = select_core_v2_pages(
        "target measure 2023",
        candidates,
        [],
        page_records=page_records,
        document_top_k=1,
        page_top_k=2,
        global_escape_pages=0,
    )
    assert selected[0]["page_number"] == 9
    assert 99 not in {item["page_number"] for item in selected}
    assert trace["page_store_rescored_pages"] == 2
