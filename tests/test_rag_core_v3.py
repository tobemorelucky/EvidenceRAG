import os

from backend.rag_core_v3 import (
    build_core_v3_evidence,
    merge_core_v3_candidate_routes,
    merge_opened_pages,
    select_core_v3_pages,
)
from backend.runtime_profile import (
    RAG_CORE_V3_OVERRIDES,
    apply_runtime_profile,
    feature_state,
    uses_rag_core_v3_path,
)
from backend.rag_orchestrator import _rank_core_v3_scoped_documents, _run_core_v3_search


def _chunk(filename, page, chunk_id, *, text="revenue 2023", score=0.1):
    return {
        "filename": filename,
        "page_number": page,
        "chunk_id": chunk_id,
        "score": score,
        "text": text,
    }


def test_v3_profiles_are_isolated_and_only_frozen_skills_are_optional(monkeypatch):
    for name in RAG_CORE_V3_OVERRIDES:
        monkeypatch.setenv(name, os.getenv(name, ""))
    assert apply_runtime_profile("rag_core_v3") == "rag_core_v3"
    assert uses_rag_core_v3_path("rag_core_v3")
    state = feature_state("rag_core_v3")
    assert state["retrieval_mode"] == "baseline"
    assert not any(state["modules"].values())

    assert apply_runtime_profile("rag_core_v3_skills") == "rag_core_v3_skills"
    modules = feature_state("rag_core_v3_skills")["modules"]
    assert modules["Explicit Formula Skill"]
    assert modules["Canonical Finance Metric Skill"]
    assert sum(bool(value) for value in modules.values()) == 2


def test_v3_caps_repeated_chunk_support_below_one_strong_page():
    repeated = [
        _chunk("repeated.pdf", 1, f"r{index}", text="annual report general information")
        for index in range(20)
    ]
    strong = _chunk("answer.pdf", 8, "answer", text="target revenue FY2023 exact disclosure")
    selected, trace = select_core_v3_pages(
        "What was target revenue in FY2023?",
        [*repeated, strong],
        [{**strong, "rerank_score": 0.95}],
        document_top_k=1,
        page_pool_k=4,
        final_page_k=1,
        global_escape_pages=0,
    )

    assert [(item["filename"], item["page_number"]) for item in selected] == [("answer.pdf", 8)]
    scores = {(item["filename"], item["page_number"]): item for item in trace["page_scores"]}
    assert scores[("answer.pdf", 8)]["page_score"] > scores[("repeated.pdf", 1)]["page_score"]


def test_v3_candidate_fusion_records_global_scoped_and_both_sources():
    shared = _chunk("a.pdf", 1, "shared")
    global_only = _chunk("b.pdf", 2, "global")
    scoped_only = _chunk("a.pdf", 3, "scoped")

    merged = merge_core_v3_candidate_routes(
        [shared, global_only], [("scoped:a.pdf", [shared, scoped_only])]
    )
    by_id = {item["chunk_id"]: item for item in merged}

    assert by_id["shared"]["candidate_source"] == "both"
    assert by_id["global"]["candidate_source"] == "global"
    assert by_id["scoped"]["candidate_source"] == "scoped:a.pdf"
    assert by_id["shared"]["core_v3_rrf_score"] > by_id["global"]["core_v3_rrf_score"]
    assert by_id["scoped"]["core_v3_rrf_score"] > by_id["global"]["core_v3_rrf_score"]


def test_v3_retrieval_context_is_only_a_soft_document_boost():
    candidates = [
        _chunk("ALPHA_2022_10K.pdf", 1, "a", text="revenue annual report"),
        _chunk("BETA_2022_10K.pdf", 1, "b", text="revenue annual report"),
    ]
    selected, scores = _rank_core_v3_scoped_documents(
        "What was revenue in 2022?",
        candidates,
        {"selected_documents": ["BETA_2022_10K"], "period": "2022"},
        limit=1,
    )

    assert selected == ["BETA_2022_10K.pdf"]
    assert {item["filename"] for item in scores} == {"ALPHA_2022_10K.pdf", "BETA_2022_10K.pdf"}
    assert scores[0]["retrieval_context_boost_reasons"] == ["selected_document_hint", "period_hint"]


def test_v3_document_local_retrieval_uses_original_query_and_one_finalizer(monkeypatch):
    global_docs = [
        _chunk("a.pdf", 1, "a1", text="target revenue 2023"),
        _chunk("b.pdf", 2, "b2", text="target revenue 2023"),
        _chunk("c.pdf", 3, "c3", text="target revenue 2023"),
    ]
    calls = []
    monkeypatch.setattr(
        "backend.rag_orchestrator.retrieve_candidate_documents",
        lambda query, candidate_k: {"docs": global_docs, "meta": {"candidate_k": candidate_k}},
    )

    def scoped(query, filenames, **kwargs):
        calls.append((query, tuple(filenames), kwargs["retrieval_scope"]))
        return [_chunk(filenames[0], 9, f"{filenames[0]}-scoped", text="target revenue 2023 exact")]

    monkeypatch.setattr("backend.rag_orchestrator.retrieve_document_scoped_candidates", scoped)
    finalize_calls = []

    def finalize(query, candidates, **kwargs):
        finalize_calls.append((query, candidates, kwargs))
        return {"final_retrieved_docs": candidates[:4], "meta": {"rerank_provider": "test"}}

    monkeypatch.setattr("backend.rag_orchestrator.finalize_retrieved_documents", finalize)
    result = _run_core_v3_search("target revenue 2023")

    assert len(calls) == 3
    assert all(item[0] == "target revenue 2023" for item in calls)
    assert all(item[2] == "rag_core_v3:document_local" for item in calls)
    assert len(finalize_calls) == 1
    assert result["rag_trace"]["core_v3_dense_bm25_calls"] == 4
    assert result["rag_trace"]["stage2_queries"] == ["target revenue 2023"] * 3
    assert result["rag_trace"]["retrieval_context_applied"] is False


def test_v3_global_escape_uses_true_global_page_rank_outside_soft_documents():
    candidates = [
        _chunk("a.pdf", 1, "a1", text="revenue 2023"),
        _chunk("a.pdf", 2, "a2", text="revenue details"),
        _chunk("b.pdf", 7, "b7", text="revenue 2023 exact table"),
        _chunk("c.pdf", 9, "c9", text="other content"),
    ]
    reranked = [
        {**candidates[0], "rerank_score": 0.95},
        {**candidates[2], "rerank_score": 0.90},
        {**candidates[1], "rerank_score": 0.20},
    ]
    selected, trace = select_core_v3_pages(
        "What was revenue in 2023?",
        candidates,
        reranked,
        document_top_k=1,
        page_pool_k=4,
        final_page_k=2,
        global_escape_pages=1,
    )

    escape = trace["global_escape_pages"]
    assert len(escape) == 1
    assert escape[0]["filename"] == "b.pdf"
    assert escape[0]["global_escape_candidate_rank"] == 2
    assert escape[0]["whether_outside_selected_documents"] is True
    assert {item["filename"] for item in selected} == {"a.pdf", "b.pdf"}


def test_v3_greedy_selection_rewards_new_explicit_period_coverage():
    candidates = [
        _chunk("report.pdf", 1, "p1", text="revenue FY2022 100"),
        _chunk("report.pdf", 2, "p2", text="revenue FY2022 repeated discussion"),
        _chunk("report.pdf", 3, "p3", text="revenue FY2021 90"),
    ]
    reranked = [
        {**candidates[0], "rerank_score": 0.90},
        {**candidates[1], "rerank_score": 0.85},
        {**candidates[2], "rerank_score": 0.75},
    ]
    selected, _ = select_core_v3_pages(
        "Revenue in FY2021 and FY2022",
        candidates,
        reranked,
        document_top_k=1,
        page_pool_k=3,
        final_page_k=2,
        global_escape_pages=0,
    )

    assert {item["page_number"] for item in selected} == {1, 3}


def test_v3_context_gives_every_page_a_window_and_keeps_table_complete():
    selected = [
        {
            "filename": "report.pdf",
            "page_number": page,
            "page_score": 1.0 / page,
            "best_chunk": {"text": f"anchor-{page}"},
        }
        for page in range(1, 7)
    ]
    opened = []
    for page in range(1, 7):
        text = f"page-{page}-header\n" + ("narrative\n" * 800) + f"anchor-{page}\npage-{page}-tail"
        opened.append({"filename": "report.pdf", "page_number": page, "page_text": text, "text": text})
    pages = merge_opened_pages(selected, opened)
    tables = [{
        "table_id": "table-6",
        "filename": "report.pdf",
        "page_number": 6,
        "title": "Revenue",
        "before_context": "USD millions",
        "columns": ["Metric", "2023", "2022"],
        "rows": [{"Metric": "Revenue", "2023": "120", "2022": "100"}],
    }]
    evidence, trace = build_core_v3_evidence(
        "What was revenue in 2023?",
        pages,
        tables,
        max_context_chars=18000,
        max_table_chars=2000,
        min_page_chars=2000,
    )

    assert len(evidence) <= 18000
    assert trace["answer_context_unit_count"] == 6
    assert {item["page_number"] for item in trace["answer_context_pages"]} == set(range(1, 7))
    assert all(item["body_chars"] >= 1900 for item in trace["answer_context_page_allocations"])
    assert "Columns: Metric | 2023 | 2022" in evidence
    assert "Metric: Revenue | 2023: 120 | 2022: 100" in evidence
    assert trace["tables_attached"] == 1


def test_v3_context_budget_supports_eight_pages_without_dropping_tail_pages():
    pages = [
        {
            "filename": "report.pdf",
            "page_number": page,
            "text": f"page {page}\n" + (f"line {page}\n" * 1000),
            "best_chunk": {"text": f"line {page}"},
        }
        for page in range(1, 9)
    ]
    evidence, trace = build_core_v3_evidence(
        "general question", pages, [], max_context_chars=28000, min_page_chars=2200,
    )

    assert len(evidence) <= 28000
    assert trace["answer_context_unit_count"] == 8
    assert trace["answer_context_pages"][-1]["page_number"] == 8
    assert all(item["body_chars"] >= 2100 for item in trace["answer_context_page_allocations"])
