from backend.finance_online_profile import finance_online_trace_fields, load_finance_online_profile


def test_finance_online_profile_matches_final100_depths():
    config = load_finance_online_profile()
    assert config["query_rewrite"] == {
        "enabled": True, "max_rewrites": 2, "original_retained": True,
    }
    assert config["retrieval"] == {
        "dense_top_k": 240, "bm25_top_k": 240, "rrf_top_k": 120, "rrf_k": 60,
    }
    assert config["rerank"] == {"provider": "jina", "input_k": 80, "output_k": 12}
    assert config["context"] == {"max_chars": 28000}
    assert finance_online_trace_fields(config) == {
        "profile_config": "finance_online_v1",
        "query_rewrite_enabled": True,
        "dense_top_k": 240,
        "bm25_top_k": 240,
        "rrf_top_k": 120,
        "jina_input_k": 80,
        "jina_output_k": 12,
        "context_budget": 28000,
    }


def test_finance_online_search_uses_fixed_profile_depths(monkeypatch):
    import backend.rag_orchestrator as orchestrator
    import query_rewriter

    monkeypatch.setattr(query_rewriter, "create_query_rewrite_model", lambda: object())
    monkeypatch.setattr(query_rewriter, "rewrite_queries", lambda question, model: {
        "queries": [question, "rewrite one", "rewrite two"],
        "generated_alternatives": ["rewrite one", "rewrite two"],
        "usage": {"total_tokens": 10},
    })
    calls = {}

    def retrieve(queries, **kwargs):
        calls["retrieve"] = (queries, kwargs)
        return {"docs": [{"chunk_id": "c1", "text": "fact"}], "meta": {"rrf_fused_candidate_count": 1}}

    def rerank(question, docs, **kwargs):
        calls["rerank"] = (question, docs, kwargs)
        return {"docs": docs, "meta": {"rerank_applied": True, "rerank_provider": "remote"}}

    monkeypatch.setattr(orchestrator, "retrieve_profile_candidates", retrieve)
    monkeypatch.setattr(orchestrator, "rerank_profile_candidates", rerank)
    result = orchestrator._run_finance_online_search("question", load_finance_online_profile())

    assert calls["retrieve"] == (
        ["question", "rewrite one", "rewrite two"],
        {"dense_top_k": 240, "bm25_top_k": 240, "rrf_top_k": 120, "rrf_k": 60},
    )
    assert calls["rerank"][2] == {"input_k": 80, "output_k": 12}
    assert result["rag_trace"]["query_rewrite_executed"] is True
    assert result["rag_trace"]["profile_config"] == "finance_online_v1"


def test_finance_online_rewrite_failure_preserves_original_query(monkeypatch):
    import backend.rag_orchestrator as orchestrator
    import query_rewriter

    monkeypatch.setattr(query_rewriter, "create_query_rewrite_model", lambda: object())
    monkeypatch.setattr(query_rewriter, "rewrite_queries", lambda question, model: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(orchestrator, "retrieve_profile_candidates", lambda queries, **kwargs: {
        "docs": [], "meta": {"seen_queries": queries},
    })
    monkeypatch.setattr(orchestrator, "rerank_profile_candidates", lambda question, docs, **kwargs: {"docs": [], "meta": {}})

    result = orchestrator._run_finance_online_search("original", load_finance_online_profile())
    assert result["rag_trace"]["retrieval_queries"] == ["original"]
    assert result["rag_trace"]["query_rewrite_executed"] is False
    assert "RuntimeError" in result["rag_trace"]["query_rewrite_error"]


def test_ranked_chunk_context_retains_relevant_financial_tail():
    from backend.evidence_context import build_ranked_chunk_evidence

    page = "\n".join([
        "CONSOLIDATED BALANCE SHEETS", "In thousands", "2015 2014", "ASSETS",
        *[f"Current asset line {index} 100 90" for index in range(40)],
        "LIABILITIES", "Current liabilities:", "Total current liabilities 2,213,556 2,494,435",
        "Long-term liabilities 1,907,231 911,086",
    ])
    evidence, included, trace = build_ranked_chunk_evidence(
        [{"filename": "ADOBE.pdf", "page_number": 58, "text": page}],
        max_context_chars=28000,
        top_k=12,
    )
    assert "Total current liabilities 2,213,556" in evidence
    assert len(included) == 1
    assert trace["answer_context_builder"] == "ranked_raw_chunks"
