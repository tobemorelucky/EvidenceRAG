from backend.finance_online_profile import finance_online_trace_fields, load_finance_online_profile


def test_finance_online_profile_matches_final100_depths():
    config = load_finance_online_profile()
    assert config["query_rewrite"] == {
        "enabled": True, "max_rewrites": 2, "original_retained": True,
    }
    assert config["retrieval"] == {
        "dense_top_k": 240, "bm25_top_k": 240, "rrf_top_k": 120, "rrf_k": 60,
    }
    assert config["rerank"] == {
        "provider": "jina", "input_k": 80, "output_k": 12, "input_max_chars": 0,
    }
    assert config["context"] == {"max_chars": 28000}
    assert config["prompt"]["name"] == "clean_baseline_v1"
    assert config["answer"] == {
        "model": "deepseek-v4-flash-ga-260731", "temperature": 0.1,
        "thinking": "disabled", "max_tokens": 1024,
    }
    trace = finance_online_trace_fields(config)
    assert {
        key: trace[key] for key in (
            "profile_config", "query_rewrite_enabled", "dense_top_k", "bm25_top_k",
            "rrf_top_k", "jina_input_k", "jina_output_k", "jina_input_max_chars",
            "context_budget", "answer_prompt_name", "answer_temperature",
            "answer_thinking", "answer_max_tokens",
        )
    } == {
        "profile_config": "finance_online_v2",
        "query_rewrite_enabled": True,
        "dense_top_k": 240,
        "bm25_top_k": 240,
        "rrf_top_k": 120,
        "jina_input_k": 80,
        "jina_output_k": 12,
        "jina_input_max_chars": 0,
        "context_budget": 28000,
        "answer_prompt_name": "clean_baseline_v1",
        "answer_temperature": 0.1,
        "answer_thinking": "disabled",
        "answer_max_tokens": 1024,
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
    assert calls["rerank"][2] == {"input_k": 80, "output_k": 12, "input_max_chars": 0}
    assert result["rag_trace"]["query_rewrite_executed"] is True
    assert result["rag_trace"]["profile_config"] == "finance_online_v2"
    assert result["rag_trace"]["rerank_status"] == "success"


def test_finance_online_rewrite_failure_preserves_original_query(monkeypatch):
    import backend.rag_orchestrator as orchestrator
    import query_rewriter

    monkeypatch.setattr(query_rewriter, "create_query_rewrite_model", lambda: object())
    monkeypatch.setattr(query_rewriter, "rewrite_queries", lambda question, model: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(orchestrator, "retrieve_profile_candidates", lambda queries, **kwargs: {
        "docs": [], "meta": {"seen_queries": queries},
    })
    monkeypatch.setattr(orchestrator, "rerank_profile_candidates", lambda question, docs, **kwargs: {
        "docs": [], "meta": {"rerank_applied": True, "rerank_status": "success"},
    })

    result = orchestrator._run_finance_online_search("original", load_finance_online_profile())
    assert result["rag_trace"]["retrieval_queries"] == ["original"]
    assert result["rag_trace"]["query_rewrite_executed"] is False
    assert "RuntimeError" in result["rag_trace"]["query_rewrite_error"]


def test_finance_online_rerank_failure_blocks_answer_and_retains_rrf_trace(monkeypatch):
    import pytest
    import backend.rag_orchestrator as orchestrator
    import query_rewriter

    monkeypatch.setattr(query_rewriter, "create_query_rewrite_model", lambda: object())
    monkeypatch.setattr(query_rewriter, "rewrite_queries", lambda question, model: {
        "queries": [question], "usage": {},
    })
    candidates = [{"chunk_id": "rrf-1", "text": "fact"}]
    monkeypatch.setattr(orchestrator, "retrieve_profile_candidates", lambda *args, **kwargs: {
        "docs": candidates, "meta": {"rrf_fused_candidate_count": 1},
    })
    monkeypatch.setattr(orchestrator, "rerank_profile_candidates", lambda *args, **kwargs: {
        "docs": candidates, "meta": {"rerank_status": "failed", "rerank_error": "offline"},
    })
    with pytest.raises(orchestrator.FinanceRerankServiceError) as caught:
        orchestrator._run_finance_online_search("question", load_finance_online_profile())
    assert caught.value.rag_trace["rerank_status"] == "failed"
    assert caught.value.rag_trace["rrf_fallback_retained"] is True
    assert caught.value.rag_trace["answer_blocked"] is True


def test_finance_baseline_uses_clean_prompt_without_policy_injection():
    from backend.answer_generator import build_answer_messages
    from backend.prompts import CLEAN_BASELINE_ANSWER_SYSTEM_PROMPT

    messages = build_answer_messages(
        "Calculate a ratio", "Source evidence", task_policy="do something extra",
        profile="finance", prompt_mode="baseline",
    )
    assert CLEAN_BASELINE_ANSWER_SYSTEM_PROMPT in messages[0].content
    assert "do something extra" not in messages[-1].content


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
