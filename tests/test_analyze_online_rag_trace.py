from scripts.analyze_online_rag_trace import analyze_trace, render_markdown


def test_trace_analysis_exposes_pipeline_settings_and_pages():
    result = analyze_trace({
        "trace_id": "t1",
        "conversation_id": "c1",
        "user_query": "Block cash flow",
        "conversation_understanding": {"need_retrieval": True, "standalone_query": "Block FY2020 cash flow"},
        "policy_decision": {
            "retrieval": True,
            "rewrite": True,
            "query_rewrite_executed": True,
            "profile": "finance",
            "execution_mode": "static",
            "answer_prompt_mode": "baseline",
            "profile_config": "finance_online_v2",
            "query_rewrite_enabled": True,
            "context_budget": 28000,
            "before_memory_context_chars": 24680,
            "after_memory_context_chars": 24680,
            "answer_prompt_name": "clean_baseline_v1",
            "answer_config": {
                "temperature": 0.1,
                "thinking": "disabled",
                "max_tokens": 1024,
            },
        },
        "retrieval_executed": True,
        "retrieval_counts": {"dense": 240, "bm25": 240, "rrf": 120},
        "rerank_parameters": {
            "enabled": True,
            "applied": True,
            "provider": "remote",
            "model": "jina-reranker-v3",
            "dense_top_k": 240,
            "bm25_top_k": 240,
            "rrf_top_k": 120,
            "input_k": 80,
            "output_k": 12,
            "input_max_chars": 0,
            "status": "success",
            "profile_config": "finance_online_v2",
            "context_budget": 28000,
        },
        "evidence_items": [{"filename": "BLOCK_2020_10K.pdf", "page_number": 89, "text": "cash flow"}],
        "answer_model": "deepseek-v4-flash-ga-260731",
    })
    assert result["matches_financebench_pipeline"]
    assert result["jina"]["called"]
    assert result["final_evidence_pages"][0]["page_number"] == 89
    assert "BLOCK_2020_10K.pdf" in render_markdown(result)


def test_legacy_trace_reports_unknown_fields_without_guessing():
    result = analyze_trace({
        "policy_decision": {"retrieval": False, "rewrite": False},
        "conversation_understanding": {"need_retrieval": False},
        "retrieval_executed": False,
        "rerank_parameters": {"applied": True, "provider": "remote", "model": "jina-reranker-v3", "candidate_k": 40, "final_top_k": 5},
    })
    assert result["profile"].startswith("unknown")
    assert not result["matches_financebench_pipeline"]
    assert any("未执行新检索" in item for item in result["pipeline_differences"])
