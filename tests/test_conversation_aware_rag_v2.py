import json
from pathlib import Path

from scripts.run_conversation_aware_rag_v2_shadow import standalone_quality, summarize


ROOT = Path(__file__).resolve().parents[1]


def test_fifty_human_scenarios_cover_all_required_categories():
    cases = json.loads((ROOT / "tests/fixtures/conversation_aware_rag_v2_cases.json").read_text(encoding="utf-8"))
    assert len(cases) == len({case["id"] for case in cases}) == 50
    assert {case["category"] for case in cases} == {
        "new_question", "pronoun_followup", "condition_modification", "challenge_previous_answer",
        "explain_citation", "supplemental_analysis", "non_rag_chat", "summary_request",
    }


def test_standalone_quality_requires_terms_and_resolved_reference():
    good = standalone_quality("Compare Acme revenue in FY2023", ["Acme", "revenue", "FY2023"], True)
    locally_resolved = standalone_quality("How did it compare for Acme revenue FY2023?", ["Acme", "revenue", "FY2023"], True)
    unresolved = standalone_quality("How did it compare for Acme revenue?", ["Acme", "revenue", "FY2023"], True)
    missing = standalone_quality("Compare Acme revenue", ["Acme", "revenue", "FY2023"], True)
    assert good["quality_pass"] is True
    assert locally_resolved["quality_pass"] is True
    assert unresolved["quality_pass"] is False
    assert missing["quality_pass"] is False


def test_summary_counts_invalid_json_as_incorrect():
    cases = [
        {"id": "a", "category": "new_question", "expected": {"need_retrieval": True, "need_previous_context": False}},
        {"id": "b", "category": "new_question", "expected": {"need_retrieval": True, "need_previous_context": False}},
    ]
    records = [
        {"id": "a", "status": "ok", "need_rag_correct": True, "need_rewrite_correct": True,
         "standalone_quality": {"quality_pass": True, "term_retention": 1}, "trace": {"usage": {}}, "latency_ms": 1},
        {"id": "b", "status": "invalid_json", "need_rag_correct": False, "need_rewrite_correct": False,
         "standalone_quality": {}, "latency_ms": 1},
    ]
    result = summarize(cases, records)
    assert result["json_legal_rate"] == result["need_rag_accuracy"] == result["need_rewrite_accuracy"] == 0.5
    assert result["errors"] == 1


def test_runner_only_contains_conversation_understanding_model_call():
    import inspect
    import scripts.run_conversation_aware_rag_v2_shadow as module

    source = inspect.getsource(module)
    forbidden = ("prepare_rag_response", "generate_answer", "judge_answer", "JinaReranker", "milvus_client", "hybrid_retrieve")
    assert all(token not in source for token in forbidden)
    assert module.MODEL == "deepseek-v4-flash-ga-260731"
