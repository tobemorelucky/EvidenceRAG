import inspect

from backend.conversation_understanding_v3 import ConversationDecisionV3
from scripts.run_conversation_aware_rag_v4_shadow import (
    build_shadow_evidence,
    execute_case,
    load_cases,
)


class FakeUnderstandingModel:
    pass


def test_v4_has_thirty_balanced_financial_scenarios():
    cases = load_cases()
    assert len(cases) == 30
    assert {case["category"] for case in cases} == {
        "follow_up", "period_modification", "entity_modification", "challenge",
        "citation_question", "calculation_explanation",
    }


def test_evidence_memory_is_bounded_and_previous_first_when_reused():
    evidence, trace = build_shadow_evidence("old" * 10000, "new" * 10000, reuse_previous=True)
    assert len(evidence) <= 28000
    assert evidence.startswith("Previous-turn evidence")
    assert trace["previous_evidence_reused"]


def test_execute_case_is_injectable_and_does_not_require_retrieval(monkeypatch):
    case = load_cases()[0]
    decision = ConversationDecisionV3(
        need_rag=True, need_retrieval=False, depends_on_history=True,
        query_resolution_needed=True, required_memory_scope="last exchange",
        standalone_query="Explain the direction using Ulta Beauty FY2023 evidence",
        response_mode="explain",
    )
    monkeypatch.setattr(
        "scripts.run_conversation_aware_rag_v4_shadow.understand_conversation_v3",
        lambda *_args, **_kwargs: (decision, {"model_called": True, "usage": {}}),
    )
    retrieval_calls = []
    result = execute_case(
        case,
        {"question": "Base question", "answer": "Base answer", "evidence": "Source: report.pdf | Page: 2\nFact"},
        FakeUnderstandingModel(),
        retrieve_fn=lambda *_args, **_kwargs: retrieval_calls.append(1),
        answer_fn=lambda question, evidence, state: ("The evidence supports the prior direction.", {}),
    )
    assert not retrieval_calls
    assert result["memory"]["previous_evidence_reused"]
    assert result["status"] if "status" in result else True


def test_v4_runner_contains_no_jina_call():
    import scripts.run_conversation_aware_rag_v4_shadow as module

    source = inspect.getsource(module)
    forbidden = ("JinaReranker", "rerank_documents(", "rerank_candidates(", "requests.post(")
    assert all(token not in source for token in forbidden)
