from backend.conversation_understanding_v3 import ConversationDecisionV3
from backend.memory.conversation_policy import build_conversation_policy


def decision(**changes):
    value = dict(
        need_rag=True, need_retrieval=False, depends_on_history=True,
        query_resolution_needed=True, required_memory_scope="last exchange",
        standalone_query="Adobe FY2017 operating cash flow ratio",
        response_mode="model advisory text",
    )
    value.update(changes)
    return ConversationDecisionV3(**value)


def test_model_need_retrieval_is_not_forwarded_directly():
    policy = build_conversation_policy("Explain that result briefly.", decision(need_retrieval=True), has_previous_evidence=True)
    assert not policy.retrieval
    assert policy.reuse_previous_evidence


def test_challenge_forces_fresh_verification():
    policy = build_conversation_policy("Are you sure? Recheck it.", decision(need_retrieval=False), has_previous_evidence=True)
    assert policy.retrieval and policy.reuse_previous_evidence
    assert policy.response_mode == "verify"


def test_citation_and_calculation_explanation_reuse_evidence():
    citation = build_conversation_policy("Which source page supports that?", decision(), has_previous_evidence=True)
    calculation = build_conversation_policy("How did you calculate it? Show the formula.", decision(), has_previous_evidence=True)
    assert not citation.retrieval and citation.reuse_previous_evidence
    assert not calculation.retrieval and calculation.reuse_previous_evidence
    assert calculation.response_mode == "explain_calculation"
    passive = build_conversation_policy("How was free cash flow calculated? Show the two evidence values.", decision(), has_previous_evidence=True)
    assert passive.response_mode == "explain_calculation"


def test_condition_change_retrieves_resolved_query():
    policy = build_conversation_policy("Use FY2017 instead.", decision(), has_previous_evidence=True)
    assert policy.retrieval and policy.rewrite and not policy.reuse_previous_evidence
    assert policy.retrieval_query == "Adobe FY2017 operating cash flow ratio"


def test_comparison_period_change_is_a_condition_modification():
    policy = build_conversation_policy("Change the comparison to FY2020 versus FY2021.", decision(), has_previous_evidence=True)
    assert policy.retrieval and policy.rewrite and not policy.reuse_previous_evidence


def test_condition_change_with_citation_request_still_retrieves():
    observed = decision(
        need_retrieval=True,
        depends_on_history=True,
        query_resolution_needed=True,
        standalone_query="Calculate Adobe FY2017 operating cash flow ratio and cite the filing pages",
    )
    policy = build_conversation_policy(
        "把财年改为 FY2017，仍使用同一公式重新计算，并引用文件和页码。",
        observed,
        has_previous_evidence=True,
    )
    assert policy.retrieval and policy.rewrite and not policy.reuse_previous_evidence
    assert policy.response_mode == "answer_with_modified_conditions"


def test_resolved_metric_change_with_citation_request_retrieves():
    observed = decision(
        need_retrieval=True,
        depends_on_history=True,
        query_resolution_needed=True,
        standalone_query="Calculate Adobe FY2021 and FY2022 operating margins and cite the filing pages",
    )
    policy = build_conversation_policy(
        "Now calculate operating margin for FY2021 and FY2022 and cite the source pages.",
        observed,
        has_previous_evidence=True,
    )
    assert policy.retrieval and policy.rewrite and not policy.reuse_previous_evidence
    assert policy.policy_reason == "resolved_followup_requires_fresh_retrieval"


def test_independent_question_retrieves_even_if_model_need_retrieval_is_false():
    observed = decision(depends_on_history=False, query_resolution_needed=False, required_memory_scope="none", need_retrieval=False, standalone_query="New finance question")
    policy = build_conversation_policy("What was revenue in FY2023?", observed, has_previous_evidence=True)
    assert policy.retrieval and not policy.rewrite and not policy.reuse_previous_evidence
