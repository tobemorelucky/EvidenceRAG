import os

import pytest

from backend.runtime_profile import CLEAN_BASELINE_OVERRIDES, apply_runtime_profile, feature_state
from backend.skill_tools.operand_search import OperandSearchResult
from backend.skills.canonical_finance_metric.skill import (
    CanonicalFinanceMetricSkill,
    build_metric_contract,
)
from backend.skills.explicit_formula.schema import SkillResult
from backend.skills.registry import execute_matching_skill


def _income(text: str, page: int = 20) -> dict:
    return {
        "filename": "EXAMPLE_CORP_2023_10K.pdf",
        "company": "Example Corp",
        "page_number": page,
        "text": "Example Corp Consolidated Statements of Operations\n(USD in millions)\n" + text,
    }


def _balance(text: str, page: int = 10) -> dict:
    return {
        "filename": "EXAMPLE_CORP_2023_10K.pdf",
        "company": "Example Corp",
        "page_number": page,
        "text": "Example Corp Consolidated Balance Sheets\n(USD in millions)\n" + text,
    }


def test_finance_skills_profile_inherits_clean_path_and_enables_only_two_skills(monkeypatch):
    for name in {*CLEAN_BASELINE_OVERRIDES, "RAG_PROFILE"}:
        monkeypatch.setenv(name, os.getenv(name, ""))
    apply_runtime_profile("finance_skills_v1")
    state = feature_state("finance_skills_v1")
    assert state["modules"]["Explicit Formula Skill"] is True
    assert state["modules"]["Canonical Finance Metric Skill"] is True
    assert all(
        not enabled for name, enabled in state["modules"].items()
        if name not in {"Explicit Formula Skill", "Canonical Finance Metric Skill"}
    )
    apply_runtime_profile("clean_baseline")
    assert not any(feature_state("clean_baseline")["modules"].values())


@pytest.mark.parametrize(
    ("question", "metric", "periods", "inferred"),
    [
        ("What was Example Corp's quick ratio in FY2023?", "quick_ratio", ("2023",), ()),
        ("Calculate Example Corp's acid-test ratio for 2023.", "quick_ratio", ("2023",), ()),
        ("What was Example Corp's inventory turnover in FY2023?", "inventory_turnover", ("2023",), ()),
        ("What was Example Corp's gross profit margin in 2022?", "gross_margin", ("2022",), ()),
        ("Did Example Corp's gross margin improve as of FY2022?", "gross_margin", ("2021", "2022"), ("2021",)),
        ("Compare Example Corp operating margin between 2021 and 2022.", "operating_margin", ("2021", "2022"), ()),
        ("What was Example Corp's operating income margin in FY2023?", "operating_margin", ("2023",), ()),
        ("Does Example Corp have a healthy profile based on its quick ratio for FY22? If the metric is not relevant, explain why.", "quick_ratio", ("2022",), ()),
    ],
)
def test_generic_metric_contracts(question, metric, periods, inferred):
    contract, failure = build_metric_contract(question)
    assert failure == ""
    assert contract is not None
    assert contract.metric_name == metric
    assert contract.requested_periods == periods
    assert contract.inferred_periods == inferred


@pytest.mark.parametrize(
    ("question", "reason"),
    [
        ("Why did Example Corp's gross margin increase in 2023?", "causal_question_excluded"),
        ("What drove Example Corp's operating margin in 2023?", "causal_question_excluded"),
        ("What were the drivers of inventory turnover in 2023?", "causal_question_excluded"),
        ("Was Example Corp's gross margin historically consistent in 2023?", "subjective_question_excluded"),
        ("Is operating margin useful for Example Corp in 2023?", "subjective_question_excluded"),
        ("Does the quick ratio show a capital-intensive business in 2023?", "subjective_question_excluded"),
        ("Does gross margin show a high-growth business in 2023?", "subjective_question_excluded"),
        ("What is Example Corp's return on assets in 2023?", "no_canonical_metric_alias"),
        ("What is Example Corp's gross margin?", "required_period_missing"),
        ("Compare gross margin in 2020, 2021 and 2022.", "trend_requires_exactly_two_periods"),
    ],
)
def test_negative_routing_is_conservative(question, reason):
    contract, failure = build_metric_contract(question)
    assert contract is None
    assert failure == reason


def test_explicit_formula_has_registry_priority(monkeypatch):
    question = (
        "What is Example Corp's FY2023 quick ratio? Define quick ratio as total current assets "
        "divided by total current liabilities."
    )
    contract, failure = build_metric_contract(question)
    assert contract is None
    assert failure == "explicit_formula_has_priority"
    class FakeExplicit:
        name = "explicit_formula"

        @staticmethod
        def can_handle(_question):
            return True

        @staticmethod
        def execute(*_args):
            return SkillResult(detected=True, trace={"skill_name": "explicit_formula"})

    class FakeCanonical:
        name = "canonical_finance_metric"

        @staticmethod
        def can_handle(_question):
            return True

        @staticmethod
        def execute(*_args):
            raise AssertionError("lower-priority skill must not execute")

    monkeypatch.setattr("backend.skills.registry._SKILLS", (FakeExplicit(), FakeCanonical()))
    result = execute_matching_skill(question, [], [], ("explicit_formula", "canonical_finance_metric"))
    assert result.trace["skill_name"] == "explicit_formula"


def test_registry_no_match_is_uniform_and_does_not_execute_explicit_skill():
    result = execute_matching_skill("Summarize Example Corp's strategy.", [], [], ("explicit_formula", "canonical_finance_metric"))
    assert result.detected is False
    assert result.trace["skill_name"] == "none"
    assert result.trace["fallback_reason"] == "no_registered_skill_matched"


def test_quick_ratio_uses_only_validated_quick_assets():
    documents = [_balance(
        "2023 2022\n"
        "Cash and cash equivalents 100 80\n"
        "Short-term investments 20 15\n"
        "Accounts receivable, net 50 40\n"
        "Due from related parties 10 8\n"
        "Inventory 999 900\nPrepaid expenses 777 700\nOther current assets 333 300\n"
        "Total current assets 2,289 2,043\nTotal current liabilities 100 90"
    )]
    result = CanonicalFinanceMetricSkill().execute(
        "What was Example Corp's quick ratio in FY2023?", documents, documents,
    )
    assert result.success is True
    assert result.trace["metric_display_result"] == "1.80"
    assert "inventory" not in {item["concept"] for item in result.trace["resolved_operands"]}


def test_quick_ratio_allows_absent_optional_rows_only_on_complete_statement():
    documents = [_balance(
        "2023 2022\nCash and cash equivalents 100 80\nAccounts receivable, net 50 40\n"
        "Inventory 30 20\nTotal current assets 180 140\nTotal current liabilities 100 90"
    )]
    result = CanonicalFinanceMetricSkill().execute(
        "What was Example Corp's quick ratio in FY2023?", documents, documents,
    )
    assert result.success is True
    assert result.trace["skill_dense_bm25_calls"] == 0
    assert set(result.trace["validated_absent_optional_operands"]) == {
        "short_term_investments_2023", "current_related_party_receivables_2023",
    }


def test_quick_ratio_health_uses_verified_result_but_keeps_single_llm_interpretation():
    documents = [_balance(
        "2023 2022\nCash and cash equivalents 100 80\nAccounts receivable, net 50 40\n"
        "Total current assets 150 120\nTotal current liabilities 100 90"
    )]
    result = CanonicalFinanceMetricSkill().execute(
        "Was Example Corp's quick ratio healthy in FY2023?", documents, documents,
    )
    assert result.success is True
    assert result.applied is False
    assert result.trace["authoritative_answer"] is False
    assert "1.50" in result.trace["verified_evidence"]
    assert result.trace["skill_llm_calls"] == 0


def test_inventory_turnover_requires_beginning_and_ending_inventory():
    documents = [
        _income("2023 2022\nCost of goods sold 600 500"),
        _balance("2023 2022\nInventories 100 80\nTotal current assets 500 450\nTotal current liabilities 200 190"),
    ]
    result = CanonicalFinanceMetricSkill().execute(
        "What was Example Corp's inventory turnover in FY2023?", documents, documents,
    )
    assert result.success is True
    assert result.trace["metric_display_result"] == "6.67"
    assert {item["period"] for item in result.trace["resolved_operands"]} == {"2022", "2023"}


def test_inventory_turnover_rejects_ending_inventory_only(monkeypatch):
    monkeypatch.setattr(
        "backend.skills.canonical_finance_metric.skill.search_missing_operands",
        lambda *_args, **_kwargs: OperandSearchResult((), (), 0),
    )
    documents = [
        _income("2023\nCost of goods sold 600"),
        _balance("2023\nInventories 100\nTotal current assets 500\nTotal current liabilities 200"),
    ]
    result = CanonicalFinanceMetricSkill().execute(
        "What was Example Corp's inventory turnover in FY2023?", documents, documents,
    )
    assert result.success is False
    assert "beginning_inventory_2023" in result.trace["operand_resolution_failure_reason"]


def test_gross_margin_prefers_direct_gross_profit():
    documents = [_income("2023 2022\nRevenue 1,000 900\nGross profit 400 350\nCost of goods sold 999 888")]
    result = CanonicalFinanceMetricSkill().execute(
        "What was Example Corp's gross margin in FY2023?", documents, documents,
    )
    assert result.success is True
    assert result.trace["metric_display_result"] == "40.00"
    assert result.trace["formula_variant"] == "gross_profit / revenue"


def test_gross_margin_uses_same_period_revenue_minus_cogs_fallback():
    documents = [_income("2023 2022\nRevenue 1,000 900\nCost of goods sold 600 550")]
    result = CanonicalFinanceMetricSkill().execute(
        "What was Example Corp's gross margin in FY2023?", documents, documents,
    )
    assert result.success is True
    assert result.trace["metric_display_result"] == "40.00"
    assert result.trace["formula_variant"] == "(revenue - cost_of_goods_sold) / revenue"


def test_operating_margin_trend_is_deterministic():
    documents = [_income("2022 2021\nRevenue 1,000 800\nOperating income 200 120")]
    result = CanonicalFinanceMetricSkill().execute(
        "Did Example Corp's operating margin improve as of FY2022?", documents, documents,
    )
    assert result.success is True
    assert result.applied is True
    assert result.trace["trend_comparison_result"] == "increase"
    assert [item["display_result"] for item in result.trace["metric_results"]] == ["15.00", "20.00"]


def test_cross_document_operands_are_rejected():
    documents = [
        _income("2023\nRevenue 1,000", 20),
        {
            "filename": "OTHER_CORP_2023_10K.pdf", "company": "Example Corp", "page_number": 21,
            "text": "Other Corp Consolidated Statements of Operations\n(USD in millions)\n2023\nOperating income 200",
        },
    ]
    result = CanonicalFinanceMetricSkill().execute(
        "What was Example Corp's operating margin in FY2023?", documents, documents,
    )
    assert result.success is False


def test_unknown_scale_fails_closed():
    documents = [{
        "filename": "EXAMPLE_CORP_2023_10K.pdf", "company": "Example Corp", "page_number": 20,
        "text": "Example Corp Consolidated Statements of Operations\nUSD\n2023\nRevenue 1,000\nOperating income 200",
    }]
    result = CanonicalFinanceMetricSkill().execute(
        "What was Example Corp's operating margin in FY2023?", documents, documents,
    )
    assert result.success is False
    assert "operand_scale_unknown" in result.trace["operand_resolution_failure_reason"]


def test_quick_ratio_does_not_borrow_cash_from_another_period(monkeypatch):
    monkeypatch.setattr(
        "backend.skills.canonical_finance_metric.skill.search_missing_operands",
        lambda *_args, **_kwargs: OperandSearchResult((), (), 0),
    )
    documents = [_balance(
        "2023 2022\nAccounts receivable, net 50 40\nTotal current assets 150 120\n"
        "Total current liabilities 100 90\nCash and cash equivalents (not reported) 80"
    )]
    result = CanonicalFinanceMetricSkill().execute(
        "What was Example Corp's quick ratio in FY2023?", documents, documents,
    )
    assert result.success is False
    assert "cash_and_equivalents_2023" in result.trace["operand_resolution_failure_reason"]


def test_metric_normalizes_thousands_and_millions():
    documents = [
        _income("2023\nRevenue 1,000", 20),
        {
            "filename": "EXAMPLE_CORP_2023_10K.pdf", "company": "Example Corp", "page_number": 21,
            "text": "Example Corp Consolidated Statements of Operations\n(USD in thousands)\n2023\nOperating income 200,000",
        },
    ]
    result = CanonicalFinanceMetricSkill().execute(
        "What was Example Corp's operating margin in FY2023?", documents, documents,
    )
    assert result.success is True
    assert result.trace["metric_display_result"] == "20.00"


def test_metric_preserves_parenthesized_negative_value():
    documents = [_income("2023\nRevenue 1,000\nOperating income (200)")]
    result = CanonicalFinanceMetricSkill().execute(
        "What was Example Corp's operating margin in FY2023?", documents, documents,
    )
    assert result.success is True
    assert result.trace["metric_display_result"] == "-20.00"


def test_metric_rejects_equal_confidence_conflicting_values(monkeypatch):
    monkeypatch.setattr(
        "backend.skills.canonical_finance_metric.skill.search_missing_operands",
        lambda *_args, **_kwargs: OperandSearchResult((), (), 0),
    )
    documents = [
        _income("2023\nRevenue 1,000\nOperating income 200", 20),
        _income("2023\nRevenue 2,000\nOperating income 200", 21),
    ]
    result = CanonicalFinanceMetricSkill().execute(
        "What was Example Corp's operating margin in FY2023?", documents, documents,
    )
    assert result.success is False
    assert "revenue_2023" in result.trace["operand_resolution_failure_reason"]


def test_restricted_guarantor_scope_does_not_conflict_with_group_statement():
    documents = [
        _income("2023\nRevenue 1,000\nOperating income 200", 20),
        {
            "filename": "EXAMPLE_CORP_2023_10K.pdf", "company": "Example Corp", "page_number": 80,
            "text": "Deed of Cross Guarantee\nConsolidated Statements of Operations\n(USD in millions)\n2023\nRevenue 50\nOperating income 40",
        },
    ]
    result = CanonicalFinanceMetricSkill().execute(
        "What was Example Corp's operating margin in FY2023?", documents, documents,
    )
    assert result.success is True
    assert result.trace["metric_display_result"] == "20.00"


def test_short_alphanumeric_document_identity_is_matched_dynamically():
    documents = [{
        "filename": "3M_2023_10K.pdf", "page_number": 10,
        "text": "3M Consolidated Balance Sheets\n(USD in millions)\n2023 2022\n"
        "Cash and cash equivalents 100 80\nAccounts receivable, net 50 40\n"
        "Total current assets 150 120\nTotal current liabilities 100 90",
    }]
    result = CanonicalFinanceMetricSkill().execute(
        "What was 3M's quick ratio in FY2023?", documents, documents,
    )
    assert result.success is True
    assert result.trace["metric_display_result"] == "1.50"


def test_conjunction_acronym_matches_full_document_name_without_company_registry():
    documents = [
        {
            "filename": "JASPER_JUNIPER_2023_10K.pdf", "page_number": 20,
            "text": "Jasper and Juniper Consolidated Statements of Operations\n(USD in millions)\n2023\nCost of goods sold 600",
        },
        {
            "filename": "JASPER_JUNIPER_2023_10K.pdf", "page_number": 10,
            "text": "Jasper and Juniper Consolidated Balance Sheets\n(USD in millions)\n2023 2022\n"
            "Inventories 100 80\nTotal current assets 500 450\nTotal current liabilities 200 190",
        },
    ]
    result = CanonicalFinanceMetricSkill().execute(
        "What was JnJ's inventory turnover in FY2023?", documents, documents,
    )
    assert result.success is True
    assert result.trace["metric_display_result"] == "6.67"
