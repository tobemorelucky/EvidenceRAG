import os
from decimal import Decimal

import pytest

from backend.runtime_profile import CLEAN_BASELINE_OVERRIDES, apply_runtime_profile, feature_state
from backend.skill_tools.decimal_calculator import DecimalCalculationError, calculate_decimal
from backend.skill_tools.operand_search import extract_operand_candidates, resolve_unique_operand
from backend.skills.explicit_formula.schema import AtomicOperand
from backend.skills.explicit_formula.skill import ExplicitFormulaSkill, build_formula_contract


def test_formula_profile_inherits_clean_path_and_enables_only_skill(monkeypatch):
    for name in {*CLEAN_BASELINE_OVERRIDES, "RAG_PROFILE", "EXPLICIT_FORMULA_SKILL_ENABLED"}:
        monkeypatch.setenv(name, os.getenv(name, ""))
    apply_runtime_profile("clean_baseline_formula_skill")
    state = feature_state("clean_baseline_formula_skill")

    assert state["profile"] == "clean_baseline_formula_skill"
    assert state["modules"]["Explicit Formula Skill"] is True
    assert all(
        not enabled for name, enabled in state["modules"].items()
        if name != "Explicit Formula Skill"
    )
    assert state["field_aware"] is False
    assert state["page_first"] is False

    # The frozen profile must restore the exact no-skill state after an A/B run.
    apply_runtime_profile("clean_baseline")
    assert not any(feature_state("clean_baseline")["modules"].values())


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "What is the FY2020 ratio? Define ratio as total current assets divided by total current liabilities. Round your answer to two decimal places.",
            "current_assets_2020 / current_liabilities_2020",
        ),
        (
            "What is FY2022 ROA? ROA is defined as: FY2022 net income / (average total assets between FY2021 and FY2022). Round your answer to two decimal places.",
            "net_income_2022 / (average(total_assets_2021, total_assets_2022))",
        ),
        (
            "What is FY2017 DPO? DPO is defined as: 365 * (average accounts payable between FY2016 and FY2017) / (FY2017 COGS + change in inventory between FY2016 and FY2017).",
            "365 * (average(accounts_payable_2016, accounts_payable_2017)) / (cogs_2017 + (inventory_2017 - inventory_2016))",
        ),
    ],
)
def test_question_defined_formula_builds_safe_contract(question, expected):
    contract, failure = build_formula_contract(question)
    assert failure == ""
    assert contract is not None
    assert contract.expression == expected


def test_standard_metric_without_explicit_definition_does_not_trigger():
    question = "Calculate the FY2022 quick ratio for Example Corp."
    contract, failure = build_formula_contract(question)
    assert contract is None
    assert failure == "no_explicit_formula_cue"
    assert ExplicitFormulaSkill().can_handle(question) is False


def test_quarter_formula_does_not_silently_resolve_as_annual_period():
    contract, failure = build_formula_contract(
        "For Q2 of FY2023, define ratio as total current assets divided by total current liabilities."
    )
    assert failure == ""
    assert contract is not None
    assert {item.period for item in contract.operands} == {"2023Q2"}


def test_decimal_calculator_supports_average_and_half_up_rounding():
    result = calculate_decimal(
        "365 * average(ap_2016, ap_2017) / (cogs + (inv_2017 - inv_2016))",
        {
            "ap_2016": Decimal("200"),
            "ap_2017": Decimal("220"),
            "cogs": Decimal("800"),
            "inv_2016": Decimal("100"),
            "inv_2017": Decimal("120"),
        },
        2,
    )
    assert result["full_precision_result"].startswith("93.475609")
    assert result["display_result"] == "93.48"


def test_decimal_calculator_rejects_calls_other_than_average():
    with pytest.raises(DecimalCalculationError):
        calculate_decimal("max(a, b)", {"a": Decimal("1"), "b": Decimal("2")})


def test_operand_extraction_preserves_period_scale_sign_and_source():
    operand = AtomicOperand(
        key="current_liabilities_2022",
        concept="current_liabilities",
        label="total current liabilities",
        aliases=("total current liabilities",),
        period="2022",
    )
    documents = [{
        "filename": "EXAMPLE_2022_10K.pdf",
        "page_number": 42,
        "text": (
            "Example Corp Consolidated Balance Sheets\n"
            "(USD in millions)\n2022 2021\n"
            "Total current liabilities (1,250.5) 1,100.0"
        ),
    }]
    candidates = extract_operand_candidates(
        operand, documents, "What were Example Corp's FY2022 current liabilities?", ["EXAMPLE_2022_10K.pdf"],
    )
    resolved, failure = resolve_unique_operand(candidates)

    assert failure == ""
    assert resolved is not None
    assert resolved.period == "2022"
    assert resolved.normalized_value == Decimal("-1250.5")
    assert resolved.currency == "USD"
    assert resolved.scale == "millions"
    assert resolved.page_number == 42
    assert "Total current liabilities" in resolved.source_text


def test_operand_extraction_drops_leading_parenthesized_footnote_marker():
    operand = AtomicOperand(
        key="cogs_2022",
        concept="cost_of_goods_sold",
        label="cost of goods sold",
        aliases=("cost of goods sold",),
        period="2022",
        statement_types=("income_statement",),
    )
    documents = [{
        "filename": "EXAMPLE_2022_10K.pdf",
        "page_number": 31,
        "text": (
            "Example Corp Consolidated Statements of Income\n"
            "(USD in millions)\n2022 2021\n"
            "Cost of goods sold (1) 7,880 6,420"
        ),
    }]
    candidates = extract_operand_candidates(
        operand, documents, "Example Corp FY2022 cost of goods sold", ["EXAMPLE_2022_10K.pdf"],
    )

    assert candidates
    assert candidates[0].normalized_value == Decimal("7880")
    assert candidates[0].raw_value == "7,880"


def test_operand_extraction_drops_leading_bracketed_footnote_marker():
    operand = AtomicOperand(
        key="cogs_2021", concept="cost_of_goods_sold", label="cost of goods sold",
        aliases=("cost of goods sold",), period="2021", statement_types=("income_statement",),
    )
    documents = [{
        "filename": "EXAMPLE_2022_10K.pdf",
        "page_number": 31,
        "text": "Consolidated Statement of Income\n2022 2021\nCost of goods sold [2] 7,880 6,420",
    }]
    candidates = extract_operand_candidates(
        operand, documents, "Example Corp FY2021 cost of goods sold", ["EXAMPLE_2022_10K.pdf"],
    )

    assert candidates
    assert candidates[0].normalized_value == Decimal("6420")


def test_operand_extraction_preserves_genuine_small_and_parenthesized_values():
    positive = AtomicOperand(
        key="income_2022", concept="net_income", label="net income",
        aliases=("net income",), period="2022", statement_types=("income_statement",),
    )
    negative = AtomicOperand(
        key="loss_2022", concept="net_loss", label="net loss",
        aliases=("net loss",), period="2022", statement_types=("income_statement",),
    )
    documents = [{
        "filename": "EXAMPLE_2022_10K.pdf",
        "page_number": 31,
        "text": (
            "Example Corp Consolidated Statements of Income\n2022 2021\n"
            "Net income 1 2\nNet loss (1) (2)"
        ),
    }]

    positive_candidates = extract_operand_candidates(
        positive, documents, "Example Corp FY2022 net income", ["EXAMPLE_2022_10K.pdf"],
    )
    negative_candidates = extract_operand_candidates(
        negative, documents, "Example Corp FY2022 net loss", ["EXAMPLE_2022_10K.pdf"],
    )

    assert positive_candidates[0].normalized_value == Decimal("1")
    assert negative_candidates[0].normalized_value == Decimal("-1")


def test_operand_extraction_rejects_unexplained_extra_leading_small_integer():
    operand = AtomicOperand(
        key="cogs_2022", concept="cost_of_goods_sold", label="cost of goods sold",
        aliases=("cost of goods sold",), period="2022", statement_types=("income_statement",),
    )
    documents = [{
        "filename": "EXAMPLE_2022_10K.pdf",
        "page_number": 31,
        "text": "Consolidated Statement of Income\n2022 2021\nCost of goods sold 1 7,880 6,420",
    }]

    assert extract_operand_candidates(
        operand, documents, "Example Corp FY2022 cost of goods sold", ["EXAMPLE_2022_10K.pdf"],
    ) == []


def test_statement_operand_rejects_note_table_cross_reference_as_primary_statement():
    operand = AtomicOperand(
        key="cogs_2022", concept="cost_of_goods_sold", label="cost of goods sold",
        aliases=("cost of goods sold",), period="2022", statement_types=("income_statement",),
    )
    note_page = {
        "filename": "EXAMPLE_2022_10K.pdf",
        "page_number": 180,
        "text": (
            "Notes to Consolidated Financial Statements (Continued)\n2022 2021 2020\n"
            "AOCL Components Affected Line Item in the Consolidated Statements of Operations\n"
            "Non-regulated cost of goods sold (1) 1 (3)"
        ),
    }
    statement_page = {
        "filename": "EXAMPLE_2022_10K.pdf",
        "page_number": 50,
        "text": (
            "Example Corp Consolidated Statements of Operations\n2022 2021\n"
            "Cost of goods sold 7,880 6,420"
        ),
    }

    candidates = extract_operand_candidates(
        operand,
        [note_page, statement_page],
        "Example Corp FY2022 cost of goods sold",
        ["EXAMPLE_2022_10K.pdf"],
    )

    assert len(candidates) == 1
    assert candidates[0].page_number == 50
    assert candidates[0].normalized_value == Decimal("7880")


def test_equal_confidence_conflicting_operand_values_are_rejected():
    operand = AtomicOperand(
        key="revenue_2022", concept="revenue", label="revenue", aliases=("revenue",), period="2022",
    )
    documents = [
        {"filename": "EXAMPLE_2022_10K.pdf", "page_number": 1, "text": "USD in millions\n2022\nRevenue 100"},
        {"filename": "EXAMPLE_2022_10K.pdf", "page_number": 2, "text": "USD in millions\n2022\nRevenue 200"},
    ]
    candidates = extract_operand_candidates(
        operand, documents, "Example FY2022 revenue", ["EXAMPLE_2022_10K.pdf"],
    )
    resolved, failure = resolve_unique_operand(candidates)
    assert resolved is None
    assert failure == "operand_value_ambiguous"


def test_skill_normalizes_known_scale_differences_before_ratio():
    question = (
        "What is Adobe's FY2017 ratio? Ratio is defined as: cash from operations / "
        "total current liabilities. Round your answer to two decimal places."
    )
    documents = [
        {
            "filename": "ADOBE_2017_10K.pdf",
            "page_number": 60,
            "text": "Adobe Consolidated Statements of Cash Flows\n(USD in millions)\n2017 2016\nNet cash provided by operating activities 2,912.9 2,199.7",
        },
        {
            "filename": "ADOBE_2017_10K.pdf",
            "page_number": 56,
            "text": "Adobe Consolidated Balance Sheets\n(USD in thousands)\n2017 2016\nTotal current liabilities 3,527,457 2,811,635",
        },
    ]
    result = ExplicitFormulaSkill().execute(question, documents, documents)
    assert result.success is True
    assert result.trace["display_result"] == "0.83"
    assert result.trace["skill_dense_bm25_calls"] == 0


def test_formula_profile_failure_preserves_clean_evidence_and_prompt(monkeypatch):
    from backend.answer_generator import build_answer_messages
    from backend.rag_orchestrator import ExecutionConfig, _prepare_clean_baseline_response
    from backend.skills.explicit_formula.schema import SkillResult

    documents = [{
        "filename": "EXAMPLE_2022_10K.pdf",
        "page_number": 7,
        "text": "Example Corp evidence for 2022.",
    }]
    monkeypatch.setattr("backend.rag_orchestrator._run_search", lambda _question: {
        "docs": documents,
        "initial_candidate_docs": documents,
        "rag_trace": {"rrf_fused_candidate_count": 1},
    })
    monkeypatch.setattr(
        "backend.rag_orchestrator._open_retrieved_pages",
        lambda docs: (docs, {"answer_page_open_requested": 1, "answer_page_opened": 1}),
    )
    monkeypatch.setattr(
        "skills.registry.execute_matching_skill",
        lambda *_args: SkillResult(detected=True, trace={
            "skill_detected": True,
            "skill_success": False,
            "skill_applied": False,
            "fallback_to_clean_baseline": True,
            "skill_latency_ms": 0,
        }),
    )
    clean = _prepare_clean_baseline_response(
        "What is explicitly defined as A / B?",
        ExecutionConfig(profile="clean_baseline", requested_mode="static"),
        0,
    )
    skill = _prepare_clean_baseline_response(
        "What is explicitly defined as A / B?",
        ExecutionConfig(profile="clean_baseline_formula_skill", requested_mode="static"),
        0,
    )

    assert skill["skill_applied"] is False
    assert skill["evidence"] == clean["evidence"]
    assert skill["docs"] == clean["docs"]
    assert skill["citations"] == clean["citations"]
    clean_messages = build_answer_messages(
        "What is explicitly defined as A / B?", clean["evidence"], profile="clean_baseline",
    )
    skill_messages = build_answer_messages(
        "What is explicitly defined as A / B?", skill["evidence"], profile="clean_baseline_formula_skill",
    )
    assert [message.content for message in skill_messages] == [message.content for message in clean_messages]
