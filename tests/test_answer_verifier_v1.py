from backend.answer_verifier_v1 import verify_answer_v1


def test_direct_lookup_number_in_evidence_is_consistent():
    result = verify_answer_v1(
        "What was revenue in FY2023?",
        "Revenue was $120 million in FY2023 [source: report.pdf, page 8].",
        "Revenue for 2023 was $120 million.",
    )
    assert result.numeric_ok is True
    assert result.metric_ok is True
    assert result.period_ok is True
    assert result.formula_ok is True
    assert result.risk_score == 0


def test_unsupported_number_and_missing_period_are_flagged():
    result = verify_answer_v1(
        "What was revenue in FY2023?",
        "Revenue was $130 million.",
        "Revenue for 2023 was $120 million.",
    )
    assert result.numeric_ok is False
    assert result.period_ok is False
    assert "answer_contains_unsupported_numbers" in result.warnings
    assert "requested_period_missing_from_answer" in result.warnings


def test_ratio_formula_result_is_allowed_when_operands_are_supported():
    result = verify_answer_v1(
        "What was the current ratio in FY2023?",
        "Current ratio: $120 / $100 = 1.2 in FY2023.",
        "Current assets were $120 and current liabilities were $100 in 2023.",
    )
    assert result.formula_ok is True
    assert result.numeric_ok is True


def test_inconsistent_growth_formula_is_detected():
    result = verify_answer_v1(
        "What was the percentage change in revenue from FY2022 to FY2023?",
        "Revenue growth was (120 - 100) / 100 * 100 = 30% from FY2022 to FY2023.",
        "Revenue was 100 in 2022 and 120 in 2023.",
    )
    assert result.formula_ok is False
    assert result.details["explicit_formulas"][0]["expected"] == "20.0000"


def test_metric_and_period_consistency_are_independent():
    result = verify_answer_v1(
        "What was inventory in FY2022?",
        "Revenue was 10 in FY2021.",
        "Inventory was 10 in 2022.",
    )
    assert result.metric_ok is False
    assert result.period_ok is False


def test_refusal_is_flagged_only_when_formula_operands_are_relevant_and_available():
    result = verify_answer_v1(
        "What was the quick ratio in FY2023?",
        "The quick ratio cannot be calculated from the provided evidence for FY2023.",
        "Quick ratio inputs for 2023 were 120 and 100.",
    )
    assert result.details["invalid_refusal"] is True
    assert "unnecessary_refusal_with_available_operands" in result.warnings


def test_refusal_without_relevant_operands_is_not_marked_unnecessary():
    result = verify_answer_v1(
        "What was the quick ratio in FY2023?",
        "The quick ratio cannot be calculated from the provided evidence for FY2023.",
        "The annual report discusses liquidity but provides no ratio inputs.",
    )
    assert result.details["invalid_refusal"] is False
