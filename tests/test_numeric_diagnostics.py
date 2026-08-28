from decimal import Decimal

from scripts.evaluate_financebench_numeric_diagnostics import _last_number, _numeric_equivalent


def test_last_number_normalizes_scale_and_ignores_citation_page():
    value = _last_number("Net PP&E was $8.738 billion [source: report.pdf, page 57].")

    assert value == (Decimal("8738000000"), False)


def test_numeric_equivalence_accepts_rounding_and_percent_decimal_forms():
    rounded, _ = _numeric_equivalent("The result is 93.95 days.", "The result is 93.86 days.")
    percent, _ = _numeric_equivalent("The ratio is 0.365.", "The answer is 36.5%.")

    assert rounded is True
    assert percent is True


def test_numeric_equivalence_is_null_for_qualitative_answers():
    equivalent, details = _numeric_equivalent("Revenue increased.", "Yes, it increased.")

    assert equivalent is None
    assert details["reason"] == "no_comparable_final_numeric_value"


def test_numeric_equivalence_prefers_bold_conclusion_over_later_operand():
    equivalent, details = _numeric_equivalent(
        "Using 120 and 80, the result is **50.0%**. The comparison spans two periods.",
        "The percentage change is 50% based on 120 and 80.",
    )

    assert equivalent is True
    assert details["candidate_extraction_method"] == "bold_result"


def test_numeric_equivalence_uses_requested_currency_scale_and_parentheses_sign():
    scaled, _ = _numeric_equivalent(
        "The result was **$1.616 billion**.",
        "$1616.00",
        "What was the amount in USD millions?",
    )
    negative, _ = _numeric_equivalent(
        "Corporate reported **$(473) million**.",
        "Corporate reported -$473 million.",
    )

    assert scaled is True
    assert negative is True
