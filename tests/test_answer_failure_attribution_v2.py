from scripts.audit_answer_failure_attribution_v2 import classify_failure, remediation


def test_refusal_precedes_calculation():
    category, _ = classify_failure(
        "Calculate the turnover ratio.",
        "I cannot calculate it because the evidence is insufficient.",
        "The candidate fails to provide the requested ratio.",
    )
    assert category == "G"


def test_calculation_failure_is_generic():
    category, _ = classify_failure(
        "Calculate the average percentage and round to one decimal place.",
        "The result is 2.0%.",
        "The candidate calculation does not match and rounds incorrectly.",
    )
    assert category == "E"


def test_direction_failure_is_generic():
    category, _ = classify_failure(
        "Did the value increase or decrease?", "It increased.", "The candidate contradicts the expected decrease."
    )
    assert category == "F"


def test_gold_span_only_affects_offline_remediation():
    assert remediation("E", True) == "structured_financial_reasoning"
    assert remediation("E", False) == "retrieval_or_evidence_flow"
