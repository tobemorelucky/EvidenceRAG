from calculation_retry_shadow_v1 import (
    RETRY_SYSTEM_INSTRUCTION,
    build_retry_messages,
    refusal_detected,
    retry_eligibility,
)


def test_retry_requires_calculation_term_and_supported_warning():
    assert retry_eligibility("What is the gross margin?", ["formula_inconsistent_or_not_verifiable"])["eligible"]
    assert retry_eligibility("What is the percentage change?", ["unnecessary_refusal_with_available_operands"])["eligible"]
    assert not retry_eligibility("What was revenue?", ["formula_inconsistent_or_not_verifiable"])["eligible"]
    assert not retry_eligibility("What is the turnover?", ["answer_contains_unsupported_numbers"])["eligible"]


def test_retry_messages_only_add_requested_instruction_and_never_reference_answer():
    messages = build_retry_messages("What is the ratio?", "Revenue was 10 and cost was 5.")
    assert len(messages) == 3
    assert messages[1].content == RETRY_SYSTEM_INSTRUCTION
    assert "What is the ratio?" in messages[2].content
    assert "Revenue was 10" in messages[2].content
    assert "reference answer" not in "\n".join(message.content for message in messages).lower()


def test_refusal_detection_is_conservative():
    assert refusal_detected("I cannot calculate the ratio from this evidence.")
    assert not refusal_detected("The ratio is 1.25 based on the cited operands.")
