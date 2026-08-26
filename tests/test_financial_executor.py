from decimal import Decimal

import pytest

from financial_executor import (
    FinancialExecutionError,
    execute_financial_operation,
    filter_evidence_frames,
)


def _frame(evidence_id, value, *, period="2024", company="Example Co", currency="USD", scale="millions", scope="consolidated"):
    return {
        "evidence_id": evidence_id,
        "normalized_value": str(value),
        "company": company,
        "period": period,
        "currency": currency,
        "scale": scale,
        "scope": scope,
        "citation": f"[source: report.pdf, page {evidence_id[-1]}]",
        "row_label": evidence_id,
    }


@pytest.mark.parametrize(
    ("operation", "values", "expected"),
    [
        ("sum", ["10", "2", "-1"], "11"),
        ("subtract", ["10", "2", "1"], "7"),
        ("multiply", ["2", "3", "4"], "24"),
        ("divide", ["1", "8"], "0.125"),
        ("average", ["2", "3"], "2.5"),
    ],
)
def test_decimal_arithmetic_operations_are_auditable(operation, values, expected):
    frames = [_frame(f"ef_{index}", value) for index, value in enumerate(values)]
    result = execute_financial_operation(
        operation,
        frames,
        operand_evidence_ids=[frame["evidence_id"] for frame in frames],
        rounding=2,
    )

    assert result["result"] == expected
    assert result["operand_evidence_ids"] == [frame["evidence_id"] for frame in frames]
    assert result["executor"] == "evidence_frame"
    assert Decimal(result["full_precision_result"]) == Decimal(expected)


def test_percentage_change_compare_and_selection_operations():
    frames = [_frame("ef_current", "120", period="2024"), _frame("ef_prior", "100", period="2023")]

    change = execute_financial_operation("percentage_change", frames, operand_evidence_ids=["ef_current", "ef_prior"])
    comparison = execute_financial_operation("compare", frames, operand_evidence_ids=["ef_current", "ef_prior"])
    maximum = execute_financial_operation("argmax", frames, operand_evidence_ids=["ef_current", "ef_prior"])
    minimum = execute_financial_operation("argmin", frames, operand_evidence_ids=["ef_current", "ef_prior"])

    assert change["result"] == "20.0"
    assert change["unit"] == "percent"
    assert comparison["direction"] == "greater"
    assert maximum["selected_evidence_id"] == "ef_current"
    assert minimum["selected_evidence_id"] == "ef_prior"


def test_filter_select_and_count_use_structured_metadata():
    frames = [_frame("ef_1", "10", period="2024"), _frame("ef_2", "20", period="2023")]

    assert [item["evidence_id"] for item in filter_evidence_frames(frames, {"period": "2024"})] == ["ef_1"]
    selected = execute_financial_operation("select", frames, criteria={"period": "2024"})
    counted = execute_financial_operation("count", frames, criteria={"company": "example co"})

    assert selected["result"] == ["ef_1"]
    assert counted["result"] == "2"


@pytest.mark.parametrize(
    ("field", "changed"),
    [("company", "Other Co"), ("currency", "EUR"), ("scale", "thousands"), ("scope", "segment")],
)
def test_incompatible_metadata_is_rejected(field, changed):
    first = _frame("ef_1", "10")
    second = _frame("ef_2", "2")
    second[field] = changed

    with pytest.raises(FinancialExecutionError):
        execute_financial_operation("divide", [first, second], operand_evidence_ids=["ef_1", "ef_2"])


def test_same_period_arithmetic_and_missing_evidence_ids_are_rejected():
    frames = [_frame("ef_1", "10", period="2024"), _frame("ef_2", "2", period="2023")]

    with pytest.raises(FinancialExecutionError, match="period"):
        execute_financial_operation("divide", frames, operand_evidence_ids=["ef_1", "ef_2"])
    with pytest.raises(FinancialExecutionError, match="unknown operand"):
        execute_financial_operation("sum", frames, operand_evidence_ids=["ef_missing"])
    with pytest.raises(FinancialExecutionError, match="division by zero"):
        execute_financial_operation("divide", [_frame("ef_a", "1"), _frame("ef_b", "0")], operand_evidence_ids=["ef_a", "ef_b"])
