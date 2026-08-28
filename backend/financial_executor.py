"""Auditable Decimal operations over EvidenceFrame operands.

This module is deliberately domain-neutral: it executes a small operation
vocabulary and validates provenance/compatibility, but it does not define
financial metrics or retrieve evidence.
"""

from __future__ import annotations

from decimal import Decimal, DecimalException, ROUND_HALF_UP
from typing import Any, Iterable


SUPPORTED_OPERATIONS = {
    "select", "filter", "sum", "subtract", "multiply", "divide", "average",
    "percentage_change", "compare", "argmax", "argmin", "count",
}


class FinancialExecutionError(ValueError):
    """Raised when an operation cannot be executed safely and audibly."""


def _frame_index(frames: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for frame in frames:
        evidence_id = str(frame.get("evidence_id") or "").strip()
        if evidence_id:
            indexed[evidence_id] = frame
    return indexed


def filter_evidence_frames(
    frames: Iterable[dict[str, Any]],
    criteria: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Filter frames by exact metadata values without inventing missing values."""
    criteria = criteria or {}
    result = []
    for frame in frames:
        if all(
            str(frame.get(key) or "").casefold() == str(expected or "").casefold()
            for key, expected in criteria.items()
        ):
            result.append(frame)
    return result


def _resolve_operands(
    frames: Iterable[dict[str, Any]],
    operand_evidence_ids: Iterable[str] | None,
    criteria: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    available = list(frames)
    if operand_evidence_ids is None:
        selected = filter_evidence_frames(available, criteria)
    else:
        indexed = _frame_index(available)
        identifiers = [str(item) for item in operand_evidence_ids]
        if not identifiers or len(set(identifiers)) != len(identifiers):
            raise FinancialExecutionError("operand evidence IDs must be non-empty and unique")
        missing = [item for item in identifiers if item not in indexed]
        if missing:
            raise FinancialExecutionError(f"unknown operand evidence IDs: {', '.join(missing)}")
        selected = [indexed[item] for item in identifiers]
        if criteria:
            allowed = {item["evidence_id"] for item in filter_evidence_frames(selected, criteria)}
            if len(allowed) != len(selected):
                raise FinancialExecutionError("one or more operands do not satisfy the filter criteria")
    return selected


def _known_values(frames: list[dict[str, Any]], field: str) -> set[str]:
    return {str(frame.get(field)).casefold() for frame in frames if frame.get(field) not in (None, "")}


def _validate_constraints(
    frames: list[dict[str, Any]],
    operation: str,
    constraints: dict[str, Any],
) -> None:
    checks = {
        "same_company": "company",
        "same_currency": "currency",
        "compatible_scale": "scale",
        "same_scope": "scope",
    }
    for constraint, field in checks.items():
        if constraints.get(constraint, True) and len(_known_values(frames, field)) > 1:
            raise FinancialExecutionError(f"incompatible {field} across operands")

    # Cross-period operations explicitly compare periods. Other arithmetic
    # requires a shared known period whenever multiple period values exist.
    if constraints.get("compatible_period", True) and operation not in {
        "percentage_change", "compare", "argmax", "argmin", "select", "filter", "count",
    }:
        if len(_known_values(frames, "period")) > 1:
            raise FinancialExecutionError("incompatible period across operands")

    expected = constraints.get("expected") or {}
    for field in ("company", "period", "currency", "scale", "scope"):
        expected_value = expected.get(field)
        if expected_value in (None, ""):
            continue
        actual = _known_values(frames, field)
        if not actual or actual != {str(expected_value).casefold()}:
            raise FinancialExecutionError(f"operands do not match expected {field}")


def _values(frames: list[dict[str, Any]]) -> list[Decimal]:
    values: list[Decimal] = []
    for frame in frames:
        try:
            values.append(Decimal(str(frame["normalized_value"])))
        except (KeyError, DecimalException) as exc:
            raise FinancialExecutionError("operand has no valid normalized numeric value") from exc
    return values


def _quantized(value: Decimal, rounding: int | None) -> str | None:
    if rounding is None:
        return None
    try:
        places = max(0, min(12, int(rounding)))
        rounded = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    except (TypeError, ValueError, DecimalException) as exc:
        raise FinancialExecutionError("invalid rounding precision") from exc
    return f"{rounded:.{places}f}"


def execute_financial_operation(
    operation: str,
    frames: Iterable[dict[str, Any]],
    *,
    operand_evidence_ids: Iterable[str] | None = None,
    criteria: dict[str, Any] | None = None,
    constraints: dict[str, Any] | None = None,
    rounding: int | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    """Execute a whitelisted operation using only cited EvidenceFrame values."""
    operation = str(operation or "").strip().lower()
    if operation not in SUPPORTED_OPERATIONS:
        raise FinancialExecutionError(f"unsupported operation: {operation}")
    selected = _resolve_operands(frames, operand_evidence_ids, criteria)
    if not selected and operation not in {"count"}:
        raise FinancialExecutionError("operation has no evidence operands")
    _validate_constraints(selected, operation, constraints or {})

    ids = [str(frame["evidence_id"]) for frame in selected]
    citations = list(dict.fromkeys(str(frame.get("citation") or "") for frame in selected if frame.get("citation")))
    base: dict[str, Any] = {
        "status": "executed",
        "executor": "evidence_frame",
        "operation": operation,
        "operand_evidence_ids": ids,
        "citations": citations,
        "unit": unit,
        "rounding": rounding,
    }
    if operation in {"select", "filter"}:
        return {**base, "selected_frames": selected, "normalized_operands": [], "result": ids}

    values = _values(selected)
    if operation == "count":
        result = Decimal(len(selected))
        expression = f"count({len(selected)})"
    elif operation == "sum":
        result = sum(values, Decimal("0"))
        expression = " + ".join(map(str, values))
    elif operation == "subtract":
        if len(values) < 2:
            raise FinancialExecutionError("subtract requires at least two operands")
        result = values[0] - sum(values[1:], Decimal("0"))
        expression = " - ".join(map(str, values))
    elif operation == "multiply":
        if len(values) < 2:
            raise FinancialExecutionError("multiply requires at least two operands")
        result = Decimal("1")
        for value in values:
            result *= value
        expression = " * ".join(map(str, values))
    elif operation == "divide":
        if len(values) != 2:
            raise FinancialExecutionError("divide requires exactly two operands")
        if values[1] == 0:
            raise FinancialExecutionError("division by zero")
        result = values[0] / values[1]
        expression = f"{values[0]} / {values[1]}"
    elif operation == "average":
        result = sum(values, Decimal("0")) / Decimal(len(values))
        expression = f"({' + '.join(map(str, values))}) / {len(values)}"
    elif operation == "percentage_change":
        if len(values) != 2:
            raise FinancialExecutionError("percentage_change requires current and prior operands")
        if values[1] == 0:
            raise FinancialExecutionError("percentage change has a zero prior value")
        result = (values[0] - values[1]) / abs(values[1]) * Decimal("100")
        expression = f"(({values[0]} - {values[1]}) / abs({values[1]})) * 100"
        base["unit"] = unit or "percent"
    elif operation == "compare":
        if len(values) != 2:
            raise FinancialExecutionError("compare requires exactly two operands")
        result = Decimal("1") if values[0] > values[1] else Decimal("-1") if values[0] < values[1] else Decimal("0")
        expression = f"compare({values[0]}, {values[1]})"
        base["direction"] = "greater" if result > 0 else "less" if result < 0 else "equal"
    elif operation in {"argmax", "argmin"}:
        target = max(values) if operation == "argmax" else min(values)
        index = values.index(target)
        result = target
        expression = f"{operation}({', '.join(map(str, values))})"
        base["selected_evidence_id"] = ids[index]
        base["selected_frame"] = selected[index]
    else:  # pragma: no cover - guarded by SUPPORTED_OPERATIONS
        raise FinancialExecutionError(f"unsupported operation: {operation}")

    return {
        **base,
        "normalized_operands": [format(value, "f") for value in values],
        "expression": expression,
        "full_precision_result": format(result, "f"),
        "result": format(result, "f"),
        "display_result": _quantized(result, rounding),
    }
