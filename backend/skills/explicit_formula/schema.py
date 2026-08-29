"""Data contracts for the explicit-formula skill."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class AtomicOperand:
    key: str
    concept: str
    label: str
    aliases: tuple[str, ...]
    period: str
    statement_types: tuple[str, ...] = ()
    cash_outflow_magnitude: bool = False


@dataclass(frozen=True)
class ResolvedOperand:
    key: str
    concept: str
    period: str
    raw_value: str
    normalized_value: Decimal
    currency: str
    scale: str
    filename: str
    page_number: int | str | None
    source_text: str
    confidence: float
    scope: str = "consolidated"

    def trace_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["normalized_value"] = format(self.normalized_value, "f")
        return result


@dataclass(frozen=True)
class FormulaContract:
    formula_text: str
    expression: str
    operands: tuple[AtomicOperand, ...]
    final_currency: str = ""
    final_scale: str = ""
    final_unit: str = ""
    rounding_decimal_places: int | None = None
    trigger_reason: str = "question_explicit_definition"

    def trace_dict(self) -> dict[str, Any]:
        return {
            "formula_text": self.formula_text,
            "expression": self.expression,
            "operands": [asdict(item) for item in self.operands],
            "final_currency": self.final_currency,
            "final_scale": self.final_scale,
            "final_unit": self.final_unit,
            "rounding_decimal_places": self.rounding_decimal_places,
            "trigger_reason": self.trigger_reason,
        }


@dataclass
class SkillResult:
    detected: bool = False
    success: bool = False
    applied: bool = False
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

