"""Contracts for deterministic canonical finance metric execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MetricContract:
    metric_name: str
    metric_alias: str
    requested_periods: tuple[str, ...]
    inferred_periods: tuple[str, ...]
    statement_types: tuple[str, ...]
    formula_variant: str
    required_operands: tuple[str, ...]
    optional_operands: tuple[str, ...] = ()
    output_unit: str = ""
    rounding_decimal_places: int | None = None
    trend_requested: bool = False
    interpretation_required: bool = False
    trigger_reason: str = "canonical_metric_alias"

    def trace_dict(self) -> dict[str, Any]:
        return asdict(self)
