"""Structured evidence coverage checks for financial answerability."""

from __future__ import annotations

import os
from typing import Any

from calculation_service import resolve_frame_operands
from query_parser import FIELD_ALIASES, STATEMENT_TYPE_LABELS


STRUCTURED_TASKS = {"calculation", "comparison", "selection"}


def structured_coverage_enabled() -> bool:
    return os.getenv("STRUCTURED_COVERAGE_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _known_compatible(frames: list[dict[str, Any]], fields: tuple[str, ...]) -> bool:
    for field in fields:
        values = [str(frame.get(field)) for frame in frames if frame.get(field) not in (None, "")]
        if len(values) != len(frames) or len(set(values)) != 1:
            return False
    return bool(frames)


def _frames_by_required_field(task_spec: dict[str, Any], frames: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    # Reuse the same deterministic alias resolution as the executor, but do
    # not require formula-level uniqueness for comparison/selection coverage.
    from calculation_service import matching_evidence_frames  # local import avoids an import cycle

    company = str(task_spec.get("company") or "").casefold().strip()
    result: dict[str, list[dict[str, Any]]] = {}
    for field in task_spec.get("required_fields") or []:
        candidates = matching_evidence_frames(str(field), frames)
        if company:
            candidates = [
                frame for frame in candidates
                if str(frame.get("company") or "").casefold().strip() == company
            ]
        result[str(field)] = candidates
    return result


def assess_structured_coverage(
    task_spec: dict[str, Any],
    documents: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    base_coverage: dict[str, Any],
) -> dict[str, Any]:
    """Add explicit support dimensions and a conservative answerable status."""
    task_type = str(task_spec.get("task_type") or "lookup")
    page_supported = bool(documents)
    base = {
        **base_coverage,
        "base_status": base_coverage.get("status"),
        "structured_coverage_enabled": True,
        "page_supported": page_supported,
        "row_supported": None,
        "period_supported": None,
        "unit_scale_supported": None,
        "scope_supported": None,
        "operands_validated": None,
        "answerable": base_coverage.get("status") == "complete" and page_supported,
        "structured_missing": [],
    }
    # Textual lookups retain the existing coverage semantics.
    if task_type not in STRUCTURED_TASKS:
        base["coverage_basis"] = "text_lookup"
        return base

    required_fields = [str(field) for field in task_spec.get("required_fields") or []]
    field_frames = _frames_by_required_field(task_spec, frames)
    row_supported = bool(required_fields) and all(field_frames.get(field) for field in required_fields)
    relevant_frames = [frame for field in required_fields for frame in field_frames.get(field, [])]
    required_periods = [str(period) for period in task_spec.get("required_periods") or []]
    if required_periods:
        period_supported = row_supported and all(
            all(any(str(frame.get("period") or "") == period for frame in field_frames.get(field, [])) for period in required_periods)
            for field in required_fields
        )
    else:
        period_supported = row_supported and bool(relevant_frames) and all(frame.get("period") for frame in relevant_frames)

    unit_scale_supported = row_supported and _known_compatible(relevant_frames, ("currency", "scale"))
    scope_supported = row_supported and _known_compatible(relevant_frames, ("scope",))
    operands_validated = False
    if task_type == "calculation" and task_spec.get("formula"):
        operands_validated = bool(resolve_frame_operands(task_spec, frames)) and unit_scale_supported and scope_supported
    elif task_type == "comparison":
        entities = {str(frame.get("company") or "") for frame in relevant_frames if frame.get("company")}
        periods = {str(frame.get("period") or "") for frame in relevant_frames if frame.get("period")}
        target_count = len(required_periods) if required_periods else max(len(periods), len(entities))
        operands_validated = row_supported and target_count >= 2 and period_supported and unit_scale_supported and scope_supported
    elif task_type == "selection":
        candidate_keys = {
            (str(frame.get("row_label") or ""), str(frame.get("period") or ""), str(frame.get("company") or ""))
            for frame in relevant_frames
        }
        operands_validated = row_supported and len(candidate_keys) >= 2 and unit_scale_supported and scope_supported

    missing = []
    dimensions = {
        "page_supported": page_supported,
        "row_supported": row_supported,
        "period_supported": period_supported,
        "unit_scale_supported": unit_scale_supported,
        "scope_supported": scope_supported,
        "operands_validated": operands_validated,
    }
    for name, supported in dimensions.items():
        if supported is False:
            missing.append(name)
    answerable = page_supported and operands_validated
    status = "complete" if answerable else "partial" if page_supported and (row_supported or bool(frames)) else "insufficient"
    return {
        **base,
        **dimensions,
        "answerable": answerable,
        "status": status,
        "structured_missing": missing,
        "coverage_basis": "evidence_frame",
        "structured_relevant_frame_count": len(relevant_frames),
    }


def build_document_scoped_supplemental_query(
    question: str,
    task_spec: dict[str, Any],
    coverage: dict[str, Any],
) -> str:
    """Build one deterministic query from missing concepts and constraints."""
    missing_fields = [str(item) for item in coverage.get("missing_fields") or []]
    structured_missing = set(coverage.get("structured_missing") or [])
    if not missing_fields and structured_missing & {
        "row_supported", "period_supported", "unit_scale_supported", "scope_supported", "operands_validated",
    }:
        missing_fields = [str(item) for item in task_spec.get("required_fields") or []]
    labels = [FIELD_ALIASES[field][0] for field in missing_fields if FIELD_ALIASES.get(field)]
    periods = [str(item) for item in task_spec.get("required_periods") or []]
    statements = [
        STATEMENT_TYPE_LABELS.get(str(item), str(item))
        for item in task_spec.get("statement_types") or []
    ]
    anchors = list(dict.fromkeys([*labels, *periods, *statements]))
    if not anchors:
        return ""
    return f"{question}\nMissing financial evidence: {'; '.join(anchors)}"
