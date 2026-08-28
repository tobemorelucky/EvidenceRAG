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


def structured_coverage_advisory_enabled() -> bool:
    return os.getenv("STRUCTURED_COVERAGE_ADVISORY_ENABLED", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _known_compatible(frames: list[dict[str, Any]], fields: tuple[str, ...]) -> bool:
    for field in fields:
        values = [str(frame.get(field)) for frame in frames if frame.get(field) not in (None, "")]
        if len(values) != len(frames) or len(set(values)) != 1:
            return False
    return bool(frames)


def _frame_concepts(task_spec: dict[str, Any]) -> list[tuple[str, list[str]]]:
    concepts: list[tuple[str, list[str]]] = []
    for field in task_spec.get("required_fields") or []:
        aliases = [str(item) for item in FIELD_ALIASES.get(str(field), [])]
        concepts.append((str(field), aliases or [str(field).replace("_", " ")]))
    target = str(task_spec.get("target_measure") or "").strip()
    task_type = str(task_spec.get("task_type") or "lookup")
    if target and (not concepts or task_type in {"comparison", "selection"}):
        existing = {item.casefold() for _, values in concepts for item in values}
        if target.casefold() not in existing:
            concepts.append(("target_measure", [target]))
    return concepts


def _frames_by_required_field(
    task_spec: dict[str, Any],
    frames: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    # Reuse the same layered resolution as the executor, but do not require
    # formula-level uniqueness for comparison/selection coverage.
    from calculation_service import match_evidence_frames_detailed  # local import avoids an import cycle

    company = str(task_spec.get("company") or "").casefold().strip()
    result: dict[str, list[dict[str, Any]]] = {}
    traces: list[dict[str, Any]] = []
    for field, concepts in _frame_concepts(task_spec):
        candidates, match_trace = match_evidence_frames_detailed(
            field,
            frames,
            concepts=concepts,
            statement_types=[str(item) for item in task_spec.get("statement_types") or []],
            scope=str(task_spec.get("scope") or ""),
        )
        if company:
            candidates = [
                frame for frame in candidates
                if str(frame.get("company") or "").casefold().strip() == company
            ]
        result[str(field)] = candidates
        traces.append(match_trace)
    return result, traces


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
        "base_answerable": base_coverage.get("status") == "complete" and page_supported,
        "structured_coverage_enabled": True,
        "page_supported": page_supported,
        "row_supported": None,
        "period_supported": None,
        "unit_scale_supported": None,
        "scope_supported": None,
        "operands_validated": None,
        "structured_answerable": False,
        "structured_execution_ready": False,
        "structured_status": "not_applicable",
        "answerable": base_coverage.get("status") == "complete" and page_supported,
        "structured_missing": [],
    }
    # Textual lookups retain the existing coverage semantics.
    if task_type not in STRUCTURED_TASKS:
        base["coverage_basis"] = "text_lookup"
        return base

    required_fields = [str(field) for field in task_spec.get("required_fields") or []]
    field_frames, match_traces = _frames_by_required_field(task_spec, frames)
    concept_keys = [field for field, _ in _frame_concepts(task_spec)]
    row_supported = bool(concept_keys) and all(field_frames.get(field) for field in concept_keys)
    relevant_frames = list({
        str(frame.get("evidence_id") or id(frame)): frame
        for field in concept_keys
        for frame in field_frames.get(field, [])
    }.values())
    required_periods = [str(period) for period in task_spec.get("required_periods") or []]
    if required_periods:
        period_supported = row_supported and all(
            all(any(str(frame.get("period") or "") == period for frame in field_frames.get(field, [])) for period in required_periods)
            for field in concept_keys
        )
    else:
        period_supported = row_supported and bool(relevant_frames) and all(frame.get("period") for frame in relevant_frames)

    unit_scale_supported = row_supported and _known_compatible(relevant_frames, ("currency", "scale"))
    scope_supported = row_supported and _known_compatible(relevant_frames, ("scope",))
    operands_validated = False
    if task_type == "calculation" and task_spec.get("formula"):
        operands_validated = bool(resolve_frame_operands(task_spec, frames)) and unit_scale_supported and scope_supported
    elif task_type == "comparison":
        from calculation_service import resolve_comparison_frames
        operands_validated = bool(resolve_comparison_frames(task_spec, frames))
    elif task_type == "selection":
        from calculation_service import resolve_selection_frames
        operands_validated = bool(resolve_selection_frames(task_spec, frames))

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
    structured_answerable = page_supported and operands_validated
    structured_status = "complete" if structured_answerable else "partial" if page_supported and (row_supported or bool(frames)) else "insufficient"
    base_answerable = bool(base["base_answerable"])
    advisory = structured_coverage_advisory_enabled()
    answerable = (base_answerable or structured_answerable) if advisory else structured_answerable
    if advisory:
        status = "complete" if answerable else str(base_coverage.get("status") or structured_status)
    else:
        status = structured_status
    failure_reason = ""
    if not relevant_frames:
        failure_reason = "no_related_frames"
    elif not row_supported:
        failure_reason = "missing_required_concept"
    elif not period_supported:
        failure_reason = "period_not_resolved"
    elif not unit_scale_supported:
        failure_reason = "unit_or_scale_not_validated"
    elif not scope_supported:
        failure_reason = "scope_not_validated"
    elif not operands_validated:
        failure_reason = "operands_not_unique_or_complete"
    frame_candidates = [candidate for trace in match_traces for candidate in trace.get("candidates") or []]
    best_candidate = max(frame_candidates, key=lambda item: float(item.get("match_score") or 0), default={})
    return {
        **base,
        **dimensions,
        "answerable": answerable,
        "status": status,
        "structured_status": structured_status,
        "structured_answerable": structured_answerable,
        "structured_execution_ready": structured_answerable,
        "structured_advisory_mode": advisory,
        "structured_missing": missing,
        "coverage_basis": "evidence_frame",
        "structured_relevant_frame_count": len(relevant_frames),
        "relevant_frame_count": len(relevant_frames),
        "queryspec_concepts": [concept for _, values in _frame_concepts(task_spec) for concept in values],
        "frame_match_candidates": frame_candidates,
        "frame_match_method": best_candidate.get("match_method") or "",
        "frame_match_score": best_candidate.get("match_score") or 0.0,
        "operand_resolution_failure_reason": failure_reason,
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
    concepts = [str(item) for item in task_spec.get("required_concepts") or [] if str(item).strip()]
    periods = [str(item) for item in task_spec.get("required_periods") or []]
    statements = [
        STATEMENT_TYPE_LABELS.get(str(item), str(item))
        for item in task_spec.get("statement_types") or []
    ]
    scope = str(task_spec.get("scope") or "").strip()
    anchors = list(dict.fromkeys([*labels, *concepts, *periods, *statements, *([scope] if scope else [])]))
    if not anchors:
        return ""
    return f"{question}\nMissing financial evidence: {'; '.join(anchors)}"
