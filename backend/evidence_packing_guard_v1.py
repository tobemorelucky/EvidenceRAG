"""Conservative regression guards for Evidence Packing v1 shadow experiments."""

from __future__ import annotations

from typing import Any

try:
    from evidence_assembly_v5 import EvidenceUnit
    from evidence_packing_v1 import (
        _key,
        _rank,
        _unit_dict,
        query_relevance_score,
        select_evidence_packing_v1,
    )
except ModuleNotFoundError:
    from backend.evidence_assembly_v5 import EvidenceUnit
    from backend.evidence_packing_v1 import (
        _key,
        _rank,
        _unit_dict,
        query_relevance_score,
        select_evidence_packing_v1,
    )


def select_anchor_keys(
    question: str,
    units: list[EvidenceUnit | dict[str, Any]],
    *,
    anchor_top_n: int = 5,
    anchor_min_query_relevance: float = 0.35,
) -> set[tuple]:
    """Protect top-ranked units only when they have high generic query relevance."""
    if anchor_top_n < 0:
        raise ValueError("anchor_top_n must be >= 0")
    if not 0.0 <= anchor_min_query_relevance <= 1.0:
        raise ValueError("anchor_min_query_relevance must be between 0 and 1")
    candidates = sorted((_unit_dict(unit) for unit in units), key=_rank)
    return {
        _key(unit)
        for unit in candidates[:anchor_top_n]
        if query_relevance_score(question, unit) >= anchor_min_query_relevance
    }


def select_evidence_packing_guard_v1(
    question: str,
    units: list[EvidenceUnit | dict[str, Any]],
    *,
    max_context_chars: int = 28000,
    replacement_threshold: float = 1.05,
    max_replacements: int | None = 5,
    anchor_top_n: int = 5,
    anchor_min_query_relevance: float = 0.35,
) -> tuple[str, list[dict], dict]:
    """Run Packing v1 with conservative replacement and anchor guards."""
    anchors = select_anchor_keys(
        question,
        units,
        anchor_top_n=anchor_top_n,
        anchor_min_query_relevance=anchor_min_query_relevance,
    )
    context, selected, trace = select_evidence_packing_v1(
        question,
        units,
        max_context_chars=max_context_chars,
        replacement_threshold=replacement_threshold,
        protected_unit_keys=anchors,
        max_replacements=max_replacements,
    )
    trace.update({
        "packing": "evidence_packing_regression_guard_v1_shadow",
        "anchor_top_n": anchor_top_n,
        "anchor_min_query_relevance": anchor_min_query_relevance,
        "anchor_keys": [list(key) for key in sorted(anchors, key=repr)],
    })
    return context, selected, trace
