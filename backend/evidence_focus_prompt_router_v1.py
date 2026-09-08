"""Deterministic shadow router for the finance evidence-focus answer prompt."""

from __future__ import annotations

import re


TREND_TERMS = re.compile(
    r"\b(?:trend|increase|decrease|improve|decline|growth|compare|comparison)\b|增长|下降|改善|趋势|比较",
    re.IGNORECASE,
)
CALCULATION_TERMS = re.compile(
    r"\b(?:ratio|margin|percentage|turnover|growth)\b",
    re.IGNORECASE,
)


def route_evidence_focus_prompt(
    question: str,
    *,
    context_chars: int,
    candidate_evidence_chunks: int,
) -> dict:
    """Return a transparent rule decision without using benchmark metadata."""
    reasons: list[str] = []
    if TREND_TERMS.search(question or ""):
        reasons.append("trend_or_comparison_language")
    if CALCULATION_TERMS.search(question or ""):
        reasons.append("calculation_metric_language")
    if int(context_chars) > 15_000:
        reasons.append("long_context_gt_15000")
    if int(candidate_evidence_chunks) >= 8:
        reasons.append("candidate_evidence_chunks_gte_8")
    return {"use_evidence_focus": bool(reasons), "reasons": reasons}

