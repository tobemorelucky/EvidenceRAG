"""Deterministic Evidence Ranking v1 shadow module."""

from __future__ import annotations

import re
from typing import Any

try:
    from evidence_assembly_v5 import EvidenceUnit, _render
except ModuleNotFoundError:
    from backend.evidence_assembly_v5 import EvidenceUnit, _render


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*|(?:19|20)\d{2}|\d+(?:\.\d+)?%?")
_NUMBER_RE = re.compile(r"(?:[$€£¥]\s*)?\(?-?\d[\d,]*(?:\.\d+)?%?\)?")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "what", "when",
    "which", "who", "with", "would",
}
WEIGHTS = {
    "retrieval_score": 0.50,
    "query_lexical_overlap": 0.25,
    "numeric_presence": 0.10,
    "period_match": 0.10,
    "unit_completeness": 0.05,
}


def _terms(value: object) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(str(value or ""))
        if token.casefold() not in _STOPWORDS
    }


def _years(value: object) -> set[str]:
    return set(_YEAR_RE.findall(str(value or "")))


def _numeric_presence(value: object) -> float:
    for match in _NUMBER_RE.finditer(str(value or "")):
        plain = match.group(0).replace(",", "").strip("$€£¥()-% ")
        if not (plain.isdigit() and 1900 <= int(plain) <= 2099):
            return 1.0
    return 0.0


def _completeness(unit: dict, query_years: set[str]) -> float:
    required = [unit.get("document_id"), unit.get("page_id"), unit.get("entity"), unit.get("source_text")]
    if query_years:
        required.append(unit.get("period"))
    if unit.get("source_type") == "table":
        required.extend([unit.get("metric"), unit.get("value"), unit.get("unit")])
    return sum(value not in (None, "", [], {}) for value in required) / max(1, len(required))


def score_evidence_unit(question: str, unit: EvidenceUnit | dict) -> dict[str, float]:
    value = unit.to_dict() if callable(getattr(unit, "to_dict", None)) else unit
    metadata = value.get("metadata") or {}
    rank = max(1, int(metadata.get("retrieval_rank") or 1))
    original_score = metadata.get("retrieval_score")
    retrieval_score = float(original_score) if original_score is not None else 1.0 / rank
    retrieval_score = max(0.0, min(1.0, retrieval_score))
    query_terms = _terms(question)
    source_terms = _terms(value.get("source_text"))
    lexical = len(query_terms & source_terms) / max(1, len(query_terms))
    question_years = _years(question)
    unit_years = set(str(item) for item in value.get("period") or []) | _years(value.get("source_text"))
    period_match = len(question_years & unit_years) / len(question_years) if question_years else 0.0
    features = {
        "retrieval_score": retrieval_score,
        "query_lexical_overlap": lexical,
        "numeric_presence": _numeric_presence(value.get("source_text")),
        "period_match": period_match,
        "unit_completeness": _completeness(value, question_years),
    }
    return {name: round(score, 8) for name, score in features.items()}


def rank_evidence_units_v1(question: str, units: list[EvidenceUnit | dict]) -> list[dict[str, Any]]:
    ranked = []
    for fallback_rank, unit in enumerate(units, 1):
        value = unit.to_dict() if callable(getattr(unit, "to_dict", None)) else dict(unit)
        features = score_evidence_unit(question, value)
        score = sum(WEIGHTS[name] * features[name] for name in WEIGHTS)
        ranked.append({
            **value,
            "ranking_v1_score": round(score, 8),
            "ranking_v1_features": features,
            "original_candidate_order": fallback_rank,
        })
    ranked.sort(key=lambda item: (
        -item["ranking_v1_score"],
        int((item.get("metadata") or {}).get("retrieval_rank") or 0),
        item["original_candidate_order"],
    ))
    for rank, item in enumerate(ranked, 1):
        item["ranking_v1_rank"] = rank
    return ranked


def select_ranked_evidence_v1(
    question: str,
    units: list[EvidenceUnit | dict],
    *,
    max_context_chars: int = 28000,
) -> tuple[str, list[dict], dict]:
    ranked = rank_evidence_units_v1(question, units)
    selected: list[dict] = []
    rendered: list[str] = []
    used = 0
    for item in ranked:
        evidence_unit = EvidenceUnit(**{
            key: item[key]
            for key in EvidenceUnit.__dataclass_fields__
        })
        value = _render(evidence_unit, len(selected) + 1)
        separator = 2 if rendered else 0
        if used + separator + len(value) > max_context_chars:
            continue
        selected.append(item)
        rendered.append(value)
        used += separator + len(value)
    context = "\n\n".join(rendered)
    return context, selected, {
        "ranking": "evidence_ranking_v1_shadow",
        "weights": WEIGHTS,
        "candidate_unit_count": len(ranked),
        "selected_unit_count": len(selected),
        "selected_text_unit_count": sum(item.get("source_type") == "text" for item in selected),
        "selected_table_unit_count": sum(item.get("source_type") == "table" for item in selected),
        "context_chars": len(context),
        "max_context_chars": max_context_chars,
        "rank_trace": [{
            "source_type": item.get("source_type"),
            "page_id": item.get("page_id"),
            "ranking_v1_rank": item["ranking_v1_rank"],
            "ranking_v1_score": item["ranking_v1_score"],
            "ranking_v1_features": item["ranking_v1_features"],
            "selected": item in selected,
        } for item in ranked],
    }
