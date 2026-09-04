"""Local requirement/evidence compatibility for offline shadow experiments only.

No domain aliases or operation inference. Scores measure visible support, not
operand validity, semantic entailment, or a guarantee of answerability.
"""

from __future__ import annotations

import copy
import re

from backend.evidence_assembly_v5 import EvidenceUnit
from backend.evidence_packing_v1 import _numbers, _rank, _score, _terms
from backend.query_requirement_v1 import QueryRequirement, _QUARTER, _YEAR, periods


# Task/instruction words only; no domain metric dictionary.
_TASK_WORDS = {
    "much", "many", "calculate", "compute", "compare", "comparison", "explain",
    "why", "please", "state", "question", "answer", "based", "data", "if", "then",
    "not", "useful", "relevant", "meaningful", "metric", "company", "like",
    "roughly", "only", "use", "details", "shown", "fiscal", "year", "period",
    "fy", "during", "between", "end", "as", "its", "it", "can",
}
_LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "company"}


def _lexical_terms(value: object) -> set[str]:
    # Use the same representation for query, entity removal and source text.
    text = re.sub(r"(?<=\w)['’]s\b", "", str(value or ""), flags=re.I)
    return {term.strip("'") for term in _terms(text) if term.strip("'")}


def _entity_tokens(value: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", value.casefold()) if t not in _LEGAL_SUFFIXES]


def entity_match(question: str, entity: str) -> tuple[float, str, set[str]]:
    """Match contiguous query tokens, allowing metadata whitespace differences.

    No acronym expansion. An unmentioned entity is unresolved, not a conflict:
    the frozen requirement specifies no independently resolved target entity.
    """
    entity_tokens = _entity_tokens(entity)
    if not entity_tokens or entity.casefold() in {"unknown", "none", "null"}:
        return 0.0, "unknown_metadata", set()
    target = "".join(entity_tokens)
    query_tokens = _entity_tokens(question)
    for start in range(len(query_tokens)):
        for end in range(start + 1, len(query_tokens) + 1):
            joined = "".join(query_tokens[start:end])
            if len(joined) > len(target):
                break
            if joined == target:
                return 1.0, "query_entity_surface_match", set(query_tokens[start:end])
    return 0.0, "target_unresolved_or_not_mentioned", set()


def _numeric_tokens(text: str) -> set[str]:
    return _numbers(_QUARTER.sub(" ", _YEAR.sub(" ", text)))


def _fragments(unit: dict) -> list[str]:
    text = str(unit.get("source_text") or "")
    if unit.get("source_type") == "table":
        # Existing table Units already contain title/header + one selected row.
        # Keep their context together; do not create or parse a new table.
        return [text] if text.strip() else []
    return [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", text) if part.strip()]


def match_requirement_evidence(question: str, requirement: QueryRequirement, unit: dict) -> dict:
    """Question + frozen requirement + one Unit -> compatibility in [0, 1]."""
    text = str(unit.get("source_text") or "")
    entity_score, entity_status, entity_words = entity_match(question, str(unit.get("entity") or ""))
    target_terms = _lexical_terms(_QUARTER.sub(" ", _YEAR.sub(" ", question))) - _TASK_WORDS - _lexical_terms(" ".join(entity_words))
    fragments = _fragments(unit)
    local = []
    requested = set(requirement.explicit_periods)
    for index, fragment in enumerate(fragments):
        overlap = len(target_terms & _lexical_terms(fragment)) / len(target_terms) if target_terms else 0.0
        local.append({"index": index, "metric_overlap": overlap,
                      "numbers": sorted(_numeric_tokens(fragment)), "periods": sorted(periods(fragment))})
    best = max(local, key=lambda f: (f["metric_overlap"], -f["index"]), default=None)
    text_metric = best["metric_overlap"] if best else 0.0
    metric_terms = _lexical_terms(unit.get("metric"))
    meta_metric = len(target_terms & metric_terms) / len(target_terms) if target_terms else 0.0
    # Metadata-only relevance receives partial credit, never full textual support.
    metric_score = max(text_metric, 0.5 * meta_metric)
    relevant = [f for f in local if f["metric_overlap"] > 0 and f["metric_overlap"] == text_metric]
    numeric_score = max((min(len(f["numbers"]), 1) for f in relevant), default=0.0)
    pair_score = max((min(len(f["numbers"]), 2) / 2 for f in relevant), default=0.0)
    local_periods = set().union(*(set(f["periods"]) for f in relevant))
    raw_periods = periods(text)
    metadata_periods = periods(" ".join(map(str, unit.get("period") or [])))
    def fraction(values):
        return len(values & requested) / len(requested) if requested else float(bool(values))
    period_score = max(fraction(local_periods), 0.5 * fraction(raw_periods | metadata_periods))
    if fraction(local_periods):
        period_status = "local_text_full" if fraction(local_periods) == 1 else "local_text_partial"
    elif fraction(raw_periods | metadata_periods):
        period_status = "nonlocal_or_metadata_only"
    elif requested and raw_periods | metadata_periods:
        period_status = "requested_period_not_visible"
    else:
        period_status = "unknown"
    active = {}
    if requirement.requires_entity:
        active["entity_match"] = entity_score
    if requirement.requires_period:
        active["period_match"] = period_score
    if requirement.requires_numeric_evidence:
        active["numeric_availability"] = numeric_score
    if requirement.requires_calculation or requirement.requires_comparison:
        active["calculation_or_comparison_support"] = pair_score
    # Same bounded bonus envelope as Query Requirement v1, no weight search.
    # Without active constraints, use only target-text relevance.
    compatibility = metric_score * (sum(active.values()) / len(active) if active else 1.0)
    return {
        "compatibility_score": compatibility, "metric_relevance": metric_score,
        "text_metric_relevance": text_metric, "metadata_metric_relevance": meta_metric,
        "target_terms": sorted(target_terms), "entity_match": entity_score, "entity_status": entity_status,
        "period_match": period_score, "period_status": period_status,
        "numeric_availability": numeric_score, "calculation_support": pair_score,
        "active_features": active, "best_fragment_index": best["index"] if best else None,
        "best_fragment_preview": fragments[best["index"]][:500] if best else "",
        "supporting_fragments": relevant, "requested_periods": sorted(requested),
        "local_periods": sorted(local_periods), "metadata_periods": sorted(metadata_periods),
        "limitation": "Co-occurrence proxy only; no verified operand binding, arithmetic, or entity disambiguation",
    }


def matching_inputs(question: str, requirement: QueryRequirement, units: list[dict]) -> tuple[list[dict], dict]:
    ranks = [_rank(unit) for unit in units]
    if any(r <= 0 for r in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("Unique positive frozen ranks required")
    adapted, traces = [], []
    for unit in units:
        clean = {key: copy.deepcopy(unit[key]) for key in (*EvidenceUnit.__dataclass_fields__, "current_ranking")}
        match = match_requirement_evidence(question, requirement, clean)
        clean["current_ranking"]["score"] = _score(unit) * (1 + match["compatibility_score"])
        adapted.append(clean)
        traces.append({"rank": _rank(unit), "original_score": _score(unit),
                       "effective_score": clean["current_ranking"]["score"], **match})
    return adapted, {"requirement": requirement.to_dict(), "unit_matches": traces,
                     "score_formula": "original_score * (1 + compatibility_score)",
                     "compatibility_formula": "local_metric_relevance * mean(active_support)",
                     "gold_used": False}
