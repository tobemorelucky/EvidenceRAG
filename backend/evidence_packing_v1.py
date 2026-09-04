"""Deterministic utility-based Evidence Packing v1 shadow implementation."""

from __future__ import annotations

import re
from typing import Any

try:
    from evidence_assembly_v5 import EvidenceUnit, _render
except ModuleNotFoundError:
    from backend.evidence_assembly_v5 import EvidenceUnit, _render


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_NUMBER_RE = re.compile(r"(?:[$€£¥]\s*)?\(?-?\d[\d,]*(?:\.\d+)?%?\)?")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "what", "when",
    "which", "who", "with", "would",
}
_FEATURE_WEIGHTS = {"query": 1.0, "period": 1.5, "number": 0.15, "structure": 0.1}


def _unit_dict(unit: EvidenceUnit | dict) -> dict:
    return unit.to_dict() if callable(getattr(unit, "to_dict", None)) else dict(unit)


def _evidence_unit(unit: dict) -> EvidenceUnit:
    return EvidenceUnit(**{name: unit.get(name) for name in EvidenceUnit.__dataclass_fields__})


def _terms(value: object) -> set[str]:
    return {
        token.casefold() for token in _TOKEN_RE.findall(str(value or ""))
        if token.casefold() not in _STOP
    }


def _numbers(value: object) -> set[str]:
    values = set()
    for match in _NUMBER_RE.finditer(str(value or "")):
        number = match.group(0).replace(",", "").replace(" ", "").strip("()$€£¥")
        plain = number.rstrip("%")
        try:
            if plain.isdigit() and 1900 <= int(plain) <= 2099:
                continue
        except ValueError:
            pass
        values.add(number)
    return values


def coverage_features(question: str, unit: EvidenceUnit | dict) -> set[str]:
    """Build a generic, gold-free coverage signature for one unit."""
    value = _unit_dict(unit)
    text = str(value.get("source_text") or "")
    query_terms = _terms(question)
    text_terms = _terms(text)
    question_years = set(_YEAR_RE.findall(question))
    unit_years = set(str(item) for item in value.get("period") or []) | set(_YEAR_RE.findall(text))
    features = {f"query:{term}" for term in query_terms & text_terms}
    features.update(f"period:{year}" for year in question_years & unit_years)
    features.update(f"number:{number}" for number in _numbers(text))
    source_type = str(value.get("source_type") or "")
    if source_type:
        features.add(f"structure:{source_type}")
    return features


def query_relevance_score(question: str, unit: EvidenceUnit | dict) -> float:
    """Return generic lexical query coverage for anchor selection."""
    query_terms = _terms(question)
    if not query_terms:
        return 0.0
    value = _unit_dict(unit)
    text_terms = _terms(value.get("source_text"))
    return len(query_terms & text_terms) / len(query_terms)


def _feature_weight(feature: str) -> float:
    return _FEATURE_WEIGHTS.get(feature.split(":", 1)[0], 0.0)


def _weighted_gain(features: set[str], covered: set[str]) -> float:
    return sum(_feature_weight(feature) for feature in features - covered)


def _normalized_tokens(value: object) -> list[str]:
    return re.findall(r"[a-z0-9%$.-]+", re.sub(r"\s+", " ", str(value or "")).strip().casefold())


def near_duplicate_similarity(left: object, right: object) -> float:
    left_tokens, right_tokens = _normalized_tokens(left), _normalized_tokens(right)
    if len(left_tokens) < 20 or len(right_tokens) < 20:
        return 0.0
    if _numbers(left) != _numbers(right):
        return 0.0
    if min(len(left_tokens), len(right_tokens)) / max(len(left_tokens), len(right_tokens)) < 0.9:
        return 0.0
    left_shingles = {tuple(left_tokens[index:index + 5]) for index in range(len(left_tokens) - 4)}
    right_shingles = {tuple(right_tokens[index:index + 5]) for index in range(len(right_tokens) - 4)}
    return len(left_shingles & right_shingles) / max(1, len(left_shingles | right_shingles))


def _key(unit: dict) -> tuple:
    metadata = unit.get("metadata") or {}
    return (
        unit.get("source_type"), unit.get("page_id"), metadata.get("chunk_id"),
        metadata.get("table_id"), metadata.get("row_index"),
    )


def _rank(unit: dict) -> int:
    return int((unit.get("current_ranking") or {}).get("rank") or unit.get("ranking_v1_rank") or 0)


def _score(unit: dict) -> float:
    return float((unit.get("current_ranking") or {}).get("score") or unit.get("ranking_v1_score") or 0.0)


def _page_key(unit: dict) -> tuple:
    metadata = unit.get("metadata") or {}
    return unit.get("page_id"), metadata.get("filename"), int(metadata.get("page_number") or 0)


def _render_selection(units: list[dict]) -> tuple[str, int]:
    values = []
    for index, unit in enumerate(sorted(units, key=_rank), 1):
        values.append(_render(_evidence_unit(unit), index))
    context = "\n\n".join(values)
    return context, len(context)


def _unit_utility(
    question: str,
    unit: dict,
    selected: list[dict],
    covered: set[str],
    *,
    feature_cache: dict[tuple, set[str]] | None = None,
    length_cache: dict[tuple, int] | None = None,
    similarity_cache: dict[tuple, float] | None = None,
) -> dict[str, float]:
    key = _key(unit)
    features = feature_cache[key] if feature_cache is not None else coverage_features(question, unit)
    gain = _weighted_gain(features, covered)
    rendered_length = (
        length_cache[key]
        if length_cache is not None else max(1, len(_render(_evidence_unit(unit), 1)))
    )
    same_page_count = sum(_page_key(existing) == _page_key(unit) for existing in selected)
    page_penalty = 1.0 / (1.0 + 0.2 * same_page_count)
    similarities = []
    for existing in selected:
        pair = tuple(sorted((key, _key(existing)), key=repr))
        if similarity_cache is not None and pair in similarity_cache:
            similarity = similarity_cache[pair]
        else:
            similarity = near_duplicate_similarity(unit.get("source_text"), existing.get("source_text"))
            if similarity_cache is not None:
                similarity_cache[pair] = similarity
        similarities.append(similarity)
    duplicate_similarity = max(similarities, default=0.0)
    duplicate_penalty = 0.1 if duplicate_similarity >= 0.92 else 1.0
    utility = _score(unit) * gain / rendered_length * page_penalty * duplicate_penalty
    return {
        "coverage_gain": gain,
        "rendered_length": float(rendered_length),
        "page_repeat_penalty": page_penalty,
        "near_duplicate_similarity": duplicate_similarity,
        "near_duplicate_penalty": duplicate_penalty,
        "utility": utility,
    }


def _selection_objective(
    question: str,
    units: list[dict],
    *,
    feature_cache: dict[tuple, set[str]] | None = None,
    length_cache: dict[tuple, int] | None = None,
    similarity_cache: dict[tuple, float] | None = None,
) -> tuple[float, dict[tuple, float]]:
    covered: set[str] = set()
    ordered: list[dict] = []
    contributions: dict[tuple, float] = {}
    total = 0.0
    for unit in sorted(units, key=_rank):
        details = _unit_utility(
            question, unit, ordered, covered,
            feature_cache=feature_cache,
            length_cache=length_cache,
            similarity_cache=similarity_cache,
        )
        contributions[_key(unit)] = details["utility"]
        total += details["utility"]
        covered.update(feature_cache[_key(unit)] if feature_cache is not None else coverage_features(question, unit))
        ordered.append(unit)
    return total, contributions


def select_evidence_packing_v1(
    question: str,
    units: list[EvidenceUnit | dict],
    *,
    max_context_chars: int = 28000,
    replacement_threshold: float = 1.0,
    protected_unit_keys: set[tuple] | None = None,
    max_replacements: int | None = None,
) -> tuple[str, list[dict], dict]:
    """Select units by score × marginal generic coverage / rendered length."""
    if replacement_threshold < 1.0:
        raise ValueError("replacement_threshold must be >= 1.0")
    if max_replacements is not None and max_replacements < 0:
        raise ValueError("max_replacements must be >= 0 or None")
    protected_unit_keys = protected_unit_keys or set()
    candidates = [_unit_dict(unit) for unit in units]
    feature_cache = {_key(unit): coverage_features(question, unit) for unit in candidates}
    length_cache = {
        _key(unit): max(1, len(_render(_evidence_unit(unit), 1))) for unit in candidates
    }
    similarity_cache: dict[tuple, float] = {}
    empty: list[dict] = []
    initial = {
        _key(unit): _unit_utility(
            question, unit, empty, set(),
            feature_cache=feature_cache,
            length_cache=length_cache,
            similarity_cache=similarity_cache,
        )
        for unit in candidates
    }
    ordered = sorted(candidates, key=lambda unit: (-initial[_key(unit)]["utility"], _rank(unit), _key(unit)))
    selected: list[dict] = []
    covered: set[str] = set()
    trace_by_key: dict[tuple, dict[str, Any]] = {}
    replacements = 0
    objective_cache: tuple[float, dict[tuple, float]] | None = None
    for unit in ordered:
        key = _key(unit)
        details = _unit_utility(
            question, unit, selected, covered,
            feature_cache=feature_cache,
            length_cache=length_cache,
            similarity_cache=similarity_cache,
        )
        trace = {
            "rank": _rank(unit),
            "score": _score(unit),
            **{name: round(value, 10) for name, value in details.items()},
            "selected": False,
            "selection_reason": None,
            "replaced_unit_rank": None,
            "anchor_protected": key in protected_unit_keys,
        }
        if details["coverage_gain"] <= 0 or details["utility"] <= 0:
            trace["selection_reason"] = "zero_marginal_coverage_gain"
            trace_by_key[key] = trace
            continue
        proposed = [*selected, unit]
        _, proposed_chars = _render_selection(proposed)
        if proposed_chars <= max_context_chars:
            selected = proposed
            covered.update(feature_cache[key])
            objective_cache = None
            trace["selected"] = True
            trace["selection_reason"] = "selected_direct"
            trace_by_key[key] = trace
            continue

        if max_replacements is not None and replacements >= max_replacements:
            trace["selection_reason"] = "replacement_budget_exhausted"
            trace_by_key[key] = trace
            continue

        if objective_cache is None:
            objective_cache = _selection_objective(
                question, selected,
                feature_cache=feature_cache,
                length_cache=length_cache,
                similarity_cache=similarity_cache,
            )
        current_objective, contributions = objective_cache
        replaceable = [item for item in selected if _key(item) not in protected_unit_keys]
        lowest = min(
            replaceable,
            key=lambda item: (contributions[_key(item)], -_rank(item)),
        ) if replaceable else None
        if lowest is None:
            trace["selection_reason"] = "budget_no_unprotected_replacement_candidate"
            trace_by_key[key] = trace
            continue
        if details["utility"] <= contributions[_key(lowest)] + 1e-12:
            trace["selection_reason"] = "budget_incoming_utility_not_above_lowest_selected"
            trace_by_key[key] = trace
            continue
        replacement = [item for item in selected if _key(item) != _key(lowest)] + [unit]
        _, replacement_chars = _render_selection(replacement)
        replacement_objective, _ = _selection_objective(
            question, replacement,
            feature_cache=feature_cache,
            length_cache=length_cache,
            similarity_cache=similarity_cache,
        )
        required_objective = current_objective * replacement_threshold
        objective_passes = (
            replacement_objective > current_objective + 1e-12
            if replacement_threshold == 1.0 else
            replacement_objective + 1e-12 >= required_objective
        )
        if replacement_chars <= max_context_chars and objective_passes:
            selected = replacement
            covered = set().union(*(feature_cache[_key(item)] for item in selected))
            objective_cache = None
            trace["selected"] = True
            trace["selection_reason"] = "selected_by_replacement"
            trace["replaced_unit_rank"] = _rank(lowest)
            old_trace = trace_by_key.get(_key(lowest))
            if old_trace is not None:
                old_trace["selected"] = False
                old_trace["selection_reason"] = "replaced_by_higher_total_utility"
                old_trace["replaced_by_rank"] = _rank(unit)
            replacements += 1
        else:
            trace["selection_reason"] = "budget_no_beneficial_single_replacement"
        trace_by_key[key] = trace

    context, context_chars = _render_selection(selected)
    selected = sorted(selected, key=_rank)
    selected_keys = {_key(unit) for unit in selected}
    traces = []
    for unit in sorted(candidates, key=_rank):
        key = _key(unit)
        value = trace_by_key[key]
        value["selected"] = key in selected_keys
        metadata = unit.get("metadata") or {}
        traces.append({
            "unit_id": metadata.get("chunk_id") or f"{metadata.get('table_id')}:{metadata.get('row_index')}",
            "page_id": unit.get("page_id"),
            "source_type": unit.get("source_type"),
            **value,
        })
    return context, selected, {
        "packing": "evidence_packing_optimization_v1_shadow",
        "utility_formula": "ranking_score * generic_marginal_coverage_gain / rendered_length * penalties",
        "gold_used_for_selection": False,
        "candidate_unit_count": len(candidates),
        "selected_unit_count": len(selected),
        "context_chars": context_chars,
        "max_context_chars": max_context_chars,
        "replacement_count": replacements,
        "replacement_threshold": replacement_threshold,
        "max_replacements": max_replacements,
        "protected_anchor_count": len(protected_unit_keys),
        "trace": traces,
    }
