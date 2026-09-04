"""Dynamic, metadata-novelty set selection for offline shadow evaluation only."""

from __future__ import annotations

import math
import re
from collections import Counter
from decimal import Decimal, InvalidOperation

from backend.evidence_packing_v1 import _rank, _render_selection, _score

FIELDS = ("entity", "period", "metric", "numeric_value", "unit")
NUMERIC_RE = re.compile(r"(?P<currency>[$€£¥])?\s*(?P<amount>[+-]?(?:\d[\d,]*)(?:\.\d+)?)(?P<percent>%)?")


def _values(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, (list, tuple)) else [value]


def _normalize(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def metadata_features(unit: dict) -> dict[str, set[str]]:
    """Read frozen fields only; do not infer new facts or parse new metadata."""
    result = {name: {_normalize(v) for v in _values(unit.get(name)) if v is not None and _normalize(v)}
              for name in ("entity", "period", "metric", "unit")}
    numeric = set()
    for raw in _values(unit.get("value")):
        text = str(raw).strip()
        # A known period duplicated in the value array is not a new amount.
        if text in result["period"]:
            continue
        negative = text.startswith("(") and text.endswith(")")
        if negative:
            text = text[1:-1].strip()
        match = NUMERIC_RE.fullmatch(text)
        if match is None:
            continue
        try:
            amount = Decimal(match["amount"].replace(",", ""))
        except InvalidOperation:
            continue
        if negative:
            amount = -abs(amount)
        normalized_amount = format(amount.normalize(), "f")
        if not match["currency"] and not match["percent"] and normalized_amount in result["period"]:
            continue
        # Preserve currency, percent and sign; no scale/percent conversion.
        numeric.add(f"{match['currency'] or ''}{normalized_amount}{match['percent'] or ''}")
    result["numeric_value"] = numeric
    return {name: result[name] for name in FIELDS}


def marginal_gain(features: dict[str, set[str]], covered: dict[str, set[str]]) -> tuple[float, dict]:
    detail = {}
    for name in FIELDS:
        new = features[name] - covered[name]
        detail[name] = {"new_values": sorted(new), "new_count": len(new), "total_count": len(features[name]),
                        "fraction": len(new) / len(features[name]) if features[name] else 0.0}
    return sum(item["fraction"] for item in detail.values()) / len(FIELDS), detail


def _shingles(text: str) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[\w%$€£¥+-]+", text.casefold())
    if not tokens:
        return set()
    width = min(3, len(tokens))
    return {tuple(tokens[index:index + width]) for index in range(len(tokens) - width + 1)}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def select_evidence_set_v1(units: list[dict], *, max_context_chars: int = 28000) -> tuple[str, list[dict], dict]:
    """Recompute utility for ALL feasible units after every selection.

    utility = (frozen ranking score + mean category novelty) /
              (1 + same-page saturation + max text Jaccard + max number Jaccard)

    No relevance reweighting, length normalization, replacements, or hard page
    caps. Cost only decides feasibility. Rendering/order remain the old format.
    """
    if max_context_chars < 0:
        raise ValueError("Context budget must be nonnegative")
    ranks = [_rank(unit) for unit in units]
    if len(set(ranks)) != len(ranks) or any(rank < 1 for rank in ranks):
        raise ValueError("Candidates must have unique positive frozen ranks")
    relevance = [_score(unit) for unit in units]
    if any(not math.isfinite(score) or score < 0 for score in relevance):
        raise ValueError("Existing ranking scores must be finite and nonnegative")
    features = [metadata_features(unit) for unit in units]
    shingles = [_shingles(str(unit.get("source_text") or "")) for unit in units]
    lengths = [len(_render_selection([unit])[0]) for unit in units]
    pages = [(unit.get("document_id"), unit.get("page_id")) if unit.get("document_id") and unit.get("page_id") else None for unit in units]
    max_text_similarity, max_number_similarity = [0.0] * len(units), [0.0] * len(units)
    page_counts: Counter = Counter()
    covered = {name: set() for name in FIELDS}
    remaining = set(range(len(units)))
    selected_indices, steps = [], []
    used, evaluations = 0, 0
    empty = {i for i, unit in enumerate(units) if not str(unit.get("source_text") or "").strip()}
    remaining -= empty
    while remaining:
        best = None
        for index in sorted(remaining, key=lambda i: ranks[i]):
            # Old renderer changes only the [Evidence Unit N] ordinal. Sum of
            # ordinal widths is independent of the final original-rank order.
            incremental_chars = lengths[index] + (2 if selected_indices else 0) + len(str(len(selected_indices) + 1)) - 1
            if used + incremental_chars > max_context_chars:
                continue
            gain, detail = marginal_gain(features[index], covered)
            same_page = page_counts[pages[index]] if pages[index] else 0
            page_penalty = same_page / (1 + same_page)
            denominator = 1 + page_penalty + max_text_similarity[index] + max_number_similarity[index]
            utility = (relevance[index] + gain) / denominator
            evaluations += 1
            if utility <= 0:
                continue
            if best is None or (utility, -ranks[index]) > (best["utility"], -best["original_rank"]):
                best = {
                    "index": index, "original_rank": ranks[index], "utility": utility,
                    "relevance": relevance[index], "marginal_information_gain": gain,
                    "novelty_by_field": detail, "redundancy_denominator": denominator,
                    "same_page_penalty": page_penalty, "max_text_similarity": max_text_similarity[index],
                    "max_numeric_similarity": max_number_similarity[index], "incremental_chars": incremental_chars,
                }
        if best is None:
            break
        index = best.pop("index")
        selected_indices.append(index)
        remaining.remove(index)
        used += best["incremental_chars"]
        steps.append(dict(best, step=len(steps) + 1, context_chars=used))
        for name in FIELDS:
            covered[name].update(features[index][name])
        if pages[index]:
            page_counts[pages[index]] += 1
        # Cached maxima are exactly the maximum over all selected units, not
        # approximations; novelty still recomputes against the full union.
        for candidate in remaining:
            max_text_similarity[candidate] = max(max_text_similarity[candidate], _jaccard(shingles[candidate], shingles[index]))
            max_number_similarity[candidate] = max(max_number_similarity[candidate], _jaccard(features[candidate]["numeric_value"], features[index]["numeric_value"]))
    selected = sorted((units[i] for i in selected_indices), key=_rank)
    context, actual_chars = _render_selection(selected)
    if actual_chars != used or actual_chars > max_context_chars:
        raise AssertionError("Evidence renderer changed: exact context budget contract violated")
    return context, selected, {
        "selector": "evidence_set_selector_v1_shadow", "formula": "(relevance + mean_metadata_novelty) / (1 + page + text + numeric_redundancy)",
        "candidate_count": len(units), "selected_count": len(selected),
        "budget": max_context_chars, "context_chars": actual_chars,
        "dynamic_utility_evaluations": evaluations, "replacement_count": 0,
        "selection_order": [ranks[i] for i in selected_indices], "steps": steps,
        "covered_feature_counts": {name: len(values) for name, values in covered.items()},
        "unselected": [{"original_rank": ranks[i], "reason": "empty_source_text" if i in empty else
                        "budget" if used + lengths[i] + (2 if selected else 0) + len(str(len(selected) + 1)) - 1 > max_context_chars
                        else "nonpositive_utility"} for i in sorted(remaining | empty, key=lambda i: ranks[i])],
        "gold_used_for_selection": False,
    }
