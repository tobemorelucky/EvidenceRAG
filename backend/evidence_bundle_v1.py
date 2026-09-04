"""Offline, gold-free bundle construction and atomic packing. Not production."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from backend.evidence_packing_v1 import (
    _key, _rank, _render_selection, _score, _terms,
    _weighted_gain, coverage_features, near_duplicate_similarity,
)


def unit_id(unit: dict) -> str:
    """Stable qualified identity; do not rely on file name or rank."""
    identity = [unit.get("document_id"), *_key(unit)]
    return "unit_" + hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()[:24]


def _periods(unit: dict) -> set[str]:
    value = unit.get("period") or []
    return {str(item).strip() for item in (value if isinstance(value, list) else [value]) if str(item).strip()}


def compatible(units: list[dict], *, require_period: bool = False) -> bool:
    entities = {str(unit.get("entity") or "").strip().casefold() for unit in units} - {""}
    periods = [_periods(unit) for unit in units]
    known = [value for value in periods if value]
    if len(entities) > 1:
        return False
    if require_period and (not known or len(known) != len(units)):
        return False
    return not known or bool(set.intersection(*known))


def _page_number(unit: dict) -> int | None:
    value = (unit.get("metadata") or {}).get("page_number")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _make_bundle(question: str, members: list[dict], kind: str, similarity: float | None = None) -> dict:
    members = sorted(members, key=lambda unit: (_rank(unit), unit_id(unit)))
    ids = [unit_id(unit) for unit in members]
    lengths = [len(_render_selection([unit])[0]) for unit in members]
    score = sum(_score(unit) * length for unit, length in zip(members, lengths)) / max(1, sum(lengths))
    features = set().union(*(coverage_features(question, unit) for unit in members))
    pages = sorted({(unit["document_id"], unit["page_id"], _page_number(unit)) for unit in members}, key=repr)
    return {
        "bundle_id": "bundle_" + hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest()[:24],
        "bundle_type": kind,
        "unit_ids": ids,
        "pages": [{"document_id": doc, "page_id": page, "page_number": number} for doc, page, number in pages],
        "source_types": sorted({unit["source_type"] for unit in members}),
        "estimated_coverage_features": sorted(features),
        "length": len(_render_selection(members)[0]),
        "score": score,
        "rank": min(_rank(unit) for unit in members),
        "common_periods": sorted(set.intersection(*[_periods(unit) for unit in members])) if members else [],
        "unknown_period_unit_count": sum(not _periods(unit) for unit in members),
        "unknown_entity_unit_count": sum(not str(unit.get("entity") or "").strip() for unit in members),
        "adjacent_lexical_similarity": similarity,
    }


def build_evidence_bundles(question: str, units: list[dict]) -> list[dict]:
    """Disjoint partition: complete-link same-page groups, then adjacent pairs.

    Unknown metadata is not a verified match. Same-page unknowns may join a
    non-conflicting group, but cannot bridge conflicting known periods/entities.
    Adjacent pairs require every member's period and a common intersection.
    """
    ids = [unit_id(unit) for unit in units]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate candidate identity")
    by_page: dict[tuple, list[dict]] = defaultdict(list)
    isolated = []
    for unit in sorted(units, key=lambda item: (_rank(item), unit_id(item))):
        if not unit.get("document_id") or not unit.get("page_id"):
            isolated.append([unit])
        else:
            by_page[(unit["document_id"], unit["page_id"])].append(unit)
    groups = list(isolated)
    for page_units in by_page.values():
        page_groups: list[list[dict]] = []
        for unit in page_units:
            target = next((group for group in page_groups if compatible([*group, unit])), None)
            if target is None:
                page_groups.append([unit])
            else:
                target.append(unit)
        groups.extend(page_groups)

    signatures = [set().union(*(_terms(unit.get("source_text")) for unit in group)) for group in groups]
    edges = []
    for left, a in enumerate(groups):
        for right in range(left + 1, len(groups)):
            b = groups[right]
            if not a[0].get("document_id") or a[0].get("document_id") != b[0].get("document_id"):
                continue
            if a[0].get("page_id") == b[0].get("page_id"):
                continue
            pa, pb = _page_number(a[0]), _page_number(b[0])
            if pa is None or pb is None or not 1 <= abs(pa - pb) <= 2:
                continue
            if not compatible([*a, *b], require_period=True):
                continue
            similarity = len(signatures[left] & signatures[right]) / max(1, len(signatures[left] | signatures[right]))
            if similarity >= 0.20:
                edges.append((-similarity, min(_rank(unit) for unit in [*a, *b]), left, right))
    used = set()
    bundles = []
    for negative_similarity, _, left, right in sorted(edges):
        if left in used or right in used:
            continue
        used.update((left, right))
        bundles.append(_make_bundle(question, [*groups[left], *groups[right]], "adjacent_page", -negative_similarity))
    for index, group in enumerate(groups):
        if index not in used:
            bundles.append(_make_bundle(question, group, "same_page" if len(group) > 1 else "singleton"))
    return sorted(bundles, key=lambda bundle: (bundle["rank"], bundle["bundle_id"]))


def pack_evidence_bundles(units: list[dict], bundles: list[dict], *, budget: int = 28000) -> tuple[str, list[dict], dict]:
    """Packing v1 utility/penalties, applied to indivisible disjoint bundles.

    No guard threshold, anchor, or replacement budget. Exact final rendering is
    checked; scoring length is the same standalone-render convention as v1.
    """
    lookup = {unit_id(unit): unit for unit in units}
    member_ids = [uid for bundle in bundles for uid in bundle["unit_ids"]]
    if len(member_ids) != len(set(member_ids)) or set(member_ids) != set(lookup):
        raise ValueError("Bundles must partition the unchanged candidate units")
    features = {b["bundle_id"]: set(b["estimated_coverage_features"]) for b in bundles}
    page_sets = {b["bundle_id"]: {(p["document_id"], p["page_id"]) for p in b["pages"]} for b in bundles}
    texts = {b["bundle_id"]: "\n\n".join(lookup[uid]["source_text"] for uid in b["unit_ids"]) for b in bundles}
    similarities = {}

    def ordered(selection):
        return sorted(selection, key=lambda b: (b["rank"], b["bundle_id"]))

    def members(selection):
        return sorted([lookup[uid] for b in selection for uid in b["unit_ids"]], key=_rank)

    def utility(bundle, selection, covered):
        bid = bundle["bundle_id"]
        repeat = sum(bool(page_sets[bid] & page_sets[old["bundle_id"]]) for old in selection)
        duplicate = 0.0
        for old in selection:
            oid = old["bundle_id"]
            pair = tuple(sorted((bid, oid)))
            if pair not in similarities:
                similarities[pair] = near_duplicate_similarity(texts[bid], texts[oid])
            duplicate = max(duplicate, similarities[pair])
        gain = _weighted_gain(features[bid], covered)
        value = bundle["score"] * gain / max(1, bundle["length"]) / (1 + 0.2 * repeat)
        return value * (0.1 if duplicate >= 0.92 else 1.0)

    def objective(selection):
        covered, previous, contributions = set(), [], {}
        for bundle in ordered(selection):
            bid = bundle["bundle_id"]
            contributions[bid] = utility(bundle, previous, covered)
            covered.update(features[bid])
            previous.append(bundle)
        return sum(contributions.values()), contributions

    selected, trace, events = [], {}, []
    covered = set()
    initial = {b["bundle_id"]: utility(b, [], set()) for b in bundles}
    for bundle in sorted(bundles, key=lambda b: (-initial[b["bundle_id"]], b["rank"], b["bundle_id"])):
        bid = bundle["bundle_id"]
        value = utility(bundle, selected, covered)
        entry = {"bundle_id": bid, "initial_utility": initial[bid], "marginal_utility": value, "reason": "zero_marginal_gain"}
        trace[bid] = entry
        if value <= 0:
            continue
        proposed = [*selected, bundle]
        if len(_render_selection(members(proposed))[0]) <= budget:
            selected = proposed
            covered.update(features[bid])
            entry["reason"] = "selected_direct"
            continue
        entry["reason"] = "oversize_atomic_bundle" if bundle["length"] > budget else "budget_no_beneficial_replacement"
        old_total, contributions = objective(selected)
        if not selected:
            continue
        lowest = min(selected, key=lambda b: (contributions[b["bundle_id"]], -b["rank"]))
        if value <= contributions[lowest["bundle_id"]] + 1e-12:
            continue
        proposed = [old for old in selected if old is not lowest] + [bundle]
        new_total, _ = objective(proposed)
        if len(_render_selection(members(proposed))[0]) <= budget and new_total > old_total + 1e-12:
            events.append({
                "removed_bundle_id": lowest["bundle_id"], "added_bundle_id": bid,
                "before_unit_ids": [unit_id(unit) for unit in members(selected)],
                "after_unit_ids": [unit_id(unit) for unit in members(proposed)],
                "old_total_utility": old_total, "new_total_utility": new_total,
            })
            trace[lowest["bundle_id"]]["reason"] = "replaced_by_higher_utility"
            entry["reason"] = "selected_by_replacement"
            selected = proposed
            covered = set().union(*(features[b["bundle_id"]] for b in selected))
    final_units = members(selected)
    context, length = _render_selection(final_units)
    selected_ids = {b["bundle_id"] for b in selected}
    return context, final_units, {
        "context_chars": length, "budget": budget,
        "replacement_count": len(events), "replacement_events": events,
        "selected_bundle_ids": sorted(selected_ids),
        "trace": [dict(entry, selected=bid in selected_ids) for bid, entry in trace.items()],
    }
