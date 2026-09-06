"""Question-scoped compression of Financial Evidence Summary facts.

Shadow-only: this module consumes already-frozen summary facts and performs no
retrieval, model call, or production mutation.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from evidence_intent_alignment_v1 import extract_question_intent_v1


_COMPETING_METRICS = {
    "wages expense as percent of sales": {"selling general and administrative expense", "SG&A"},
    "gross margin": {"operating margin"},
    "interest coverage": {"EBITDAR", "adjusted EBITDAR"},
    "operating cash flow ratio": {"current ratio"},
}


def _compact(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _entity_matches(required: str | None, observed: str | None) -> bool:
    if not required:
        return True
    target, candidate = _compact(required), _compact(observed)
    if not candidate:
        return False
    if target in candidate or candidate in target:
        return True
    if len(target) <= 5:
        iterator = iter(candidate)
        return all(character in iterator for character in target)
    return False


def _period_matches(required: str, observed: str | None) -> bool:
    if not observed:
        return False
    if "Q" in required:
        return required == observed
    return observed == required or observed.startswith(required + "Q")


def _span_key(fact: dict[str, Any]) -> tuple[Any, ...]:
    span = fact["source_span"]
    return span["document"], span["page"], span["chunk_id"], span["line_number"], span["text"]


def compress_financial_evidence_v1(question: str, summary_record: dict[str, Any]) -> dict[str, Any]:
    """Filter facts by entity/period/metric intent and group identical spans."""
    intent = extract_question_intent_v1(question)
    target_metric = str(summary_record.get("target_metric") or "")
    entity_candidates = intent.get("entity_candidates") or []
    required_entity = entity_candidates[0]["value"] if entity_candidates else None
    required_periods = [item["value"] for item in intent.get("period_candidates") or []]
    competitors = {_compact(value) for value in _COMPETING_METRICS.get(target_metric, set())}
    candidates = list(summary_record.get("financial_evidence_summary") or summary_record.get("facts") or [])

    entity_metric_candidates = []
    dropped = []
    for fact in candidates:
        if not _entity_matches(required_entity, fact.get("entity")):
            dropped.append({"fact": fact, "reason": "other_entity"})
            continue
        if _compact(fact.get("metric")) in competitors:
            dropped.append({"fact": fact, "reason": "competing_metric"})
            continue
        # Financial Summary v1 only emits target facts or explicitly marked
        # operands.  Reject any externally supplied fact outside that contract.
        flags = set(fact.get("ambiguity_flags") or [])
        if fact.get("metric") != target_metric and "operand_for_target_metric" not in flags:
            # Direct fact labels can be normalized descendants of the target
            # (e.g. the target cash-flow activity has three activity facts).
            if target_metric not in {"cash flow activity", "inventory balance drivers", "acquisitions", "business separation", "store count", "wages expense as percent of sales", "restructuring liability"}:
                dropped.append({"fact": fact, "reason": "unrelated_metric"})
                continue
        entity_metric_candidates.append(fact)

    if required_periods:
        directly_relevant_spans = {
            _span_key(fact)
            for fact in entity_metric_candidates
            if any(_period_matches(required, fact.get("period")) for required in required_periods)
        }
        kept = []
        for fact in entity_metric_candidates:
            period_relevant = any(_period_matches(required, fact.get("period")) for required in required_periods)
            # A prior-period value on the exact same financial row is a related
            # comparator/average operand, not unrelated period noise.
            same_span_comparator = _span_key(fact) in directly_relevant_spans
            if period_relevant or same_span_comparator:
                kept.append(fact)
            else:
                dropped.append({"fact": fact, "reason": "unrelated_period"})
    else:
        kept = entity_metric_candidates

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for fact in kept:
        grouped.setdefault(_span_key(fact), []).append(fact)
    lines = [
        "Compressed Financial Evidence (derived only from the frozen Jina context):",
        "Each item includes its exact original source span; no outside facts were added.",
    ]
    for index, facts in enumerate(grouped.values(), 1):
        span = facts[0]["source_span"]
        fact_values = []
        for fact in facts:
            fact_values.append(
                f"entity={fact.get('entity') or 'unknown'}; period={fact.get('period') or 'unknown'}; "
                f"metric={fact.get('metric') or 'unknown'}; value={fact.get('value') if fact.get('value') is not None else 'not explicitly numeric'}; "
                f"unit={fact.get('unit') or 'unknown'}"
            )
        lines.extend([
            f"Evidence {index}:",
            f"  Source: {span['document']} | Page: {span['page']} | Chunk: {span['chunk_id']} | Line: {span['line_number']}",
            f"  Facts: {' | '.join(fact_values)}",
            f"  Original source span: {span['text']}",
        ])
    text = "\n".join(lines)
    return {
        "text": text,
        "kept_facts": kept,
        "kept_span_count": len(grouped),
        "dropped_facts": dropped,
        "drop_reasons": dict(Counter(item["reason"] for item in dropped)),
        "required_entity": required_entity,
        "required_periods": required_periods,
        "target_metric": target_metric,
        "source_contract": "financial_evidence_summary_v1_from_frozen_jina_context",
    }

