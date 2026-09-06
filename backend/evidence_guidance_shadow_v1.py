"""Build a small, deterministic guidance block from frozen shadow artifacts."""

from __future__ import annotations

import math
from typing import Any

from evidence_compression_shadow_v1 import compress_financial_evidence_v1


_COMPETING_METRICS = {
    "wages expense as percent of sales": ("SG&A",),
    "gross margin": ("operating margin",),
    "interest coverage": ("EBITDAR",),
    "operating cash flow ratio": ("current ratio",),
}


def estimate_guidance_tokens(text: str) -> int:
    """Conservative experiment budget used without a model/tokenizer call."""
    return math.ceil(len(text) / 4)


def _frame_payload(frame_record: dict[str, Any]) -> dict[str, Any]:
    return frame_record.get("frame") or frame_record


def _schema_payload(schema_record: dict[str, Any]) -> dict[str, Any]:
    return schema_record.get("schema") or schema_record


def _target_entity(frame: dict[str, Any], facts: list[dict[str, Any]]) -> str | None:
    matched = [item.get("value") for item in frame.get("entity_candidates") or [] if item.get("question_match")]
    if matched:
        return str(matched[0])
    return next((str(fact["entity"]) for fact in facts if fact.get("entity")), None)


def _target_periods(frame: dict[str, Any]) -> list[str]:
    return [str(item["value"]) for item in frame.get("period_candidates") or [] if item.get("requested")]


def _fact_score(fact: dict[str, Any], target_metric: str, target_periods: list[str]) -> float:
    score = 0.0
    if str(fact.get("metric") or "").casefold() == target_metric.casefold():
        score += 4.0
    if fact.get("period") in target_periods:
        score += 3.0
    elif fact.get("period") and any(str(fact["period"]).startswith(period) for period in target_periods):
        score += 2.0
    if fact.get("value") is not None:
        score += 2.0
    if "operand_for_target_metric" in (fact.get("ambiguity_flags") or []):
        score += 1.0
    score -= 0.2 * len(fact.get("ambiguity_flags") or [])
    return score


def _fact_line(fact: dict[str, Any]) -> str:
    span = fact["source_span"]
    return (
        f"- entity={fact.get('entity') or 'unknown'}; period={fact.get('period') or 'unknown'}; "
        f"metric={fact.get('metric') or 'unknown'}; value={fact.get('value') if fact.get('value') is not None else 'not explicitly numeric'}; "
        f"unit={fact.get('unit') or 'unknown'}; source={span['document']} p.{span['page']} "
        f"chunk={span['chunk_id']} line={span['line_number']}; source_span=\"{span['text']}\""
    )


def build_evidence_guidance_v1(
    question: str,
    summary_record: dict[str, Any],
    frame_record: dict[str, Any],
    schema_record: dict[str, Any],
    *,
    max_tokens: int = 800,
    max_facts: int = 6,
) -> dict[str, Any]:
    """Return <=max_tokens estimated guidance without using gold/reference data."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    frame = _frame_payload(frame_record)
    schema = _schema_payload(schema_record)
    compressed = compress_financial_evidence_v1(question, summary_record)
    facts = compressed["kept_facts"]
    target_metric = str(summary_record.get("target_metric") or "unknown")
    target_periods = _target_periods(frame) or list(compressed["required_periods"])
    target_entity = _target_entity(frame, facts) or compressed.get("required_entity") or "unknown"

    competing_entities = [
        str(item["value"]) for item in frame.get("entity_candidates") or []
        if not item.get("question_match") and item.get("value")
    ][:3]
    competing_periods = [
        str(item["value"]) for item in frame.get("period_candidates") or []
        if not item.get("requested") and item.get("value")
    ][:4]
    competing_metrics = list(_COMPETING_METRICS.get(target_metric, ()))
    forbidden = [
        f"- Do not substitute metrics: {', '.join(competing_metrics)}." if competing_metrics else "- Do not substitute a different financial metric for the target metric.",
        f"- Do not use other entities as the answer entity: {', '.join(competing_entities)}." if competing_entities else "- Keep the answer entity consistent with the question.",
        (
            f"- Do not treat other periods as the target period: {', '.join(competing_periods)}. "
            "They may be used only when explicitly needed as comparators or calculation operands."
            if competing_periods else "- Keep reporting periods consistent with the question."
        ),
    ]
    formula_lines = []
    if schema.get("recognized") and schema.get("formula"):
        labels = [operand.get("label") or operand.get("key") for operand in schema.get("required_operands") or []]
        formula_lines = [
            f"Calculation: operation={schema.get('operation_type')}; formula={schema['formula']}; "
            f"required_operands={', '.join(str(label) for label in labels)}."
        ]

    prefix = [
        "Evidence Guidance (derived only from the frozen evidence; not an independent factual source):",
        "Question target:",
        f"- entity={target_entity}",
        f"- period={', '.join(target_periods) if target_periods else 'not explicitly specified'}",
        f"- metric={target_metric}",
        "Forbidden confusion:", *forbidden, *formula_lines,
        "Relevant evidence (verify against each exact source span):",
    ]
    max_chars = max_tokens * 4
    selected_facts = []
    lines = list(prefix)
    seen_spans = set()
    ranked = sorted(facts, key=lambda fact: -_fact_score(fact, target_metric, target_periods))
    for fact in ranked:
        span = fact["source_span"]
        span_key = (span["document"], span["page"], span["chunk_id"], span["line_number"], span["text"], fact.get("metric"), fact.get("period"), fact.get("value"))
        if span_key in seen_spans:
            continue
        candidate = _fact_line(fact)
        proposed = "\n".join([*lines, candidate])
        if len(proposed) > max_chars:
            continue
        lines.append(candidate)
        selected_facts.append(fact)
        seen_spans.add(span_key)
        if len(selected_facts) >= max_facts:
            break
    if not selected_facts:
        lines.append("- No fact passed entity/period/metric alignment; do not infer missing facts.")
    text = "\n".join(lines)
    if estimate_guidance_tokens(text) > max_tokens:
        raise AssertionError("Guidance exceeded its token estimate budget")
    return {
        "text": text,
        "estimated_tokens": estimate_guidance_tokens(text),
        "characters": len(text),
        "target": {"entity": target_entity, "periods": target_periods, "metric": target_metric},
        "selected_facts": selected_facts,
        "available_relevant_facts": len(facts),
        "forbidden_confusion": {
            "metrics": competing_metrics, "entities": competing_entities, "periods": competing_periods,
        },
        "formula_included": bool(formula_lines),
        "source_contract": "summary_frame_schema_from_frozen_jina_context_only",
    }

