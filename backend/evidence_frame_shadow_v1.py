"""Rule-only EvidenceFrame extraction for frozen answer contexts.

This shadow module has no dependency on retrieval, reranking, production
EvidenceFrame execution, or model calls.  It parses only text that was actually
present in the frozen answer context.  Page metadata is provenance, not an
additional text source.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


_SOURCE_RE = re.compile(r"(?m)^Source:\s*(.+?)\s*\|\s*Page:\s*(\d+)\s*$")
_YEAR_RE = re.compile(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", re.IGNORECASE)
_SHORT_FY_RE = re.compile(r"\bFY\s*['’]?(\d{2})\b", re.IGNORECASE)
_QUARTER_RE = re.compile(r"\bQ([1-4])(?:['’]?\s*(\d{2,4}))?\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\(?\s*[-+]?\s*[$€£]?\s*\d[\d,]*(?:\.\d+)?\s*%?\s*\)?(?![A-Za-z0-9])")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9&'-]*")
_REFUSAL_RE = re.compile(
    r"\b(?:cannot|can't|unable to|insufficient|not enough|does not provide|"
    r"do not provide|cannot determine|cannot calculate|not meaningful)\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "based", "be", "between", "by", "calculate",
    "company", "did", "do", "does", "for", "from", "have", "has", "how", "if", "in",
    "is", "it", "its", "many", "of", "on", "or", "please", "roughly", "state", "than",
    "that", "the", "then", "this", "to", "using", "was", "were", "what", "when", "which",
    "with", "year",
}
_METRIC_HEADS = {
    "asset", "assets", "capital", "cash", "companies", "company", "ebit", "ebitda", "eps",
    "expense", "expenses", "flow", "growth", "income", "inventory", "liabilities", "liability",
    "margin", "profit", "ratio", "sales", "segments", "stores", "turnover",
}


def _normalized_words(value: str) -> list[str]:
    return [word.casefold().strip("'-") for word in _WORD_RE.findall(str(value or ""))]


def _content_words(value: str) -> set[str]:
    return {word for word in _normalized_words(value) if word not in _STOPWORDS and len(word) > 1}


def _periods(value: str) -> list[str]:
    periods = [match.group(1) for match in _YEAR_RE.finditer(str(value or ""))]
    periods.extend(str(2000 + int(match.group(1))) for match in _SHORT_FY_RE.finditer(str(value or "")))
    for match in _QUARTER_RE.finditer(str(value or "")):
        quarter, year = match.groups()
        if year:
            year = str(2000 + int(year)) if len(year) == 2 else year
            periods.append(f"{year}Q{quarter}")
        else:
            periods.append(f"Q{quarter}")
    return list(dict.fromkeys(periods))


def _numbers(value: str) -> list[dict[str, Any]]:
    result = []
    for match in _NUMBER_RE.finditer(str(value or "")):
        raw = re.sub(r"\s+", "", match.group(0))
        percent = "%" in raw
        negative = raw.startswith("(") and raw.endswith(")")
        cleaned = raw.replace("$", "").replace("€", "").replace("£", "").replace(",", "").replace("%", "").strip("()")
        if negative and not cleaned.startswith("-"):
            cleaned = f"-{cleaned}"
        try:
            number = float(cleaned)
        except ValueError:
            continue
        if not percent and number.is_integer() and 1900 <= abs(number) <= 2100:
            continue
        result.append({"raw": match.group(0).strip(), "normalized": f"{number:.8f}".rstrip("0").rstrip("."), "percent": percent})
    return result


def _question_type(question: str) -> str:
    lowered = str(question or "").casefold()
    if re.search(r"\b(?:calculate|defined as|numerator|denominator|ratio|turnover|how many times)\b", lowered):
        return "calculation"
    if re.search(r"\b(?:among|which brought|which .* most|which .* least|three main)\b", lowered):
        return "selection"
    if re.search(r"\b(?:compare|between|increase|decrease|change|improv|accelerat|decelerat|most|least)\b", lowered):
        return "comparison"
    if re.search(r"^(?:is|does|did|was|were)\b|\b(?:useful|relevant|capital-intensive)\b", lowered):
        return "judgment"
    return "lookup"


def _metric_candidates(question: str) -> list[dict[str, Any]]:
    words = _normalized_words(question)
    candidates: list[str] = []
    for index, word in enumerate(words):
        if word not in _METRIC_HEADS:
            continue
        start = max(0, index - 3)
        phrase_words = [item for item in words[start : index + 1] if item not in _STOPWORDS and not re.fullmatch(r"fy?\d{2,4}", item)]
        while phrase_words and (phrase_words[0].endswith("'s") or phrase_words[0] in {"improving", "annual", "total"}):
            phrase_words.pop(0)
        if phrase_words:
            candidates.append(" ".join(phrase_words))
    # Preserve the most specific phrase while retaining shorter alternatives.
    unique = list(dict.fromkeys(candidates))
    unique.sort(key=lambda value: (-len(value.split()), value))
    return [{"value": value, "source": "question_lexical", "confidence": 0.8} for value in unique[:8]]


def _explicit_formula(question: str) -> list[dict[str, Any]]:
    source = str(question or "")
    formulas = []
    numerator = re.search(
        r"using\s+(.{2,100}?)\s+as\s+the\s+numerator\s+and\s+(.{2,100}?)\s+as\s+the\s+denominator",
        source,
        re.IGNORECASE,
    )
    if numerator:
        left, right = (re.sub(r"\s+", " ", value).strip(" .?:") for value in numerator.groups())
        formulas.append({"operation": "divide", "expression": f"{left} / {right}", "operands": [left, right], "source": "question_numerator_denominator", "confidence": 1.0})
    defined = re.search(r"(?:defined|calculated|computed)\s+as\s*:?\s*([^?.]+)", source, re.IGNORECASE)
    if defined and "/" in defined.group(1):
        expression = re.sub(r"\s+", " ", defined.group(1)).strip(" .")
        left, right = (part.strip(" ()") for part in expression.split("/", 1))
        if left and right:
            formulas.append({"operation": "divide", "expression": expression, "operands": [left, right], "source": "question_explicit_formula", "confidence": 1.0})
    if not formulas and _question_type(source) == "comparison":
        formulas.append({"operation": "compare", "expression": "compare like-for-like evidence values", "operands": [], "source": "question_comparison_cue", "confidence": 0.65})
    return formulas


def _context_blocks(evidence: str) -> list[dict[str, Any]]:
    matches = list(_SOURCE_RE.finditer(str(evidence or "")))
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(evidence)
        blocks.append({"document": match.group(1).strip(), "page": int(match.group(2)), "text": evidence[match.end() : end].strip()})
    return blocks


def _entity_candidates(question: str, blocks: list[dict], page_metadata: list[dict]) -> list[dict[str, Any]]:
    metadata = {(str(item.get("filename") or ""), int(item.get("page_number") or 0)): item for item in (page_metadata or [])}
    counts: Counter[str] = Counter()
    for block in blocks:
        item = metadata.get((block["document"], block["page"]), {})
        entity = str(item.get("company") or "").strip()
        if entity:
            counts[entity] += 1
    query_compact = re.sub(r"[^a-z0-9]+", "", str(question or "").casefold())
    result = []
    for entity, count in counts.most_common():
        entity_compact = re.sub(r"[^a-z0-9]+", "", entity.casefold())
        matched = bool(entity_compact and (entity_compact in query_compact or query_compact.find(entity_compact[:4]) >= 0))
        result.append({"value": entity, "source": "page_metadata", "supporting_pages": count, "question_match": matched, "confidence": 0.95 if matched else 0.55})
    return result


def _span_candidates(question: str, blocks: list[dict], metrics: list[dict], requested_periods: set[str]) -> list[dict[str, Any]]:
    query_words = _content_words(question)
    metric_word_sets = [_content_words(item["value"]) for item in metrics]
    spans = []
    for block in blocks:
        recent_periods: list[str] = []
        for line_number, raw_line in enumerate(block["text"].splitlines(), 1):
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            line_periods = _periods(line)
            if line_periods:
                recent_periods = line_periods
            if len(line) < 12:
                continue
            words = _content_words(line)
            overlap = len(words & query_words) / max(1, len(query_words))
            metric_overlap = max(
                (len(words & metric_words) / len(metric_words) for metric_words in metric_word_sets if metric_words),
                default=0.0,
            )
            values = _numbers(line)
            periods = line_periods or recent_periods
            period_match = bool(requested_periods & set(periods))
            score = overlap + 0.8 * metric_overlap + 0.25 * period_match + 0.15 * bool(values)
            if score < 0.14:
                continue
            spans.append({
                "document": block["document"], "page": block["page"], "line_number": line_number,
                "text": line[:900], "score": round(score, 4), "periods": periods, "values": values,
                "query_token_overlap": round(overlap, 4), "metric_token_overlap": round(metric_overlap, 4),
            })
    spans.sort(key=lambda item: (-item["score"], item["document"], item["page"], item["line_number"]))
    return spans[:30]


def _operand_candidates(formulas: list[dict], spans: list[dict], question_type: str) -> tuple[list[dict], list[str], str]:
    required = list(dict.fromkeys(operand for formula in formulas for operand in formula.get("operands", [])))
    candidates = []
    matched_required = set()
    for span in spans:
        if not span["values"]:
            continue
        span_words = _content_words(span["text"])
        matched = []
        for operand in required:
            operand_words = _content_words(operand)
            if operand_words and len(operand_words & span_words) / len(operand_words) >= 0.75:
                matched.append(operand)
                matched_required.add(operand)
        candidates.append({
            "label": re.split(r"(?=\(?\s*[-+]?\s*[$€£]?\s*\d)", span["text"], maxsplit=1)[0].strip(" :|-"),
            "values": span["values"], "periods": span["periods"], "document": span["document"],
            "page": span["page"], "source_span": span["text"], "matched_required_operands": matched,
            "score": span["score"],
        })
    if not required:
        status = "formula_not_specified" if question_type == "calculation" else "not_applicable"
    elif len(matched_required) == len(required):
        status = "complete"
    elif matched_required:
        status = "partial"
    else:
        status = "missing"
    return candidates[:20], required, status


def build_evidence_frame_shadow_v1(question: str, evidence: str, page_metadata: list[dict] | None = None) -> dict[str, Any]:
    blocks = _context_blocks(evidence)
    question_type = _question_type(question)
    metrics = _metric_candidates(question)
    formulas = _explicit_formula(question)
    question_periods = set(_periods(question))
    spans = _span_candidates(question, blocks, metrics, question_periods)
    operands, required_operands, operand_status = _operand_candidates(formulas, spans, question_type)
    evidence_periods = set(period for span in spans for period in span["periods"])
    metric_found = any(item["metric_token_overlap"] >= 0.5 for item in spans) if metrics else False
    period_found = bool(question_periods) and question_periods <= evidence_periods
    return {
        "question_type": question_type,
        "entity_candidates": _entity_candidates(question, blocks, page_metadata or []),
        "period_candidates": [
            {"value": period, "source": "question" if period in question_periods else "evidence_span", "requested": period in question_periods}
            for period in sorted(question_periods | evidence_periods)
        ],
        "metric_candidates": metrics,
        "operand_candidates": operands,
        "formula_candidates": formulas,
        "evidence_spans": spans,
        "diagnostics": {
            "context_blocks": len(blocks),
            "key_metric_found": metric_found,
            "requested_periods": sorted(question_periods),
            "requested_period_found": period_found,
            "required_operand_names": required_operands,
            "required_operands_status": operand_status,
            "required_operands_found": operand_status == "complete" if required_operands else None,
        },
    }


def explain_answer_failure(frame: dict[str, Any], answer: str) -> dict[str, Any]:
    diagnostics = frame["diagnostics"]
    refusal = bool(_REFUSAL_RE.search(str(answer or "")))
    signals = []
    if not diagnostics["key_metric_found"]:
        signals.append("key_metric_not_located_in_context")
    if diagnostics["requested_periods"] and not diagnostics["requested_period_found"]:
        signals.append("requested_period_not_located_in_relevant_spans")
    if diagnostics["required_operands_status"] in {"missing", "partial"}:
        signals.append("required_operands_incomplete")
    if frame["question_type"] == "calculation" and diagnostics["required_operands_status"] == "formula_not_specified":
        signals.append("formula_not_explicit_in_question")
    if refusal and diagnostics["required_operands_found"]:
        signals.append("unnecessary_refusal_candidate")
    if not signals and diagnostics["key_metric_found"] and (
        not diagnostics["requested_periods"] or diagnostics["requested_period_found"]
    ):
        signals.append("evidence_utilization_or_reasoning_failure")

    explained_types = set()
    if any(signal in signals for signal in ("key_metric_not_located_in_context", "requested_period_not_located_in_relevant_spans", "required_operands_incomplete")):
        explained_types.add("evidence_not_sufficient")
    if "unnecessary_refusal_candidate" in signals:
        explained_types.add("refusal_failure")
    if frame["question_type"] == "calculation" and diagnostics["required_operands_found"] and not refusal:
        explained_types.add("calculation_failure")
    if "evidence_utilization_or_reasoning_failure" in signals:
        explained_types.update({"reasoning_failure", "terminology_failure"})
    return {"refusal_detected": refusal, "signals": signals, "explained_failure_types": sorted(explained_types)}
