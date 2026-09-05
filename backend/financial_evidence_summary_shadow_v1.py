"""Rule-only financial fact summaries over an already-frozen answer context.

This is a shadow-analysis module.  It does not retrieve documents, call a model,
or generate an answer.  Every extracted fact retains the exact line and source
location from which it was derived.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from evidence_intent_alignment_v1 import (
    align_context_chunk_v1,
    build_frozen_context_chunks_v1,
    extract_question_intent_v1,
)


_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<negative>\()?\s*(?P<currency>[$€£])?\s*"
    r"(?P<number>[-+]?\d[\d,]*(?:\.\d+)?)\s*(?P<percent>%)?\s*(?(negative)\))(?![A-Za-z0-9])"
)
_YEAR_RE = re.compile(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", re.I)
_SHORT_FY_RE = re.compile(r"\bFY\s*['’]?(\d{2})\b", re.I)
_QY_RE = re.compile(r"(?<![A-Za-z0-9])Q([1-4])(?:\s+of)?[_\s-]*(?:FY[_\s-]*)?['’]?(\d{2,4})(?!\d)", re.I)
_YQ_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})[_\s-]*Q([1-4])(?!\d)", re.I)
_SCALE_RE = re.compile(r"\b(in\s+)?(thousands?|millions?|billions?)\b", re.I)


@dataclass(frozen=True)
class FinancialEvidenceSummary:
    entity: str | None
    period: str | None
    metric: str
    value: str | None
    unit: str | None
    source_span: dict[str, Any]
    ambiguity_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ambiguity_flags"] = list(self.ambiguity_flags)
        return result


def _fact(metric: str, aliases: tuple[str, ...], *, role: str = "target") -> dict[str, Any]:
    return {"metric": metric, "aliases": aliases, "role": role}


# Generic financial concepts only.  These mappings contain no company, benchmark
# ID, answer, or report-specific rule.
_FACT_SPECS: dict[str, tuple[dict[str, Any], ...]] = {
    "wages expense as percent of sales": (
        _fact("wages expense", ("wages expense", "store payroll and benefits", "payroll and benefits", "wage investments")),
    ),
    "interest coverage": (
        _fact("interest coverage", ("interest coverage",)),
        _fact("adjusted EBIT", ("adjusted ebit",), role="operand"),
        _fact("operating income", ("operating income", "income from operations"), role="operand"),
        _fact("interest expense", ("interest expense", "interest expense, net"), role="operand"),
    ),
    "gross margin": (
        _fact("gross margin", ("gross margin",)),
        _fact("gross profit", ("gross profit",), role="operand"),
        _fact("revenue", ("total revenues", "total revenue", "net revenues", "net revenue", "net sales", "revenue"), role="operand"),
        _fact("cost of sales", ("cost of goods sold", "cost of sales", "cost of products sold", "cost of products"), role="operand"),
    ),
    "operating cash flow ratio": (
        _fact("operating cash flow ratio", ("operating cash flow ratio",)),
        _fact("cash from operations", ("net cash provided by operating activities", "cash from operations"), role="operand"),
        _fact("current liabilities", ("total current liabilities", "current liabilities"), role="operand"),
    ),
    "capital intensity": (
        _fact("capital intensity", ("capital intensity", "capital-intensive")),
        _fact("return on assets", ("return on assets", "roa"), role="operand"),
        _fact("property plant and equipment", ("property, plant and equipment", "property plant and equipment", "fixed assets"), role="operand"),
        _fact("total assets", ("total assets",), role="operand"),
    ),
    "cash flow activity": (
        _fact("operating activities cash flow", ("operating activities", "cash provided by operating activities")),
        _fact("investing activities cash flow", ("investing activities", "cash used in investing activities")),
        _fact("financing activities cash flow", ("financing activities", "cash used in financing activities")),
    ),
    "inventory turnover": (
        _fact("inventory turnover", ("inventory turnover",)),
        _fact("cost of goods sold", ("cost of goods sold", "cost of sales", "cost of products sold"), role="operand"),
        _fact("inventory", ("average inventory", "merchandise inventories", "inventories", "inventory"), role="operand"),
    ),
    "EPS growth": (
        _fact("EPS growth", ("eps growth",)),
        _fact("earnings per share", ("adjusted eps", "diluted earnings per share", "earnings per share", "eps"), role="operand"),
    ),
    "restructuring liability": (
        _fact("restructuring liability", ("restructuring liability", "restructuring liabilities", "employee liabilities", "restructuring")),
    ),
    "store count": (_fact("store count", ("number of stores", "total stores", "store count", "stores")),),
    "inventory balance drivers": (
        _fact("inventory balance drivers", ("merchandise inventories", "inventory balance", "new stores", "inventory")),
    ),
    "working capital": (
        _fact("working capital", ("working capital",)),
        _fact("current assets", ("total current assets", "current assets"), role="operand"),
        _fact("current liabilities", ("total current liabilities", "current liabilities"), role="operand"),
    ),
    "acquisitions": (_fact("acquisition", ("companies acquired", "acquired", "acquisition", "business combination")),),
    "business separation": (_fact("business separation", ("spinning off", "spin-off", "spinoff", "separation")),),
    "quick ratio": (
        _fact("quick ratio", ("quick ratio",)),
        _fact("cash and cash equivalents", ("cash and cash equivalents", "cash equivalents"), role="operand"),
        _fact("short-term investments", ("short-term investments", "marketable securities"), role="operand"),
        _fact("accounts receivable", ("accounts receivable", "trade receivables", "receivables, net"), role="operand"),
        _fact("current liabilities", ("total current liabilities", "current liabilities"), role="operand"),
    ),
    "current ratio": (
        _fact("current ratio", ("current ratio",)),
        _fact("current assets", ("total current assets", "current assets"), role="operand"),
        _fact("current liabilities", ("total current liabilities", "current liabilities"), role="operand"),
    ),
    "operating margin": (
        _fact("operating margin", ("operating margin",)),
        _fact("operating income", ("operating income", "income from operations"), role="operand"),
        _fact("revenue", ("total revenues", "net revenues", "net sales", "revenue"), role="operand"),
    ),
    "revenue growth": (
        _fact("revenue growth", ("revenue growth", "net sales growth")),
        _fact("revenue", ("total revenues", "net revenues", "net sales", "revenue"), role="operand"),
    ),
}


def _phrase(line: str, alias: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", line, re.I))


def _periods(text: str) -> list[str]:
    values: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in _QY_RE.finditer(text):
        quarter, year = match.groups()
        year = str(2000 + int(year)) if len(year) == 2 else year
        values.append(f"{year}Q{quarter}")
        spans.append(match.span())
    for match in _YQ_RE.finditer(text):
        year, quarter = match.groups()
        values.append(f"{year}Q{quarter}")
        spans.append(match.span())
    masked = list(text)
    for start, end in spans:
        masked[start:end] = " " * (end - start)
    remaining = "".join(masked)
    values.extend(match.group(1) for match in _YEAR_RE.finditer(remaining))
    values.extend(str(2000 + int(match.group(1))) for match in _SHORT_FY_RE.finditer(remaining))
    return list(dict.fromkeys(values))


def _numbers(line: str) -> list[dict[str, str | None]]:
    results = []
    for match in _NUMBER_RE.finditer(line):
        raw = match.group(0).strip()
        number = float(match.group("number").replace(",", ""))
        if not match.group("percent") and not match.group("currency") and number.is_integer() and 1900 <= abs(number) <= 2100:
            continue
        if match.group("negative") and not match.group("number").startswith("-"):
            raw = f"-{match.group('number')}" + ("%" if match.group("percent") else "")
        results.append({"raw": raw, "currency": match.group("currency"), "percent": match.group("percent")})
    return results


def _unit(line: str, number: dict[str, str | None]) -> str | None:
    scale_match = _SCALE_RE.search(line)
    scale = scale_match.group(2).lower().rstrip("s") if scale_match else None
    if number.get("percent"):
        return "percent"
    currency = {"$": "USD", "€": "EUR", "£": "GBP"}.get(number.get("currency") or "")
    if currency and scale:
        return f"{currency} {scale}"
    return currency or scale


def _entity(intent: dict[str, Any], chunk: dict[str, Any]) -> str | None:
    company = str(chunk.get("company") or "")
    if company:
        return company
    requested = intent.get("entity_candidates") or []
    return str(requested[0]["value"]) if requested else None


def _source_span(chunk: dict[str, Any], line_number: int, line: str) -> dict[str, Any]:
    return {
        "document": chunk["document"], "page": chunk["page"], "chunk_id": chunk["chunk_id"],
        "line_number": line_number, "text": line,
    }


def _chunk_entity_matches(intent: dict[str, Any], chunk: dict[str, Any]) -> bool:
    if align_context_chunk_v1(intent, chunk)["entity_match"]:
        return True
    requested = intent.get("entity_candidates") or []
    if not requested:
        return True
    target = re.sub(r"[^a-z0-9]+", "", str(requested[0]["value"]).casefold())
    document = re.sub(r"[^a-z0-9]+", "", str(chunk.get("document") or "").casefold())
    return bool(target and target in document)


def extract_financial_evidence_summary_v1(
    question: str,
    evidence: str,
    page_metadata: list[dict] | None = None,
    *,
    max_facts: int = 40,
) -> dict[str, Any]:
    """Extract question-relevant facts from the exact frozen context."""
    intent = extract_question_intent_v1(question)
    target = (intent.get("metric_candidates") or [{}])[0].get("value")
    specs = _FACT_SPECS.get(str(target), ())
    chunks = build_frozen_context_chunks_v1(evidence, page_metadata or [])
    facts: list[FinancialEvidenceSummary] = []
    seen: set[tuple[Any, ...]] = set()
    facts_per_metric: dict[str, int] = {}

    for chunk in chunks:
        # Contexts can contain multiple companies.  Entity extraction is allowed
        # to fall back when the question has no explicit entity, but a known
        # question entity must never be assigned to another company's source.
        if not _chunk_entity_matches(intent, chunk):
            continue
        recent_periods: list[str] = []
        recent_period_line = -100
        document_periods = _periods(chunk.get("document", ""))
        if chunk.get("report_year"):
            document_periods.append(str(chunk["report_year"]))
        for line_number, raw_line in enumerate(str(chunk.get("text") or "").splitlines(), 1):
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            line_periods = _periods(line)
            if line_periods:
                recent_periods = line_periods
                recent_period_line = line_number
            matched_specs = [spec for spec in specs if any(_phrase(line, alias) for alias in spec["aliases"])]
            if not matched_specs:
                continue
            values = _numbers(line)
            local_header_periods = recent_periods if line_number - recent_period_line <= 3 else []
            periods = line_periods or local_header_periods or list(dict.fromkeys(document_periods))
            if not periods:
                periods = [None]
            flags_base: list[str] = []
            if not line_periods and periods:
                flags_base.append("period_inferred_from_local_or_document_context")
            if len(matched_specs) > 1:
                flags_base.append("multiple_metrics_on_source_span")
            for spec in matched_specs:
                if facts_per_metric.get(spec["metric"], 0) >= 8:
                    continue
                role_flags = list(flags_base)
                if spec["role"] == "operand":
                    role_flags.append("operand_for_target_metric")
                pairs: list[tuple[str | None, dict[str, str | None] | None]] = []
                if values:
                    if len(periods) == len(values):
                        pairs = list(zip(periods, values))
                    elif len(periods) > 1 and len(values) > 1:
                        pairs = list(zip(periods, values[: len(periods)]))
                        role_flags.append("period_value_count_mismatch")
                    else:
                        pairs = [(periods[0], value) for value in values]
                        if len(values) > 1:
                            role_flags.append("multiple_values_same_period")
                else:
                    pairs = [(periods[0], None)]
                    role_flags.append("qualitative_value")
                for period, number in pairs:
                    unit = _unit(line, number) if number else None
                    flags = list(role_flags)
                    if number and unit is None:
                        flags.append("unit_missing")
                    value = number["raw"] if number else None
                    key = (chunk["document"], chunk["page"], line_number, spec["metric"], period, value)
                    if key in seen:
                        continue
                    seen.add(key)
                    facts.append(FinancialEvidenceSummary(
                        entity=_entity(intent, chunk), period=period, metric=spec["metric"], value=value,
                        unit=unit, source_span=_source_span(chunk, line_number, line),
                        ambiguity_flags=tuple(dict.fromkeys(flags)),
                    ))
                    facts_per_metric[spec["metric"]] = facts_per_metric.get(spec["metric"], 0) + 1
                    if len(facts) >= max_facts:
                        break
                if len(facts) >= max_facts:
                    break
            if len(facts) >= max_facts:
                break
        if len(facts) >= max_facts:
            break

    target_metrics = {spec["metric"] for spec in specs if spec["role"] == "target"}
    operand_metrics = {spec["metric"] for spec in specs if spec["role"] == "operand"}
    found_metrics = {fact.metric for fact in facts}
    extraction_status = "direct" if found_metrics & target_metrics else "operand_supported" if found_metrics & operand_metrics else "absent"
    return {
        "target_metric": target,
        "target_metric_source": "question_intent",
        "extraction_status": extraction_status,
        "facts": [fact.to_dict() for fact in facts],
        "context_chunk_count": len(chunks),
        "source_contract": "exact_frozen_answer_context_only",
    }


def detect_metric_substitution_v1(target_metric: str | None, answer: str) -> list[str]:
    """Flag a small set of generic, easily-confused financial concepts in an answer."""
    competitors = {
        "wages expense as percent of sales": ("selling, general and administrative", "sg&a"),
        "gross margin": ("operating margin",),
        "interest coverage": ("ebitdar",),
        "operating cash flow ratio": ("current ratio",),
    }
    return [term for term in competitors.get(str(target_metric), ()) if _phrase(answer, term)]
