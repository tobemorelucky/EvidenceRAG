"""Deterministic financial query expansion for BM25 shadow experiments only."""

from __future__ import annotations

import re
from typing import Any


_ABBREVIATIONS = (
    (re.compile(r"\bPP\s*&\s*E\b", re.I), "property plant and equipment"),
    (re.compile(r"\bSG\s*&\s*A\b", re.I), "selling general and administrative expense"),
    (re.compile(r"\bCOGS\b", re.I), "cost of goods sold cost of sales"),
    (re.compile(r"\bEPS\b", re.I), "earnings per share"),
    (re.compile(r"\bEBITDAR\b", re.I), "earnings before interest taxes depreciation amortization and rent"),
    (re.compile(r"\bEBITDA\b", re.I), "earnings before interest taxes depreciation and amortization"),
    (re.compile(r"\bEBIT\b", re.I), "earnings before interest and taxes operating income"),
    (re.compile(r"\bROA\b", re.I), "return on assets"),
    (re.compile(r"\bROE\b", re.I), "return on equity"),
    (re.compile(r"\bFCF\b", re.I), "free cash flow"),
)

_METRICS = (
    (re.compile(r"\bgross margin\b", re.I), "gross profit gross margin percentage"),
    (re.compile(r"\boperating margin\b", re.I), "operating income operating profit margin"),
    (re.compile(r"\binventory turnover\b", re.I), "inventory turnover cost of goods sold average inventory"),
    (re.compile(r"\binterest coverage(?: ratio)?\b", re.I), "interest coverage earnings before interest and taxes interest expense"),
    (re.compile(r"\bquick ratio\b", re.I), "quick ratio acid test ratio liquid assets current liabilities"),
    (re.compile(r"\bcurrent ratio\b", re.I), "current ratio current assets current liabilities"),
    (re.compile(r"\bworking capital\b", re.I), "working capital current assets current liabilities"),
    (re.compile(r"\boperating cash flow ratio\b", re.I), "cash from operations net cash provided by operating activities current liabilities"),
    (re.compile(r"\bcapital intensity\b|\bcapital[- ]intensive\b", re.I), "capital intensity fixed assets property plant and equipment total assets asset turnover"),
    (re.compile(r"\brevenue growth\b|\bnet sales growth\b", re.I), "revenue net sales year over year growth"),
    (re.compile(r"\brestructuring liabilit(?:y|ies)\b", re.I), "restructuring liability employee liabilities restructuring reserve"),
)

_FORMULA_OPERANDS = (
    (re.compile(r"\bquick ratio\b", re.I), "cash cash equivalents short term investments accounts receivable total current liabilities"),
    (re.compile(r"\bcurrent ratio\b", re.I), "total current assets total current liabilities"),
    (re.compile(r"\binventory turnover\b", re.I), "cost of goods sold beginning inventory ending inventory average inventory"),
    (re.compile(r"\binterest coverage(?: ratio)?\b", re.I), "EBIT adjusted EBIT interest expense"),
    (re.compile(r"\bgross margin\b", re.I), "revenue net sales cost of sales gross profit"),
    (re.compile(r"\boperating margin\b", re.I), "operating income revenue net sales"),
    (re.compile(r"\boperating cash flow ratio\b", re.I), "net cash provided by operating activities total current liabilities"),
    (re.compile(r"\bworking capital\b", re.I), "total current assets total current liabilities"),
)

_FY_RE = re.compile(r"\bFY\s*['’]?(\d{2,4})\b", re.I)
_QFY_RE = re.compile(r"\bQ([1-4])\s+(?:of\s+)?FY\s*['’]?(\d{2,4})\b", re.I)


def _year(value: str) -> str:
    return str(2000 + int(value)) if len(value) == 2 else value


def explain_financial_query_expansion(query: str) -> dict[str, Any]:
    """Return an append-only BM25 expansion and a transparent trace."""
    original = re.sub(r"\s+", " ", str(query or "")).strip()
    additions: list[tuple[str, str]] = []
    for pattern, phrase in _ABBREVIATIONS:
        if pattern.search(original):
            additions.append(("accounting_abbreviation", phrase))
    for pattern, phrase in _METRICS:
        if pattern.search(original):
            additions.append(("metric_synonym", phrase))
    for pattern, phrase in _FORMULA_OPERANDS:
        if pattern.search(original):
            additions.append(("formula_operand", phrase))

    normalized_periods = []
    quarter_spans = []
    for match in _QFY_RE.finditer(original):
        quarter, raw_year = match.groups()
        year = _year(raw_year)
        normalized_periods.append(f"fiscal year {year} fiscal quarter {quarter} Q{quarter} {year}")
        quarter_spans.append(match.span())
    masked = list(original)
    for start, end in quarter_spans:
        masked[start:end] = " " * (end - start)
    for match in _FY_RE.finditer("".join(masked)):
        year = _year(match.group(1))
        normalized_periods.append(f"fiscal year {year} FY{year} {year}")
    additions.extend(("fiscal_year_normalization", phrase) for phrase in normalized_periods)

    deduplicated = []
    seen = set()
    original_folded = original.casefold()
    for source, phrase in additions:
        key = phrase.casefold()
        if key in seen or key in original_folded:
            continue
        seen.add(key)
        deduplicated.append({"source": source, "text": phrase})
    expanded = " ".join([original, *(item["text"] for item in deduplicated)]).strip()
    return {
        "original_query": original,
        "expanded_query": expanded,
        "additions": deduplicated,
        "changed": expanded != original,
    }


def expand_financial_bm25_query(query: str) -> str:
    return explain_financial_query_expansion(query)["expanded_query"]

