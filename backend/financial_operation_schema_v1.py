"""Rule-only financial operation schemas for offline shadow evaluation."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


_SOURCE_RE = re.compile(r"(?m)^Source:\s*(.+?)\s*\|\s*Page:\s*(\d+)\s*$")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\(?\s*[-+]?\s*[$€£]?\s*\d[\d,]*(?:\.\d+)?\s*%?\s*\)?(?![A-Za-z0-9])")
_YEAR_RE = re.compile(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", re.IGNORECASE)
_SHORT_FY_RE = re.compile(r"\bFY\s*['’]?(\d{2})\b", re.IGNORECASE)


def _operand(key: str, label: str, aliases: list[str], *, min_values: int = 1, period_role: str = "current") -> dict:
    return {"key": key, "label": label, "aliases": aliases, "min_values": min_values, "period_role": period_role}


_SCHEMAS = {
    "quick_ratio": {
        "metric": "quick ratio", "operation_type": "divide",
        "required_operands": [
            _operand("cash", "cash and cash equivalents", ["cash and cash equivalents", "cash equivalents", "cash"]),
            _operand("short_term_investments", "short-term investments", ["short-term investments", "short term investments", "marketable securities"]),
            _operand("receivables", "accounts receivable", ["accounts receivable", "trade receivables", "receivables, net"]),
            _operand("current_liabilities", "total current liabilities", ["total current liabilities", "current liabilities"]),
        ],
        "formula": "(cash + short-term investments + accounts receivable) / total current liabilities",
        "period_mode": "single_period",
    },
    "current_ratio": {
        "metric": "current ratio", "operation_type": "divide",
        "required_operands": [
            _operand("current_assets", "total current assets", ["total current assets", "current assets"]),
            _operand("current_liabilities", "total current liabilities", ["total current liabilities", "current liabilities"]),
        ],
        "formula": "total current assets / total current liabilities", "period_mode": "single_period",
    },
    "inventory_turnover": {
        "metric": "inventory turnover", "operation_type": "divide",
        "required_operands": [
            _operand("cost_of_sales", "cost of goods sold", ["cost of goods sold", "cost of sales", "cost of products sold"]),
            _operand("average_inventory", "average inventory", ["average inventory", "inventories", "inventory"], min_values=2, period_role="current_and_prior"),
        ],
        "formula": "cost of goods sold / average inventory", "period_mode": "current_and_prior_balance",
    },
    "interest_coverage": {
        "metric": "interest coverage", "operation_type": "divide",
        "required_operands": [
            _operand("ebit", "EBIT", ["earnings before interest and taxes", "ebit", "operating income"]),
            _operand("interest_expense", "interest expense", ["interest expense", "interest expense, net"]),
        ],
        "formula": "EBIT / interest expense", "period_mode": "single_period",
    },
    "gross_margin": {
        "metric": "gross margin", "operation_type": "divide",
        "required_operands": [
            _operand("revenue", "revenue", ["total revenues", "total revenue", "net revenues", "net revenue", "net sales", "revenue"]),
            _operand("cost_of_sales", "cost of goods sold", ["cost of goods sold", "cost of sales", "cost of products sold", "cost of products", "total cost of products"]),
        ],
        "formula": "(revenue - cost of goods sold) / revenue * 100", "period_mode": "current_and_prior",
    },
    "operating_margin": {
        "metric": "operating margin", "operation_type": "divide",
        "required_operands": [
            _operand("operating_income", "operating income", ["operating income", "income from operations", "operating profit"]),
            _operand("revenue", "revenue", ["total revenues", "total revenue", "net revenues", "net revenue", "net sales", "revenue"]),
        ],
        "formula": "operating income / revenue * 100", "period_mode": "single_period",
    },
    "revenue_growth": {
        "metric": "revenue growth", "operation_type": "percentage_change",
        "required_operands": [
            _operand("current_revenue", "current-period revenue", ["total revenues", "total revenue", "net revenues", "net revenue", "net sales", "revenue"], period_role="current"),
            _operand("prior_revenue", "prior-period revenue", ["total revenues", "total revenue", "net revenues", "net revenue", "net sales", "revenue"], period_role="prior"),
        ],
        "formula": "(current-period revenue - prior-period revenue) / abs(prior-period revenue) * 100", "period_mode": "current_and_prior",
    },
    "eps_growth": {
        "metric": "EPS growth", "operation_type": "percentage_change",
        "required_operands": [
            _operand("current_eps", "current-period EPS", ["adjusted eps", "diluted earnings per share", "earnings per share", "eps"], period_role="current"),
            _operand("prior_eps", "prior-period EPS", ["adjusted eps", "diluted earnings per share", "earnings per share", "eps"], period_role="prior"),
        ],
        "formula": "(current-period EPS - prior-period EPS) / abs(prior-period EPS) * 100", "period_mode": "current_and_prior",
    },
}


_METRIC_PATTERNS = (
    ("quick_ratio", re.compile(r"\bquick ratio\b", re.IGNORECASE)),
    ("current_ratio", re.compile(r"\bcurrent ratio\b", re.IGNORECASE)),
    ("inventory_turnover", re.compile(r"\binventory turnover\b|\bsold (?:its|the) inventory\b", re.IGNORECASE)),
    ("interest_coverage", re.compile(r"\binterest coverage(?: ratio)?\b", re.IGNORECASE)),
    ("gross_margin", re.compile(r"\bgross margin\b", re.IGNORECASE)),
    ("operating_margin", re.compile(r"\boperating margin\b", re.IGNORECASE)),
    ("revenue_growth", re.compile(r"^(?=.*\b(?:revenue|net sales)\b)(?=.*\b(?:growth|grow\w*|accelerat\w*|decelerat\w*)\b)", re.IGNORECASE | re.DOTALL)),
    ("eps_growth", re.compile(r"^(?=.*\beps\b)(?=.*\b(?:growth|grow\w*|accelerat\w*|decelerat\w*)\b)", re.IGNORECASE | re.DOTALL)),
)


def _periods(question: str) -> list[str]:
    values = [match.group(1) for match in _YEAR_RE.finditer(question)]
    values.extend(str(2000 + int(match.group(1))) for match in _SHORT_FY_RE.finditer(question))
    return list(dict.fromkeys(values))


def _entity(question: str) -> str | None:
    possessive = re.search(r"\b([A-Z][A-Za-z0-9&.]*?(?:\s+[A-Z][A-Za-z0-9&.]*){0,2})['’]s\b", question)
    if possessive:
        return possessive.group(1)
    patterns = (
        r"^(?:Does|Did|Is|Was|Were|Has|Have)\s+([A-Z][A-Za-z0-9&.]*(?:\s+[A-Z][A-Za-z0-9&.]*){0,2})\b",
        r"\bfor\s+([A-Z][A-Za-z0-9&.]*(?:\s+[A-Z][A-Za-z0-9&.]*){0,2})(?:\?|\s+in\b|\s+as\b)",
        r"\bhas\s+([A-Z][A-Za-z0-9&.]*(?:\s+[A-Z][A-Za-z0-9&.]*){0,2})\s+(?:sold|reported)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            return match.group(1).strip()
    return None


def build_financial_operation_schema_v1(question: str) -> dict[str, Any]:
    schema_key = next((key for key, pattern in _METRIC_PATTERNS if pattern.search(str(question or ""))), None)
    periods = _periods(str(question or ""))
    entity = _entity(str(question or ""))
    if schema_key is None:
        return {
            "metric": None, "operation_type": "unsupported", "required_operands": [], "formula": None,
            "period_requirement": {"explicit_periods": periods, "mode": "unspecified"},
            "entity_requirement": entity, "recognized": False,
        }
    result = deepcopy(_SCHEMAS[schema_key])
    if schema_key == "interest_coverage" and re.search(r"\badjusted EBIT\b", question, re.IGNORECASE):
        result["required_operands"][0] = _operand("adjusted_ebit", "Adjusted EBIT", ["adjusted ebit"])
        result["formula"] = "Adjusted EBIT / interest expense"
    result.update(
        period_requirement={"explicit_periods": periods, "mode": result.pop("period_mode")},
        entity_requirement=entity,
        recognized=True,
    )
    return result


def _blocks(evidence: str, page_metadata: list[dict]) -> list[dict[str, Any]]:
    metadata = {(str(item.get("filename") or ""), int(item.get("page_number") or 0)): item for item in page_metadata}
    matches = list(_SOURCE_RE.finditer(str(evidence or "")))
    blocks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(evidence)
        document, page = match.group(1).strip(), int(match.group(2))
        item = metadata.get((document, page), {})
        blocks.append({"document": document, "page": page, "company": str(item.get("company") or ""), "text": evidence[match.end() : end].strip()})
    return blocks


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _entity_match(required: str | None, company: str, text: str) -> bool:
    if not required:
        return True
    target, candidate = _compact(required), _compact(company)
    if target and candidate and (target in candidate or candidate in target):
        return True
    if target and len(target) <= 5 and candidate:
        iterator = iter(candidate)
        if all(character in iterator for character in target):
            return True
    return bool(target and target in _compact(text[:1200]))


def _line_numbers(line: str) -> list[str]:
    values = []
    for match in _NUMBER_RE.finditer(line):
        raw = match.group(0).strip()
        cleaned = re.sub(r"[$€£,%()\s]", "", raw)
        try:
            number = float(cleaned)
        except ValueError:
            continue
        if not "%" in raw and number.is_integer() and 1900 <= abs(number) <= 2100:
            continue
        values.append(raw)
    return values


def _alias_match(line: str, alias: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", line, re.IGNORECASE))


def find_required_operands_v1(schema: dict[str, Any], evidence: str, page_metadata: list[dict] | None = None) -> dict[str, Any]:
    if not schema.get("recognized"):
        return {"complete": False, "found_operands": {}, "missing_operands": [], "entity_scoped_blocks": 0}
    blocks = _blocks(evidence, page_metadata or [])
    scoped = [block for block in blocks if _entity_match(schema.get("entity_requirement"), block["company"], block["text"])]
    found: dict[str, list[dict]] = {}
    missing = []
    repeated_roles: dict[tuple[str, ...], list[dict]] = {}
    for operand in schema["required_operands"]:
        aliases = tuple(operand["aliases"])
        candidates = repeated_roles.get(aliases)
        if candidates is None:
            candidates = []
            for block in scoped:
                recent_periods: list[str] = []
                for line_number, raw_line in enumerate(block["text"].splitlines(), 1):
                    line = re.sub(r"\s+", " ", raw_line).strip()
                    line_periods = _periods(line)
                    if line_periods:
                        recent_periods = line_periods
                    if not any(_alias_match(line, alias) for alias in aliases):
                        continue
                    values = _line_numbers(line)
                    if len(values) < int(operand.get("min_values") or 1):
                        continue
                    candidates.append({
                        "document": block["document"], "page": block["page"], "company": block["company"],
                        "line_number": line_number, "text": line[:700], "values": values,
                        "periods": line_periods or recent_periods, "entity_match": True,
                    })
            repeated_roles[aliases] = candidates
        role = operand.get("period_role")
        if role in {"current", "prior"} and sum(item.get("period_role") in {"current", "prior"} for item in schema["required_operands"] if tuple(item["aliases"]) == aliases) > 1:
            multi_period = [candidate for candidate in candidates if len(candidate["values"]) >= 2 or len(candidate["periods"]) >= 2]
            candidates = multi_period or candidates
        if candidates:
            found[operand["key"]] = candidates[:5]
        else:
            missing.append(operand["key"])
    return {"complete": not missing, "found_operands": found, "missing_operands": missing, "entity_scoped_blocks": len(scoped)}
