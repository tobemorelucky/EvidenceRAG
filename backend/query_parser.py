import re
from dataclasses import dataclass
from typing import Dict, List, Optional


COMPANY_SPECS = {
    "adobe": {
        "name_aliases": ["adobe", "adbe"],
        "ticker_aliases": ["ADBE"],
    },
    "amd": {
        "name_aliases": ["amd", "advanced micro devices"],
        "ticker_aliases": ["AMD"],
    },
    "boeing": {
        "name_aliases": ["boeing"],
        "ticker_aliases": ["BA"],
    },
    "best_buy": {
        "name_aliases": ["best buy"],
        "ticker_aliases": ["BBY"],
    },
    "johnson_johnson": {
        "name_aliases": ["johnson & johnson", "johnson and johnson"],
        "ticker_aliases": ["JNJ"],
    },
    "jpmorgan": {
        "name_aliases": ["jpmorgan", "jpmorgan chase", "jp morgan"],
        "ticker_aliases": ["JPM"],
    },
    "amcor": {
        "name_aliases": ["amcor"],
        "ticker_aliases": ["AMCR"],
    },
    "3m": {
        "name_aliases": ["3m"],
        "ticker_aliases": ["MMM"],
    },
    "pfizer": {
        "name_aliases": ["pfizer"],
        "ticker_aliases": ["PFE"],
    },
    "verizon": {
        "name_aliases": ["verizon"],
        "ticker_aliases": ["VZ"],
    },
    "pepsico": {
        "name_aliases": ["pepsico"],
        "ticker_aliases": ["PEP"],
    },
    "cvs_health": {
        "name_aliases": ["cvs health"],
        "ticker_aliases": ["CVS"],
    },
    "ulta_beauty": {
        "name_aliases": ["ulta beauty"],
        "ticker_aliases": ["ULTA"],
    },
    "american_express": {
        "name_aliases": ["american express", "amex"],
        "ticker_aliases": ["AXP"],
    },
}

METRIC_ALIASES = {
    "gross margin": ["gross margin"],
    "operating margin": ["operating margin"],
    "quick ratio": ["quick ratio"],
    "current ratio": ["current ratio"],
    "ebitda": ["ebitda"],
    "adjusted ebitda": ["adjusted ebitda"],
    "capex": ["capex", "capital expenditures", "capital expenditure"],
    "eps": ["eps", "earnings per share"],
    "adjusted eps": ["adjusted eps"],
    "effective tax rate": ["effective tax rate"],
    "inventory": ["inventory"],
    "revenue": ["revenue"],
    "net sales": ["net sales"],
    "free cash flow": ["free cash flow"],
    "store count": ["store count", "stores"],
    "shareholders' equity": ["shareholders' equity", "stockholders' equity"],
    "assets": ["assets"],
    "liabilities": ["liabilities"],
}

FIELD_ALIASES = {
    "cash_from_operations": ["net cash provided by operating activities", "cash from operations", "cash flows from operating activities"],
    "current_assets": ["total current assets", "current assets"],
    "current_liabilities": ["total current liabilities", "current liabilities"],
    "cash_and_equivalents": ["cash and cash equivalents", "cash equivalents"],
    "short_term_investments": ["short-term investments", "short term investments", "marketable securities"],
    "accounts_receivable": ["accounts receivable", "trade receivables", "receivables, net"],
    "inventory": ["inventory", "inventories"],
    "revenue": ["net revenue", "net revenues", "net sales", "total revenue", "total revenues", "revenue"],
    "operating_income": ["operating income", "income from operations", "operating profit"],
    "net_income": ["net income", "net earnings"],
    "total_assets": ["total assets"],
    "ppe": ["property, plant and equipment", "property and equipment", "net pp&e", "net ppe"],
    "depreciation_amortization": ["depreciation and amortization", "depreciation, amortization"],
    "capital_expenditures": ["capital expenditures", "capital expenditure", "capital spending", "purchases of property"],
}

DOC_TYPE_ALIASES = {
    "10-K": ["10-k", "10 k"],
    "10-Q": ["10-q", "10 q"],
    "8-K": ["8-k", "8 k"],
    "earnings": ["earnings", "earnings release"],
}


@dataclass
class CompanyMatch:
    company: str
    matched_alias: str
    method: str
    confidence: float


def _contains_phrase(text_lower: str, phrase: str) -> bool:
    return phrase.lower() in text_lower


def _contains_ticker_token(text: str, ticker: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(ticker)}(?![A-Za-z0-9])", text))


def _normalize_company_text(text: str) -> tuple[str, str]:
    lowered = (text or "").lower()
    spaced = re.sub(r"[_\-]+", " ", lowered)
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    return spaced, compact


def detect_company(question: str) -> Optional[CompanyMatch]:
    text = question or ""
    text_lower = text.lower()
    matches: List[CompanyMatch] = []

    for company, spec in COMPANY_SPECS.items():
        for alias in spec["name_aliases"]:
            if _contains_phrase(text_lower, alias):
                confidence = 0.98 if " " in alias or "&" in alias else 0.9
                matches.append(CompanyMatch(company, alias, "name_alias", confidence))
        for ticker in spec["ticker_aliases"]:
            if _contains_ticker_token(text, ticker):
                confidence = 0.96 if len(ticker) <= 4 else 0.9
                matches.append(CompanyMatch(company, ticker, "ticker_alias", confidence))

    if not matches:
        return None

    matches.sort(key=lambda item: (item.confidence, len(item.matched_alias)), reverse=True)
    return matches[0]


def extract_years(question: str) -> List[int]:
    text_lower = (question or "").lower()
    years = {int(item) for item in re.findall(r"\b(20\d{2})\b", text_lower)}
    for match in re.findall(r"\bfy\s?(\d{2,4})\b", text_lower):
        if len(match) == 2:
            years.add(2000 + int(match))
        else:
            years.add(int(match))
    return sorted(years)


def extract_quarters(question: str) -> List[str]:
    text_lower = (question or "").lower()
    quarters = []
    for match in re.finditer(r"\b(20\d{2})\s*q([1-4])\b", text_lower):
        quarters.append(f"{match.group(1)}Q{match.group(2)}")
    for match in re.finditer(r"\bq([1-4])\b", text_lower):
        value = f"Q{match.group(1)}"
        if value not in quarters:
            quarters.append(value)
    return quarters


def extract_doc_types(question: str) -> List[str]:
    text_lower = (question or "").lower()
    doc_types = []
    for canonical, aliases in DOC_TYPE_ALIASES.items():
        if any(alias in text_lower for alias in aliases):
            doc_types.append(canonical)
    return doc_types


def extract_metrics(question: str) -> List[str]:
    text_lower = (question or "").lower()
    metrics = []
    for canonical, aliases in METRIC_ALIASES.items():
        if any(alias in text_lower for alias in aliases):
            metrics.append(canonical)
    return metrics


def infer_task_spec(question: str) -> Dict[str, object]:
    """Infer deterministic FinanceBench-style calculation requirements without an LLM call."""
    text = (question or "").lower()
    if "quick ratio" in text:
        return {"task_type": "calculation", "required_fields": ["cash_and_equivalents", "short_term_investments", "accounts_receivable", "current_liabilities"], "formula": "(cash_and_equivalents + short_term_investments + accounts_receivable) / current_liabilities"}
    if "operating cash flow ratio" in text:
        return {"task_type": "calculation", "required_fields": ["cash_from_operations", "current_liabilities"], "formula": "cash_from_operations / current_liabilities"}
    if "working capital ratio" in text:
        return {"task_type": "calculation", "required_fields": ["current_assets", "current_liabilities"], "formula": "current_assets / current_liabilities"}
    if "fixed asset turnover" in text:
        return {"task_type": "calculation", "required_fields": ["revenue", "ppe"], "formula": "revenue / average(ppe)"}
    if "return on assets" in text or re.search(r"\broa\b", text):
        return {"task_type": "calculation", "required_fields": ["net_income", "total_assets"], "formula": "net_income / average(total_assets)"}
    if "operating margin" in text:
        return {"task_type": "calculation", "required_fields": ["operating_income", "revenue"], "formula": "operating_income / revenue"}
    if "depreciation" in text and "margin" in text:
        return {"task_type": "calculation", "required_fields": ["depreciation_amortization", "revenue"], "formula": "depreciation_amortization / revenue"}
    if "ebitda" in text and ("capex" in text or "capital expenditure" in text or "capital spending" in text):
        return {"task_type": "calculation", "required_fields": ["operating_income", "depreciation_amortization", "capital_expenditures"], "formula": "operating_income + depreciation_amortization - capital_expenditures"}
    if any(marker in text for marker in ("highest", "lowest", "largest", "smallest", "which region", "which segment")):
        return {"task_type": "selection", "required_fields": [], "formula": ""}
    if any(marker in text for marker in ("increase", "decrease", "change", "compare", "between", "versus", " vs ")):
        return {"task_type": "comparison", "required_fields": [], "formula": ""}
    return {"task_type": "lookup", "required_fields": [], "formula": ""}


def build_required_field_query(question: str) -> str:
    """Add required statement labels to a calculation query without an LLM rewrite."""
    task_spec = infer_task_spec(question)
    required_fields = list(task_spec.get("required_fields") or [])
    if not required_fields:
        return ""
    labels = [FIELD_ALIASES[field][0] for field in required_fields if FIELD_ALIASES.get(field)]
    return f"{question}\nRequired financial statement fields: {'; '.join(labels)}"


def assess_required_field_coverage(task_spec: Dict[str, object], documents: List[Dict[str, object]]) -> Dict[str, object]:
    """Report whether retrieved evidence contains each required field; never triggers another search."""
    required_fields = list(task_spec.get("required_fields") or [])
    context = "\n".join(str(document.get("text") or document.get("page_text") or "") for document in documents)
    lowered = context.lower()
    matched = {
        field: next((alias for alias in FIELD_ALIASES.get(field, []) if alias in lowered), "")
        for field in required_fields
    }
    missing = [field for field, alias in matched.items() if not alias]
    if not required_fields or not missing:
        status = "complete"
    elif len(missing) == len(required_fields):
        status = "insufficient"
    else:
        status = "partial"
    return {
        "task_type": task_spec.get("task_type", "lookup"),
        "formula": task_spec.get("formula", ""),
        "required_fields": required_fields,
        "matched_fields": {field: alias for field, alias in matched.items() if alias},
        "missing_fields": missing,
        "status": status,
        "supplemental_search_attempted": False,
    }


def parse_query(question: str) -> Dict[str, object]:
    company_match = detect_company(question)
    parsed = {
        "company": company_match.company if company_match else "",
        "matched_company_alias": company_match.matched_alias if company_match else "",
        "company_match_method": company_match.method if company_match else "",
        "company_confidence": company_match.confidence if company_match else 0.0,
        "years": extract_years(question),
        "quarters": extract_quarters(question),
        "doc_types": extract_doc_types(question),
        "metrics": extract_metrics(question),
    }
    return {**parsed, **infer_task_spec(question)}


def company_aliases_for(company: str) -> List[str]:
    spec = COMPANY_SPECS.get(company or "", {})
    aliases = []
    aliases.extend(spec.get("name_aliases", []))
    aliases.extend(spec.get("ticker_aliases", []))
    return aliases


def matches_company_text(text: str, company: str) -> bool:
    spec = COMPANY_SPECS.get(company or "")
    if not spec:
        return False
    original = text or ""
    lowered, compact = _normalize_company_text(original)
    for alias in spec["name_aliases"]:
        alias_lower = alias.lower()
        alias_compact = re.sub(r"[^a-z0-9]+", "", alias_lower)
        if _contains_phrase(lowered, alias_lower) or (alias_compact and alias_compact in compact):
            return True
    for ticker in spec["ticker_aliases"]:
        if _contains_ticker_token(original, ticker):
            return True
    return False
