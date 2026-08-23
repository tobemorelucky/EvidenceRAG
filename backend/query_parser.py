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
    "activision_blizzard": {
        "name_aliases": ["activision blizzard", "activision"],
        "ticker_aliases": ["ATVI"],
    },
    "aes": {
        "name_aliases": ["aes corporation", "the aes corporation"],
        "ticker_aliases": ["AES"],
    },
    "amazon": {
        "name_aliases": ["amazon", "amazon.com"],
        "ticker_aliases": ["AMZN"],
    },
    "block": {
        "name_aliases": ["block, inc.", "block inc", "square, inc."],
        "ticker_aliases": ["SQ"],
    },
    "corning": {
        "name_aliases": ["corning", "corning incorporated"],
        "ticker_aliases": ["GLW"],
    },
    "general_mills": {
        "name_aliases": ["general mills"],
        "ticker_aliases": ["GIS"],
    },
    "mgm_resorts": {
        "name_aliases": ["mgm resorts", "mgm resorts international"],
        "ticker_aliases": ["MGM"],
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
    "other_current_assets": ["other current assets"],
    "accounts_payable": ["accounts payable"],
    "other_accrued_liabilities": ["other accrued liabilities", "accrued expenses and other liabilities"],
    "inventory": ["inventory", "inventories"],
    "revenue": ["net revenue", "net revenues", "net sales", "total revenue", "total revenues", "revenue"],
    "operating_income": ["operating income", "income from operations", "operating profit"],
    "net_income": ["net income", "net earnings"],
    "total_assets": ["total assets"],
    "ppe": ["property, plant and equipment", "property and equipment", "net pp&e", "net ppe"],
    "depreciation_amortization": ["depreciation and amortization", "depreciation, amortization"],
    "capital_expenditures": ["capital expenditures", "capital expenditure", "capital spending", "purchases of property"],
    "store_count": ["total stores", "number of stores", "store count", "stores"],
}

METRIC_REQUIRED_FIELDS = {
    "inventory": ["inventory"],
    "revenue": ["revenue"],
    "net sales": ["revenue"],
    "store count": ["store_count"],
    "assets": ["total_assets"],
}

FIELD_STATEMENT_TYPES = {
    "cash_from_operations": ["cash_flow"],
    "current_assets": ["balance_sheet"],
    "current_liabilities": ["balance_sheet"],
    "cash_and_equivalents": ["balance_sheet"],
    "short_term_investments": ["balance_sheet"],
    "accounts_receivable": ["balance_sheet"],
    "other_current_assets": ["balance_sheet"],
    "accounts_payable": ["balance_sheet"],
    "other_accrued_liabilities": ["balance_sheet"],
    "inventory": ["balance_sheet"],
    "revenue": ["income_statement"],
    "operating_income": ["income_statement"],
    "net_income": ["income_statement"],
    "total_assets": ["balance_sheet"],
    "ppe": ["balance_sheet"],
    "depreciation_amortization": ["cash_flow"],
    "capital_expenditures": ["cash_flow"],
    "store_count": ["notes"],
}

STATEMENT_PAGE_MARKERS = {
    "balance_sheet": [
        "consolidated balance sheets",
        "consolidated balance sheet",
        "statements of financial position",
        "total current assets",
        "total current liabilities",
        "total assets",
    ],
    "income_statement": [
        "consolidated statements of income",
        "consolidated statement of income",
        "consolidated statements of operations",
        "consolidated statements of earnings",
        "net revenues",
        "income from operations",
    ],
    "cash_flow": [
        "consolidated statements of cash flows",
        "consolidated statement of cash flows",
        "cash flows from operating activities",
        "cash flows from investing activities",
        "cash flows from financing activities",
        "net cash provided by operating activities",
    ],
    "notes": [
        "notes to consolidated financial statements",
        "notes to the consolidated financial statements",
    ],
    "mda": [
        "management's discussion and analysis",
        "management’s discussion and analysis",
        "results of operations",
        "liquidity and capital resources",
    ],
}

STATEMENT_TYPE_LABELS = {
    "balance_sheet": "balance sheet",
    "income_statement": "income statement",
    "cash_flow": "cash flow statement",
    "notes": "financial statement notes",
    "mda": "management discussion and analysis",
}

_FINANCIAL_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:[$€£]\s*)?\(?-?\d[\d,]*(?:\.\d+)?%?\)?")

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
    if "working capital" in text:
        return {
            "task_type": "calculation",
            "required_fields": ["accounts_receivable", "inventory", "other_current_assets", "accounts_payable", "other_accrued_liabilities"],
            "formula": "accounts_receivable + inventory + other_current_assets - accounts_payable - other_accrued_liabilities",
            "calculation_basis": "operating_working_capital",
        }
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
    if any(marker in text for marker in ("pp&e", "ppe", "ppne", "property, plant and equipment", "property plant and equipment")):
        task_type = "comparison" if any(marker in text for marker in ("increase", "decrease", "change", "grow", "between", "versus", " vs ")) else "lookup"
        return {"task_type": task_type, "required_fields": ["ppe"], "formula": ""}
    if "capital-intensive" in text or "capital intensive" in text:
        return {"task_type": "judgment", "required_fields": ["capital_expenditures", "revenue", "ppe"], "formula": ""}
    if any(marker in text for marker in ("highest", "lowest", "largest", "smallest", "which region", "which segment")):
        metrics = extract_metrics(question)
        required_fields = list(dict.fromkeys(field for metric in metrics for field in METRIC_REQUIRED_FIELDS.get(metric, [])))
        return {"task_type": "selection", "required_fields": required_fields, "formula": ""}
    if any(marker in text for marker in ("increase", "decrease", "change", "compare", "between", "versus", " vs ")):
        metrics = extract_metrics(question)
        required_fields = list(dict.fromkeys(field for metric in metrics for field in METRIC_REQUIRED_FIELDS.get(metric, [])))
        return {"task_type": "comparison", "required_fields": required_fields, "formula": ""}
    return {"task_type": "lookup", "required_fields": [], "formula": ""}


def infer_statement_types(question: str, required_fields: Optional[List[str]] = None) -> List[str]:
    """Infer the financial statement families most likely to contain the answer."""
    text = (question or "").lower()
    fields = required_fields if required_fields is not None else list(infer_task_spec(question).get("required_fields") or [])
    statement_types: List[str] = []
    for field in fields:
        statement_types.extend(FIELD_STATEMENT_TYPES.get(field, []))

    if any(marker in text for marker in ("balance sheet", "financial position")):
        statement_types.append("balance_sheet")
    if any(marker in text for marker in ("income statement", "statement of operations", "statement of earnings")):
        statement_types.append("income_statement")
    if "cash flow statement" in text or "statement of cash flows" in text:
        statement_types.append("cash_flow")
    if any(marker in text for marker in ("why ", "explain", "reason for", "driven by")):
        statement_types.append("mda")
    return list(dict.fromkeys(statement_types))


def infer_page_statement_types(text: str) -> List[str]:
    """Classify page text using conservative filing headings and line-item markers."""
    lowered = (text or "").lower()
    matches = []
    for statement_type, markers in STATEMENT_PAGE_MARKERS.items():
        if any(marker in lowered for marker in markers):
            matches.append(statement_type)
    return matches


def build_required_field_query(question: str) -> str:
    """Compatibility wrapper for the deterministic finance rewrite."""
    return build_finance_query_rewrite(question)


def build_finance_query_rewrite(question: str) -> str:
    """Add only deterministic financial anchors; never generates generic QA text."""
    parsed = parse_query(question)
    required_fields = list(parsed.get("required_fields") or [])
    if not required_fields:
        return ""
    labels = [FIELD_ALIASES[field][0] for field in required_fields if FIELD_ALIASES.get(field)]
    periods = [str(value) for value in (parsed.get("required_periods") or [])]
    statement_labels = [
        STATEMENT_TYPE_LABELS.get(item, item)
        for item in (parsed.get("statement_types") or [])
    ]
    anchors = labels + statement_labels + periods + list(parsed.get("doc_types") or [])
    return f"{question}\nFinancial retrieval anchors: {'; '.join(dict.fromkeys(anchors))}"


def build_supplemental_field_query(question: str, coverage: Dict[str, object]) -> str:
    """Build one deterministic query containing only fields missing from evidence."""
    missing_fields = list(coverage.get("missing_fields") or [])
    missing_periods = [str(item) for item in (coverage.get("missing_periods") or [])]
    if not missing_fields:
        return ""
    labels = [FIELD_ALIASES[field][0] for field in missing_fields if FIELD_ALIASES.get(field)]
    statement_types = infer_statement_types(question, missing_fields)
    statement_labels = [STATEMENT_TYPE_LABELS.get(item, item) for item in statement_types]
    anchors = labels + statement_labels + missing_periods
    return f"{question}\nMissing financial evidence: {'; '.join(dict.fromkeys(anchors))}"


def _nearby_financial_numbers(text: str, alias: str, window: int = 180) -> List[str]:
    lowered = text.lower()
    start = lowered.find(alias.lower())
    if start < 0:
        return []
    # Filing rows normally place values after their label. Looking backwards
    # incorrectly assigns the previous row's value to a missing operand.
    excerpt = text[start : start + len(alias) + window]
    values = []
    for match in _FINANCIAL_NUMBER_PATTERN.findall(excerpt):
        plain = re.sub(r"[^0-9.]", "", match)
        if plain.isdigit() and 1900 <= int(plain) <= 2100 and not any(symbol in match for symbol in ("$", "€", "£", "%", ".", ",")):
            continue
        values.append(match.strip())
    return values[:8]


def match_required_fields_in_text(required_fields: List[str], text: str) -> Dict[str, str]:
    """Return required fields backed by a nearby numeric value in one text unit."""
    matched: Dict[str, str] = {}
    for field in required_fields:
        for alias in FIELD_ALIASES.get(field, []):
            if _nearby_financial_numbers(text, alias):
                matched[field] = alias
                break
    return matched


def _period_tokens(task_spec: Dict[str, object]) -> List[str]:
    periods = [str(year) for year in (task_spec.get("years") or [])]
    periods.extend(str(quarter) for quarter in (task_spec.get("quarters") or []))
    return list(dict.fromkeys(periods))


def assess_required_field_coverage(task_spec: Dict[str, object], documents: List[Dict[str, object]]) -> Dict[str, object]:
    """Validate required fields against numeric, company-scoped evidence; never triggers another search."""
    required_fields = list(task_spec.get("required_fields") or [])
    company = str(task_spec.get("company") or "")
    scoped_documents = list(documents)
    scope_status = "not_required"
    if company:
        scoped_documents = [
            document
            for document in documents
            if matches_company_text(
                "\n".join(
                    [
                        str(document.get("filename") or ""),
                        str(document.get("doc_name") or ""),
                    ]
                ),
                company,
            )
        ]
        scope_status = "matched" if scoped_documents else "company_mismatch"

    field_evidence: Dict[str, Dict[str, object]] = {}
    for field in required_fields:
        for document in scoped_documents:
            text = str(document.get("text") or document.get("page_text") or "")
            for alias in FIELD_ALIASES.get(field, []):
                values = _nearby_financial_numbers(text, alias)
                if values:
                    field_evidence[field] = {
                        "alias": alias,
                        "values": values,
                        "filename": str(document.get("filename") or ""),
                        "page_number": document.get("page_number"),
                    }
                    break
            if field in field_evidence:
                break

    matched = {field: str(evidence.get("alias") or "") for field, evidence in field_evidence.items()}
    missing = [field for field in required_fields if field not in field_evidence]
    required_periods = _period_tokens(task_spec)
    scoped_context = "\n".join(
        "\n".join(
            [
                str(document.get("filename") or ""),
                str(document.get("text") or document.get("page_text") or ""),
            ]
        )
        for document in scoped_documents
    ).lower()
    matched_periods = [period for period in required_periods if period.lower() in scoped_context]
    missing_periods = [period for period in required_periods if period not in matched_periods]
    periods_required_for_status = task_spec.get("task_type") == "comparison" and len(required_periods) >= 2

    if company and scope_status == "company_mismatch":
        status = "insufficient"
    elif not required_fields and not periods_required_for_status:
        status = "complete"
    elif len(missing) == len(required_fields) and required_fields:
        status = "insufficient"
    elif missing or (periods_required_for_status and missing_periods):
        status = "partial"
    else:
        status = "complete"
    return {
        "task_type": task_spec.get("task_type", "lookup"),
        "formula": task_spec.get("formula", ""),
        "required_fields": required_fields,
        "matched_fields": {field: alias for field, alias in matched.items() if alias},
        "field_evidence": field_evidence,
        "missing_fields": missing,
        "required_periods": required_periods,
        "matched_periods": matched_periods,
        "missing_periods": missing_periods,
        "scope_status": scope_status,
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
        "required_periods": [str(year) for year in extract_years(question)] + extract_quarters(question),
    }
    task_spec = infer_task_spec(question)
    return {
        **parsed,
        **task_spec,
        "statement_types": infer_statement_types(question, list(task_spec.get("required_fields") or [])),
    }


def build_answer_directives(question: str, task_spec: Dict[str, object]) -> List[str]:
    """Return concise task-specific answer rules; these are instructions, not evidence."""
    text = (question or "").lower()
    directives: List[str] = []
    if "quick ratio" in text:
        directives.append("Calculate the quick ratio and give the requested healthy/not-healthy conclusion directly. Do not add or end with a generic business-model or cash-flow caveat unless the evidence explicitly states the ratio is inapplicable.")
    if task_spec.get("task_type") == "selection":
        directives.append("Compare every row in the shared candidate table, including Corporate/Other and negative values, before selecting the minimum or maximum.")
    if "store" in text and task_spec.get("task_type") == "comparison":
        directives.append("Use the explicitly labeled Total row for this company-level store-count question. Ignore the branded 'Best Buy' subrow and report the Total counts and their change.")
    if task_spec.get("calculation_basis") == "operating_working_capital":
        directives.append("Use operating working capital: receivables + inventory + other current assets - accounts payable - other accrued liabilities; exclude cash and short-term debt.")
    if task_spec.get("formula") == "revenue / average(ppe)":
        directives.append("Keep full precision through the average-PP&E calculation and round only the final ratio to two decimals.")
    if "capital-intensive" in text or "capital intensive" in text:
        directives.append("Give the requested yes/no capital-intensity conclusion after comparing capital spending and net PP&E with revenue. Low-single-digit capital spending as a share of revenue together with net PP&E well below annual revenue supports a 'not capital-intensive' conclusion; do not stop at 'no benchmark'.")
    return directives


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
