"""Deterministic shadow verification for evidence-grounded financial answers."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation


_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\(?\s*[-+]?\s*\$?\s*\d[\d,]*(?:\.\d+)?\s*%?\s*\)?(?![A-Za-z0-9])")
_YEAR_RE = re.compile(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", re.IGNORECASE)
_CITATION_RE = re.compile(r"\[(?:source|citation):[^\]]+\]", re.IGNORECASE)
_REFUSAL_RE = re.compile(
    r"\b(?:cannot|can't|unable to|insufficient|not enough|does not provide|do not provide|"
    r"no information|cannot be determined|cannot determine|cannot calculate|not possible to calculate)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z][a-z0-9&'-]{2,}", re.IGNORECASE)
_METRIC_TERMS = {
    "asset", "assets", "capital", "cash", "cogs", "cost", "debt", "earnings", "ebit", "ebitda",
    "equity", "expense", "expenses", "income", "inventory", "liabilities", "liability", "margin",
    "ppe", "pp&e", "ppne", "profit", "ratio", "receivables", "revenue", "sales", "tax", "taxes",
    "turnover", "wages", "working",
}
_STOP = {
    "what", "which", "does", "based", "using", "according", "from", "with", "between", "during",
    "have", "has", "were", "was", "that", "this", "company", "fiscal", "year", "calculate", "roughly",
    "answer", "evidence", "provided", "percent", "percentage", "increase", "decrease", "change",
}


@dataclass(frozen=True)
class VerificationResult:
    numeric_ok: bool
    metric_ok: bool
    period_ok: bool
    formula_ok: bool
    warnings: list[str]
    risk_score: float
    details: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _decimal(raw: str) -> tuple[Decimal, bool] | None:
    text = re.sub(r"\s+", "", str(raw or ""))
    negative = text.startswith("(") and text.endswith(")")
    percent = "%" in text
    text = text.replace("(", "").replace(")", "").replace("$", "").replace(",", "").replace("%", "")
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return (-abs(value) if negative else value), percent


def _numbers(text: str, *, exclude_years: bool = True) -> list[tuple[Decimal, bool, str]]:
    values = []
    for match in _NUMBER_RE.finditer(str(text or "")):
        parsed = _decimal(match.group(0))
        if parsed is None:
            continue
        value, percent = parsed
        if exclude_years and not percent and value == value.to_integral_value() and 1900 <= abs(int(value)) <= 2099:
            continue
        values.append((value, percent, match.group(0).strip()))
    return values


def _same_number(left: tuple[Decimal, bool, str], right: tuple[Decimal, bool, str], tolerance=Decimal("0.0001")) -> bool:
    return left[1] == right[1] and abs(left[0] - right[0]) <= tolerance


def _formula_type(question: str) -> str | None:
    value = str(question or "").casefold()
    if re.search(r"\b(?:growth rate|percentage change|percent change|changed by|grew by)\b", value):
        return "growth"
    if "margin" in value:
        return "margin"
    if re.search(r"\b(?:ratio|turnover|times)\b", value):
        return "ratio"
    if re.search(r"\b(?:difference|change between|increased by|decreased by)\b", value):
        return "change"
    return None


def _metric_tokens(text: str) -> set[str]:
    tokens = {token.casefold().replace("&", "") for token in _WORD_RE.findall(str(text or ""))}
    metric = {token for token in tokens if token in _METRIC_TERMS}
    if metric:
        return metric
    return {token for token in tokens if token not in _STOP and not token.isdigit()}


def _formula_check(question: str, answer: str, evidence: str) -> tuple[bool, dict, list[tuple[Decimal, bool, str]]]:
    kind = _formula_type(question)
    evidence_numbers = _numbers(evidence)
    candidates = []
    for line in str(answer or "").splitlines():
        if "=" not in line or not any(symbol in line for symbol in ("/", "÷", "-", "−")):
            continue
        left, right = line.rsplit("=", 1)
        operand_text = re.sub(r"(?<=\d)\s*[-−]\s*(?=\$?\s*\d)", " ", left)
        operands = _numbers(operand_text)
        results = _numbers(right)
        if len(operands) < 2 or not results:
            continue
        # Only a terminal multiplier is a formula constant. A denominator of
        # 100 remains a genuine evidence operand.
        evidence_operands = list(operands)
        if re.search(r"(?:\*|×|x)\s*100\s*$", left.strip(), re.IGNORECASE) and evidence_operands[-1][0] == 100:
            evidence_operands.pop()
        if len(evidence_operands) < 2:
            continue
        first, second = evidence_operands[0], evidence_operands[1]
        result = results[0]
        try:
            if kind == "growth" or ("/" in left and ("-" in left or "−" in left)):
                expected = (first[0] - second[0]) / abs(second[0]) * Decimal(100)
                expected_percent = True
            elif "/" in left or "÷" in left:
                expected_percent = bool(
                    result[1] or kind == "margin" or re.search(r"(?:\*|×|x)\s*100\s*$", left.strip(), re.IGNORECASE)
                )
                expected = first[0] / second[0] * (Decimal(100) if expected_percent else Decimal(1))
            else:
                expected = first[0] - second[0]
                expected_percent = result[1]
        except (InvalidOperation, ZeroDivisionError):
            continue
        operands_supported = all(any(_same_number(operand, source) for source in evidence_numbers) for operand in evidence_operands[:2])
        tolerance = Decimal("0.11") if expected_percent else max(Decimal("0.01"), abs(expected) * Decimal("0.005"))
        consistent = result[1] == expected_percent and abs(result[0] - expected) <= tolerance
        candidates.append({
            "line": line.strip(),
            "formula_type": kind or "explicit",
            "operands_supported": operands_supported,
            "consistent": consistent,
            "expected": str(expected.quantize(Decimal("0.0001"))),
            "reported": str(result[0]),
            "reported_percent": result[1],
        })
    if candidates:
        ok = all(item["consistent"] and item["operands_supported"] for item in candidates)
        allowed_results = [
            (Decimal(item["reported"]), bool(item["reported_percent"]), item["reported"])
            for item in candidates if item["consistent"] and item["operands_supported"]
        ]
        return ok, {"formula_type": kind, "explicit_formulas": candidates, "formula_required": bool(kind)}, allowed_results
    if kind:
        return False, {"formula_type": kind, "explicit_formulas": [], "formula_required": True}, []
    return True, {"formula_type": None, "explicit_formulas": [], "formula_required": False}, []


def _relevant_operand_count(question: str, evidence: str) -> int:
    metric = _metric_tokens(question)
    values = set()
    for line in str(evidence or "").splitlines():
        if metric and not (metric & _metric_tokens(line)):
            continue
        for value, percent, _ in _numbers(line):
            values.add((value, percent))
    return len(values)


def verify_answer_v1(question: str, answer: str, evidence: str) -> VerificationResult:
    warnings = []
    formula_ok, formula_details, allowed_results = _formula_check(question, answer, evidence)
    if not formula_ok:
        warnings.append("formula_inconsistent_or_not_verifiable")

    answer_without_citations = _CITATION_RE.sub("", str(answer or ""))
    answer_numbers = _numbers(answer_without_citations)
    evidence_numbers = _numbers(evidence)
    unsupported = []
    for number in answer_numbers:
        if number[0] == 100 and not number[1]:
            continue
        if any(_same_number(number, source) for source in evidence_numbers):
            continue
        if any(_same_number(number, result, tolerance=Decimal("0.11")) for result in allowed_results):
            continue
        unsupported.append(number[2])
    numeric_ok = not unsupported
    if not numeric_ok:
        warnings.append("answer_contains_unsupported_numbers")

    question_metric = _metric_tokens(question)
    answer_metric = _metric_tokens(answer_without_citations)
    metric_ok = not question_metric or bool(question_metric & answer_metric)
    if not metric_ok:
        warnings.append("question_metric_missing_from_answer")

    requested_periods = set(_YEAR_RE.findall(question))
    answer_periods = set(_YEAR_RE.findall(answer_without_citations))
    period_ok = not requested_periods or requested_periods <= answer_periods
    if not period_ok:
        warnings.append("requested_period_missing_from_answer")

    refusal = bool(_REFUSAL_RE.search(answer_without_citations))
    relevant_operands = _relevant_operand_count(question, evidence)
    invalid_refusal = refusal and formula_details["formula_required"] and relevant_operands >= 2
    if invalid_refusal:
        warnings.append("unnecessary_refusal_with_available_operands")

    risk = min(1.0, round(
        (0.30 if not numeric_ok else 0)
        + (0.20 if not metric_ok else 0)
        + (0.15 if not period_ok else 0)
        + (0.25 if not formula_ok else 0)
        + (0.30 if invalid_refusal else 0),
        4,
    ))
    return VerificationResult(
        numeric_ok=numeric_ok,
        metric_ok=metric_ok,
        period_ok=period_ok,
        formula_ok=formula_ok,
        warnings=warnings,
        risk_score=risk,
        details={
            "unsupported_numbers": unsupported,
            "question_metric_tokens": sorted(question_metric),
            "answer_metric_tokens": sorted(answer_metric),
            "requested_periods": sorted(requested_periods),
            "answer_periods": sorted(answer_periods),
            "refusal_detected": refusal,
            "relevant_operand_count": relevant_operands,
            "invalid_refusal": invalid_refusal,
            **formula_details,
        },
    )
