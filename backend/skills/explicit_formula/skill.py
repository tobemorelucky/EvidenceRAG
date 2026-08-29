"""High-precision skill for formulas explicitly defined by the user question."""

from __future__ import annotations

import ast
import re
import time
from decimal import Decimal
from typing import Any

from query_parser import FIELD_ALIASES, FIELD_STATEMENT_TYPES, extract_explicit_formula_contract
from skill_tools.decimal_calculator import DecimalCalculationError, calculate_decimal
from skill_tools.operand_search import (
    extract_operand_candidates,
    infer_target_filenames,
    resolve_unique_operand,
    search_missing_operands,
)
from skills.explicit_formula.schema import AtomicOperand, FormulaContract, ResolvedOperand, SkillResult


_SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "cash_from_operations": (
        "net cash provided by operating activities", "cash from operations",
        "cash flows from operating activities", "cash provided by operating activities",
        "cash provided by operations", "net cash provided by operations",
    ),
    "cogs": ("cost of goods sold", "cost of sales", "cost of revenues", "cogs"),
    "ppe": ("property, plant and equipment", "property and equipment", "net pp&e", "pp&e", "ppe"),
    "depreciation_amortization": (
        "depreciation and amortization", "depreciation, amortization", "d&a",
    ),
    "capital_expenditures": (
        "capital expenditures", "capital expenditure", "capital spending", "capex",
        "purchases of property, plant and equipment", "purchases of property",
    ),
}
_SKILL_STATEMENTS: dict[str, tuple[str, ...]] = {
    "cogs": ("income_statement",),
    "ppe": ("balance_sheet",),
    "depreciation_amortization": ("cash_flow",),
    "capital_expenditures": ("cash_flow",),
}
_OUTFLOW_MAGNITUDE_CONCEPTS = {"capital_expenditures"}
_ROUNDING_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
}


def _question_years(question: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", question, re.IGNORECASE)))


def _default_period(question: str) -> str:
    quarter = re.search(
        r"\b(?:FY\s*)?((?:19|20)\d{2})\s*Q([1-4])\b|\bQ([1-4])\s+(?:of\s+)?(?:FY\s*)?((?:19|20)\d{2})\b",
        question,
        re.IGNORECASE,
    )
    if quarter:
        year = quarter.group(1) or quarter.group(4)
        value = quarter.group(2) or quarter.group(3)
        return f"{year}Q{value}"
    years = _question_years(question)
    return years[0] if years else ""


def _canonical_concept(item: dict[str, Any]) -> str:
    field = str(item.get("field") or item.get("key") or "").strip()
    return {"pp_e": "ppe"}.get(field, field)


def _aliases(concept: str, item: dict[str, Any]) -> tuple[str, ...]:
    values = [
        str(item.get("label") or ""),
        *(str(value) for value in (item.get("aliases") or [])),
        *(FIELD_ALIASES.get(concept) or []),
        *(_SKILL_ALIASES.get(concept) or ()),
    ]
    return tuple(dict.fromkeys(value for value in values if value))


def _rounding_places(question: str) -> int | None:
    match = re.search(
        r"round(?:ed)?(?:\s+your answer)?\s+to\s+(\d+|zero|one|two|three|four|five|six)\s+decimal places?",
        question,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group(1).casefold()
    return int(value) if value.isdigit() else _ROUNDING_WORDS[value]


def _result_metadata(question: str) -> tuple[str, str, str]:
    lower = question.casefold()
    currency = "USD" if re.search(r"\bUSD\b|U\.S\. dollars?", question, re.IGNORECASE) else ""
    scale = "billions" if "billion" in lower else "millions" if "million" in lower else "thousands" if "thousand" in lower else ""
    unit = "percent" if "%" in question or "percentage" in lower or "percent " in lower else ""
    if not unit and re.search(r"\bdays?\b", lower.split("defined as", 1)[0]):
        unit = "days"
    return currency, scale, unit


def _atomic_operands(question: str, parsed: dict[str, Any]) -> list[AtomicOperand]:
    default_period = _default_period(question)
    atoms: list[AtomicOperand] = []
    for item in parsed.get("explicit_formula_operands") or []:
        concept = _canonical_concept(item)
        transform = str(item.get("transform") or "direct")
        periods = list(dict.fromkeys(str(value) for value in (item.get("periods") or []) if value))
        if not periods:
            periods = [default_period] if default_period else [""]
        if transform in {"average", "change"} and len(periods) != 2:
            return []
        for period in periods:
            key = f"{concept}_{period}" if period else concept
            atoms.append(AtomicOperand(
                key=key,
                concept=concept,
                label=str(item.get("label") or concept),
                aliases=_aliases(concept, item),
                period=period,
                statement_types=tuple(FIELD_STATEMENT_TYPES.get(concept) or _SKILL_STATEMENTS.get(concept) or ()),
                cash_outflow_magnitude=concept in _OUTFLOW_MAGNITUDE_CONCEPTS,
            ))
    # A target such as "defined EBITDA ... EBITDA less capex" contains an
    # explicit outer subtraction even though capex follows outside the definition.
    prefix = question.split(str(parsed.get("explicit_formula_text") or ""), 1)[0]
    if re.search(r"\bless\s+cap(?:ex|ital expenditures?)\b", prefix, re.IGNORECASE):
        period = default_period
        key = f"capital_expenditures_{period}" if period else "capital_expenditures"
        if all(item.key != key for item in atoms):
            atoms.append(AtomicOperand(
                key=key,
                concept="capital_expenditures",
                label="capex",
                aliases=_aliases("capital_expenditures", {"label": "capex"}),
                period=period,
                statement_types=tuple(FIELD_STATEMENT_TYPES.get("capital_expenditures") or ()),
                cash_outflow_magnitude=True,
            ))
    return atoms


def _replace_formula_operands(formula_text: str, parsed: dict[str, Any], atoms: list[AtomicOperand]) -> str:
    expression = re.sub(r"\[[^]]*]", " ", formula_text).strip()
    atom_by_concept_period = {(item.concept, item.period): item.key for item in atoms}
    operand_items = list(parsed.get("explicit_formula_operands") or [])
    for item in sorted(operand_items, key=lambda value: len(str(value.get("label") or "")), reverse=True):
        concept = _canonical_concept(item)
        aliases = _aliases(concept, item)
        alias_pattern = "|".join(re.escape(value) for value in sorted(aliases, key=len, reverse=True))
        periods = list(dict.fromkeys(str(value) for value in (item.get("periods") or []) if value))
        transform = str(item.get("transform") or "direct")
        if transform in {"average", "change"} and len(periods) == 2:
            keys = [atom_by_concept_period.get((concept, period), "") for period in periods]
            if not all(keys):
                return ""
            replacement = f"average({keys[0]}, {keys[1]})" if transform == "average" else f"({keys[1]} - {keys[0]})"
            expression, count = re.subn(
                rf"\b{transform}(?:\s+in)?\s+(?:{alias_pattern})\s+between\s+(?:FY\s*)?{periods[0]}\s+and\s+(?:FY\s*)?{periods[1]}",
                replacement,
                expression,
                count=1,
                flags=re.IGNORECASE,
            )
            if count == 0:
                return ""
        else:
            period = periods[0] if periods else next((item.period for item in atoms if item.concept == concept), "")
            key = atom_by_concept_period.get((concept, period), concept)
            expression, count = re.subn(
                rf"(?:\bFY\s*{period}\s+)?(?:{alias_pattern})",
                key,
                expression,
                count=1,
                flags=re.IGNORECASE,
            )
            if count == 0:
                return ""
    expression = re.sub(r"\bdivided\s+by\b", "/", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\bmultiplied\s+by\b|\btimes\b", "*", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\bplus\b", "+", expression, flags=re.IGNORECASE)
    expression = re.sub(r"\bless\b|\bminus\b", "-", expression, flags=re.IGNORECASE)
    expression = re.sub(
        r"\b(?:unadjusted|adjusted)\s+(?=[a-z_][a-z0-9_]*(?:\s*[+\-*/)]|$))",
        "",
        expression,
        flags=re.IGNORECASE,
    )
    expression = re.sub(r"\s+", " ", expression).strip()
    capex = next((item.key for item in atoms if item.concept == "capital_expenditures"), "")
    if capex and capex not in expression:
        expression = f"({expression}) - {capex}"
    return expression


def _expression_is_safe(expression: str, atoms: list[AtomicOperand]) -> bool:
    if not expression:
        return False
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    operand_names = {item.key for item in atoms}
    allowed_names = operand_names | {"average"}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    if not operand_names.issubset(names) or not names.issubset(allowed_names):
        return False
    allowed_nodes = (
        ast.Expression, ast.Name, ast.Load, ast.Constant, ast.UnaryOp, ast.UAdd, ast.USub,
        ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Call,
    )
    return all(isinstance(node, allowed_nodes) for node in ast.walk(tree))


def build_formula_contract(question: str) -> tuple[FormulaContract | None, str]:
    parsed = extract_explicit_formula_contract(question)
    if not parsed.get("explicit_formula_present"):
        return None, "no_explicit_formula_cue"
    atoms = _atomic_operands(question, parsed)
    if len(atoms) < 2:
        return None, "formula_operands_not_parseable"
    expression = _replace_formula_operands(str(parsed.get("explicit_formula_text") or ""), parsed, atoms)
    if not _expression_is_safe(expression, atoms):
        return None, "formula_expression_not_parseable"
    currency, scale, unit = _result_metadata(question)
    return FormulaContract(
        formula_text=str(parsed.get("explicit_formula_text") or ""),
        expression=expression,
        operands=tuple(atoms),
        final_currency=currency,
        final_scale=scale,
        final_unit=unit,
        rounding_decimal_places=_rounding_places(question),
    ), ""


def _validate_operands(contract: FormulaContract, operands: list[ResolvedOperand]) -> str:
    if len(operands) != len(contract.operands):
        return "required_operands_incomplete"
    filenames = {item.filename for item in operands}
    identities = {re.sub(r"(?:19|20)\d{2}|10[-_ ]?[kq]|[^a-z0-9]", "", name.casefold()) for name in filenames}
    if len({item for item in identities if item}) > 1:
        return "cross_company_or_document_operands"
    expected_periods = {item.key: item.period for item in contract.operands}
    if any(item.period != expected_periods.get(item.key) for item in operands):
        return "operand_period_mismatch"
    currencies = {item.currency for item in operands if item.currency}
    scales = {item.scale for item in operands if item.scale}
    if len(currencies) > 1:
        return "operand_currency_mismatch"
    if any(not item.scale for item in operands):
        return "operand_scale_unknown"
    return ""


def _base_values(operands: list[ResolvedOperand]) -> dict[str, Decimal]:
    factors = {
        "thousands": Decimal("1000"),
        "millions": Decimal("1000000"),
        "billions": Decimal("1000000000"),
    }
    return {item.key: item.normalized_value * factors[item.scale] for item in operands}


def _monetary_result(contract: FormulaContract) -> bool:
    try:
        tree = ast.parse(contract.expression, mode="eval")
    except SyntaxError:
        return False
    return not any(isinstance(node, ast.Div) for node in ast.walk(tree))


def _format_answer(contract: FormulaContract, operands: list[ResolvedOperand], calculation: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    value = str(calculation["display_result"])
    if contract.final_unit == "percent":
        percent = Decimal(str(calculation["full_precision_result"])) * Decimal("100")
        places = contract.rounding_decimal_places
        converted = calculate_decimal("value", {"value": percent}, places)
        value = str(converted["display_result"])
        calculation["full_precision_result"] = converted["full_precision_result"]
        calculation["display_result"] = value
    suffix = "%" if contract.final_unit == "percent" else f" {contract.final_unit}" if contract.final_unit else ""
    if contract.final_currency:
        suffix = f" {contract.final_currency}{(' ' + contract.final_scale) if contract.final_scale else ''}"
    citations: list[dict[str, Any]] = []
    operand_parts: list[str] = []
    for item in operands:
        key = (item.filename, item.page_number)
        if not any((citation["filename"], citation["page_number"]) == key for citation in citations):
            citations.append({
                "id": f"evidence-{len(citations) + 1}",
                "filename": item.filename,
                "page_number": item.page_number,
                "text": item.source_text,
                "score": item.confidence,
            })
        label = f"[source: {item.filename}, page {item.page_number}]"
        operand_parts.append(f"{item.concept} ({item.period}) = {item.raw_value} {label}")
    answer = (
        f"The result is {value}{suffix}. Using {contract.formula_text}, the supported operands are "
        + "; ".join(operand_parts)
        + f". The verified calculation is {contract.expression} = {value}{suffix}."
    )
    return answer, citations


class ExplicitFormulaSkill:
    name = "explicit_formula"

    def can_handle(self, question: str) -> bool:
        contract, _ = build_formula_contract(question)
        return contract is not None

    def prepare(self, question: str) -> tuple[FormulaContract | None, str]:
        return build_formula_contract(question)

    def execute(
        self,
        question: str,
        baseline_documents: list[dict[str, Any]],
        candidate_documents: list[dict[str, Any]],
    ) -> SkillResult:
        started = time.perf_counter()
        trace: dict[str, Any] = {
            "skill_detected": False,
            "skill_name": self.name,
            "trigger_reason": "",
            "formula_text": "",
            "formula_operands": [],
            "operands_found_in_baseline_evidence": [],
            "missing_operands_before_tool": [],
            "operand_search_calls": 0,
            "operand_search_queries": [],
            "operand_candidates": {},
            "operand_resolution_status": "not_started",
            "operand_resolution_failure_reason": "",
            "calculator_called": False,
            "full_precision_result": "",
            "display_result": "",
            "skill_success": False,
            "skill_applied": False,
            "fallback_to_clean_baseline": True,
            "skill_dense_bm25_calls": 0,
            "skill_jina_calls": 0,
        }
        contract, failure = self.prepare(question)
        if contract is None:
            trace["operand_resolution_failure_reason"] = failure
            trace["skill_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return SkillResult(trace=trace)
        trace.update({
            "skill_detected": True,
            "trigger_reason": contract.trigger_reason,
            "formula_text": contract.formula_text,
            "formula_operands": [item.key for item in contract.operands],
            "formula_contract": contract.trace_dict(),
        })
        target_filenames = infer_target_filenames(question, [*baseline_documents, *candidate_documents])
        resolved: dict[str, ResolvedOperand] = {}
        missing: list[AtomicOperand] = []
        for operand in contract.operands:
            candidates = extract_operand_candidates(operand, baseline_documents, question, target_filenames)
            trace["operand_candidates"][operand.key] = [item.trace_dict() for item in candidates[:5]]
            value, reason = resolve_unique_operand(candidates)
            if value:
                resolved[operand.key] = value
                trace["operands_found_in_baseline_evidence"].append(operand.key)
            else:
                missing.append(operand)
        trace["missing_operands_before_tool"] = [item.key for item in missing]
        if missing:
            search = search_missing_operands(
                question, missing, baseline_documents, candidate_documents, max_queries=4,
            )
            trace["operand_search_calls"] = len(search.calls)
            trace["operand_search_queries"] = [dict(item) for item in search.calls]
            trace["skill_dense_bm25_calls"] = search.dense_bm25_calls
            trace["skill_jina_calls"] = search.jina_calls
            search_documents = list(search.documents)
            search_targets = infer_target_filenames(question, search_documents) or target_filenames
            for operand in missing:
                candidates = extract_operand_candidates(operand, search_documents, question, search_targets)
                trace["operand_candidates"][operand.key] = [item.trace_dict() for item in candidates[:5]]
                value, reason = resolve_unique_operand(candidates)
                if value:
                    resolved[operand.key] = value
        unresolved = [item.key for item in contract.operands if item.key not in resolved]
        if unresolved:
            trace["operand_resolution_status"] = "failed"
            trace["operand_resolution_failure_reason"] = f"unresolved_operands:{','.join(unresolved)}"
            trace["skill_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return SkillResult(detected=True, trace=trace)
        ordered = [resolved[item.key] for item in contract.operands]
        failure = _validate_operands(contract, ordered)
        if failure:
            trace["operand_resolution_status"] = "failed"
            trace["operand_resolution_failure_reason"] = failure
            trace["skill_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return SkillResult(detected=True, trace=trace)
        trace["operand_resolution_status"] = "validated"
        trace["resolved_operands"] = [item.trace_dict() for item in ordered]
        trace["calculator_called"] = True
        try:
            calculation = calculate_decimal(contract.expression, _base_values(ordered), None)
            numeric_result = Decimal(str(calculation["full_precision_result"]))
            if _monetary_result(contract):
                scale = contract.final_scale or ordered[0].scale
                numeric_result /= {
                    "thousands": Decimal("1000"),
                    "millions": Decimal("1000000"),
                    "billions": Decimal("1000000000"),
                }[scale]
            display = calculate_decimal(
                "value", {"value": numeric_result}, contract.rounding_decimal_places,
            )
            calculation["full_precision_result"] = display["full_precision_result"]
            calculation["display_result"] = display["display_result"]
        except DecimalCalculationError as exc:
            trace["operand_resolution_failure_reason"] = f"calculator_error:{exc}"
            trace["skill_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return SkillResult(detected=True, trace=trace)
        answer, citations = _format_answer(contract, ordered, calculation)
        trace.update({
            "full_precision_result": calculation["full_precision_result"],
            "display_result": calculation["display_result"],
            "skill_success": True,
            "skill_applied": True,
            "fallback_to_clean_baseline": False,
            "skill_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        return SkillResult(
            detected=True, success=True, applied=True, answer=answer, citations=citations, trace=trace,
        )
