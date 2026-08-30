"""Deterministic execution for a deliberately small set of canonical metrics.

The skill is intentionally conservative: metric definitions come from local
configuration, operands come from cited statement rows, and every calculation
uses the existing restricted Decimal calculator.
"""

from __future__ import annotations

import json
import re
import time
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from query_parser import extract_explicit_formula_contract
from skill_tools.decimal_calculator import DecimalCalculationError, calculate_decimal
from skill_tools.operand_search import (
    extract_operand_candidates,
    infer_target_filenames,
    resolve_unique_operand,
    search_missing_operands,
)
from skills.canonical_finance_metric.schema import MetricContract
from skills.explicit_formula.schema import AtomicOperand, ResolvedOperand, SkillResult


_CAUSE_RE = re.compile(r"\b(?:why|what drove|drivers?|reasons?|due to|primarily driven)\b", re.I)
_SUBJECTIVE_RE = re.compile(
    r"\b(?:is|was|are|were)\b.{0,45}\b(?:useful|relevant)\b|"
    r"\b(?:capital[- ]intensive|high[- ]growth|historically consistent)\b",
    re.I,
)
_TREND_RE = re.compile(r"\b(?:improv\w*|declin\w*|increas\w*|decreas\w*|trend|compar\w*|change\w*)\b", re.I)
_INTERPRET_RE = re.compile(r"\b(?:healthy|health|strong|weak|reasonable|adequate|sufficient)\b", re.I)
_ROUNDING_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_SCALE = {"thousands": Decimal("1000"), "millions": Decimal("1000000"), "billions": Decimal("1000000000")}


@lru_cache(maxsize=1)
def _definitions() -> dict[str, dict[str, Any]]:
    path = Path(__file__).with_name("metrics.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _primary_clause(question: str) -> str:
    # Finance questions often append "if the metric is not meaningful, explain
    # why" as a fallback. Routing is based only on the actual request before it.
    return re.split(r"(?:[;,.?]\s*)?\bif\b", question, maxsplit=1, flags=re.I)[0]


def _metric_match(question: str) -> tuple[str, str]:
    lower = question.casefold()
    matches: list[tuple[int, str, str]] = []
    for metric_name, definition in _definitions().items():
        for alias in definition["aliases"]:
            if re.search(rf"(?<![a-z0-9]){re.escape(alias.casefold())}(?![a-z0-9])", lower):
                matches.append((len(alias), metric_name, alias))
    if not matches:
        return "", ""
    _, metric_name, alias = max(matches)
    return metric_name, alias


def detect_metric_alias(question: str) -> tuple[str, str]:
    """Return the registry metric/alias without deciding whether execution is safe."""
    return _metric_match(question)


def _question_periods(question: str) -> list[str]:
    periods: list[str] = []
    for match in re.finditer(
        r"\b(?:FY\s*)?((?:19|20)\d{2})\s*Q([1-4])\b|\bQ([1-4])\s+(?:of\s+)?(?:FY\s*)?((?:19|20)\d{2})\b",
        question,
        re.I,
    ):
        period = f"{match.group(1) or match.group(4)}Q{match.group(2) or match.group(3)}"
        if period not in periods:
            periods.append(period)
    without_quarters = re.sub(r"\b(?:FY\s*)?(?:19|20)\d{2}\s*Q[1-4]\b|\bQ[1-4]\s+(?:of\s+)?(?:FY\s*)?(?:19|20)\d{2}\b", " ", question, flags=re.I)
    for year in re.findall(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", without_quarters, re.I):
        if year not in periods:
            periods.append(year)
    for short_year in re.findall(r"\bFY\s*'?([0-9]{2})\b", without_quarters, re.I):
        year = str(2000 + int(short_year))
        if year not in periods:
            periods.append(year)
    return periods


def _previous_period(period: str) -> str:
    quarter = re.fullmatch(r"((?:19|20)\d{2})Q([1-4])", period)
    if quarter:
        year, value = int(quarter.group(1)), int(quarter.group(2))
        return f"{year - 1}Q4" if value == 1 else f"{year}Q{value - 1}"
    return str(int(period) - 1) if re.fullmatch(r"(?:19|20)\d{2}", period) else ""


def _rounding_places(question: str, default: int) -> int:
    match = re.search(
        r"round(?:ed)?(?:\s+your answer)?\s+to\s+(\d+|zero|one|two|three|four|five|six)\s+decimal places?",
        question,
        re.I,
    )
    if not match:
        return default
    value = match.group(1).casefold()
    return int(value) if value.isdigit() else _ROUNDING_WORDS[value]


def build_metric_contract(question: str) -> tuple[MetricContract | None, str]:
    if extract_explicit_formula_contract(question).get("explicit_formula_present"):
        return None, "explicit_formula_has_priority"
    metric_name, alias = _metric_match(question)
    if not metric_name:
        return None, "no_canonical_metric_alias"
    primary = _primary_clause(question)
    if _CAUSE_RE.search(primary):
        return None, "causal_question_excluded"
    if _SUBJECTIVE_RE.search(primary):
        return None, "subjective_question_excluded"
    periods = _question_periods(question)
    if not periods:
        return None, "required_period_missing"
    trend_requested = bool(_TREND_RE.search(primary))
    if trend_requested and "between" in primary.casefold():
        periods.sort()
    inferred: list[str] = []
    if trend_requested and len(periods) == 1:
        previous = _previous_period(periods[0])
        if not previous:
            return None, "trend_comparison_period_not_inferable"
        inferred.append(previous)
        periods.insert(0, previous)
    if trend_requested and len(periods) != 2:
        return None, "trend_requires_exactly_two_periods"
    if not trend_requested and len(periods) != 1:
        return None, "point_metric_requires_one_period"
    definition = _definitions()[metric_name]
    operands = {
        "quick_ratio": ("cash_and_equivalents", "current_liabilities"),
        "inventory_turnover": ("cost_of_goods_sold", "beginning_inventory", "ending_inventory"),
        "gross_margin": ("gross_profit", "revenue", "cost_of_goods_sold"),
        "operating_margin": ("operating_income", "revenue"),
    }[metric_name]
    optional = (
        "short_term_investments", "accounts_receivable", "current_related_party_receivables",
    ) if metric_name == "quick_ratio" else ()
    interpretation = metric_name == "quick_ratio" and bool(_INTERPRET_RE.search(primary))
    statements = ("balance_sheet",) if metric_name == "quick_ratio" else (
        "balance_sheet", "income_statement",
    ) if metric_name == "inventory_turnover" else ("income_statement",)
    return MetricContract(
        metric_name=metric_name,
        metric_alias=alias,
        requested_periods=tuple(periods),
        inferred_periods=tuple(inferred),
        statement_types=statements,
        formula_variant=str(definition["formula"]),
        required_operands=operands,
        optional_operands=optional,
        output_unit=str(definition["output_unit"]),
        rounding_decimal_places=_rounding_places(question, int(definition["default_decimal_places"])),
        trend_requested=trend_requested,
        interpretation_required=interpretation,
    ), ""


def _atomic(key: str, concept: str, period: str, aliases: Iterable[str], statement: str) -> AtomicOperand:
    return AtomicOperand(
        key=key,
        concept=concept,
        label=concept.replace("_", " "),
        aliases=tuple(aliases),
        period=period,
        statement_types=(statement,),
        cash_outflow_magnitude=concept == "cost_of_goods_sold",
    )


def _operand_atoms(contract: MetricContract) -> list[AtomicOperand]:
    fields = _definitions()[contract.metric_name]["fields"]
    atoms: list[AtomicOperand] = []
    for period in contract.requested_periods:
        if contract.metric_name == "quick_ratio":
            concepts = (*contract.required_operands, *contract.optional_operands)
            atoms.extend(_atomic(f"{concept}_{period}", concept, period, fields[concept], "balance_sheet") for concept in concepts)
        elif contract.metric_name == "inventory_turnover":
            previous = _previous_period(period)
            atoms.extend((
                _atomic(f"cost_of_goods_sold_{period}", "cost_of_goods_sold", period, fields["cost_of_goods_sold"], "income_statement"),
                _atomic(f"beginning_inventory_{period}", "inventory", previous, fields["inventory"], "balance_sheet"),
                _atomic(f"ending_inventory_{period}", "inventory", period, fields["inventory"], "balance_sheet"),
            ))
        else:
            concepts = contract.required_operands
            atoms.extend(
                _atomic(f"{concept}_{period}", concept, period, fields[concept], "income_statement")
                for concept in concepts
            )
    return list({item.key: item for item in atoms}.values())


def _complete_current_assets_section(document: dict[str, Any], period: str) -> bool:
    text = str(document.get("text") or document.get("page_text") or "")
    head = text[:3500].casefold()
    if not re.search(r"consolidated (?:balance sheets?|statements? of financial position)", head):
        return False
    if period not in text or "current assets" not in text.casefold() or "total current assets" not in text.casefold():
        return False
    return "total current liabilities" in text.casefold() or "current liabilities" in text.casefold()


def _optional_absence_validated(
    atom: AtomicOperand,
    documents: list[dict[str, Any]],
    target_filenames: list[str],
) -> bool:
    candidates = [item for item in documents if not target_filenames or item.get("filename") in target_filenames]
    complete = [item for item in candidates if _complete_current_assets_section(item, atom.period)]
    if not complete:
        return False
    aliases = tuple(alias.casefold() for alias in atom.aliases)
    return all(not any(alias in str(item.get("text") or item.get("page_text") or "").casefold() for alias in aliases) for item in complete)


def _canonical_target_filenames(question: str, documents: list[dict[str, Any]]) -> list[str]:
    targets = infer_target_filenames(question, documents)
    if targets:
        return targets
    normalized_question = re.sub(r"[^a-z0-9]+", "", question.casefold())
    result: list[str] = []
    generic_tokens = {"annual", "earnings", "financial", "form", "quarterly", "report", "results", "pdf", "10k", "10q", "20f"}
    for document in documents:
        filename = str(document.get("filename") or "")
        tokens = [
            token for token in re.findall(r"[a-z0-9]+", Path(filename).stem.casefold())
            if token not in generic_tokens and not re.fullmatch(r"20\d{2}(?:q[1-4])?|q[1-4]", token)
        ]
        variants = set(tokens)
        if len(tokens) >= 2:
            initials = "".join(token[0] for token in tokens)
            variants.update({initials, initials[0] + "n" + initials[1:]})
        if len(tokens) == 2:
            variants.add("".join(token[:2] for token in tokens))
        for token in variants:
            if len(token) == 2 and any(character.isdigit() for character in token) and token in normalized_question:
                if filename and filename not in result:
                    result.append(filename)
            elif len(token) >= 3 and token in normalized_question:
                if filename and filename not in result:
                    result.append(filename)
    return result


def _scope_filtered_candidates(
    candidates: list[ResolvedOperand], documents: list[dict[str, Any]],
) -> list[ResolvedOperand]:
    by_page = {
        (str(item.get("filename") or ""), item.get("page_number")):
            str(item.get("text") or item.get("page_text") or "")
        for item in documents
    }
    excluded = re.compile(
        r"\b(?:deed of cross guarantee|guarantor financial statements?|"
        r"financial information of (?:the )?parent|parent company only)\b",
        re.I,
    )
    return [
        item for item in candidates
        if not excluded.search(by_page.get((item.filename, item.page_number), "")[:2500])
    ]


def _base_value(item: ResolvedOperand) -> Decimal:
    if item.scale not in _SCALE:
        raise ValueError("operand_scale_unknown")
    return item.normalized_value * _SCALE[item.scale]


def _validate_resolved(atoms: list[AtomicOperand], values: list[ResolvedOperand]) -> str:
    if len(atoms) != len(values):
        return "required_operands_incomplete"
    filenames = {item.filename for item in values}
    if len(filenames) != 1:
        return "cross_document_operands"
    expected = {item.key: item.period for item in atoms}
    if any(item.period != expected.get(item.key) for item in values):
        return "operand_period_mismatch"
    currencies = {item.currency for item in values if item.currency}
    if len(currencies) > 1:
        return "operand_currency_mismatch"
    if any(not item.currency for item in values):
        return "operand_currency_unknown"
    if any(item.scale not in _SCALE for item in values):
        return "operand_scale_unknown"
    if len({(item.filename, item.page_number, item.source_text, item.period) for item in values}) != len(values):
        return "operand_row_reused"
    return ""


def _citations(values: list[ResolvedOperand]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in values:
        key = (item.filename, item.page_number)
        if any((entry["filename"], entry["page_number"]) == key for entry in result):
            continue
        result.append({
            "id": f"evidence-{len(result) + 1}",
            "filename": item.filename,
            "page_number": item.page_number,
            "text": item.source_text,
            "score": item.confidence,
        })
    return result


def _calculate_period(
    contract: MetricContract,
    period: str,
    resolved: dict[str, ResolvedOperand],
) -> tuple[dict[str, Any], list[ResolvedOperand], str]:
    metric = contract.metric_name
    used: list[ResolvedOperand]
    if metric == "quick_ratio":
        keys = [f"cash_and_equivalents_{period}", *(f"{name}_{period}" for name in contract.optional_operands)]
        numerator_values = [resolved[key] for key in keys if key in resolved]
        denominator = resolved[f"current_liabilities_{period}"]
        used = [*numerator_values, denominator]
        expression = "quick_assets / current_liabilities"
        values = {
            "quick_assets": sum((_base_value(item) for item in numerator_values), Decimal("0")),
            "current_liabilities": _base_value(denominator),
        }
        variant = "validated_quick_assets/current_liabilities"
    elif metric == "inventory_turnover":
        used = [resolved[f"cost_of_goods_sold_{period}"], resolved[f"beginning_inventory_{period}"], resolved[f"ending_inventory_{period}"]]
        expression = "cost_of_goods_sold / average(beginning_inventory, ending_inventory)"
        values = {
            "cost_of_goods_sold": _base_value(used[0]),
            "beginning_inventory": _base_value(used[1]),
            "ending_inventory": _base_value(used[2]),
        }
        variant = expression
    elif metric == "gross_margin":
        revenue = resolved[f"revenue_{period}"]
        gross = resolved.get(f"gross_profit_{period}")
        if gross:
            used = [gross, revenue]
            expression = "gross_profit / revenue"
            values = {"gross_profit": _base_value(gross), "revenue": _base_value(revenue)}
        else:
            cogs = resolved[f"cost_of_goods_sold_{period}"]
            used = [revenue, cogs]
            expression = "(revenue - cost_of_goods_sold) / revenue"
            values = {"revenue": _base_value(revenue), "cost_of_goods_sold": _base_value(cogs)}
        variant = expression
    else:
        used = [resolved[f"operating_income_{period}"], resolved[f"revenue_{period}"]]
        expression = "operating_income / revenue"
        values = {"operating_income": _base_value(used[0]), "revenue": _base_value(used[1])}
        variant = expression
    result = calculate_decimal(expression, values, None)
    numeric = Decimal(str(result["full_precision_result"]))
    if contract.output_unit == "percent":
        numeric *= Decimal("100")
    display = calculate_decimal("value", {"value": numeric}, contract.rounding_decimal_places)
    return {
        "period": period,
        "formula_variant": variant,
        "full_precision_result": str(display["full_precision_result"]),
        "display_result": str(display["display_result"]),
        "unit": contract.output_unit,
    }, used, variant


def _answer(contract: MetricContract, results: list[dict[str, Any]], values: list[ResolvedOperand]) -> str:
    suffix = {"percent": "%", "times": " times", "ratio": ""}.get(contract.output_unit, "")
    parts = [f"{item['period']}: {item['display_result']}{suffix}" for item in results]
    conclusion = "; ".join(parts)
    if len(results) == 2:
        left = Decimal(results[0]["full_precision_result"])
        right = Decimal(results[1]["full_precision_result"])
        direction = "increased" if right > left else "decreased" if right < left else "was unchanged"
        conclusion += f". The {contract.metric_alias} {direction} from {results[0]['period']} to {results[1]['period']}"
    sources = "; ".join(
        f"{item.concept} ({item.period}) = {item.raw_value} [source: {item.filename}, page {item.page_number}]"
        for item in values
    )
    formulas = "; ".join(dict.fromkeys(str(item["formula_variant"]) for item in results))
    return f"The verified {contract.metric_alias} is {conclusion}. Formula: {formulas}. Operands: {sources}."


class CanonicalFinanceMetricSkill:
    name = "canonical_finance_metric"

    def can_handle(self, question: str) -> bool:
        contract, _ = build_metric_contract(question)
        return contract is not None

    def prepare(self, question: str) -> tuple[MetricContract | None, str]:
        return build_metric_contract(question)

    def execute(
        self,
        question: str,
        baseline_documents: list[dict[str, Any]],
        candidate_documents: list[dict[str, Any]],
    ) -> SkillResult:
        started = time.perf_counter()
        trace: dict[str, Any] = {
            "skill_detected": False, "skill_name": self.name, "canonical_metric_name": "", "metric_name": "",
            "metric_alias_matched": "", "matched_metric_alias": "", "metric_definition_source": "metrics.json",
            "metric_definition_version": "v1", "requested_periods": [], "target_periods": [], "inferred_periods": [],
            "comparison_period_inferred": "",
            "formula_variant": "", "required_operands": [], "optional_operands": [],
            "operands_found_in_baseline_evidence": [], "missing_operands_before_tool": [], "missing_operands_before_search": [],
            "operand_search_calls": 0, "operand_search_queries": [], "operand_candidates": {},
            "operand_resolution_status": "not_started", "operand_resolution_failure_reason": "",
            "statement_scope_validation": "not_started", "period_validation": "not_started",
            "currency_scale_validation": "not_started", "calculator_called": False,
            "metric_full_precision_result": "", "metric_display_result": "", "metric_results_by_period": {},
            "trend_comparison_result": "", "trend_direction": "", "authoritative_numeric": False,
            "authoritative_answer": False,
            "fallback_to_clean_baseline": True, "fallback_reason": "", "skill_success": False,
            "skill_applied": False, "skill_dense_bm25_calls": 0, "skill_jina_calls": 0,
            "skill_llm_calls": 0,
        }
        contract, failure = self.prepare(question)
        if contract is None:
            trace["operand_resolution_failure_reason"] = failure
            trace["fallback_reason"] = failure
            trace["skill_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return SkillResult(trace=trace)
        trace.update({
            "skill_detected": True, "canonical_metric_name": contract.metric_name, "metric_name": contract.metric_name,
            "metric_alias_matched": contract.metric_alias, "matched_metric_alias": contract.metric_alias,
            "requested_periods": list(contract.requested_periods), "target_periods": list(contract.requested_periods),
            "inferred_periods": list(contract.inferred_periods), "formula_variant": contract.formula_variant,
            "comparison_period_inferred": "previous_fiscal_year" if contract.inferred_periods else "",
            "required_operands": list(contract.required_operands), "optional_operands": list(contract.optional_operands),
            "metric_contract": contract.trace_dict(), "fallback_reason": "operand_resolution_incomplete",
        })
        atoms = _operand_atoms(contract)
        all_documents = [*baseline_documents, *candidate_documents]
        target_filenames = _canonical_target_filenames(question, all_documents)
        if not target_filenames:
            trace["operand_resolution_failure_reason"] = "target_document_not_identified"
            trace["skill_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return SkillResult(detected=True, trace=trace)
        resolved: dict[str, ResolvedOperand] = {}
        missing: list[AtomicOperand] = []
        for atom in atoms:
            candidates = extract_operand_candidates(atom, baseline_documents, question, target_filenames)
            candidates = _scope_filtered_candidates(candidates, baseline_documents)
            trace["operand_candidates"][atom.key] = [item.trace_dict() for item in candidates[:5]]
            value, _ = resolve_unique_operand(candidates)
            if value:
                resolved[atom.key] = value
                trace["operands_found_in_baseline_evidence"].append(atom.key)
            else:
                missing.append(atom)
        if contract.metric_name == "gross_margin":
            needed_missing: list[AtomicOperand] = []
            for atom in missing:
                period = atom.period
                if atom.concept == "gross_profit" and {
                    f"revenue_{period}", f"cost_of_goods_sold_{period}",
                }.issubset(resolved):
                    continue
                if atom.concept == "cost_of_goods_sold" and {
                    f"gross_profit_{period}", f"revenue_{period}",
                }.issubset(resolved):
                    continue
                needed_missing.append(atom)
            missing = needed_missing
        trace["missing_operands_before_tool"] = [item.key for item in missing]
        trace["missing_operands_before_search"] = [item.key for item in missing]
        optional_keys = {f"{name}_{period}" for name in contract.optional_operands for period in contract.requested_periods}
        prevalidated_absent = [
            atom.key for atom in missing
            if atom.key in optional_keys and _optional_absence_validated(atom, all_documents, target_filenames)
        ]
        missing = [atom for atom in missing if atom.key not in prevalidated_absent]
        searched_documents: list[dict[str, Any]] = []
        if missing:
            search = search_missing_operands(question, missing, baseline_documents, candidate_documents, max_queries=6)
            searched_documents = list(search.documents)
            trace["operand_search_calls"] = len(search.calls)
            trace["operand_search_queries"] = [dict(item) for item in search.calls]
            trace["skill_dense_bm25_calls"] = search.dense_bm25_calls
            trace["skill_jina_calls"] = search.jina_calls
            documents = [*baseline_documents, *searched_documents]
            for atom in missing:
                candidates = extract_operand_candidates(atom, documents, question, target_filenames)
                candidates = _scope_filtered_candidates(candidates, documents)
                trace["operand_candidates"][atom.key] = [item.trace_dict() for item in candidates[:5]]
                value, _ = resolve_unique_operand(candidates)
                if value:
                    resolved[atom.key] = value
        validation_documents = [*baseline_documents, *searched_documents, *candidate_documents]
        allowed_absent: list[str] = list(prevalidated_absent)
        for atom in atoms:
            if atom.key in resolved or atom.key not in optional_keys:
                continue
            if atom.key not in allowed_absent and _optional_absence_validated(atom, validation_documents, target_filenames):
                allowed_absent.append(atom.key)
        trace["validated_absent_optional_operands"] = allowed_absent
        required_atoms: list[AtomicOperand] = []
        for atom in atoms:
            if atom.key in optional_keys:
                continue
            if contract.metric_name == "gross_margin" and atom.concept == "cost_of_goods_sold":
                period = atom.period
                if f"gross_profit_{period}" in resolved:
                    continue
            if contract.metric_name == "gross_margin" and atom.concept == "gross_profit":
                period = atom.period
                if f"cost_of_goods_sold_{period}" in resolved:
                    continue
            required_atoms.append(atom)
        unresolved = [item.key for item in required_atoms if item.key not in resolved]
        unsafe_optional = [key for key in optional_keys if key not in resolved and key not in allowed_absent]
        if unresolved or unsafe_optional:
            missing_keys = [*unresolved, *unsafe_optional]
            trace["operand_resolution_status"] = "failed"
            trace["operand_resolution_failure_reason"] = f"unresolved_operands:{','.join(sorted(missing_keys))}"
            trace["skill_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return SkillResult(detected=True, trace=trace)
        used_values: list[ResolvedOperand] = []
        results: list[dict[str, Any]] = []
        variants: list[str] = []
        try:
            for period in contract.requested_periods:
                result, period_values, variant = _calculate_period(contract, period, resolved)
                failure = _validate_resolved(
                    [next(atom for atom in atoms if atom.key == item.key) for item in period_values], period_values,
                )
                if failure:
                    raise ValueError(failure)
                results.append(result)
                used_values.extend(period_values)
                variants.append(variant)
        except (DecimalCalculationError, KeyError, StopIteration, ValueError) as exc:
            trace["operand_resolution_status"] = "failed"
            trace["operand_resolution_failure_reason"] = f"validation_or_calculator_error:{exc}"
            trace["skill_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return SkillResult(detected=True, trace=trace)
        trace["operand_resolution_status"] = "validated"
        trace["statement_scope_validation"] = "validated"
        trace["period_validation"] = "validated"
        trace["currency_scale_validation"] = "validated"
        trace["calculator_called"] = True
        trace["resolved_operands"] = [item.trace_dict() for item in used_values]
        trace["formula_variant"] = variants[0] if len(set(variants)) == 1 else variants
        trace["metric_results"] = results
        trace["metric_results_by_period"] = {item["period"]: dict(item) for item in results}
        trace["metric_full_precision_result"] = results[-1]["full_precision_result"]
        trace["metric_display_result"] = results[-1]["display_result"]
        if len(results) == 2:
            left, right = (Decimal(item["full_precision_result"]) for item in results)
            trace["trend_comparison_result"] = "increase" if right > left else "decrease" if right < left else "unchanged"
            trace["trend_direction"] = trace["trend_comparison_result"]
        citations = _citations(used_values)
        answer = _answer(contract, results, used_values)
        direct = not contract.interpretation_required
        suffix = {"percent": "%", "times": " times", "ratio": ""}.get(contract.output_unit, "")
        verified = (
            "Verified metric evidence (calculation policy only; facts remain grounded in cited evidence):\n"
            + "; ".join(f"{item['period']} {contract.metric_alias} = {item['display_result']}{suffix}" for item in results)
            + ". Use this verified numeric result when answering the question; do not recalculate it. "
            "Interpret only the metric the user requested. Do not speculate that management prefers another metric "
            "unless the cited evidence explicitly says this metric is irrelevant."
        )
        trace.update({
            "verified_evidence": verified if not direct else "", "authoritative_numeric": True,
            "authoritative_answer": direct,
            "fallback_to_clean_baseline": not direct, "fallback_reason": "interpretation_requires_normal_answer_generator" if not direct else "",
            "skill_success": True, "skill_applied": direct,
            "skill_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        return SkillResult(
            detected=True, success=True, applied=direct, answer=answer if direct else "",
            citations=citations, trace=trace,
        )
