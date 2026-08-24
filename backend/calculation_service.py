"""Build conservative, auditable Decimal calculations from validated evidence fields."""

import re
from typing import Dict, List

from agent_tools import calculate


_SUPPORTED_FORMULA = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\s*[+\-*/]\s*[A-Za-z_][A-Za-z0-9_]*)*$")


def _normalized_number(value: str) -> str:
    source = str(value or "").strip()
    negative = source.startswith("(") and source.endswith(")")
    source = source.replace("$", "").replace("€", "").replace("£", "").replace(",", "").replace("%", "")
    source = source.strip("() ")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", source):
        return ""
    return f"-{source}" if negative and not source.startswith("-") else source


_ROW_NUMBER_PATTERN = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?")


def _row_values(documents: List[dict], aliases: List[str], field: str = "") -> tuple[List[str], dict]:
    candidates: List[tuple[int, List[str], dict]] = []
    for document_index, document in enumerate(documents):
        text = str(document.get("text") or document.get("page_text") or "")
        lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
        for line in lines:
            lowered = line.lower()
            matches = [(index, item) for index, item in enumerate(aliases) if item.lower() in lowered]
            if not matches:
                continue
            if field == "revenue" and re.search(r"(?:sales\s+)?(?:as a\s+)?percentage of (?:net |total )?revenue", lowered):
                continue
            if field == "revenue" and "deferred revenue" in lowered:
                continue
            alias_index, alias = min(matches, key=lambda item: (item[0], -len(item[1])))
            tail = line[lowered.find(alias.lower()) + len(alias) :]
            tail = re.sub(r"\([^)]*notes?[^)]*\)", "", tail, flags=re.IGNORECASE)
            # Some balance-sheet labels include allowance/accumulated-depreciation
            # amounts before the actual period columns, e.g. ``- $40 and $42``.
            tail = re.sub(
                r"\s*[-–—]\s*\$?\(?\d[\d,]*(?:\.\d+)?\)?\s+and\s+\$?\(?\d[\d,]*(?:\.\d+)?\)?",
                "",
                tail,
                flags=re.IGNORECASE,
            )
            if "respectively" in tail.lower():
                tail = tail[tail.lower().find("respectively") + len("respectively") :]
            values = list(filter(None, (_normalized_number(value) for value in _ROW_NUMBER_PATTERN.findall(tail))))
            if values:
                score = 100 - alias_index * 5 - document_index
                normalized_line = re.sub(r"\s+", " ", lowered)
                if field == "current_liabilities":
                    if "total current liabilities" in normalized_line:
                        score += 200
                    elif "other current liabilities" in normalized_line:
                        score -= 100
                if field == "accounts_receivable":
                    if "accounts receivable, net" in normalized_line or "receivable, net" in normalized_line:
                        score += 100
                    if "allowance for" in normalized_line and not normalized_line.startswith(("accounts", "trade")):
                        score -= 100
                if field == "inventory":
                    if re.search(r"\binventory\s+(?:turn|turns|turnover|days)\b", normalized_line):
                        score -= 200
                    if re.match(r"^inventor(?:y|ies)\s*(?:\(|\$|\d)", normalized_line):
                        score += 100
                candidates.append((score, values, {
                    "filename": document.get("filename") or "",
                    "page_number": document.get("page_number"),
                    "matched_alias": alias,
                }))
    if not candidates:
        return [], {}
    _, values, source = max(candidates, key=lambda item: item[0])
    return values, source


def _build_row_calculation(task_spec: Dict[str, object], documents: List[dict]) -> Dict[str, object] | None:
    from query_parser import FIELD_ALIASES

    formula = str(task_spec.get("formula") or "").strip()
    fields = list(task_spec.get("required_fields") or [])
    if not formula or not fields:
        return None

    operands: Dict[str, dict] = {}
    value_series: Dict[str, List[str]] = {}
    expression = formula
    for field in fields:
        values, source = _row_values(documents, FIELD_ALIASES.get(field, []), str(field))
        if not values:
            return None
        value_series[str(field)] = values
        if formula == "revenue / average(ppe)" and field == "ppe":
            if len(values) < 2:
                return None
            replacement = f"(({values[0]}) + ({values[1]})) / 2"
            operand_value: object = values[:2]
        else:
            operand_value = values[0]
            if field == "capital_expenditures" and re.search(rf"-\s*{re.escape(str(field))}\b", formula):
                operand_value = operand_value.lstrip("-")
            replacement = f"({operand_value})"
        expression = re.sub(rf"\b{re.escape(str(field))}\b", replacement, expression)
        operands[field] = {"value": operand_value, **source}

    expression = expression.replace("average(", "(") if "average(" in expression else expression
    try:
        calculated = calculate(expression)
    except ValueError:
        return None
    result = {
        "formula": formula,
        "expression": calculated["expression"],
        "operands": operands,
        "result": calculated["result"],
        "status": "calculated",
        "source": "structured_row_decimal",
    }
    if task_spec.get("compare_periods") and all(len(value_series.get(str(field), [])) >= 2 for field in fields):
        expressions = []
        values = []
        for period_index in range(2):
            period_expression = formula
            for field in fields:
                period_expression = re.sub(
                    rf"\b{re.escape(str(field))}\b",
                    f"({value_series[str(field)][period_index]})",
                    period_expression,
                )
            period_result = calculate(period_expression)
            expressions.append(period_result["expression"])
            values.append(period_result["result"])
        from decimal import Decimal

        latest, prior = Decimal(values[0]), Decimal(values[1])
        result["comparison"] = {
            "reported_order": "latest_then_prior",
            "values": values,
            "expressions": expressions,
            "direction": "increased" if latest > prior else "decreased" if latest < prior else "unchanged",
        }
    return result


def build_calculation_result(
    task_spec: Dict[str, object],
    coverage: Dict[str, object],
    documents: List[dict] | None = None,
) -> Dict[str, object] | None:
    """Calculate only when every required field resolves to one unambiguous numeric value."""
    if task_spec.get("task_type") != "calculation" or coverage.get("status") != "complete":
        return None
    formula = str(task_spec.get("formula") or "").strip()
    if documents:
        row_calculation = _build_row_calculation(task_spec, documents)
        if row_calculation:
            return row_calculation
    if not formula or not _SUPPORTED_FORMULA.fullmatch(formula):
        return None

    operands: Dict[str, dict] = {}
    used_sources: set[tuple[str, object, str]] = set()
    used_values: set[str] = set()
    expression = formula
    for field in task_spec.get("required_fields") or []:
        evidence = dict((coverage.get("field_evidence") or {}).get(field) or {})
        values = list(dict.fromkeys(filter(None, (_normalized_number(value) for value in evidence.get("values") or []))))
        if len(values) != 1:
            return None
        value = values[0]
        if field == "capital_expenditures" and re.search(rf"-\s*{re.escape(str(field))}\b", formula):
            value = value.lstrip("-")
        source_key = (str(evidence.get("filename") or ""), evidence.get("page_number"), value)
        if source_key in used_sources:
            return None
        if value in used_values:
            return None
        used_sources.add(source_key)
        used_values.add(value)
        expression = re.sub(rf"\b{re.escape(str(field))}\b", f"({value})", expression)
        operands[str(field)] = {
            "value": value,
            "filename": evidence.get("filename") or "",
            "page_number": evidence.get("page_number"),
            "matched_alias": evidence.get("alias") or "",
        }

    calculated = calculate(expression)
    return {
        "formula": formula,
        "expression": calculated["expression"],
        "operands": operands,
        "result": calculated["result"],
        "status": "calculated",
    }


def format_calculation_evidence(calculation: Dict[str, object] | None) -> str:
    if not calculation:
        return ""
    lines = ["Validated calculation (Decimal):"]
    for field, operand in (calculation.get("operands") or {}).items():
        lines.append(
            f"- {field} = {operand.get('value')} "
            f"[source: {operand.get('filename')}, page {operand.get('page_number')}]"
        )
    lines.extend(
        [
            f"- Formula: {calculation.get('formula')}",
            f"- Expression: {calculation.get('expression')}",
            f"- Result: {calculation.get('result')}",
        ]
    )
    comparison = calculation.get("comparison") or {}
    if comparison:
        lines.append(
            "- Validated comparison (latest reported period vs prior reported period): "
            f"{comparison.get('values', ['?', '?'])[0]} vs {comparison.get('values', ['?', '?'])[1]}; "
            f"direction = {comparison.get('direction')}"
        )
    return "\n".join(lines)
