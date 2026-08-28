"""Build conservative, auditable Decimal calculations from validated evidence fields."""

import os
import re
from decimal import Decimal, DecimalException, ROUND_HALF_UP
from typing import Dict, List

from agent_tools import calculate
from financial_executor import FinancialExecutionError, execute_financial_operation


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
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_INCOME_FIELDS = {"revenue", "operating_income", "net_income"}
_BALANCE_FIELDS = {
    "current_assets", "current_liabilities", "cash_and_equivalents", "short_term_investments",
    "accounts_receivable", "other_current_assets", "accounts_payable", "other_accrued_liabilities",
    "inventory", "total_assets", "ppe",
}
_CASH_FLOW_FIELDS = {"cash_from_operations", "depreciation_amortization", "capital_expenditures"}


def _with_display_result(result: Dict[str, object], task_spec: Dict[str, object]) -> Dict[str, object]:
    places = task_spec.get("rounding_decimal_places")
    if places is None:
        return result
    try:
        precision = max(0, min(12, int(places)))
        quantum = Decimal(1).scaleb(-precision)
        display = Decimal(str(result["result"])).quantize(quantum, rounding=ROUND_HALF_UP)
    except (KeyError, TypeError, ValueError, DecimalException):
        return result
    result["display_result"] = f"{display:.{precision}f}"
    result["rounding_decimal_places"] = precision
    return result


def _requested_years(task_spec: Dict[str, object]) -> List[str]:
    return list(dict.fromkeys(
        year
        for period in (task_spec.get("required_periods") or [])
        for year in _YEAR_PATTERN.findall(str(period))
    ))


def _normalized_label(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _label_tokens(value: object) -> set[str]:
    return {
        token
        for token in _normalized_label(value).split()
        if len(token) > 1 and token not in {"and", "the", "of", "total", "net"}
    }


def match_evidence_frames_detailed(
    field: str,
    frames: List[dict],
    *,
    concepts: List[str] | None = None,
    statement_types: List[str] | None = None,
    scope: str = "",
) -> tuple[List[dict], Dict[str, object]]:
    """Layered short-text frame recall; metadata validation remains downstream."""
    from query_parser import FIELD_ALIASES

    field_label = str(field or "").replace("_", " ")
    canonical_aliases = list(dict.fromkeys([
        field_label,
        *[str(item) for item in FIELD_ALIASES.get(field, [])],
    ]))
    query_concepts = list(dict.fromkeys([
        *[str(item) for item in (concepts or []) if str(item).strip()],
        *canonical_aliases,
    ]))
    normalized_concepts = [_normalized_label(item) for item in query_concepts if _normalized_label(item)]
    normalized_aliases = {_normalized_label(item) for item in canonical_aliases if _normalized_label(item)}
    expected_statements = {str(item) for item in (statement_types or []) if item}
    expected_scope = _normalized_label(scope)
    alignment_enabled = os.getenv("FRAME_ALIGNMENT_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not alignment_enabled:
        aliases = [_normalized_label(alias) for alias in canonical_aliases if _normalized_label(alias)]
        legacy = []
        for frame in frames:
            label = _normalized_label(frame.get("row_label"))
            scores = [1.0 if label == alias else 0.8 if label.startswith(alias) else 0.6 if alias in label else 0.0 for alias in aliases]
            score = max(scores, default=0.0)
            if score:
                legacy.append((score, frame))
        best = max((score for score, _ in legacy), default=0.0)
        selected = [frame for score, frame in legacy if score == best]
        return selected, {
            "field": field,
            "concepts": query_concepts,
            "candidates": [
                {
                    "evidence_id": str(frame.get("evidence_id") or ""),
                    "row_label": str(frame.get("row_label") or ""),
                    "period": str(frame.get("period") or ""),
                    "match_method": "legacy_alias",
                    "match_score": score,
                }
                for score, frame in legacy[:12]
            ],
            "selected_count": len(selected),
            "alignment_enabled": False,
        }
    scored: List[tuple[float, str, dict]] = []
    for frame in frames:
        label = _normalized_label(frame.get("row_label"))
        descriptor = _normalized_label(
            frame.get("descriptor")
            or " | ".join(
                str(item or "")
                for item in [
                    frame.get("row_label"),
                    " ".join(frame.get("row_path") or []),
                    " ".join(frame.get("column_path") or []),
                    frame.get("section"),
                    frame.get("statement_type"),
                    frame.get("scope"),
                ]
            )
        )
        method = ""
        score = 0.0
        if label and label in normalized_concepts:
            method, score = "normalized_exact", 1.0
        elif label and label in normalized_aliases:
            method, score = "canonical_alias", 0.96
        else:
            for concept in normalized_concepts:
                concept_tokens = _label_tokens(concept)
                label_tokens = _label_tokens(label)
                descriptor_tokens = _label_tokens(descriptor)
                if not concept_tokens:
                    continue
                if concept in label or label in concept:
                    candidate_score = 0.82 + 0.12 * len(concept_tokens & label_tokens) / len(concept_tokens)
                    candidate_method = "phrase_overlap"
                else:
                    overlap = len(concept_tokens & descriptor_tokens) / len(concept_tokens)
                    candidate_score = 0.5 + 0.3 * overlap if overlap >= 0.5 else 0.0
                    candidate_method = "descriptor_context"
                if candidate_score > score:
                    method, score = candidate_method, candidate_score
        if not score:
            continue
        if expected_statements and str(frame.get("statement_type") or "") in expected_statements:
            score += 0.03
        if expected_scope and _normalized_label(frame.get("scope")) == expected_scope:
            score += 0.02
        scored.append((min(score, 1.0), method, frame))

    method_priority = {
        "normalized_exact": 4,
        "canonical_alias": 3,
        "phrase_overlap": 2,
        "descriptor_context": 1,
    }
    deduplicated: Dict[str, tuple[float, str, dict]] = {}
    for score, method, frame in scored:
        evidence_id = str(frame.get("evidence_id") or "")
        key = evidence_id or f"object:{id(frame)}"
        previous = deduplicated.get(key)
        if previous is None or (method_priority.get(method, 0), score) > (method_priority.get(previous[1], 0), previous[0]):
            deduplicated[key] = (score, method, frame)
    scored = list(deduplicated.values())
    scored.sort(
        key=lambda item: (method_priority.get(item[1], 0), item[0], str(item[2].get("evidence_id") or "")),
        reverse=True,
    )
    best_priority = method_priority.get(scored[0][1], 0) if scored else 0
    same_layer = [item for item in scored if method_priority.get(item[1], 0) == best_priority]
    best = max((score for score, _, _ in same_layer), default=0.0)
    selected = [frame for score, _, frame in same_layer if score >= 0.55 and score >= best - 0.08]
    trace = {
        "field": field,
        "concepts": query_concepts,
        "candidates": [
            {
                "evidence_id": str(frame.get("evidence_id") or ""),
                "row_label": str(frame.get("row_label") or ""),
                "period": str(frame.get("period") or ""),
                "match_method": method,
                "match_score": round(score, 4),
            }
            for score, method, frame in scored[:12]
        ],
        "selected_count": len(selected),
        "alignment_enabled": True,
    }
    return selected, trace


def matching_evidence_frames(field: str, frames: List[dict]) -> List[dict]:
    """Compatibility wrapper for the layered matcher."""
    matches, _ = match_evidence_frames_detailed(field, frames)
    return matches


def resolve_frame_operands(
    task_spec: Dict[str, object],
    evidence_frames: List[dict],
) -> Dict[str, List[dict]] | None:
    """Resolve formula fields conservatively; ambiguity always falls back."""
    requested_years = _requested_years(task_spec)
    expected_company = _normalized_label(task_spec.get("company"))
    formula = str(task_spec.get("formula") or "")
    selected: Dict[str, List[dict]] = {}
    for field in task_spec.get("required_fields") or []:
        candidates = matching_evidence_frames(str(field), evidence_frames)
        if expected_company:
            candidates = [
                frame for frame in candidates
                if _normalized_label(frame.get("company")) == expected_company
            ]
        average_field = bool(re.search(rf"average\(\s*{re.escape(str(field))}\s*\)", formula))
        if requested_years:
            # Explicit periods may never be inferred from column order.
            matching = [frame for frame in candidates if str(frame.get("period") or "") in requested_years]
            if average_field and matching:
                anchor = matching[0] if len(matching) == 1 else None
                peers = [
                    frame for frame in candidates
                    if anchor
                    and frame.get("table_id") == anchor.get("table_id")
                    and frame.get("row_label") == anchor.get("row_label")
                    and frame.get("period")
                    and frame.get("period") != anchor.get("period")
                ]
                try:
                    anchor_year = int(str(anchor.get("period"))) if anchor else 0
                    prior = [frame for frame in peers if int(str(frame.get("period"))) < anchor_year]
                    prior.sort(key=lambda frame: int(str(frame.get("period"))), reverse=True)
                except (TypeError, ValueError):
                    prior = []
                candidates = [anchor, prior[0]] if anchor and prior else []
            else:
                candidates = matching
        elif average_field:
            # Without a requested period, exactly two explicitly-labelled cells
            # from one row/table are required.
            groups: Dict[tuple, List[dict]] = {}
            for frame in candidates:
                if frame.get("period"):
                    groups.setdefault((frame.get("table_id"), frame.get("row_label")), []).append(frame)
            valid_groups = [items for items in groups.values() if len(items) == 2]
            candidates = valid_groups[0] if len(valid_groups) == 1 else []
        if (average_field and len(candidates) != 2) or (not average_field and len(candidates) != 1):
            return None
        selected[str(field)] = candidates
    return selected


def _task_frame_matches(task_spec: Dict[str, object], evidence_frames: List[dict]) -> List[dict]:
    target = str(task_spec.get("target_measure") or "").strip()
    fields = [str(item) for item in task_spec.get("required_fields") or []]
    matches: Dict[str, dict] = {}
    for field in fields:
        candidates, _ = match_evidence_frames_detailed(
            field,
            evidence_frames,
            statement_types=[str(item) for item in task_spec.get("statement_types") or []],
            scope=str(task_spec.get("scope") or ""),
        )
        matches.update({str(item.get("evidence_id") or ""): item for item in candidates})
    if target:
        candidates, _ = match_evidence_frames_detailed(
            "target_measure",
            evidence_frames,
            concepts=[target],
            statement_types=[str(item) for item in task_spec.get("statement_types") or []],
            scope=str(task_spec.get("scope") or ""),
        )
        matches.update({str(item.get("evidence_id") or ""): item for item in candidates})
    selected = [item for key, item in matches.items() if key]
    expected_company = _normalized_label(task_spec.get("company"))
    if expected_company:
        selected = [item for item in selected if _normalized_label(item.get("company")) == expected_company]
    expected_scope = _normalized_label(task_spec.get("scope"))
    if expected_scope:
        selected = [item for item in selected if _normalized_label(item.get("scope")) == expected_scope]
    return selected


def _compatible_matrix(frames: List[dict]) -> bool:
    if len(frames) < 2:
        return False
    for field in ("company", "currency", "scale", "scope"):
        values = [str(item.get(field) or "") for item in frames]
        if any(not value for value in values) or len(set(value.casefold() for value in values)) != 1:
            return False
    return True


def resolve_comparison_frames(task_spec: Dict[str, object], evidence_frames: List[dict]) -> List[dict]:
    """Resolve exactly one compatible value for each explicitly requested period."""
    periods = [str(item) for item in task_spec.get("required_periods") or []]
    if len(periods) != 2:
        return []
    candidates = _task_frame_matches(task_spec, evidence_frames)
    groups: Dict[tuple, List[dict]] = {}
    for frame in candidates:
        groups.setdefault(
            (frame.get("table_id"), tuple(frame.get("row_path") or [frame.get("row_label")]), frame.get("scope")),
            [],
        ).append(frame)
    valid: List[List[dict]] = []
    for group in groups.values():
        selected = []
        for period in periods:
            matches = [item for item in group if str(item.get("period") or "") == period]
            if len(matches) != 1:
                selected = []
                break
            selected.append(matches[0])
        if selected and _compatible_matrix(selected):
            valid.append(selected)
    return valid[0] if len(valid) == 1 else []


def resolve_selection_frames(task_spec: Dict[str, object], evidence_frames: List[dict]) -> List[dict]:
    """Resolve a complete, same-table candidate matrix without inventing members."""
    if not task_spec.get("candidate_dimension") or task_spec.get("selection_direction") not in {"max", "min"}:
        return []
    periods = [str(item) for item in task_spec.get("required_periods") or []]
    candidates = _task_frame_matches(task_spec, evidence_frames)
    if periods:
        if len(periods) != 1:
            return []
        candidates = [item for item in candidates if str(item.get("period") or "") == periods[0]]
    groups: Dict[tuple, List[dict]] = {}
    for frame in candidates:
        groups.setdefault(
            (frame.get("table_id"), tuple(frame.get("column_path") or []), frame.get("period"), frame.get("scope")),
            [],
        ).append(frame)
    valid = []
    for group in groups.values():
        labels = [str(item.get("row_label") or "").strip() for item in group]
        if len(group) >= 2 and all(labels) and len(set(label.casefold() for label in labels)) == len(labels) and _compatible_matrix(group):
            valid.append(group)
    return valid[0] if len(valid) == 1 else []


def _build_non_calculation_frame_result(
    task_spec: Dict[str, object],
    evidence_frames: List[dict],
) -> Dict[str, object] | None:
    if os.getenv("STRUCTURED_TASK_EXECUTOR_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    task_type = str(task_spec.get("task_type") or "")
    if task_type == "comparison":
        selected = resolve_comparison_frames(task_spec, evidence_frames)
        operation = "compare"
    elif task_type == "selection":
        selected = resolve_selection_frames(task_spec, evidence_frames)
        operation = "argmax" if task_spec.get("selection_direction") == "max" else "argmin"
    else:
        return None
    if not selected:
        return None
    try:
        result = execute_financial_operation(
            operation,
            evidence_frames,
            operand_evidence_ids=[str(item["evidence_id"]) for item in selected],
            rounding=task_spec.get("rounding_decimal_places"),
            unit=str(task_spec.get("result_unit") or selected[0].get("scale") or ""),
        )
    except FinancialExecutionError:
        return None
    result["source"] = "evidence_frame_decimal"
    result["task_type"] = task_type
    result["authoritative"] = True
    result["candidate_matrix"] = [
        {
            "evidence_id": item.get("evidence_id"),
            "label": item.get("row_label"),
            "period": item.get("period"),
            "raw_value": item.get("raw_value"),
            "normalized_value": item.get("normalized_value"),
            "currency": item.get("currency"),
            "scale": item.get("scale"),
            "scope": item.get("scope"),
            "citation": item.get("citation"),
            "document": item.get("document"),
            "page_number": item.get("page_number"),
        }
        for item in selected
    ]
    if task_type == "comparison":
        result["comparison_direction"] = {
            "greater": "increased",
            "less": "decreased",
            "equal": "unchanged",
        }.get(str(result.get("direction") or ""), "")
    else:
        result["selected_entity"] = str((result.get("selected_frame") or {}).get("row_label") or "")
    return result


def _build_frame_calculation(
    task_spec: Dict[str, object],
    evidence_frames: List[dict],
) -> Dict[str, object] | None:
    """Execute a QuerySpec formula only when every EvidenceFrame is auditable."""
    if os.getenv("STRUCTURED_EXECUTOR_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    formula = str(task_spec.get("formula") or "").strip()
    if not formula or not evidence_frames:
        return None
    selected = resolve_frame_operands(task_spec, evidence_frames)
    if not selected:
        return None
    operand_frames = [frame for field_frames in selected.values() for frame in field_frames]
    ids = [str(frame.get("evidence_id") or "") for frame in operand_frames]
    if not all(ids):
        return None

    # The select operation performs the shared provenance/metadata checks. A
    # cross-table formula with unknown units/scopes is intentionally rejected.
    table_ids = {str(frame.get("table_id") or "") for frame in operand_frames}
    for field in ("currency", "scale", "scope"):
        known = [frame.get(field) for frame in operand_frames if frame.get(field) not in (None, "")]
        if len(table_ids) > 1 and (len(known) != len(operand_frames) or len(set(known)) != 1):
            return None
    expected: Dict[str, object] = {}
    if task_spec.get("company"):
        expected["company"] = task_spec.get("company")
    requested_years = _requested_years(task_spec)
    if len(requested_years) == 1 and "average(" not in formula:
        expected["period"] = requested_years[0]
    try:
        validation = execute_financial_operation(
            "select",
            evidence_frames,
            operand_evidence_ids=ids,
            constraints={"expected": expected},
        )
    except FinancialExecutionError:
        return None

    expression = formula
    operands: Dict[str, dict] = {}
    for field, frames in selected.items():
        if len(frames) == 2:
            values = [str(frame["normalized_value"]) for frame in frames]
            expression = re.sub(
                rf"average\(\s*{re.escape(field)}\s*\)",
                f"((({values[0]}) + ({values[1]})) / 2)",
                expression,
            )
            operand_value: object = values
        else:
            operand_value = str(frames[0]["normalized_value"])
            expression = re.sub(rf"\b{re.escape(field)}\b", f"({operand_value})", expression)
        first = frames[0]
        operands[field] = {
            "value": operand_value,
            "raw_value": [frame.get("raw_value") for frame in frames] if len(frames) > 1 else first.get("raw_value"),
            "normalized_value": operand_value,
            "filename": first.get("document") or "",
            "page_number": first.get("page_number"),
            "period": first.get("period") or "",
            "currency": first.get("currency"),
            "scale": first.get("scale"),
            "scope": first.get("scope"),
            "citation": first.get("citation"),
            "matched_alias": first.get("row_label") or "",
            "evidence_ids": [frame["evidence_id"] for frame in frames],
        }
    try:
        calculated = calculate(expression)
    except ValueError:
        return None
    root_operation = "divide" if "/" in formula else "multiply" if "*" in formula else "subtract" if "-" in formula else "sum"
    return _with_display_result({
        "formula": formula,
        "operation": root_operation,
        "expression": calculated["expression"],
        "operands": operands,
        "normalized_operands": [str(frame["normalized_value"]) for frame in operand_frames],
        "operand_evidence_ids": ids,
        "citations": validation["citations"],
        "result": calculated["result"],
        "full_precision_result": calculated["result"],
        "status": "calculated",
        "executor": "evidence_frame",
        "source": "evidence_frame_decimal",
        "task_type": "calculation",
        "authoritative": True,
        "result_unit": task_spec.get("result_unit") or "",
    }, task_spec)


def _header_periods(lines: List[str], row_index: int, value_count: int) -> List[str]:
    """Return the nearest table-header years aligned left-to-right with row values."""
    # Long statement tables can place later rows dozens of lines below the
    # shared period header (for example Selected Financial Data pages).
    header = "\n".join(lines[max(0, row_index - 80) : row_index])
    years = list(dict.fromkeys(_YEAR_PATTERN.findall(header)))
    if len(years) < value_count:
        return []
    return years[-value_count:]


def _align_values_to_periods(
    values: List[str],
    periods: List[str],
    requested_years: List[str],
) -> tuple[List[str], List[str]]:
    if not requested_years:
        return values, periods
    if len(periods) != len(values) or any(year not in periods for year in requested_years):
        return [], []
    indexes = [periods.index(year) for year in requested_years]
    indexes.extend(index for index in range(len(values)) if index not in indexes)
    return [values[index] for index in indexes], [periods[index] for index in indexes]


def _row_values(
    documents: List[dict],
    aliases: List[str],
    field: str = "",
    requested_years: List[str] | None = None,
) -> tuple[List[str], dict]:
    candidates: List[tuple[int, List[str], dict]] = []
    requested_years = requested_years or []
    for document_index, document in enumerate(documents):
        text = str(document.get("text") or document.get("page_text") or "")
        document_lower = text.lower()
        filename_lower = str(document.get("filename") or "").lower()
        lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
        document_heading = "\n".join(lines[:12]).lower()
        income_statement_page = bool(re.search(
            r"(?m)^\s*consolidated statements? of (?:income|operations|earnings)\b",
            document_heading,
        ))
        balance_sheet_page = bool(re.search(
            r"(?m)^\s*(?:consolidated balance sheets?|statements? of financial position)\b",
            document_heading,
        ))
        cash_flow_page = bool(re.search(
            r"(?m)^\s*consolidated statements? of cash flows?\b",
            document_heading,
        ))
        if "schedule i condensed financial information of parent" in document_lower:
            continue
        if "reclassifications out of aocl" in document_lower:
            continue
        if "segment" in document_heading and not (income_statement_page or balance_sheet_page or cash_flow_page):
            continue
        if (
            field in _BALANCE_FIELDS
            and "consolidated statement" in document_heading
            and not balance_sheet_page
        ):
            continue
        for row_index, line in enumerate(lines):
            lowered = line.lower()
            matches = [(index, item) for index, item in enumerate(aliases) if item.lower() in lowered]
            if not matches:
                continue
            if field == "revenue" and re.search(r"(?:sales\s+)?(?:as a\s+)?percentage of (?:net |total )?revenue", lowered):
                continue
            if field == "revenue" and "deferred revenue" in lowered:
                continue
            alias_index, alias = min(matches, key=lambda item: (item[0], -len(item[1])))
            alias_position = lowered.find(alias.lower())
            prefix_words = re.findall(r"[a-z]+", lowered[:alias_position])
            # Deterministic operands must come from a table-like row label, not
            # narrative such as "foreign exchange decreased operating income by ...".
            if alias_position > 40 or len(prefix_words) > 4:
                continue
            tail = line[alias_position + len(alias) :]
            tail = re.sub(r"\([^)]*notes?[^)]*\)", "", tail, flags=re.IGNORECASE)
            # Some balance-sheet labels include allowance/accumulated-depreciation
            # amounts before the actual period columns, e.g. ``- $40 and $42``.
            tail = re.sub(
                r"\s*[-–—]\s*\$?\(?\d[\d,]*(?:\.\d+)?\)?\s+and\s+\$?\(?\d[\d,]*(?:\.\d+)?\)?",
                "",
                tail,
                flags=re.IGNORECASE,
            )
            tail = re.sub(
                r"\s*(?:net\s+of\s+)?allowances?(?:\s+for\s+[^$\d]{0,40})?\s+of\s+"
                r"\$?\(?\d[\d,]*(?:\.\d+)?\)?\s+and\s+\$?\(?\d[\d,]*(?:\.\d+)?\)?",
                "",
                tail,
                flags=re.IGNORECASE,
            )
            if "respectively" in tail.lower():
                tail = tail[tail.lower().find("respectively") + len("respectively") :]
            values = list(filter(None, (_normalized_number(value) for value in _ROW_NUMBER_PATTERN.findall(tail))))
            if values:
                periods = _header_periods(lines, row_index, len(values))
                values, periods = _align_values_to_periods(values, periods, requested_years)
                if not values:
                    continue
                score = 100 - alias_index * 5 - document_index
                if alias_position <= 8:
                    score += 150
                if requested_years and periods and periods[0] == requested_years[0]:
                    score += 300
                primary_year = requested_years[0] if requested_years else ""
                if primary_year and primary_year in filename_lower and "10k" in filename_lower.replace("-", ""):
                    score += 350
                if field in _INCOME_FIELDS and income_statement_page:
                    score += 300
                if field in _BALANCE_FIELDS and balance_sheet_page:
                    score += 300
                if field in _CASH_FLOW_FIELDS and cash_flow_page:
                    score += 300
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
                if field == "net_income" and "attributable to" in normalized_line:
                    score += 100
                candidates.append((score, values, {
                    "filename": document.get("filename") or "",
                    "page_number": document.get("page_number"),
                    "matched_alias": alias,
                    "period": periods[0] if periods else "",
                    "periods": periods,
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
    requested_years = _requested_years(task_spec)
    for field in fields:
        values, source = _row_values(
            documents,
            FIELD_ALIASES.get(field, []),
            str(field),
            requested_years,
        )
        if not values:
            return None
        value_series[str(field)] = values
        average_field = bool(re.search(rf"average\(\s*{re.escape(str(field))}\s*\)", formula))
        if average_field:
            if len(values) < 2:
                return None
            replacement = f"((({values[0]}) + ({values[1]})) / 2)"
            operand_value: object = values[:2]
            expression = re.sub(
                rf"average\(\s*{re.escape(str(field))}\s*\)",
                replacement,
                expression,
            )
        else:
            operand_value = values[0]
            if field == "capital_expenditures" and re.search(rf"-\s*{re.escape(str(field))}\b", formula):
                operand_value = operand_value.lstrip("-")
            replacement = f"({operand_value})"
            expression = re.sub(rf"\b{re.escape(str(field))}\b", replacement, expression)
        operands[field] = {"value": operand_value, **source}

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
    evidence_frames: List[dict] | None = None,
) -> Dict[str, object] | None:
    """Calculate only when every required field resolves to one unambiguous numeric value."""
    if task_spec.get("task_type") != "calculation":
        return _build_non_calculation_frame_result(task_spec, evidence_frames or [])
    structured_coverage_required = os.getenv("STRUCTURED_COVERAGE_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }
    frame_calculation = None
    if not structured_coverage_required or coverage.get("operands_validated") is True:
        frame_calculation = _build_frame_calculation(task_spec, evidence_frames or [])
    if frame_calculation:
        return frame_calculation
    # Structured coverage may be partial solely because the table adapter
    # could not recover metadata. Preserve the documented second-priority
    # fallback to the existing, independently validated text-row calculator.
    if coverage.get("base_status", coverage.get("status")) != "complete":
        return None
    formula = str(task_spec.get("formula") or "").strip()
    if documents:
        row_calculation = _build_row_calculation(task_spec, documents)
        if row_calculation:
            row_calculation["result_unit"] = task_spec.get("result_unit") or ""
            return _with_display_result(row_calculation, task_spec)
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
    return _with_display_result({
        "formula": formula,
        "expression": calculated["expression"],
        "operands": operands,
        "result": calculated["result"],
        "status": "calculated",
        "result_unit": task_spec.get("result_unit") or "",
    }, task_spec)


def format_calculation_evidence(calculation: Dict[str, object] | None) -> str:
    if not calculation:
        return ""
    lines = ["Validated calculation contract (authoritative Decimal result):"]
    for field, operand in (calculation.get("operands") or {}).items():
        period = f", period {operand.get('period')}" if operand.get("period") else ""
        lines.append(
            f"- {field} = {operand.get('value')} "
            f"[source: {operand.get('filename')}, page {operand.get('page_number')}{period}]"
        )
    lines.extend(
        [
            f"- Formula: {calculation.get('formula')}",
            f"- Expression: {calculation.get('expression')}",
            f"- Full-precision result: {calculation.get('result')}",
        ]
    )
    if calculation.get("display_result") is not None:
        lines.append(
            f"- Required final display result: {calculation.get('display_result')} "
            f"({calculation.get('rounding_decimal_places')} decimal places)"
        )
    if calculation.get("result_unit") == "percent":
        try:
            percent_result = Decimal(str(calculation.get("result"))) * Decimal("100")
            lines.append(f"- Required final unit: percent; unrounded percentage = {percent_result}%")
        except DecimalException:
            pass
    comparison = calculation.get("comparison") or {}
    if comparison:
        lines.append(
            "- Validated comparison (latest reported period vs prior reported period): "
            f"{comparison.get('values', ['?', '?'])[0]} vs {comparison.get('values', ['?', '?'])[1]}; "
            f"direction = {comparison.get('direction')}"
        )
    return "\n".join(lines)


def _answer_numbers(text: str) -> List[Decimal]:
    values: List[Decimal] = []
    for raw in re.findall(r"(?<![A-Za-z0-9])\(?-?\d[\d,]*(?:\.\d+)?\)?%?", text or ""):
        normalized = _normalized_number(raw)
        if normalized:
            try:
                values.append(Decimal(normalized))
            except DecimalException:
                continue
    return values


def _result_candidates(calculation: Dict[str, object]) -> List[Decimal]:
    values: List[Decimal] = []
    for key in ("display_result", "result", "full_precision_result"):
        if calculation.get(key) in (None, ""):
            continue
        try:
            values.append(Decimal(str(calculation[key])))
        except DecimalException:
            continue
    if calculation.get("result_unit") == "percent" or calculation.get("unit") == "percent":
        values.extend(value * Decimal("100") for value in list(values))
    return list(dict.fromkeys(values))


def _citations_text(calculation: Dict[str, object]) -> str:
    citations = [str(item) for item in calculation.get("citations") or [] if str(item).strip()]
    if not citations:
        citations = list(dict.fromkeys(
            str(item.get("citation") or "")
            for item in calculation.get("candidate_matrix") or []
            if item.get("citation")
        ))
    return " ".join(citations)


def _authoritative_template(task_spec: Dict[str, object], calculation: Dict[str, object]) -> str:
    task_type = str(calculation.get("task_type") or task_spec.get("task_type") or "")
    citation = _citations_text(calculation)
    if task_type == "comparison":
        matrix = list(calculation.get("candidate_matrix") or [])
        values = "；".join(
            f"{item.get('period')}: {item.get('normalized_value')} {item.get('scale') or ''}".strip()
            for item in matrix
        )
        direction = str(calculation.get("comparison_direction") or "")
        return f"{values}。比较结果：{direction}。 {citation}".strip()
    if task_type == "selection":
        selected = str(calculation.get("selected_entity") or "")
        value = str(calculation.get("result") or "")
        unit = str(calculation.get("unit") or "")
        return f"{selected}，数值为 {value} {unit}。 {citation}".strip()
    result = str(calculation.get("display_result") or calculation.get("result") or "")
    unit = str(calculation.get("result_unit") or calculation.get("unit") or "")
    formula = str(calculation.get("formula") or calculation.get("expression") or "")
    return f"根据已验证操作数，{formula} = {result}{'%' if unit == 'percent' else (' ' + unit if unit else '')}。 {citation}".strip()


def validate_or_repair_structured_answer(
    answer: str,
    task_spec: Dict[str, object],
    calculation: Dict[str, object] | None,
) -> tuple[str, Dict[str, object]]:
    """Repair only direct conflicts with a high-confidence local result."""
    trace: Dict[str, object] = {
        "enabled": os.getenv("ANSWER_CONSISTENCY_VALIDATOR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        "authoritative_result": bool((calculation or {}).get("authoritative")),
        "checked": False,
        "conflict": False,
        "repaired": False,
        "reason": "",
    }
    if not trace["enabled"] or not trace["authoritative_result"]:
        return answer, trace
    calculation = calculation or {}
    trace["checked"] = True
    task_type = str(calculation.get("task_type") or task_spec.get("task_type") or "")
    normalized_answer = _normalized_label(answer)
    conflict = False
    reason = ""
    if task_type == "calculation":
        actual = _answer_numbers(answer)
        expected = _result_candidates(calculation)
        tolerance = Decimal("0.000001")
        if not any(abs(item - target) <= tolerance * max(Decimal("1"), abs(target)) for item in actual for target in expected):
            conflict, reason = True, "calculation_result_mismatch"
    elif task_type == "comparison":
        expected = str(calculation.get("comparison_direction") or "")
        synonyms = {
            "increased": {"increase", "increased", "grew", "higher"},
            "decreased": {"decrease", "decreased", "declined", "lower"},
            "unchanged": {"unchanged", "no change", "equal", "flat"},
        }
        expected_terms = synonyms.get(expected, {expected})
        opposing = set().union(*(values for key, values in synonyms.items() if key != expected))
        has_expected = any(term in normalized_answer for term in expected_terms)
        has_opposing = any(term in normalized_answer for term in opposing)
        if not has_expected or has_opposing:
            conflict, reason = True, "comparison_direction_mismatch"
    elif task_type == "selection":
        selected = _normalized_label(calculation.get("selected_entity"))
        conclusion = _normalized_label((answer or "").split(".", 1)[0].split("。", 1)[0])
        if not selected or selected not in conclusion:
            conflict, reason = True, "selection_entity_mismatch"
    if not conflict:
        return answer, trace
    trace.update({"conflict": True, "repaired": True, "reason": reason})
    return _authoritative_template(task_spec, calculation), trace
