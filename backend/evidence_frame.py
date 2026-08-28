"""Conservative structured evidence adapter for existing financial tables."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

try:
    from query_parser import infer_page_statement_types
    from table_reconstructor import normalize_financial_table
except ImportError:  # Package imports in unit tests.
    from backend.query_parser import infer_page_statement_types
    from backend.table_reconstructor import normalize_financial_table


SUPPORTED_STATEMENTS = {"balance_sheet", "income_statement", "cash_flow"}
_YEAR = re.compile(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", re.IGNORECASE)
_NUMERIC = re.compile(r"^\(?\s*[-+]?\s*[$€£]?\s*\d[\d,]*(?:\.\d+)?\s*%?\s*\)?$")
_UNIT_SCALE = (
    (re.compile(r"\bbillions?\b", re.IGNORECASE), "billions"),
    (re.compile(r"\bmillions?\b", re.IGNORECASE), "millions"),
    (re.compile(r"\bthousands?\b", re.IGNORECASE), "thousands"),
)


@dataclass(frozen=True)
class EvidenceFrame:
    evidence_id: str
    source_type: str
    company: str | None
    document: str
    page_number: int
    section: str | None
    statement_type: str
    table_id: str
    row_label: str
    row_path: list[str]
    column_label: str
    column_path: list[str]
    period: str | None
    period_provenance: dict[str, Any] | None
    raw_value: str
    normalized_value: str
    currency: str | None
    scale: str | None
    scope: str | None
    descriptor: str
    sign: str
    value_type: str
    citation: str
    bbox: Any | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clean(value: Any) -> str:
    return "" if value is None else re.sub(r"\s+", " ", str(value)).strip()


def _table_matrix(table: dict[str, Any]) -> list[list[str]]:
    columns = [_clean(value) for value in (table.get("normalized_columns") or table.get("columns") or [])]
    rows = table.get("normalized_rows") or table.get("rows") or []
    matrix: list[list[str]] = []
    title = _clean(table.get("normalized_title") or table.get("title") or table.get("caption"))
    context = " ".join(
        filter(
            None,
            [
                _clean(table.get("normalized_unit")),
                _clean(table.get("before_context")),
                _clean(table.get("after_context")),
            ],
        )
    )
    if title:
        matrix.append([title])
    unit_match = re.search(
        r"(?:\(?\s*(?:USD|US\$|\$|EUR|€|GBP|£)?\s*(?:in\s+)?(?:thousands?|millions?|billions?)\s*\)?|except per share amounts)",
        context,
        re.IGNORECASE,
    )
    if unit_match:
        matrix.append([unit_match.group(0)])
    if columns:
        matrix.append(columns)
    for row in rows:
        if isinstance(row, dict):
            ordered = [_clean(row.get(column)) for column in columns]
        elif isinstance(row, (list, tuple)):
            ordered = [_clean(value) for value in row]
        else:
            continue
        if any(ordered):
            matrix.append(ordered)
    return matrix


def _normalized_table(table: dict[str, Any]) -> dict[str, Any]:
    if table.get("normalized_rows") and table.get("normalized_columns"):
        return table
    matrix = _table_matrix(table)
    return normalize_financial_table({**table, "raw_matrix": matrix})


def _table_text(table: dict[str, Any], normalized: dict[str, Any]) -> str:
    rows = normalized.get("normalized_rows") or table.get("rows") or []
    return "\n".join(
        [
            _clean(normalized.get("normalized_title") or table.get("title")),
            _clean(table.get("caption")),
            _clean(table.get("before_context")),
            _clean(table.get("after_context")),
            _clean(table.get("evidence_page_context")),
            *(
                " | ".join(_clean(value) for value in row.values() if _clean(value))
                for row in rows
                if isinstance(row, dict)
            ),
        ]
    )


def _statement_type(table: dict[str, Any], normalized: dict[str, Any]) -> str:
    matches = [
        item
        for item in infer_page_statement_types(_table_text(table, normalized))
        if item in SUPPORTED_STATEMENTS
    ]
    return matches[0] if len(matches) == 1 else ""


def _number(raw_value: str) -> tuple[str, str] | None:
    source = _clean(raw_value)
    if not source or not _NUMERIC.fullmatch(source):
        return None
    parenthesized = source.startswith("(") and source.endswith(")")
    cleaned = source.replace("$", "").replace("€", "").replace("£", "").replace(",", "").replace("%", "")
    cleaned = re.sub(r"\s+", "", cleaned).strip("()")
    if parenthesized and not cleaned.startswith("-"):
        cleaned = f"-{cleaned}"
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    normalized = format(value, "f")
    sign = "negative" if value < 0 else "positive" if value > 0 else "zero"
    return normalized, sign


def _currency(raw_value: str, context: str) -> str | None:
    if "%" in raw_value:
        return None
    combined = f"{raw_value} {context}"
    if "$" in combined or re.search(r"\b(?:USD|US dollars?)\b", combined, re.IGNORECASE):
        return "USD"
    if "€" in combined or re.search(r"\bEUR\b", combined, re.IGNORECASE):
        return "EUR"
    if "£" in combined or re.search(r"\bGBP\b", combined, re.IGNORECASE):
        return "GBP"
    return None


def _scale(raw_value: str, context: str) -> str | None:
    if "%" in raw_value:
        return "percent"
    for pattern, scale in _UNIT_SCALE:
        if pattern.search(context):
            return scale
    return None


def _scope(context: str) -> str | None:
    lowered = context.lower()
    if "parent company only" in lowered or "financial information of parent" in lowered:
        return "parent_only"
    if "consolidated" in lowered:
        return "consolidated"
    if re.search(r"\b(?:geographic|geography|domestic|international|outside the u\.s\.)\b", lowered):
        return "geography"
    if re.search(r"\b(?:reportable segment|operating segment|segment information)\b", lowered):
        return "segment"
    return None


def _period(column_label: str, column_path: list[str]) -> str | None:
    explicit = " | ".join([*column_path, column_label])
    years = _YEAR.findall(explicit)
    if not years:
        return None
    quarter = re.search(r"\b(?:Q[1-4]|first|second|third|fourth)\b", explicit, re.IGNORECASE)
    return f"{quarter.group(0)} {years[-1]}" if quarter else years[-1]


def _descriptor(
    row_label: str,
    row_path: list[str],
    column_path: list[str],
    table_title: str | None,
    section: str | None,
    statement_type: str,
    scope: str | None,
) -> str:
    """Build one stable, auditable short-text representation per frame."""
    statement_label = statement_type.replace("_", " ")
    values = [
        row_label,
        " > ".join(row_path),
        " > ".join(column_path),
        table_title or "",
        section or "",
        statement_label,
        scope or "",
    ]
    return " | ".join(dict.fromkeys(value for value in values if value))


def _explicit_header_periods(
    table: dict[str, Any],
    normalized: dict[str, Any],
    value_columns: list[str],
) -> list[str]:
    """Recover periods only from an explicit table header on the matched page."""
    if not value_columns or not all(re.fullmatch(r"value_\d+", column, re.IGNORECASE) for column in value_columns):
        return []
    context = str(table.get("evidence_page_context") or "").strip()
    rows = normalized.get("normalized_rows") or table.get("rows") or []
    first_label = ""
    if rows and isinstance(rows[0], dict):
        first_label = _clean(next(iter(rows[0].values()), ""))
    if not context or not first_label:
        return []
    position = context.casefold().find(first_label.casefold())
    if position < 0:
        return []
    prefix = context[:position]
    # Require a primary-statement heading near the header. A narrative with
    # multiple years is not sufficient evidence for column alignment.
    if not infer_page_statement_types(prefix[-1500:]):
        return []
    header_lines = [item.strip() for item in prefix[-1200:].splitlines() if item.strip()]
    recovered: list[str] = []
    for line in header_lines:
        years = list(dict.fromkeys(_YEAR.findall(line)))
        explicit_dates = bool(re.search(
            r"\b(?:fiscal\s+year|year\s+ended|as\s+of|"
            r"january|february|march|april|may|june|july|august|september|october|november|december)\b",
            line,
            re.IGNORECASE,
        ))
        only_year_columns = bool(re.fullmatch(r"(?:\s*(?:FY\s*)?(?:19|20)\d{2}\s*)+", line, re.IGNORECASE))
        if len(years) == len(value_columns) and (explicit_dates or only_year_columns):
            return years
        if len(years) == 1 and (explicit_dates or only_year_columns):
            recovered.extend(years)
    recovered = list(dict.fromkeys(recovered))
    if len(recovered) == len(value_columns):
        return recovered
    return []


def _column_schema(
    table: dict[str, Any],
    normalized: dict[str, Any],
    value_columns: list[str],
    inherited_periods: list[str] | None = None,
    inherited_from_page: int | None = None,
) -> dict[str, dict[str, Any]]:
    schema_by_label = {
        _clean(item.get("label")): item
        for item in (normalized.get("column_schema") or [])
        if isinstance(item, dict) and _clean(item.get("label"))
    }
    schema = {
        column: schema_by_label.get(column, {"label": column, "path": [column], "value_type": "unknown", "unit": ""})
        for column in value_columns
    }
    recovered_periods = _explicit_header_periods(table, normalized, value_columns)
    period_source = "matched_page_explicit_header"
    periods = recovered_periods
    if not periods and inherited_periods and len(inherited_periods) == len(value_columns):
        periods = inherited_periods
        period_source = "same_table_continuation"
    if periods:
        for column, period in zip(value_columns, periods):
            if not _YEAR.search(" | ".join([*schema[column].get("path", []), str(schema[column].get("label") or "")])):
                provenance = {
                    "source": period_source,
                    "table_id": _clean(table.get("table_id")),
                    "page_number": int(table.get("evidence_page_number") or table.get("page_number") or 0),
                }
                if inherited_from_page is not None:
                    provenance["inherited_from_page"] = inherited_from_page
                schema[column] = {
                    **schema[column],
                    "label": period,
                    "path": [period],
                    "period_provenance": provenance,
                }
    return schema


def _period_provenance(
    schema_item: dict[str, Any],
    column_label: str,
    column_path: list[str],
    table: dict[str, Any],
) -> dict[str, Any] | None:
    explicit = schema_item.get("period_provenance")
    if isinstance(explicit, dict):
        return dict(explicit)
    header = " | ".join([*column_path, column_label])
    if not _YEAR.search(header):
        return None
    return {
        "source": "column_schema",
        "table_id": _clean(table.get("table_id")),
        "page_number": int(table.get("evidence_page_number") or table.get("page_number") or 0),
        "header": header,
    }


def is_evidence_frame_eligible_table(table: dict[str, Any]) -> bool:
    """Accept parser-approved tables or structurally stable primary statements."""
    normalized = _normalized_table(table)
    if table.get("accepted", True) is not False:
        return True
    rows = normalized.get("normalized_rows") or []
    columns = normalized.get("normalized_columns") or []
    return len(rows) >= 2 and len(columns) >= 2 and bool(_statement_type(table, normalized))


def build_evidence_frames(
    tables: list[dict[str, Any]],
    *,
    company: str | None = None,
    max_frames: int = 500,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt accepted primary-statement tables into auditable numeric frames."""
    frames: list[dict[str, Any]] = []
    skipped = {"unsupported_statement": 0, "unstructured_table": 0, "non_numeric_value": 0}
    accepted_tables = 0
    explicit_periods_by_table: dict[str, tuple[str, int, list[str]]] = {}
    for table in tables or []:
        if not is_evidence_frame_eligible_table(table):
            continue
        normalized = _normalized_table(table)
        statement_type = _statement_type(table, normalized)
        if not statement_type:
            skipped["unsupported_statement"] += 1
            continue
        rows = normalized.get("normalized_rows") or []
        columns = normalized.get("normalized_columns") or []
        if len(columns) < 2 or not rows:
            skipped["unstructured_table"] += 1
            continue
        accepted_tables += 1
        metric_column = columns[0]
        value_columns = list(columns[1:])
        table_id = _clean(table.get("table_id"))
        filename = _clean(table.get("filename"))
        page_number = int(table.get("evidence_page_number") or table.get("page_number") or 0)
        inherited_periods: list[str] | None = None
        inherited_from_page: int | None = None
        inherited = explicit_periods_by_table.get(table_id) if table_id else None
        if inherited and inherited[0] == filename and page_number == inherited[1] + 1:
            inherited_periods = inherited[2]
            inherited_from_page = inherited[1]
        schema = _column_schema(
            table,
            normalized,
            value_columns,
            inherited_periods=inherited_periods,
            inherited_from_page=inherited_from_page,
        )
        schema_periods = [
            _period(column, [_clean(value) for value in (schema[column].get("path") or [column]) if _clean(value)])
            for column in value_columns
        ]
        if table_id and all(schema_periods):
            explicit_periods_by_table[table_id] = (filename, page_number, [str(item) for item in schema_periods])
        context = _table_text(table, normalized)
        section = _clean(normalized.get("normalized_title") or table.get("title") or table.get("caption")) or None
        table_title = _clean(table.get("title") or table.get("caption") or normalized.get("normalized_title")) or None
        scope = _scope(context)
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            row_label = _clean(row.get(metric_column))
            if not row_label:
                continue
            for column_label in value_columns:
                raw_value = _clean(row.get(column_label))
                parsed = _number(raw_value)
                if parsed is None:
                    if raw_value:
                        skipped["non_numeric_value"] += 1
                    continue
                normalized_value, sign = parsed
                schema_item = schema[column_label]
                column_path = [_clean(value) for value in (schema_item.get("path") or [column_label]) if _clean(value)]
                period = _period(column_label, column_path)
                unit_context = " ".join(
                    filter(
                        None,
                        [
                            _clean(normalized.get("normalized_unit")),
                            _clean(schema_item.get("unit")),
                        ],
                    )
                )
                scale = _scale(raw_value, f"{unit_context} {context}")
                value_type = "percentage" if scale == "percent" else _clean(schema_item.get("value_type")) or "number"
                identity = json.dumps(
                    [table_id, row_index, row_label, column_label, raw_value],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                frame = EvidenceFrame(
                    evidence_id=f"ef_{hashlib.sha256(identity).hexdigest()[:24]}",
                    source_type="table_cell",
                    company=_clean(company) or None,
                    document=filename,
                    page_number=page_number,
                    section=section,
                    statement_type=statement_type,
                    table_id=table_id,
                    row_label=row_label,
                    row_path=[row_label],
                    column_label=column_label,
                    column_path=column_path or [column_label],
                    period=period,
                    period_provenance=_period_provenance(
                        schema_item,
                        column_label,
                        column_path or [column_label],
                        table,
                    ),
                    raw_value=raw_value,
                    normalized_value=normalized_value,
                    currency=_currency(raw_value, context),
                    scale=scale,
                    scope=scope,
                    descriptor=_descriptor(
                        row_label,
                        [row_label],
                        column_path or [column_label],
                        table_title,
                        section,
                        statement_type,
                        scope,
                    ),
                    sign=sign,
                    value_type=value_type,
                    citation=f"[source: {filename}, page {page_number}]",
                    bbox=table.get("bbox"),
                )
                frames.append(frame.to_dict())
                if len(frames) >= max(1, max_frames):
                    break
            if len(frames) >= max(1, max_frames):
                break
        if len(frames) >= max(1, max_frames):
            break
    trace = {
        "evidence_frame_count": len(frames),
        "table_frame_count": len(frames),
        "evidence_frame_tables_considered": len(tables or []),
        "evidence_frame_tables_accepted": accepted_tables,
        "frames_with_period": sum(bool(frame.get("period")) for frame in frames),
        "frames_with_unit_scale": sum(bool(frame.get("currency") or frame.get("scale")) for frame in frames),
        "frames_used_for_execution": 0,
        "evidence_frame_skipped": skipped,
    }
    return frames, trace
