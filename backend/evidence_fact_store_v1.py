"""Deterministic, offline financial fact extraction from structured tables.

This module is intentionally independent from retrieval and answer generation.
It emits a fact only when a table row can be aligned to an explicit period
header without guessing a financial formula or metric synonym.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from typing import Iterable

try:
    from table_quality import structural_quality_score
except ModuleNotFoundError:  # pragma: no cover - package import
    from backend.table_quality import structural_quality_score


_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_VALUE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<value>\(?\s*[-+]?\s*\$?\s*\d[\d,]*(?:\.\d+)?\s*%?\s*\)?)(?![A-Za-z0-9])"
)
_SPACE_RE = re.compile(r"\s+")
_METRIC_WORD_RE = re.compile(r"[A-Za-z][A-Za-z&'’/.,()\- ]{1,}")


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    document_id: str
    page_id: str
    table_id: str
    entity: str
    period: str
    metric: str
    value: str
    unit: str
    source_table: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def _periods(value: object) -> list[str]:
    return list(dict.fromkeys(match.group(1) for match in _YEAR_RE.finditer(str(value or ""))))


def _period_sequence(value: object) -> list[str]:
    return [match.group(1) for match in _YEAR_RE.finditer(str(value or ""))]


def _normalize_value(value: str) -> str | None:
    raw = _clean(value)
    negative = raw.startswith("(") and raw.endswith(")")
    percent = "%" in raw
    numeric = raw.replace("(", "").replace(")", "").replace("$", "").replace(",", "").replace("%", "")
    numeric = re.sub(r"\s+", "", numeric)
    try:
        number = Decimal(numeric)
    except InvalidOperation:
        return None
    if negative:
        number = -abs(number)
    rendered = format(number, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return f"{rendered}%" if percent else rendered


def _raw_line(row: object) -> str:
    if isinstance(row, dict):
        raw = _clean(row.get("_raw_line"))
        if raw:
            return raw
        return " ".join(
            _clean(value) for key, value in row.items()
            if not str(key).startswith("_") and _clean(value)
        )
    if isinstance(row, (list, tuple)):
        return " ".join(_clean(value) for value in row if _clean(value))
    return ""


def _explicit_periods(table: dict) -> tuple[list[str], int | None, bool]:
    """Return ordered periods and the row that supplied them.

    Parser output often contains duplicate column names (for example three
    ``December 31`` columns) that collapse in row dictionaries. The preserved
    raw header line is therefore the safest period/value alignment source.
    """
    best: tuple[list[str], int | None] = ([], None)
    for index, row in enumerate(table.get("rows") or []):
        found = _period_sequence(_raw_line(row))
        if len(found) > len(best[0]):
            best = (found, index)
    if best[0]:
        return best[0], best[1], len(set(best[0])) != len(best[0])
    found = _period_sequence(" ".join(_clean(item) for item in table.get("columns") or []))
    return found, None, len(set(found)) != len(found)


def _row_facts(line: str, periods: list[str]) -> tuple[str, list[str]] | None:
    matches = list(_VALUE_RE.finditer(line))
    # More numeric cells than periods commonly means the table also contains a
    # second measure (for example percentage change). Without a hierarchical
    # header those values cannot be aligned safely, so reject instead of taking
    # an arbitrary suffix.
    if not periods or len(matches) != len(periods):
        return None
    selected = matches
    metric = _clean(line[: selected[0].start()]).strip(" :;|.-")
    if not metric or not _METRIC_WORD_RE.search(metric) or _periods(metric):
        return None
    values = [_normalize_value(match.group("value")) for match in selected]
    if any(value is None for value in values):
        return None
    return metric, [str(value) for value in values]


def table_is_clear(table: dict, *, quality_threshold: float = 0.65) -> tuple[bool, str, dict]:
    columns = [_clean(item) for item in table.get("columns") or [] if _clean(item)]
    rows = [item for item in table.get("rows") or [] if isinstance(item, (dict, list, tuple))]
    structural = structural_quality_score(table)
    stored = float(table.get("quality_score") or 0.0)
    effective = stored if stored > 0 else structural
    periods, header_row, ambiguous_periods = _explicit_periods(table)
    details = {
        "stored_quality_score": round(stored, 4),
        "structural_quality_score": structural,
        "effective_quality_score": round(effective, 4),
        "periods": periods,
        "period_header_row": header_row,
        "ambiguous_period_header": ambiguous_periods,
    }
    if not all(_clean(table.get(key)) for key in ("table_id", "document_id", "page_id")):
        return False, "missing_identity", details
    if len(columns) < 2 or not rows:
        return False, "empty_or_narrow_structure", details
    if structural < quality_threshold or effective < quality_threshold:
        return False, "quality_below_threshold", details
    if not periods:
        return False, "explicit_period_missing", details
    if ambiguous_periods:
        return False, "ambiguous_period_header", details
    return True, "clear_structure", details


def facts_from_table(table: dict, *, quality_threshold: float = 0.65) -> tuple[list[EvidenceFact], dict]:
    eligible, reason, details = table_is_clear(table, quality_threshold=quality_threshold)
    trace = {"table_id": _clean(table.get("table_id")), "eligible": eligible, "reason": reason, **details}
    if not eligible:
        return [], trace

    periods = details["periods"]
    header_row = details["period_header_row"]
    entity = _clean(table.get("entity") or table.get("company") or table.get("doc_name") or table.get("filename") or table.get("document_id"))
    scale_unit = " ".join(dict.fromkeys(
        item for item in (_clean(table.get("unit")), _clean(table.get("scale"))) if item
    ))
    facts: list[EvidenceFact] = []
    for row_index, row in enumerate(table.get("rows") or []):
        if row_index == header_row:
            continue
        line = _raw_line(row)
        parsed = _row_facts(line, periods)
        if parsed is None:
            continue
        metric, values = parsed
        for period, value in zip(periods, values):
            identity = "|".join((_clean(table["table_id"]), str(row_index), period, metric, value))
            facts.append(EvidenceFact(
                fact_id="fact_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
                document_id=_clean(table["document_id"]),
                page_id=_clean(table["page_id"]),
                table_id=_clean(table["table_id"]),
                entity=entity,
                period=period,
                metric=metric,
                value=value,
                unit=scale_unit,
                source_table={
                    "table_id": _clean(table.get("table_id")),
                    "title": _clean(table.get("title") or table.get("caption")),
                    "header": [_clean(item) for item in table.get("columns") or []],
                    "filename": _clean(table.get("filename")),
                    "page_number": int(table.get("page_number") or 0),
                    "row_index": row_index,
                    "raw_row": line,
                },
            ))
    trace["fact_count"] = len(facts)
    if not facts:
        trace.update(eligible=False, reason="no_aligned_numeric_rows")
    return facts, trace


def build_fact_index(tables: Iterable[dict], *, quality_threshold: float = 0.65) -> tuple[list[dict], dict]:
    facts: list[EvidenceFact] = []
    traces = []
    for table in tables:
        table_facts, trace = facts_from_table(table, quality_threshold=quality_threshold)
        facts.extend(table_facts)
        traces.append(trace)
    reason_counts: dict[str, int] = {}
    for trace in traces:
        reason = trace["reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return [fact.to_dict() for fact in facts], {
        "tables_seen": len(traces),
        "tables_indexed": sum(bool(trace.get("fact_count")) for trace in traces),
        "facts": len(facts),
        "quality_threshold": quality_threshold,
        "table_reason_counts": reason_counts,
    }


def fact_text(fact: dict) -> str:
    source = fact.get("source_table") or {}
    return (
        f"Entity: {fact.get('entity', '')}\n"
        f"Period: {fact.get('period', '')}\n"
        f"Metric: {fact.get('metric', '')}\n"
        f"Value: {fact.get('value', '')}\n"
        f"Unit: {fact.get('unit', '')}\n"
        f"Source: {source.get('filename', '')}, internal page {source.get('page_number', 0)}, "
        f"table {fact.get('table_id', '')}\n"
        f"Source row: {source.get('raw_row', '')}"
    ).strip()
