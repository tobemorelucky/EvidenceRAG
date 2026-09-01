"""Deterministic quality checks for table evidence assembly."""

from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{2,}|\(?\d[\d,]*(?:\.\d+)?%?\)?", re.IGNORECASE)


def _tokens(value) -> set[str]:
    return {match.group(0).casefold().replace(",", "") for match in _TOKEN_RE.finditer(str(value or ""))}


def table_structure_text(table: dict, *, max_rows: int = 20) -> str:
    parts = [str(table.get("title") or ""), str(table.get("caption") or "")]
    parts.extend(str(column or "") for column in (table.get("columns") or []))
    for row in list(table.get("rows") or [])[:max_rows]:
        if isinstance(row, dict):
            parts.extend(str(key or "") for key in row if not str(key).startswith("_"))
            parts.extend(str(value or "") for key, value in row.items() if not str(key).startswith("_"))
        elif isinstance(row, (list, tuple)):
            parts.extend(str(value or "") for value in row)
    return "\n".join(part for part in parts if part)


def structural_quality_score(table: dict) -> float:
    columns = [str(item or "").strip() for item in (table.get("columns") or [])]
    rows = [item for item in (table.get("rows") or []) if isinstance(item, (dict, list, tuple))]
    if not columns or not rows:
        return 0.0
    nonempty_columns = sum(bool(item) for item in columns) / len(columns)
    row_widths = []
    nonempty_cells = 0
    total_cells = 0
    for row in rows:
        values = list(row.values()) if isinstance(row, dict) else list(row)
        row_widths.append(len(values))
        nonempty_cells += sum(bool(str(value or "").strip()) for value in values)
        total_cells += len(values)
    width_consistency = sum(width == len(columns) for width in row_widths) / len(row_widths)
    nonempty_ratio = nonempty_cells / max(1, total_cells)
    row_support = min(1.0, len(rows) / 4.0)
    return round(
        0.25 * nonempty_columns
        + 0.30 * width_consistency
        + 0.30 * nonempty_ratio
        + 0.15 * row_support,
        4,
    )


def table_page_match_score(table: dict, page_text: str) -> float:
    table_tokens = _tokens(table_structure_text(table))
    page_tokens = _tokens(page_text)
    if not table_tokens or not page_tokens:
        return 0.0
    return round(len(table_tokens & page_tokens) / len(table_tokens), 4)


def table_is_eligible(
    table: dict,
    page: dict,
    *,
    quality_threshold: float = 0.65,
    page_match_threshold: float = 0.35,
) -> tuple[bool, str, dict]:
    columns = table.get("columns") or []
    rows = table.get("rows") or []
    structure_score = structural_quality_score(table)
    stored_score = float(table.get("quality_score") or 0.0)
    quality_score = stored_score if stored_score > 0 else structure_score
    match_score = table_page_match_score(table, page.get("page_text") or page.get("text") or "")
    details = {
        "quality_score": round(quality_score, 4),
        "structural_quality_score": structure_score,
        "page_match_score": match_score,
    }
    if not table.get("document_id") or table.get("document_id") != page.get("document_id"):
        return False, "document_id_mismatch", details
    if not table.get("page_id") or table.get("page_id") != page.get("page_id"):
        return False, "page_id_mismatch", details
    page_number = int(page.get("page_number") or 0)
    if not (int(table.get("start_page") or 0) <= page_number <= int(table.get("end_page") or 0)):
        return False, "page_range_mismatch", details
    if not columns or not rows:
        return False, "empty_structure", details
    if quality_score < quality_threshold:
        return False, "quality_below_threshold", details
    if match_score < page_match_threshold:
        return False, "page_content_mismatch", details
    return True, "trusted_table", details

