"""Evidence Assembly v1: trusted table evidence with page-text fallback."""

from __future__ import annotations

from collections import defaultdict
import os
import re

try:
    from table_quality import table_is_eligible
except ModuleNotFoundError:
    from backend.table_quality import table_is_eligible


_WORD_RE = re.compile(r"[a-z][a-z0-9_-]{2,}|\d{4}", re.IGNORECASE)


def _terms(value) -> set[str]:
    return {match.group(0).casefold() for match in _WORD_RE.finditer(str(value or ""))}


def _row_text(row: dict, columns: list[str]) -> str:
    parts = []
    for column in columns or list(row):
        if str(column).startswith("_"):
            continue
        value = str(row.get(column, "") or "").strip()
        if value:
            parts.append(f"{column}: {value}")
    return "; ".join(parts)


def _select_target_rows(question: str, table: dict, *, max_rows: int = 8) -> list[str]:
    columns = [str(item or "").strip() for item in (table.get("columns") or [])]
    query_terms = _terms(question)
    scored = []
    for index, row in enumerate(table.get("rows") or []):
        if not isinstance(row, dict):
            continue
        text = _row_text(row, columns)
        overlap = len(query_terms & _terms(text))
        scored.append((overlap, -index, text))
    matched = [item for item in scored if item[0] > 0]
    selected = sorted(matched or scored, reverse=True)[:max_rows]
    selected.sort(key=lambda item: -item[1])
    return [item[2] for item in selected if item[2]]


def _format_table(question: str, table: dict) -> str:
    title = str(table.get("title") or table.get("caption") or "").strip()
    columns = [str(item or "").strip() for item in (table.get("columns") or [])]
    rows = _select_target_rows(question, table)
    lines = [
        "[Trusted Table Evidence]",
        f"Table ID: {table.get('table_id', '')}",
    ]
    if title:
        lines.append(f"Table title: {title}")
    lines.append(f"Header/columns: {' | '.join(columns)}")
    unit = str(table.get("unit") or "").strip()
    scale = str(table.get("scale") or "").strip()
    if unit:
        lines.append(f"Unit: {unit}")
    if scale:
        lines.append(f"Scale: {scale}")
    if rows:
        lines.append("Target rows:")
        lines.extend(f"- {row}" for row in rows)
    before = str(table.get("before_context") or "").strip()
    after = str(table.get("after_context") or "").strip()
    if before:
        lines.extend(["Nearby text before:", before[:600]])
    if after:
        lines.extend(["Nearby text after:", after[:600]])
    return "\n".join(lines)


def _fit(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n... evidence truncated to existing context budget ..."
    if limit <= len(marker):
        return marker[:limit]
    return value[: limit - len(marker)] + marker


def build_evidence_assembly_v1(
    question: str,
    selected_pages: list[dict],
    tables: list[dict],
    *,
    max_context_chars: int | None = None,
    quality_threshold: float | None = None,
    page_match_threshold: float | None = None,
) -> tuple[str, dict]:
    """Build evidence from trusted tables, otherwise preserve original page text."""
    max_context_chars = max_context_chars or int(os.getenv("RAG_CORE_V3_MAX_CONTEXT_CHARS", "28000"))
    quality_threshold = quality_threshold if quality_threshold is not None else float(
        os.getenv("EVIDENCE_ASSEMBLY_TABLE_QUALITY_THRESHOLD", "0.65")
    )
    page_match_threshold = page_match_threshold if page_match_threshold is not None else float(
        os.getenv("EVIDENCE_ASSEMBLY_PAGE_MATCH_THRESHOLD", "0.35")
    )
    pages = [page for page in selected_pages or [] if str(page.get("page_text") or page.get("text") or "").strip()]
    tables_by_page: dict[str, list[dict]] = defaultdict(list)
    for table in tables or []:
        page_id = str(table.get("page_id") or "").strip()
        if page_id:
            tables_by_page[page_id].append(table)

    slot_count = max(1, len(pages))
    page_slot = max(1, max_context_chars // slot_count)
    units = []
    trusted_table_ids = []
    rejected_tables = []
    fallback_pages = []
    for page in pages:
        page_id = str(page.get("page_id") or "").strip()
        source = f"Source: {page.get('filename', '')}, internal page {int(page.get('page_number') or 0)}"
        trusted = []
        for table in tables_by_page.get(page_id, []):
            eligible, reason, details = table_is_eligible(
                table,
                page,
                quality_threshold=quality_threshold,
                page_match_threshold=page_match_threshold,
            )
            if eligible:
                trusted.append(table)
                trusted_table_ids.append(table.get("table_id"))
            else:
                rejected_tables.append({
                    "table_id": table.get("table_id"),
                    "page_id": page_id,
                    "reason": reason,
                    **details,
                })
        if trusted:
            body = "\n\n".join(_format_table(question, table) for table in trusted)
            unit = f"{source}\nPage ID: {page_id}\n{body}"
        else:
            fallback_pages.append(page_id or f"missing:{page.get('filename', '')}:{page.get('page_number', 0)}")
            page_text = str(page.get("page_text") or page.get("text") or "").strip()
            unit = f"{source}\nPage ID: {page_id or '(missing)'}\n[Page Text Evidence]\n{page_text}"
        units.append(_fit(unit, page_slot))

    evidence = _fit("\n\n".join(units), max_context_chars)
    return evidence, {
        "evidence_assembly_version": "v1",
        "answer_context_chars": len(evidence),
        "answer_context_max_chars": max_context_chars,
        "selected_page_count": len(pages),
        "trusted_table_count": len(trusted_table_ids),
        "trusted_table_ids": trusted_table_ids,
        "rejected_table_count": len(rejected_tables),
        "rejected_tables": rejected_tables,
        "page_text_fallback_count": len(fallback_pages),
        "page_text_fallback_page_ids": fallback_pages,
        "quality_threshold": quality_threshold,
        "page_match_threshold": page_match_threshold,
    }

