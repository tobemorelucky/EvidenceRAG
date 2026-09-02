"""Evidence Fusion v3 with deterministic trusted-table row selection."""

from __future__ import annotations

try:
    from evidence_fusion_v2 import build_evidence_fusion_v2
    from row_relevance_selector import select_relevant_rows
except ModuleNotFoundError:
    from backend.evidence_fusion_v2 import build_evidence_fusion_v2
    from backend.row_relevance_selector import select_relevant_rows


def _format_table_v3(question: str, table: dict) -> tuple[str, dict]:
    title = str(table.get("title") or table.get("caption") or "").strip()
    headers = [str(item or "").strip() for item in (table.get("columns") or [])]
    selected, trace = select_relevant_rows(
        question,
        title,
        headers,
        table.get("rows") or [],
    )
    lines = [
        "[Trusted Table Evidence]",
        f"Table ID: {table.get('table_id', '')}",
    ]
    if title:
        lines.append(f"Table title: {title}")
    lines.append(f"Header/columns: {' | '.join(headers)}")
    unit = str(table.get("unit") or "").strip()
    scale = str(table.get("scale") or "").strip()
    if unit:
        lines.append(f"Unit: {unit}")
    if scale:
        lines.append(f"Scale: {scale}")
    if selected:
        lines.append("Relevant rows:")
        lines.extend(f"- {item['text']}" for item in selected)
    before = str(table.get("before_context") or "").strip()
    after = str(table.get("after_context") or "").strip()
    if before:
        lines.extend(["Nearby text before:", before[:600]])
    if after:
        lines.extend(["Nearby text after:", after[:600]])
    return "\n".join(lines), trace


def build_evidence_fusion_v3(
    question: str,
    selected_pages: list[dict],
    tables: list[dict],
    *,
    max_context_chars: int | None = None,
    quality_threshold: float | None = None,
    page_match_threshold: float | None = None,
) -> tuple[str, dict]:
    """Build v2 fusion using BM25/lexical/synonym-ranked table rows."""
    return build_evidence_fusion_v2(
        question,
        selected_pages,
        tables,
        max_context_chars=max_context_chars,
        quality_threshold=quality_threshold,
        page_match_threshold=page_match_threshold,
        _table_formatter=_format_table_v3,
        _version="fusion_v3",
    )
