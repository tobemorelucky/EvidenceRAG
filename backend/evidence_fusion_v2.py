"""Evidence Fusion v2: page text plus quality-gated table evidence."""

from __future__ import annotations

from collections import defaultdict
import os
import re

try:
    from evidence_assembly_v1 import _format_table, _terms
    from table_quality import table_is_eligible
except ModuleNotFoundError:
    from backend.evidence_assembly_v1 import _format_table, _terms
    from backend.table_quality import table_is_eligible


def _fit(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    marker = "\n... evidence truncated to existing context budget ..."
    if limit <= len(marker):
        return marker[:limit]
    return value[: limit - len(marker)] + marker


def _fit_units(units: list[str], budget: int) -> str:
    if not units or budget <= 0:
        return ""
    separator_chars = 2 * (len(units) - 1)
    content_budget = max(0, budget - separator_chars)
    slot = max(1, content_budget // len(units))
    return _fit("\n\n".join(_fit(unit, slot) for unit in units), budget)


def _focus_page_text(question: str, page_text: str, limit: int) -> str:
    """Keep query-relevant line windows when a page does not fit its slot."""
    if len(page_text) <= limit:
        return page_text
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if not lines:
        return _fit(page_text, limit)
    query_terms = _terms(question)
    ranked = []
    for index, line in enumerate(lines):
        overlap = len(query_terms & _terms(line))
        if overlap:
            numeric_density = min(6, len(re.findall(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?", line)))
            ranked.append((overlap * 10 + numeric_density, index))
    if not ranked:
        return _fit(page_text, limit)

    chosen = set(range(min(3, len(lines))))
    for _, index in sorted(ranked, reverse=True):
        window = set(range(max(0, index - 2), min(len(lines), index + 3)))
        candidate = chosen | window
        rendered = "\n".join(lines[item] for item in sorted(candidate))
        if len(rendered) > limit:
            continue
        chosen = candidate
    focused = "\n".join(lines[index] for index in sorted(chosen))
    return _fit(focused, limit)


def _build_page_evidence(question: str, pages: list[tuple[str, str]], budget: int) -> str:
    if not pages or budget <= 0:
        return ""
    separator_chars = 2 * (len(pages) - 1)
    slot = max(1, max(0, budget - separator_chars) // len(pages))
    units = []
    for reference, page_text in pages:
        header = f"[Page Text Evidence {reference}]\n"
        body = _focus_page_text(question, page_text, max(0, slot - len(header)))
        units.append(_fit(f"{header}{body}", slot))
    return _fit("\n\n".join(units), budget)


def build_evidence_fusion_v2(
    question: str,
    selected_pages: list[dict],
    tables: list[dict],
    *,
    max_context_chars: int | None = None,
    quality_threshold: float | None = None,
    page_match_threshold: float | None = None,
) -> tuple[str, dict]:
    """Fuse page text with eligible table evidence inside the existing budget."""
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

    metadata_units = []
    page_sources = []
    table_units = []
    trusted_table_ids = []
    rejected_tables = []
    for page_index, page in enumerate(pages, 1):
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

        reference = f"P{page_index}"
        trusted_ids = ", ".join(str(table.get("table_id") or "") for table in trusted)
        metadata_units.append(
            f"[{reference}] {source}\nPage ID: {page_id or '(missing)'}"
            + (f"\nTrusted table IDs: {trusted_ids}" if trusted_ids else "")
        )
        page_text = str(page.get("page_text") or page.get("text") or "").strip()
        page_sources.append((reference, page_text))
        for table in trusted:
            table_units.append(f"{source}\nPage reference: {reference}\n{_format_table(question, table)}")

    metadata_cap = int(max_context_chars * 0.05)
    table_cap = int(max_context_chars * 0.25)
    metadata_header = "[Evidence Metadata]\n"
    table_header = "[Table Evidence Layer]\n"
    metadata_body = _fit_units(metadata_units, max(0, metadata_cap - len(metadata_header)))
    table_body = _fit_units(table_units, max(0, table_cap - len(table_header)))
    metadata_evidence = f"{metadata_header}{metadata_body}" if metadata_body else ""
    table_evidence = f"{table_header}{table_body}" if table_body else ""
    section_count = sum(bool(value) for value in (metadata_evidence, table_evidence)) + bool(page_sources)
    separator_chars = 2 * max(0, section_count - 1)
    page_budget = max(0, max_context_chars - len(metadata_evidence) - len(table_evidence) - separator_chars)
    page_evidence = _build_page_evidence(question, page_sources, page_budget)

    sections = []
    if metadata_evidence:
        sections.append(metadata_evidence)
    if page_evidence:
        sections.append(page_evidence)
    if table_evidence:
        sections.append(table_evidence)
    evidence = _fit("\n\n".join(sections), max_context_chars)

    return evidence, {
        "evidence_assembly_version": "fusion_v2",
        "answer_context_chars": len(evidence),
        "answer_context_max_chars": max_context_chars,
        "selected_page_count": len(pages),
        "page_text_included_count": len(page_sources),
        "trusted_table_count": len(trusted_table_ids),
        "trusted_table_ids": trusted_table_ids,
        "rejected_table_count": len(rejected_tables),
        "rejected_tables": rejected_tables,
        "quality_threshold": quality_threshold,
        "page_match_threshold": page_match_threshold,
        "budget": {
            "metadata_cap_chars": metadata_cap,
            "metadata_used_chars": len(metadata_evidence),
            "table_cap_chars": table_cap,
            "table_used_chars": len(table_evidence),
            "page_text_budget_chars": page_budget,
            "page_text_used_chars": len(page_evidence),
        },
        "table_contribution_chars": len(table_evidence),
        "table_contribution_ratio": round(len(table_evidence) / max(1, len(evidence)), 4),
    }
