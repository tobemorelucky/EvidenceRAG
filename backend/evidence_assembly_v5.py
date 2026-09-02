"""Evidence Assembly v5 shadow package built from frozen Top120 chunks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*|(?:19|20)\d{2}|\d+(?:\.\d+)?%?")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_VALUE_RE = re.compile(r"(?:[$€£¥]\s*)?\(?-?\d[\d,]*(?:\.\d+)?%?\)?")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "what", "when",
    "which", "who", "with", "would",
}


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _integer(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _terms(value: object) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(str(value or ""))
        if token.casefold() not in _STOPWORDS
    }


def _years(value: object) -> list[str]:
    return list(dict.fromkeys(_YEAR_RE.findall(str(value or ""))))


def _values(value: object) -> list[str]:
    return list(dict.fromkeys(match.group(0) for match in _VALUE_RE.finditer(str(value or ""))))


def _row_values(row: object, columns: list[str]) -> list[str]:
    if isinstance(row, dict):
        ordered = [row.get(column) for column in columns if column in row]
        ordered.extend(value for key, value in row.items() if key not in columns and not str(key).startswith("_"))
        return [_clean(value) for value in ordered if _clean(value)]
    if isinstance(row, list):
        return [_clean(value) for value in row if _clean(value)]
    return [_clean(row)] if _clean(row) else []


def _metric(values: list[str]) -> str | None:
    return next((value for value in values if re.search(r"[A-Za-z]", value) and not _VALUE_RE.fullmatch(value)), None)


@dataclass(frozen=True)
class EvidenceUnit:
    document_id: str
    page_id: str
    source_type: str
    entity: str
    period: list[str]
    metric: str | None
    value: list[str] | None
    unit: str | None
    source_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def build_evidence_units(
    question: str,
    chunks: list[dict],
    *,
    pages: list[dict],
    tables: list[dict],
) -> list[EvidenceUnit]:
    """Create uniform text and same-page table-row units without model calls."""
    page_by_key = {
        (_clean(page.get("filename")).casefold(), _integer(page.get("page_number"))): page
        for page in pages
    }
    best_page_rank: dict[str, int] = {}
    units: list[EvidenceUnit] = []
    seen_chunks: set[str] = set()
    for fallback_rank, chunk in enumerate(chunks, 1):
        text = str(chunk.get("text") or "").strip()
        chunk_id = _clean(chunk.get("chunk_id"))
        dedupe_key = chunk_id or re.sub(r"\s+", " ", text).casefold()
        if not text or dedupe_key in seen_chunks:
            continue
        seen_chunks.add(dedupe_key)
        page = page_by_key.get((_clean(chunk.get("filename")).casefold(), _integer(chunk.get("page_number"))), {})
        page_id = _clean(page.get("page_id"))
        rank = _integer(chunk.get("merged_rank")) or fallback_rank
        if page_id:
            best_page_rank[page_id] = min(rank, best_page_rank.get(page_id, rank))
        entity = _clean(chunk.get("company") or page.get("company") or page.get("doc_name") or chunk.get("filename"))
        section = _clean(chunk.get("section") or chunk.get("section_title")) or None
        numbers = _values(text)
        units.append(EvidenceUnit(
            document_id=_clean(page.get("document_id") or chunk.get("document_id")),
            page_id=page_id,
            source_type="text",
            entity=entity,
            period=_years(text),
            metric=section,
            value=numbers or None,
            unit=None,
            source_text=text,
            metadata={
                "filename": _clean(page.get("filename") or chunk.get("filename")),
                "page_number": _integer(page.get("page_number") if page else chunk.get("page_number")),
                "chunk_id": chunk_id,
                "retrieval_rank": rank,
            },
        ))

    query_terms = _terms(question)
    for table in tables:
        page_id = _clean(table.get("page_id"))
        rank = best_page_rank.get(page_id)
        if rank is None:
            continue
        page = next((value for value in pages if _clean(value.get("page_id")) == page_id), {})
        columns = [_clean(column) for column in table.get("columns") or [] if _clean(column)]
        if not columns or not table.get("rows"):
            continue
        title = _clean(table.get("title") or table.get("caption"))
        unit = _clean(" ".join(part for part in (_clean(table.get("unit")), _clean(table.get("scale"))) if part)) or None
        header = " | ".join(columns)
        entity = _clean(page.get("company") or page.get("doc_name") or table.get("filename"))
        for row_index, row in enumerate(table.get("rows") or []):
            values = _row_values(row, columns)
            if not values:
                continue
            row_text = " | ".join(values)
            # Headers are retained as evidence structure, but a shared year in
            # the header must not make every row in the table relevant.
            searchable = f"{title} {row_text}"
            overlap = len(query_terms & _terms(searchable))
            if overlap <= 0:
                continue
            source_text = "\n".join(part for part in (
                f"Table title: {title}" if title else "",
                f"Header: {header}",
                f"Row: {row_text}",
                f"Unit/scale: {unit}" if unit else "",
            ) if part)
            numbers = _values(row_text)
            units.append(EvidenceUnit(
                document_id=_clean(page.get("document_id") or table.get("document_id")),
                page_id=page_id,
                source_type="table",
                entity=entity,
                period=_years(f"{header} {row_text}"),
                metric=_metric(values),
                value=numbers or None,
                unit=unit,
                source_text=source_text,
                metadata={
                    "filename": _clean(page.get("filename") or table.get("filename")),
                    "page_number": _integer(page.get("page_number") if page else table.get("page_number")),
                    "table_id": _clean(table.get("table_id")),
                    "row_index": row_index,
                    "retrieval_rank": rank,
                    "query_overlap": overlap,
                    "quality_score": float(table.get("quality_score") or 0.0),
                },
            ))
    return units


def _render(unit: EvidenceUnit, index: int) -> str:
    metadata = unit.metadata
    source = f"{metadata.get('filename', '')}, internal page {_integer(metadata.get('page_number'))}"
    parts = [
        f"[Evidence Unit {index}]",
        f"Source: {source}",
        f"Page ID: {unit.page_id or '(missing)'}",
        f"Source type: {unit.source_type}",
    ]
    if unit.entity:
        parts.append(f"Entity: {unit.entity}")
    if unit.period:
        parts.append("Period: " + ", ".join(unit.period))
    if unit.metric:
        parts.append(f"Metric: {unit.metric}")
    if unit.unit:
        parts.append(f"Unit: {unit.unit}")
    parts.extend(["Source text:", unit.source_text])
    return "\n".join(parts)


def assemble_evidence_v5(
    question: str,
    chunks: list[dict],
    *,
    pages: list[dict],
    tables: list[dict],
    max_context_chars: int = 28000,
    text_budget_ratio: float = 0.78,
) -> tuple[str, list[dict], dict]:
    """Assemble high-rank raw chunks plus compact same-page table rows."""
    units = build_evidence_units(question, chunks, pages=pages, tables=tables)
    text_units = sorted(
        (unit for unit in units if unit.source_type == "text"),
        key=lambda unit: (_integer(unit.metadata.get("retrieval_rank")), unit.page_id),
    )
    table_units = sorted(
        (unit for unit in units if unit.source_type == "table"),
        key=lambda unit: (
            -_integer(unit.metadata.get("query_overlap")),
            _integer(unit.metadata.get("retrieval_rank")),
            -float(unit.metadata.get("quality_score") or 0.0),
            _integer(unit.metadata.get("row_index")),
        ),
    )
    selected: list[EvidenceUnit] = []
    rendered: list[str] = []
    used = 0

    def add(unit: EvidenceUnit) -> bool:
        nonlocal used
        value = _render(unit, len(selected) + 1)
        separator = 2 if rendered else 0
        if used + separator + len(value) > max_context_chars:
            return False
        selected.append(unit)
        rendered.append(value)
        used += separator + len(value)
        return True

    text_target = int(max_context_chars * min(1.0, max(0.0, text_budget_ratio)))
    deferred_text: list[EvidenceUnit] = []
    for unit in text_units:
        value = _render(unit, len(selected) + 1)
        if used + (2 if rendered else 0) + len(value) <= text_target:
            add(unit)
        else:
            deferred_text.append(unit)
    for unit in table_units:
        add(unit)
    for unit in deferred_text:
        add(unit)

    context = "\n\n".join(rendered)
    selected_dicts = [unit.to_dict() for unit in selected]
    return context, selected_dicts, {
        "evidence_assembly_version": "v5_shadow",
        "candidate_chunk_count": len(chunks),
        "candidate_text_unit_count": len(text_units),
        "candidate_table_unit_count": len(table_units),
        "selected_text_unit_count": sum(unit.source_type == "text" for unit in selected),
        "selected_table_unit_count": sum(unit.source_type == "table" for unit in selected),
        "selected_unit_count": len(selected),
        "context_chars": len(context),
        "max_context_chars": max_context_chars,
        "text_budget_ratio": text_budget_ratio,
        "selected_pages": sorted({
            (unit.metadata.get("filename", ""), _integer(unit.metadata.get("page_number")))
            for unit in selected
        }),
    }
