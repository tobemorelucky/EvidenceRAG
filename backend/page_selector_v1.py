"""Independent, deterministic page selector for offline evaluation.

The selector only consumes existing retrieval candidates and local structural
metadata. It has no benchmark IDs, company rules, finance metric aliases, or
external model calls, and is not wired into the production pipeline.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%'-]*")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "which", "who", "with", "would",
}
_WEIGHTS = {
    "best_chunk": 0.50,
    "multi_chunk_support": 0.20,
    "title_section_lexical": 0.15,
    "table_structure": 0.10,
    "period_year": 0.05,
}


def _tokens(value: object) -> set[str]:
    return {
        token.casefold().removesuffix("'s")
        for token in _TOKEN_RE.findall(str(value or ""))
        if len(token) > 1 and token.casefold().removesuffix("'s") not in _STOPWORDS
    }


def _float(value: object) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _filename(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]


def _page_key(item: dict) -> tuple[str, object]:
    page_id = str(item.get("page_id") or "").strip()
    if page_id:
        return "page_id", page_id
    document_id = str(item.get("document_id") or "").strip()
    if document_id:
        return document_id, _int(item.get("page_number"))
    return _filename(item.get("filename") or item.get("doc_name")).casefold(), _int(item.get("page_number"))


def _fallback_page_key(item: dict) -> tuple[str, int]:
    return _filename(item.get("filename") or item.get("doc_name")).casefold(), _int(item.get("page_number"))


def _chunk_id(item: dict) -> str:
    return str(item.get("chunk_id") or item.get("id") or "").strip()


def _coverage(query_terms: set[str], value: object) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & _tokens(value)) / len(query_terms)


def _heading_text(page: dict) -> str:
    explicit = "\n".join(str(page.get(name) or "") for name in ("title", "section", "section_title", "heading"))
    lines = [line.strip() for line in str(page.get("page_text") or page.get("text") or "").splitlines() if line.strip()]
    return f"{explicit}\n{' '.join(lines[:12])}".strip()


def _row_label(row: object) -> str:
    if isinstance(row, dict):
        for name in ("row_label", "label", "name", "title"):
            if str(row.get(name) or "").strip():
                return str(row[name])
        cells = row.get("cells") or row.get("values")
        if isinstance(cells, list) and cells:
            first = cells[0]
            return str(first.get("text") if isinstance(first, dict) else first)
        return ""
    if isinstance(row, list) and row:
        first = row[0]
        return str(first.get("text") if isinstance(first, dict) else first)
    return str(row or "")


def _table_description(table: dict) -> str:
    columns = table.get("columns") if isinstance(table.get("columns"), list) else []
    rows = table.get("rows") if isinstance(table.get("rows"), list) else []
    labels = [_row_label(row) for row in rows]
    return "\n".join([
        str(table.get("title") or table.get("caption") or ""),
        " ".join(str(item) for item in columns),
        "\n".join(label for label in labels if label),
        str(table.get("unit") or ""),
        str(table.get("scale") or ""),
    ])


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def select_pages_v1(
    question: str,
    chunk_candidates: list[dict],
    *,
    page_records: list[dict],
    table_metadata: list[dict] | None = None,
    top_k: int = 8,
) -> tuple[list[dict], dict[str, Any]]:
    """Rank existing candidate pages with generic structural signals."""
    if top_k <= 0 or not page_records:
        return [], {"selector": "page_selector_v1", "weights": dict(_WEIGHTS), "page_scores": [], "selected_pages": []}

    chunks_by_id = {_chunk_id(item): item for item in chunk_candidates if _chunk_id(item)}
    chunks_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for chunk in chunk_candidates:
        chunks_by_page[_fallback_page_key(chunk)].append(chunk)

    tables_by_page: dict[tuple[str, object], list[dict]] = defaultdict(list)
    tables_by_fallback: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for table in table_metadata or []:
        tables_by_page[_page_key(table)].append(table)
        tables_by_fallback[_fallback_page_key(table)].append(table)

    query_terms = _tokens(question)
    query_years = set(_YEAR_RE.findall(question))
    pages: list[dict] = []
    raw_best_scores: list[float] = []
    for upstream_rank, page in enumerate(page_records, 1):
        supports: list[tuple[dict, int, int]] = []
        for source in page.get("expanded_from") or []:
            chunk = chunks_by_id.get(str(source.get("chunk_id") or "")) or source
            supports.append((chunk, _int(source.get("merged_rank")) or len(chunk_candidates) + 1, _int(source.get("distance"))))
        if not supports:
            for chunk in chunks_by_page.get(_fallback_page_key(page), []):
                supports.append((chunk, _int(chunk.get("merged_rank")) or upstream_rank, 0))
        if not supports and isinstance(page.get("original_chunk"), dict):
            supports.append((page["original_chunk"], _int(page.get("best_seed_rank")) or upstream_rank, _int(page.get("neighbor_distance"))))

        support_values = []
        direct_support_ids = set()
        for chunk, rank, distance in supports:
            # Route-local scores restart at one and are not comparable across
            # documents. The final merged rank is the common score contract.
            rank_score = 1.0 / math.log2(max(1, rank) + 2.0)
            support_values.append(rank_score / (1.0 + 0.35 * max(0, distance)))
            if distance == 0:
                direct_support_ids.add(_chunk_id(chunk) or f"rank:{rank}")
        raw_best = max(support_values, default=0.0)
        raw_best_scores.append(raw_best)
        page_tables = tables_by_page.get(_page_key(page)) or tables_by_fallback.get(_fallback_page_key(page), [])
        table_matches = []
        for table in page_tables:
            columns = table.get("columns") if isinstance(table.get("columns"), list) else []
            rows = table.get("rows") if isinstance(table.get("rows"), list) else []
            structural_completeness = (bool(columns) + bool(rows)) / 2.0
            table_matches.append(_coverage(query_terms, _table_description(table)) * structural_completeness)
        page_years = set(_YEAR_RE.findall("\n".join([
            str(page.get("page_text") or page.get("text") or ""),
            str(page.get("page_years") or ""),
            "\n".join(_table_description(table) for table in page_tables),
        ])))
        pages.append({
            **page,
            "_raw_best_chunk": raw_best,
            "_support_count": len(direct_support_ids),
            "_title_section_lexical": _coverage(query_terms, _heading_text(page)),
            "_table_structure": max(table_matches, default=0.0),
            "_period_year": len(query_years & page_years) / len(query_years) if query_years else 0.0,
            "_table_count": len(page_tables),
            "_upstream_rank": _int(page.get("page_candidate_rank")) or upstream_rank,
        })

    normalized_best = _normalize(raw_best_scores)
    for page, best_chunk_score in zip(pages, normalized_best):
        multi_support = (
            min(1.0, math.log2(page["_support_count"]) / math.log2(5.0))
            if page["_support_count"] > 1 else 0.0
        )
        components = {
            "best_chunk": best_chunk_score,
            "multi_chunk_support": multi_support,
            "title_section_lexical": page["_title_section_lexical"],
            "table_structure": page["_table_structure"],
            "period_year": page["_period_year"],
        }
        contributions = {name: round(_WEIGHTS[name] * value, 8) for name, value in components.items()}
        page["page_selector_v1_score"] = round(sum(contributions.values()), 8)
        page["page_selector_v1_components"] = {name: round(value, 8) for name, value in components.items()}
        page["page_selector_v1_contributions"] = contributions

    pages.sort(key=lambda item: (-item["page_selector_v1_score"], item["_upstream_rank"], _fallback_page_key(item)))
    for rank, page in enumerate(pages, 1):
        page["page_selector_v1_rank"] = rank
    selected = pages[:top_k]
    snapshots = []
    for page in pages:
        snapshots.append({
            "document_id": page.get("document_id"),
            "page_id": page.get("page_id"),
            "filename": page.get("filename"),
            "page_number": page.get("page_number"),
            "rank": page["page_selector_v1_rank"],
            "score": page["page_selector_v1_score"],
            "components": page["page_selector_v1_components"],
            "contributions": page["page_selector_v1_contributions"],
            "supporting_chunks": page["_support_count"],
            "table_count": page["_table_count"],
            "upstream_page_rank": page["_upstream_rank"],
        })
    return selected, {
        "selector": "page_selector_v1",
        "weights": dict(_WEIGHTS),
        "candidate_page_count": len(pages),
        "page_scores": snapshots,
        "selected_pages": snapshots[:top_k],
    }
