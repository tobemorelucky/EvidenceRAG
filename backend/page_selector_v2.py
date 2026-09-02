"""Shadow evidence-group selector with deterministic coverage selection.

This module is intentionally disconnected from production orchestration. It
uses no benchmark identifiers, company rules, metric aliases, or external
model calls.
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
_COVERAGE_CATEGORIES = ("query_token", "title", "table_header", "year", "multi_chunk")


def _tokens(value: object) -> set[str]:
    return {
        token.casefold().removesuffix("'s")
        for token in _TOKEN_RE.findall(str(value or ""))
        if len(token) > 1 and token.casefold().removesuffix("'s") not in _STOPWORDS
    }


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _filename(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]


def _document_key(item: dict) -> str:
    return str(item.get("document_id") or "").strip() or _filename(
        item.get("filename") or item.get("doc_name")
    ).casefold()


def _page_key(item: dict) -> tuple[str, int]:
    return _document_key(item), _int(item.get("page_number"))


def _fallback_page_key(item: dict) -> tuple[str, int]:
    return _filename(item.get("filename") or item.get("doc_name")).casefold(), _int(item.get("page_number"))


def _chunk_id(item: dict) -> str:
    return str(item.get("chunk_id") or item.get("id") or "").strip()


def _row_label(row: object) -> str:
    if isinstance(row, dict):
        for name in ("row_label", "label", "name", "title"):
            if str(row.get(name) or "").strip():
                return str(row[name])
        values = row.get("cells") or row.get("values")
        if isinstance(values, list) and values:
            first = values[0]
            return str(first.get("text") if isinstance(first, dict) else first)
        return ""
    if isinstance(row, list) and row:
        first = row[0]
        return str(first.get("text") if isinstance(first, dict) else first)
    return str(row or "")


def _table_structure_text(table: dict) -> str:
    columns = table.get("columns") if isinstance(table.get("columns"), list) else []
    rows = table.get("rows") if isinstance(table.get("rows"), list) else []
    return "\n".join([
        str(table.get("title") or table.get("caption") or ""),
        " ".join(str(column) for column in columns),
        "\n".join(label for label in (_row_label(row) for row in rows) if label),
        str(table.get("unit") or ""),
        str(table.get("scale") or ""),
    ])


def _title_text(page: dict) -> str:
    explicit = "\n".join(str(page.get(name) or "") for name in ("title", "section", "section_title", "heading"))
    lines = [line.strip() for line in str(page.get("page_text") or page.get("text") or "").splitlines() if line.strip()]
    return f"{explicit}\n{' '.join(lines[:12])}".strip()


def _coverage_score(coverage: dict[str, set[str]], universe: dict[str, set[str]]) -> float:
    score = 0.0
    for category in _COVERAGE_CATEGORIES:
        available = universe[category]
        if available:
            score += len(coverage[category] & available) / len(available)
    return score


def _snapshot_coverage(coverage: dict[str, set[str]]) -> dict[str, list[str]]:
    return {category: sorted(coverage[category]) for category in _COVERAGE_CATEGORIES}


def _related(left: dict, right: dict) -> bool:
    if left["document_key"] != right["document_key"] or abs(left["page_number"] - right["page_number"]) != 1:
        return False
    if left["source_chunk_ids"] & right["source_chunk_ids"]:
        return True
    for category in ("query_token", "title", "table_header", "year"):
        if left["coverage"][category] & right["coverage"][category]:
            return True
    return False


def _build_page_features(
    question: str,
    chunk_candidates: list[dict],
    page_records: list[dict],
    table_metadata: list[dict],
) -> tuple[list[dict], dict[str, set[str]]]:
    query_terms = _tokens(question)
    query_years = set(_YEAR_RE.findall(question))
    chunks_by_id = {_chunk_id(item): item for item in chunk_candidates if _chunk_id(item)}
    chunks_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for chunk in chunk_candidates:
        chunks_by_page[_fallback_page_key(chunk)].append(chunk)
    tables_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    tables_by_fallback: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for table in table_metadata:
        tables_by_page[_page_key(table)].append(table)
        tables_by_fallback[_fallback_page_key(table)].append(table)

    pages = []
    for upstream_rank, page in enumerate(page_records, 1):
        sources = list(page.get("expanded_from") or [])
        if not sources:
            sources = chunks_by_page.get(_fallback_page_key(page), [])
        source_chunk_ids = {
            str(source.get("chunk_id") or source.get("id") or "").strip()
            for source in sources if str(source.get("chunk_id") or source.get("id") or "").strip()
        }
        direct_chunk_ids = {
            str(source.get("chunk_id") or source.get("id") or "").strip()
            for source in sources
            if _int(source.get("distance")) == 0 and str(source.get("chunk_id") or source.get("id") or "").strip()
        }
        ranks = []
        for source in sources:
            chunk = chunks_by_id.get(str(source.get("chunk_id") or "")) or source
            ranks.append(_int(chunk.get("merged_rank")) or _int(source.get("merged_rank")) or len(chunk_candidates) + 1)
        page_tables = tables_by_page.get(_page_key(page)) or tables_by_fallback.get(_fallback_page_key(page), [])
        page_text = str(page.get("page_text") or page.get("text") or "")
        title_text = _title_text(page)
        table_text = "\n".join(_table_structure_text(table) for table in page_tables)
        years = set(_YEAR_RE.findall(f"{page_text}\n{table_text}\n{page.get('page_years') or ''}"))
        coverage = {
            "query_token": query_terms & _tokens(page_text),
            "title": query_terms & _tokens(title_text),
            "table_header": query_terms & _tokens(table_text),
            "year": query_years & years,
            "multi_chunk": {"multi_chunk"} if len(direct_chunk_ids) > 1 else set(),
        }
        pages.append({
            "record": page,
            "document_key": _document_key(page),
            "page_number": _int(page.get("page_number")),
            "page_key": _page_key(page),
            "source_chunk_ids": source_chunk_ids,
            "direct_chunk_ids": direct_chunk_ids,
            "best_chunk_rank": min(ranks, default=len(chunk_candidates) + 1),
            "upstream_rank": _int(page.get("page_candidate_rank")) or upstream_rank,
            "coverage": coverage,
            "table_count": len(page_tables),
        })
    universe = {
        "query_token": set(query_terms),
        "title": set(query_terms),
        "table_header": set(query_terms),
        "year": set(query_years),
        "multi_chunk": {"multi_chunk"},
    }
    return pages, universe


def build_evidence_groups(
    question: str,
    chunk_candidates: list[dict],
    *,
    page_records: list[dict],
    table_metadata: list[dict] | None = None,
) -> tuple[list[dict], dict[str, set[str]]]:
    """Build singleton and related adjacent-page groups from existing pages."""
    pages, universe = _build_page_features(question, chunk_candidates, page_records, table_metadata or [])
    by_document: dict[str, list[dict]] = defaultdict(list)
    for page in pages:
        by_document[page["document_key"]].append(page)
    groups: dict[tuple[tuple[str, int], ...], dict] = {}
    for document_pages in by_document.values():
        ordered = sorted(document_pages, key=lambda item: (item["page_number"], item["upstream_rank"]))
        for index, anchor in enumerate(ordered):
            members = [anchor]
            if index > 0 and _related(anchor, ordered[index - 1]):
                members.insert(0, ordered[index - 1])
            if index + 1 < len(ordered) and _related(anchor, ordered[index + 1]):
                members.append(ordered[index + 1])
            for candidate_members in ([anchor], members):
                unique = {item["page_key"]: item for item in candidate_members}
                candidate_members = sorted(unique.values(), key=lambda item: item["page_number"])
                group_key = tuple(item["page_key"] for item in candidate_members)
                if group_key in groups:
                    continue
                coverage = {category: set() for category in _COVERAGE_CATEGORIES}
                for member in candidate_members:
                    for category in _COVERAGE_CATEGORIES:
                        coverage[category].update(member["coverage"][category])
                document_key = candidate_members[0]["document_key"]
                page_numbers = [item["page_number"] for item in candidate_members]
                groups[group_key] = {
                    "group_id": f"{document_key}:{page_numbers[0]}-{page_numbers[-1]}",
                    "document_id": candidate_members[0]["record"].get("document_id"),
                    "document_key": document_key,
                    "pages": candidate_members,
                    "page_keys": set(group_key),
                    "coverage": coverage,
                    "best_chunk_rank": min(item["best_chunk_rank"] for item in candidate_members),
                    "chunk_support": len(set().union(*(item["direct_chunk_ids"] for item in candidate_members))),
                    "upstream_rank": min(item["upstream_rank"] for item in candidate_members),
                    "table_count": sum(item["table_count"] for item in candidate_members),
                }
    return list(groups.values()), universe


def select_page_groups_v2(
    question: str,
    chunk_candidates: list[dict],
    *,
    page_records: list[dict],
    table_metadata: list[dict] | None = None,
    page_budget: int = 8,
) -> tuple[list[dict], dict[str, Any]]:
    """Greedily select groups that add the most previously unseen coverage."""
    if page_budget <= 0 or not page_records:
        return [], {"selector": "page_selector_v2", "evidence_groups": [], "selected_groups": [], "selected_pages": []}
    groups, universe = build_evidence_groups(
        question, chunk_candidates, page_records=page_records, table_metadata=table_metadata,
    )
    selected_groups = []
    selected_page_keys: set[tuple[str, int]] = set()
    covered = {category: set() for category in _COVERAGE_CATEGORIES}
    selection_step = 0
    while len(selected_page_keys) < page_budget:
        eligible = [
            group for group in groups
            if not (group["page_keys"] & selected_page_keys)
            and len(group["page_keys"]) <= page_budget - len(selected_page_keys)
        ]
        if not eligible:
            break
        ranked = []
        for group in eligible:
            new_coverage = {
                category: group["coverage"][category] - covered[category]
                for category in _COVERAGE_CATEGORIES
            }
            gain = _coverage_score(new_coverage, universe)
            total = _coverage_score(group["coverage"], universe)
            ranked.append((
                gain,
                total,
                -group["best_chunk_rank"],
                group["chunk_support"],
                -len(group["page_keys"]),
                -group["upstream_rank"],
                group["group_id"],
                group,
                new_coverage,
            ))
        *_, winner, new_coverage = max(ranked, key=lambda item: item[:-2])
        selection_step += 1
        winner = {**winner, "selection_step": selection_step, "coverage_gain": _coverage_score(new_coverage, universe), "new_coverage": new_coverage}
        selected_groups.append(winner)
        selected_page_keys.update(winner["page_keys"])
        for category in _COVERAGE_CATEGORIES:
            covered[category].update(winner["coverage"][category])

    selected_pages = []
    for group in selected_groups:
        for page in group["pages"]:
            record = dict(page["record"])
            record["page_selector_v2_group_id"] = group["group_id"]
            record["page_selector_v2_step"] = group["selection_step"]
            selected_pages.append(record)
    group_snapshots = []
    selected_ids = {group["group_id"] for group in selected_groups}
    for group in sorted(groups, key=lambda item: (item["best_chunk_rank"], item["upstream_rank"], item["group_id"])):
        group_snapshots.append({
            "group_id": group["group_id"],
            "document_id": group["document_id"],
            "pages": [
                {
                    "page_id": page["record"].get("page_id"),
                    "filename": page["record"].get("filename"),
                    "page_number": page["page_number"],
                }
                for page in group["pages"]
            ],
            "coverage": _snapshot_coverage(group["coverage"]),
            "best_chunk_rank": group["best_chunk_rank"],
            "chunk_support": group["chunk_support"],
            "table_count": group["table_count"],
            "upstream_rank": group["upstream_rank"],
            "selected": group["group_id"] in selected_ids,
        })
    selected_snapshots = [
        {
            "group_id": group["group_id"],
            "selection_step": group["selection_step"],
            "coverage_gain": round(group["coverage_gain"], 8),
            "new_coverage": _snapshot_coverage(group["new_coverage"]),
            "pages": [
                {
                    "page_id": page["record"].get("page_id"),
                    "filename": page["record"].get("filename"),
                    "page_number": page["page_number"],
                }
                for page in group["pages"]
            ],
        }
        for group in selected_groups
    ]
    return selected_pages[:page_budget], {
        "selector": "page_selector_v2",
        "page_budget": page_budget,
        "coverage_universe": _snapshot_coverage(universe),
        "final_coverage": _snapshot_coverage(covered),
        "evidence_groups": group_snapshots,
        "selected_groups": selected_snapshots,
        "selected_pages": [page for group in selected_snapshots for page in group["pages"]][:page_budget],
    }
