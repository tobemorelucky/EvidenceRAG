"""Page-oriented evidence selection for the isolated RAG Core v2 profiles.

This module operates only on candidates produced by the existing hybrid/RRF/Jina
pipeline.  It does not issue retrieval or model calls and contains no benchmark,
company, or finance-metric rules.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from typing import Iterable


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%'-]*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "which", "who", "with", "would",
}


def _terms(text: object) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN.findall(str(text or ""))
        if len(token) > 1 and token.lower() not in _STOPWORDS
    }


def _float(value: object) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _page_key(item: dict) -> tuple[str, int]:
    try:
        page = int(item.get("page_number") or 0)
    except (TypeError, ValueError):
        page = 0
    return str(item.get("filename") or "").strip(), page


def _doc_key(item: dict) -> str:
    return str(item.get("filename") or "").strip()


def _chunk_key(item: dict) -> tuple:
    return (
        str(item.get("chunk_id") or ""),
        *_page_key(item),
        str(item.get("text") or "")[:160],
    )


def select_core_v2_pages(
    question: str,
    candidate_chunks: list[dict],
    reranked_chunks: list[dict],
    *,
    page_records: list[dict] | None = None,
    document_top_k: int | None = None,
    page_top_k: int | None = None,
    global_escape_pages: int | None = None,
) -> tuple[list[dict], dict]:
    """Aggregate chunk evidence into soft document and page selections."""
    document_top_k = max(1, document_top_k or int(os.getenv("RAG_CORE_V2_DOCUMENT_TOP_K", "4")))
    page_top_k = max(1, page_top_k or int(os.getenv("RAG_CORE_V2_PAGE_POOL_K", "10")))
    global_escape_pages = max(
        0, global_escape_pages if global_escape_pages is not None
        else int(os.getenv("RAG_CORE_V2_GLOBAL_ESCAPE_PAGES", "2")),
    )
    query_terms = _terms(question)
    merged: dict[tuple, dict] = {}
    for rank, chunk in enumerate(candidate_chunks, 1):
        key = _chunk_key(chunk)
        merged[key] = {**chunk, "core_candidate_rank": rank}
    for rank, chunk in enumerate(reranked_chunks, 1):
        key = _chunk_key(chunk)
        merged[key] = {
            **merged.get(key, {}),
            **chunk,
            "core_rerank_rank": rank,
        }

    pages: dict[tuple[str, int], dict] = {}
    for chunk in merged.values():
        filename, page_number = _page_key(chunk)
        if not filename:
            continue
        candidate_rank = int(chunk.get("core_candidate_rank") or len(candidate_chunks) + 1)
        rerank_rank = int(chunk.get("core_rerank_rank") or 0)
        text_terms = _terms(chunk.get("text"))
        lexical = len(query_terms & text_terms) / max(1, len(query_terms))
        contribution = 1.0 / (20.0 + candidate_rank)
        if rerank_rank:
            contribution += 0.45 / rerank_rank
        contribution += 0.55 * max(0.0, _float(chunk.get("rerank_score")))
        contribution += 0.08 * lexical
        page = pages.setdefault(
            (filename, page_number),
            {
                "filename": filename,
                "page_number": page_number,
                "page_score": 0.0,
                "best_chunk": chunk,
                "chunk_count": 0,
            },
        )
        page["page_score"] += contribution
        page["chunk_count"] += 1
        if contribution > _float(page.get("best_chunk_score")):
            page["best_chunk"] = chunk
            page["best_chunk_score"] = contribution

    ranked_pages = sorted(
        pages.values(),
        key=lambda item: (-_float(item.get("page_score")), item["filename"].lower(), item["page_number"]),
    )
    document_pages: dict[str, list[dict]] = defaultdict(list)
    for page in ranked_pages:
        document_pages[page["filename"]].append(page)
    document_scores = []
    for filename, doc_pages in document_pages.items():
        top_scores = [_float(page["page_score"]) for page in doc_pages[:3]]
        document_scores.append({
            "filename": filename,
            "document_score": sum(top_scores),
            "matched_page_count": len(doc_pages),
            "best_page": doc_pages[0]["page_number"],
        })
    document_scores.sort(key=lambda item: (-item["document_score"], item["filename"].lower()))
    selected_documents = [item["filename"] for item in document_scores[:document_top_k]]

    page_rescore_count = 0
    if page_records:
        for record in page_records:
            key = _page_key(record)
            page = pages.get(key)
            if not page or key[0] not in selected_documents:
                continue
            page_terms = _terms(record.get("page_text") or record.get("text"))
            lexical = len(query_terms & page_terms) / max(1, len(query_terms))
            page["page_store_lexical_score"] = lexical
            page["page_score"] += 0.35 * lexical
            page_rescore_count += 1
        ranked_pages = sorted(
            pages.values(),
            key=lambda item: (-_float(item.get("page_score")), item["filename"].lower(), item["page_number"]),
        )

    selected: list[dict] = []
    selected_keys: set[tuple[str, int]] = set()
    main_slots = max(1, page_top_k - global_escape_pages)
    for page in ranked_pages:
        if page["filename"] not in selected_documents:
            continue
        selected.append(page)
        selected_keys.add(_page_key(page))
        if len(selected) >= main_slots:
            break
    escape = []
    final_page_k = max(1, int(os.getenv("RAG_CORE_V2_FINAL_PAGE_K", "6")))
    context_leader_slots = max(1, min(main_slots, final_page_k - min(global_escape_pages, final_page_k)))
    context_leader_keys = {_page_key(page) for page in selected[:context_leader_slots]}
    for page in ranked_pages:
        key = _page_key(page)
        if key in context_leader_keys:
            continue
        escape.append(page)
        if key not in selected_keys:
            selected.append(page)
            selected_keys.add(key)
        if len(escape) >= global_escape_pages:
            break
    if len(selected) < page_top_k:
        for page in ranked_pages:
            key = _page_key(page)
            if key in selected_keys:
                continue
            selected.append(page)
            selected_keys.add(key)
            if len(selected) >= page_top_k:
                break

    for rank, page in enumerate(selected, 1):
        page["selected_rank"] = rank
    trace = {
        "document_scores": document_scores,
        "selected_documents": selected_documents,
        "page_scores": [
            {
                "filename": page["filename"],
                "page_number": page["page_number"],
                "page_score": round(_float(page["page_score"]), 8),
                "chunk_count": page["chunk_count"],
            }
            for page in ranked_pages[: max(12, page_top_k)]
        ],
        "selected_pages": [
            {
                "filename": page["filename"],
                "page_number": page["page_number"],
                "page_score": round(_float(page["page_score"]), 8),
            }
            for page in selected
        ],
        "global_escape_pages": [
            {"filename": page["filename"], "page_number": page["page_number"]}
            for page in escape
        ],
        "page_store_rescore_method": "candidate_page_full_text_token_overlap" if page_rescore_count else "none",
        "page_store_rescored_pages": page_rescore_count,
    }
    return selected, trace


def choose_core_v2_context_pages(
    selected_pages: list[dict],
    global_escape_pages: list[dict],
    *,
    final_page_k: int | None = None,
) -> list[dict]:
    """Keep soft-document leaders plus globally scored escape pages."""
    final_page_k = max(1, final_page_k or int(os.getenv("RAG_CORE_V2_FINAL_PAGE_K", "6")))
    escape_keys = {_page_key(page) for page in global_escape_pages}
    escape = [page for page in selected_pages if _page_key(page) in escape_keys]
    leaders = [page for page in selected_pages if _page_key(page) not in escape_keys]
    leader_slots = max(0, final_page_k - min(len(escape), final_page_k))
    return (leaders[:leader_slots] + escape[:final_page_k])[:final_page_k]


def merge_opened_pages(selected_pages: list[dict], opened_pages: list[dict]) -> list[dict]:
    opened = {_page_key(page): page for page in opened_pages}
    return [{**page, **opened.get(_page_key(page), {})} for page in selected_pages]


def _continuous_window(text: str, anchor: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text.strip()
    needle = (anchor or "").strip()[:180]
    position = text.find(needle) if needle else -1
    if position < 0 and needle:
        position = text.lower().find(needle.lower())
    if position < 0:
        position = 0
    start = max(0, position - max_chars // 4)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    if start:
        newline = text.find("\n", start, min(end, start + 240))
        if newline >= 0:
            start = newline + 1
    if end < len(text):
        newline = text.rfind("\n", max(start, end - 240), end)
        if newline > start:
            end = newline
    prefix = "[… contiguous page text …]\n" if start else ""
    suffix = "\n[… contiguous page text …]" if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


def _row_text(row: object, columns: list) -> str:
    if isinstance(row, dict):
        ordered = []
        seen = set()
        for column in columns:
            key = str(column)
            if key in row:
                ordered.append(f"{key}: {row[key]}")
                seen.add(key)
        ordered.extend(f"{key}: {value}" for key, value in row.items() if str(key) not in seen)
        return " | ".join(ordered)
    if isinstance(row, (list, tuple)):
        return " | ".join(str(value) for value in row)
    return str(row or "")


def _format_relevant_table(question: str, table: dict, max_rows: int = 7) -> tuple[str, int, float]:
    query_terms = _terms(question)
    columns = list(table.get("columns") or [])
    rows = list(table.get("rows") or [])
    table_header = " ".join(
        str(table.get(key) or "") for key in ("title", "caption", "before_context")
    ) + " " + " ".join(str(item) for item in columns)
    header_overlap = len(query_terms & _terms(table_header)) / max(1, len(query_terms))
    scored = []
    for index, row in enumerate(rows):
        text = _row_text(row, columns)
        overlap = len(query_terms & _terms(text)) / max(1, len(query_terms))
        scored.append((overlap, index, text))
    if len(rows) <= max_rows:
        chosen = scored
    else:
        relevant = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_rows]
        chosen = sorted(relevant, key=lambda item: item[1])
    row_overlap = max((item[0] for item in scored), default=0.0)
    score = header_overlap + row_overlap
    title = str(table.get("title") or table.get("caption") or "Structured table").strip()
    parts = [f"Table: {title}"]
    before = str(table.get("before_context") or "").strip()
    if before:
        parts.append(f"Context/Unit: {before[:500]}")
    if columns:
        parts.append("Columns: " + " | ".join(str(item) for item in columns))
    parts.extend(item[2] for item in chosen if item[2].strip())
    return "\n".join(parts), len(chosen), score


def build_core_v2_evidence(
    question: str,
    selected_pages: list[dict],
    tables: Iterable[dict],
    *,
    max_context_chars: int | None = None,
    max_table_chars: int | None = None,
) -> tuple[str, dict]:
    """Build contiguous page evidence and attach structured same-page tables."""
    max_context_chars = max(4000, max_context_chars or int(os.getenv("RAG_CORE_V2_MAX_CONTEXT_CHARS", "28000")))
    max_table_chars = max(0, max_table_chars if max_table_chars is not None else int(os.getenv("RAG_CORE_V2_MAX_TABLE_CHARS", "5000")))
    tables_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for table in tables:
        tables_by_page[_page_key(table)].append(table)

    blocks = []
    used_chars = 0
    original_chars = 0
    attached_ids = []
    attached_rows = 0
    table_chars = 0
    tables_available = 0
    attach_reasons = []
    for index, page in enumerate(selected_pages):
        filename, page_number = _page_key(page)
        page_text = str(page.get("page_text") or page.get("text") or "").strip()
        if not page_text:
            continue
        original_chars += len(page_text)
        anchor = str((page.get("best_chunk") or {}).get("text") or "")
        page_limit = 7000 if index < 2 else 3800
        body = _continuous_window(page_text, anchor, page_limit)
        header = f"Source: {filename} | Page: {page_number}\n"
        page_tables = tables_by_page.get((filename, page_number), [])
        tables_available += len(page_tables)
        formatted_tables = []
        if page_tables and table_chars < max_table_chars:
            ranked_tables = []
            for table in page_tables:
                formatted, row_count, score = _format_relevant_table(question, table)
                ranked_tables.append((score, formatted, row_count, table))
            ranked_tables.sort(key=lambda item: item[0], reverse=True)
            for score, formatted, row_count, table in ranked_tables[:1]:
                remaining_table = max_table_chars - table_chars
                if remaining_table < 250:
                    break
                formatted = formatted[:remaining_table]
                formatted_tables.append(formatted)
                table_chars += len(formatted)
                attached_rows += row_count
                attached_ids.append(str(table.get("table_id") or ""))
                attach_reasons.append({
                    "table_id": str(table.get("table_id") or ""),
                    "filename": filename,
                    "page_number": page_number,
                    "reason": "same_selected_page_query_rows" if score > 0 else "same_selected_page_small_table",
                })
        table_block = ""
        if formatted_tables:
            table_block = "\n\nStructured table from the same page:\n" + "\n\n".join(formatted_tables)
        block = f"{header}{body}{table_block}"
        remaining = max_context_chars - used_chars
        if remaining < 300:
            break
        if len(block) > remaining:
            block = block[:remaining].rstrip()
        blocks.append(block)
        used_chars += len(block) + (7 if len(blocks) > 1 else 0)

    evidence = "\n\n---\n\n".join(blocks)
    return evidence, {
        "answer_context_strategy": "rag_core_v2_contiguous_pages",
        "answer_context_compressed": False,
        "answer_context_original_chars": original_chars,
        "answer_context_chars": len(evidence),
        "answer_context_unit_count": len(blocks),
        "answer_context_pages": [
            {"filename": _page_key(page)[0], "page_number": _page_key(page)[1]}
            for page in selected_pages[: len(blocks)]
        ],
        "tables_available_on_selected_pages": tables_available,
        "tables_attached": len(attached_ids),
        "table_ids": attached_ids,
        "table_rows_attached": attached_rows,
        "table_context_chars": table_chars,
        "table_attach_reason": attach_reasons,
        "answer_context_task_rules_used": False,
    }
