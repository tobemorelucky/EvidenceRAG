"""Generic evidence-flow selection for the isolated RAG Core v3 profiles.

The selector consumes existing hybrid/RRF/rerank candidates.  It contains no
task classifier, finance metric registry, benchmark IDs, company rules, or gold
data.  Selection is based on relevance, explicit query coverage, document/page
structure, and redundancy only.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from typing import Iterable


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%'-]*")
_EXPLICIT = re.compile(r"\b(?:FY\s*)?(?:19|20)\d{2}\b|\bQ[1-4]\b|\b\d+(?:\.\d+)?%?\b", re.I)
_CAPITALIZED = re.compile(r"\b[A-Z][A-Za-z&.']{2,}\b")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "which", "who", "with", "would",
}
_ENTITY_EXCLUSIONS = {
    "among", "calculate", "does", "fiscal", "for", "from", "has", "how", "if",
    "roughly", "the", "what", "when", "which", "why", "would", "fy", "usd",
}


def _terms(text: object) -> set[str]:
    result = set()
    for token in _TOKEN.findall(str(text or "")):
        normalized = token.casefold().removesuffix("'s")
        if len(normalized) > 1 and normalized not in _STOPWORDS:
            result.add(normalized)
        match = re.fullmatch(r"fy((?:19|20)\d{2})", normalized)
        if match:
            result.add(match.group(1))
    return result


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


def _chunk_key(item: dict) -> tuple:
    return (
        str(item.get("chunk_id") or ""),
        *_page_key(item),
        str(item.get("text") or "")[:160],
    )


def merge_core_v3_candidate_routes(
    global_candidates: list[dict],
    scoped_routes: list[tuple[str, list[dict]]],
    *,
    rrf_k: int = 60,
) -> list[dict]:
    """Fuse global and document-local results while preserving their provenance."""
    fused: dict[tuple, dict] = {}
    routes = [("global", global_candidates), *scoped_routes]
    for source, documents in routes:
        route_weight = 1.0 if source == "global" else 1.25
        for rank, document in enumerate(documents, 1):
            key = _chunk_key(document)
            item = fused.setdefault(key, {
                **document,
                "candidate_sources": [],
                "candidate_source_ranks": {},
                "core_v3_rrf_score": 0.0,
            })
            item.update({key: value for key, value in document.items() if value not in (None, "")})
            if source not in item["candidate_sources"]:
                item["candidate_sources"].append(source)
            item["candidate_source_ranks"][source] = rank
            item["core_v3_rrf_score"] += route_weight / (max(1, rrf_k) + rank)
    result = sorted(
        fused.values(),
        key=lambda item: (
            -_float(item["core_v3_rrf_score"]),
            min(item["candidate_source_ranks"].values()),
            _page_key(item),
        ),
    )
    for rank, item in enumerate(result, 1):
        item["score"] = item["core_v3_rrf_score"]
        item["core_v3_merged_rank"] = rank
        item["candidate_source"] = "both" if len(item["candidate_sources"]) > 1 else item["candidate_sources"][0]
    return result


def _explicit_query_signals(question: str) -> set[str]:
    signals = _terms(question)
    signals.update(re.sub(r"\s+", "", item).casefold() for item in _EXPLICIT.findall(question))
    return signals


def _explicit_entity_terms(question: str) -> set[str]:
    return {
        token.casefold().removesuffix("'s")
        for token in _CAPITALIZED.findall(question)
        if token.casefold().removesuffix("'s") not in _ENTITY_EXCLUSIONS
        and not re.fullmatch(r"(?:fy)?(?:19|20)\d{2}|q[1-4]", token, re.I)
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def _page_snapshot(page: dict) -> dict:
    return {
        "filename": page["filename"],
        "page_number": page["page_number"],
        "page_score": round(_float(page.get("page_score")), 8),
        "global_rank": int(page.get("global_rank") or 0),
        "query_coverage": sorted(page.get("query_coverage") or []),
        "chunk_count": int(page.get("chunk_count") or 0),
        "redundancy": round(_float(page.get("selection_redundancy")), 6),
    }


def select_core_v3_pages(
    question: str,
    candidate_chunks: list[dict],
    reranked_chunks: list[dict],
    *,
    page_records: list[dict] | None = None,
    document_top_k: int | None = None,
    page_pool_k: int | None = None,
    final_page_k: int | None = None,
    global_escape_pages: int | None = None,
) -> tuple[list[dict], dict]:
    """Select pages with capped support, explicit coverage, and diversity."""
    document_top_k = max(1, document_top_k or int(os.getenv("RAG_CORE_V3_DOCUMENT_TOP_K", "4")))
    page_pool_k = max(1, page_pool_k or int(os.getenv("RAG_CORE_V3_PAGE_POOL_K", "12")))
    final_page_k = max(1, final_page_k or int(os.getenv("RAG_CORE_V3_FINAL_PAGE_K", "6")))
    global_escape_pages = max(
        0,
        global_escape_pages if global_escape_pages is not None
        else int(os.getenv("RAG_CORE_V3_GLOBAL_ESCAPE_PAGES", "2")),
    )
    escape_slots = min(global_escape_pages, final_page_k)
    query_signals = _explicit_query_signals(question)
    entity_terms = _explicit_entity_terms(question)
    merged: dict[tuple, dict] = {}
    for rank, chunk in enumerate(candidate_chunks, 1):
        merged[_chunk_key(chunk)] = {**chunk, "core_candidate_rank": rank}
    for rank, chunk in enumerate(reranked_chunks, 1):
        key = _chunk_key(chunk)
        merged[key] = {**merged.get(key, {}), **chunk, "core_rerank_rank": rank}

    pages: dict[tuple[str, int], dict] = {}
    for chunk in merged.values():
        filename, page_number = _page_key(chunk)
        if not filename:
            continue
        candidate_rank = int(chunk.get("core_candidate_rank") or len(candidate_chunks) + 1)
        rerank_rank = int(chunk.get("core_rerank_rank") or 0)
        chunk_terms = _terms(chunk.get("text"))
        chunk_terms.update(_terms(chunk.get("filename")))
        chunk_terms.update(_terms(chunk.get("doc_name")))
        chunk_terms.update(_terms(chunk.get("company")))
        lexical = len(query_signals & chunk_terms) / max(1, len(query_signals))
        entity_coverage = len(entity_terms & chunk_terms) / max(1, len(entity_terms)) if entity_terms else 0.0
        strength = 1.0 / (12.0 + candidate_rank)
        if rerank_rank:
            strength += 0.55 / rerank_rank
        strength += 0.70 * max(0.0, _float(chunk.get("rerank_score")))
        strength += 0.20 * lexical
        strength += 0.32 * entity_coverage
        page = pages.setdefault((filename, page_number), {
            "filename": filename,
            "page_number": page_number,
            "best_chunk": chunk,
            "chunk_strengths": [],
            "chunk_count": 0,
            "page_terms": set(),
        })
        page["chunk_strengths"].append(strength)
        page["chunk_count"] += 1
        page["page_terms"].update(chunk_terms)
        if strength > _float(page.get("best_chunk_strength")):
            page["best_chunk"] = chunk
            page["best_chunk_strength"] = strength

    records = {_page_key(record): record for record in (page_records or [])}
    for page in pages.values():
        record = records.get(_page_key(page))
        if record:
            page["page_terms"].update(_terms(record.get("page_text") or record.get("text")))
    for page in pages.values():
        strengths = sorted(page.pop("chunk_strengths"), reverse=True)
        strongest = strengths[0]
        second_support = min(strongest * 0.35, (strengths[1] * 0.50) if len(strengths) > 1 else 0.0)
        additional_support = min(strongest * 0.12, sum(strengths[2:]) * 0.08)
        coverage = query_signals & page["page_terms"]
        page["query_coverage"] = coverage
        page["entity_coverage"] = entity_terms & page["page_terms"]
        page["strongest_chunk_score"] = strongest
        page["capped_multi_chunk_support"] = second_support + additional_support
        page["explicit_query_coverage"] = len(coverage) / max(1, len(query_signals))
        page["page_score"] = (
            strongest + second_support + additional_support
            + 0.24 * page["explicit_query_coverage"]
            + 0.30 * (len(page["entity_coverage"]) / max(1, len(entity_terms)) if entity_terms else 0.0)
        )

    ranked_pages = sorted(
        pages.values(),
        key=lambda item: (-_float(item["page_score"]), item["filename"].casefold(), item["page_number"]),
    )
    for rank, page in enumerate(ranked_pages, 1):
        page["global_rank"] = rank

    by_document: dict[str, list[dict]] = defaultdict(list)
    for page in ranked_pages:
        by_document[page["filename"]].append(page)
    document_scores = []
    for filename, doc_pages in by_document.items():
        scores = [_float(page["page_score"]) for page in doc_pages[:3]]
        score = scores[0] + (0.35 * scores[1] if len(scores) > 1 else 0) + (0.15 * scores[2] if len(scores) > 2 else 0)
        document_scores.append({
            "filename": filename,
            "document_score": score,
            "matched_page_count": len(doc_pages),
            "best_page": doc_pages[0]["page_number"],
        })
    document_scores.sort(key=lambda item: (-item["document_score"], item["filename"].casefold()))
    selected_documents = [item["filename"] for item in document_scores[:document_top_k]]

    eligible = [page for page in ranked_pages if page["filename"] in selected_documents][:page_pool_k]
    main_slots = max(0, final_page_k - escape_slots)
    selected: list[dict] = []
    covered: set[str] = set()
    max_score = max((_float(page["page_score"]) for page in ranked_pages), default=1.0)
    while eligible and len(selected) < main_slots:
        scored = []
        for page in eligible:
            new_coverage = set(page["query_coverage"]) - covered
            redundancy = max(
                (_jaccard(set(page["page_terms"]), set(other["page_terms"])) for other in selected),
                default=0.0,
            )
            new_document = page["filename"] not in {item["filename"] for item in selected}
            adjacent = any(
                page["filename"] == other["filename"]
                and abs(page["page_number"] - other["page_number"]) == 1
                for other in selected
            )
            marginal = (
                (_float(page["page_score"]) / max(0.000001, max_score))
                + 0.32 * (len(new_coverage) / max(1, len(query_signals)))
                + (0.07 if new_document else 0.0)
                + (0.035 if adjacent else 0.0)
                - 0.22 * redundancy
            )
            scored.append((marginal, -int(page["global_rank"]), page, redundancy, new_coverage))
        _, _, winner, redundancy, new_coverage = max(scored, key=lambda item: (item[0], item[1]))
        winner["selection_redundancy"] = redundancy
        winner["selection_new_coverage"] = sorted(new_coverage)
        selected.append(winner)
        covered.update(winner["query_coverage"])
        eligible.remove(winner)

    selected_keys = {_page_key(page) for page in selected}
    escape = []
    for page in ranked_pages:
        if _page_key(page) in selected_keys:
            continue
        escape.append(page)
        selected_keys.add(_page_key(page))
        if len(escape) >= escape_slots:
            break
    selected.extend(escape)
    if len(selected) < final_page_k:
        for page in ranked_pages:
            if _page_key(page) in selected_keys:
                continue
            selected.append(page)
            selected_keys.add(_page_key(page))
            if len(selected) >= final_page_k:
                break
    selected = selected[:final_page_k]
    for rank, page in enumerate(selected, 1):
        page["selected_rank"] = rank

    trace = {
        "page_selector_version": "rag_core_v3_generic_v1",
        "document_scores": [
            {**item, "document_score": round(_float(item["document_score"]), 8)}
            for item in document_scores
        ],
        "selected_documents": selected_documents,
        "page_scores": [_page_snapshot(page) for page in ranked_pages[: max(16, page_pool_k)]],
        "selected_pages": [_page_snapshot(page) for page in selected],
        "global_escape_pages": [
            {
                **_page_snapshot(page),
                "global_escape_candidate_rank": page["global_rank"],
                "global_escape_reason": "global_page_rank_outside_greedy_main_selection",
                "whether_outside_selected_documents": page["filename"] not in selected_documents,
            }
            for page in escape
        ],
        "query_explicit_signals": sorted(query_signals),
        "query_explicit_entity_terms": sorted(entity_terms),
        "selected_page_redundancy": [round(_float(page.get("selection_redundancy")), 6) for page in selected],
        "page_store_rescore_method": "candidate_page_full_text_signals" if page_records else "none",
        "page_store_rescored_pages": sum(_page_key(page) in records for page in ranked_pages),
    }
    return selected, trace


def merge_opened_pages(selected_pages: list[dict], opened_pages: list[dict]) -> list[dict]:
    opened = {_page_key(page): page for page in opened_pages}
    return [{**page, **opened.get(_page_key(page), {})} for page in selected_pages]


def _continuous_window(text: str, anchor: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text.strip()
    needle = (anchor or "").strip()[:180]
    position = text.find(needle) if needle else -1
    if position < 0 and needle:
        position = text.casefold().find(needle.casefold())
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
    content = text[start:end].strip()
    content_limit = max(1, max_chars - len(prefix) - len(suffix))
    if len(content) > content_limit:
        content = content[:content_limit].rstrip()
    return f"{prefix}{content}{suffix}"


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


def _format_relevant_table(question: str, table: dict, max_rows: int) -> tuple[str, int, float]:
    query_terms = _terms(question)
    columns = list(table.get("columns") or [])
    rows = list(table.get("rows") or [])
    header = " ".join(str(table.get(key) or "") for key in ("title", "caption", "before_context"))
    header += " " + " ".join(str(item) for item in columns)
    header_overlap = len(query_terms & _terms(header)) / max(1, len(query_terms))
    scored = []
    for index, row in enumerate(rows):
        text = _row_text(row, columns)
        overlap = len(query_terms & _terms(text)) / max(1, len(query_terms))
        scored.append((overlap, index, text))
    chosen = scored if len(rows) <= max_rows else sorted(
        sorted(scored, key=lambda item: (-item[0], item[1]))[:max_rows], key=lambda item: item[1]
    )
    title = str(table.get("title") or table.get("caption") or "Structured table").strip()
    parts = [f"Table: {title}"]
    before = str(table.get("before_context") or "").strip()
    if before:
        parts.append(f"Context/Unit: {before[:500]}")
    if columns:
        parts.append("Columns: " + " | ".join(str(item) for item in columns))
    parts.extend(item[2] for item in chosen if item[2].strip())
    return "\n".join(parts), len(chosen), header_overlap + max((item[0] for item in scored), default=0.0)


def _best_complete_table(question: str, tables: list[dict], char_limit: int) -> tuple[str, int, dict | None, float]:
    candidates = []
    for table in tables:
        for max_rows in range(7, 0, -1):
            formatted, row_count, score = _format_relevant_table(question, table, max_rows)
            if len(formatted) <= char_limit:
                candidates.append((score, row_count, formatted, table))
                break
    if not candidates:
        return "", 0, None, 0.0
    score, row_count, formatted, table = max(candidates, key=lambda item: (item[0], item[1]))
    return formatted, row_count, table, score


def build_core_v3_evidence(
    question: str,
    selected_pages: list[dict],
    tables: Iterable[dict],
    *,
    max_context_chars: int | None = None,
    max_table_chars: int | None = None,
    min_page_chars: int | None = None,
) -> tuple[str, dict]:
    """Guarantee each selected page a window before distributing extra budget."""
    max_context_chars = max(4000, max_context_chars or int(os.getenv("RAG_CORE_V3_MAX_CONTEXT_CHARS", "28000")))
    max_table_chars = max(0, max_table_chars if max_table_chars is not None else int(os.getenv("RAG_CORE_V3_MAX_TABLE_CHARS", "5000")))
    min_page_chars = max(500, min_page_chars or int(os.getenv("RAG_CORE_V3_MIN_PAGE_CHARS", "2200")))
    tables_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for table in tables:
        tables_by_page[_page_key(table)].append(table)

    prepared = []
    table_budget = max_table_chars
    tables_available = 0
    attached_ids = []
    attached_rows = 0
    attach_reasons = []
    for page in selected_pages:
        filename, page_number = _page_key(page)
        page_text = str(page.get("page_text") or page.get("text") or "").strip()
        if not page_text:
            continue
        page_tables = tables_by_page.get((filename, page_number), [])
        tables_available += len(page_tables)
        table_text, row_count, table, score = _best_complete_table(
            question, page_tables, min(1600, table_budget)
        ) if page_tables and table_budget >= 250 else ("", 0, None, 0.0)
        if table_text:
            table_budget -= len(table_text)
            attached_rows += row_count
            table_id = str((table or {}).get("table_id") or "")
            attached_ids.append(table_id)
            attach_reasons.append({
                "table_id": table_id,
                "filename": filename,
                "page_number": page_number,
                "reason": "same_selected_page_query_rows" if score > 0 else "same_selected_page_small_table",
            })
        prepared.append({
            "page": page,
            "filename": filename,
            "page_number": page_number,
            "page_text": page_text,
            "anchor": str((page.get("best_chunk") or {}).get("text") or ""),
            "table_text": table_text,
        })

    separator_chars = max(0, len(prepared) - 1) * len("\n\n---\n\n")
    fixed_chars = separator_chars + sum(
        len(f"Source: {item['filename']} | Page: {item['page_number']}\n")
        + (len("\n\nStructured table from the same page:\n") + len(item["table_text"]) if item["table_text"] else 0)
        for item in prepared
    )
    body_budget = max(0, max_context_chars - fixed_chars)
    allocations = [min(len(item["page_text"]), min_page_chars) for item in prepared]
    minimum_total = sum(allocations)
    if minimum_total > body_budget and minimum_total:
        scale = body_budget / minimum_total
        allocations = [max(200, int(value * scale)) for value in allocations]
    remaining = max(0, body_budget - sum(allocations))
    desired_caps = [7000 if index < 2 else 4200 for index in range(len(prepared))]
    for index, item in enumerate(prepared):
        if remaining <= 0:
            break
        desired = min(len(item["page_text"]), desired_caps[index])
        addition = min(remaining, max(0, desired - allocations[index]))
        allocations[index] += addition
        remaining -= addition

    blocks = []
    page_allocations = []
    for item, allocation in zip(prepared, allocations):
        body = _continuous_window(item["page_text"], item["anchor"], allocation)
        block = f"Source: {item['filename']} | Page: {item['page_number']}\n{body}"
        if item["table_text"]:
            block += "\n\nStructured table from the same page:\n" + item["table_text"]
        blocks.append(block)
        page_allocations.append({
            "filename": item["filename"],
            "page_number": item["page_number"],
            "body_char_budget": allocation,
            "body_chars": len(body),
            "table_chars": len(item["table_text"]),
        })
    evidence = "\n\n---\n\n".join(blocks)
    if len(evidence) > max_context_chars:
        raise AssertionError("rag_core_v3 context allocation exceeded its fixed budget")
    return evidence, {
        "answer_context_strategy": "rag_core_v3_fair_page_budget",
        "answer_context_compressed": False,
        "answer_context_original_chars": sum(len(item["page_text"]) for item in prepared),
        "answer_context_chars": len(evidence),
        "answer_context_unit_count": len(blocks),
        "answer_context_pages": [
            {"filename": item["filename"], "page_number": item["page_number"]}
            for item in prepared
        ],
        "answer_context_page_allocations": page_allocations,
        "answer_context_min_page_chars": min_page_chars,
        "answer_context_budget_chars": max_context_chars,
        "tables_available_on_selected_pages": tables_available,
        "tables_attached": len(attached_ids),
        "table_ids": attached_ids,
        "table_rows_attached": attached_rows,
        "table_context_chars": sum(len(item["table_text"]) for item in prepared),
        "table_attach_reason": attach_reasons,
        "answer_context_task_rules_used": False,
    }
