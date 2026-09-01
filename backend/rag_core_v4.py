"""Experimental Retrieval Core v4 primitives.

This module is isolated from the production Core v3 path. It contains retrieval
mechanics only; answer prompts, skills, validators, and Context Budget v3 are
not changed here.
"""

from __future__ import annotations

import time
import math
from typing import Any

from document_page_store import DocumentPageStore
from embedding import embedding_service
from milvus_client import MilvusManager
from retrieval_ablation import chunk_key, lexical_terms, page_key


_LEAF_FILTER = '(evidence_type == "text_chunk" or evidence_type == "") and chunk_level == 3'


def merge_dense_primary(
    dense: list[dict],
    bm25: list[dict],
    *,
    dense_k: int = 120,
    bm25_k: int = 30,
) -> list[dict]:
    """Keep Dense order and append only BM25 results absent from Dense."""
    dense_slice = list(dense[: max(0, dense_k)])
    bm25_slice = list(bm25[: max(0, bm25_k)])
    bm25_ranks = {chunk_key(item): rank for rank, item in enumerate(bm25_slice, 1)}
    merged: list[dict] = []
    seen: set[tuple] = set()

    for dense_rank, document in enumerate(dense_slice, 1):
        key = chunk_key(document)
        if key in seen:
            continue
        seen.add(key)
        merged.append({
            **document,
            "dense_rank": dense_rank,
            "bm25_rank": bm25_ranks.get(key),
            "candidate_source": "both" if key in bm25_ranks else "dense",
        })

    for bm25_rank, document in enumerate(bm25_slice, 1):
        key = chunk_key(document)
        if key in seen:
            continue
        seen.add(key)
        merged.append({
            **document,
            "dense_rank": None,
            "bm25_rank": bm25_rank,
            "candidate_source": "bm25_supplement",
        })

    for merged_rank, document in enumerate(merged, 1):
        document["merged_rank"] = merged_rank
        document["score"] = 1.0 / merged_rank
    return merged


def retrieve_dense_primary(
    question: str,
    *,
    dense_k: int = 120,
    bm25_k: int = 30,
    manager: MilvusManager | None = None,
    dense_vector: list[float] | None = None,
) -> dict[str, Any]:
    """Retrieve independent routes and apply the append-only merge policy."""
    manager = manager or MilvusManager()
    started = time.perf_counter()
    embedding_started = time.perf_counter()
    dense_vector = dense_vector or embedding_service.get_embeddings([question])[0]
    embedding_ms = (time.perf_counter() - embedding_started) * 1000

    dense_started = time.perf_counter()
    dense = manager.dense_retrieve(dense_vector, top_k=dense_k, filter_expr=_LEAF_FILTER)
    dense_ms = (time.perf_counter() - dense_started) * 1000

    bm25_started = time.perf_counter()
    bm25 = manager.bm25_retrieve(question, top_k=bm25_k, filter_expr=_LEAF_FILTER)
    bm25_ms = (time.perf_counter() - bm25_started) * 1000

    merge_started = time.perf_counter()
    merged = merge_dense_primary(dense, bm25, dense_k=dense_k, bm25_k=bm25_k)
    merge_ms = (time.perf_counter() - merge_started) * 1000
    return {
        "query_embedding": dense_vector,
        "dense": dense,
        "bm25": bm25,
        "merged": merged,
        "latency_ms": {
            "embedding": round(embedding_ms, 2),
            "dense": round(dense_ms, 2),
            "bm25": round(bm25_ms, 2),
            "merge": round(merge_ms, 2),
            "total": round((time.perf_counter() - started) * 1000, 2),
        },
    }


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _best_page_anchor(question: str, page_text: str, *, window_chars: int = 600) -> str:
    """Choose a contiguous query-relevant window without domain rules."""
    text = str(page_text or "").strip()
    if not text:
        return ""
    query_terms = lexical_terms(question)
    if len(text) <= window_chars:
        return text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text[:window_chars]
    best = (0.0, 0, "")
    for index in range(len(lines)):
        block = "\n".join(lines[index:index + 8])[:window_chars]
        overlap = len(query_terms & lexical_terms(block)) / max(1, len(query_terms))
        candidate = (overlap, -index, block)
        if candidate > best:
            best = candidate
    return best[2] or text[:window_chars]


def expand_and_rank_pages(
    question: str,
    merged_chunks: list[dict],
    query_embedding: list[float],
    *,
    neighbor_window: int = 1,
    page_store: DocumentPageStore | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Expand candidate pages within-document and rank page records.

    Ranking uses only precomputed page embeddings, lexical overlap, and the
    originating merged rank. No task parser, finance metric alias, or gold data
    is consulted.
    """
    page_store = page_store or DocumentPageStore()
    seeds: dict[tuple[str, int], list[dict]] = {}
    requested: list[tuple[str, int]] = []
    for chunk in merged_chunks:
        filename, source_page = page_key(chunk)
        if not filename:
            continue
        for offset in range(-max(0, neighbor_window), max(0, neighbor_window) + 1):
            target_page = source_page + offset
            if target_page < 0:
                continue
            key = (filename, target_page)
            requested.append(key)
            seeds.setdefault(key, []).append({
                "source_page": source_page,
                "expanded_page": target_page,
                "distance": abs(offset),
                "merged_rank": int(chunk.get("merged_rank") or len(merged_chunks) + 1),
                "chunk_id": chunk.get("chunk_id") or chunk.get("id"),
                "text": chunk.get("text") or "",
            })
    unique_requested = list(dict.fromkeys(requested))
    opened = page_store.get_pages_by_keys(unique_requested)
    query_terms = lexical_terms(question)
    details: list[dict] = []
    for page in opened:
        key = page_key(page)
        sources = sorted(seeds.get(key, []), key=lambda item: (item["merged_rank"], item["distance"]))
        if not sources:
            continue
        text = "\n".join([
            str(page.get("filename") or ""),
            str(page.get("page_text") or ""),
            str(page.get("table_text") or ""),
        ])
        best_source = sources[0]
        best_rank = min(item["merged_rank"] for item in sources)
        best_distance = min(item["distance"] for item in sources)
        details.append({
            **page,
            "dense_raw": _cosine_similarity(query_embedding, list(page.get("page_dense_embedding") or [])),
            "lexical_raw": len(query_terms & lexical_terms(text)) / max(1, len(query_terms)),
            "seed_raw": 1.0 / math.log2(best_rank + 2.0),
            "best_seed_rank": best_rank,
            "neighbor_distance": best_distance,
            "exact_candidate_page": any(item["distance"] == 0 for item in sources),
            "original_chunk": best_source,
            "expanded_from": [
                {
                    "source_page": item["source_page"],
                    "expanded_page": item["expanded_page"],
                    "distance": item["distance"],
                    "merged_rank": item["merged_rank"],
                    "chunk_id": item["chunk_id"],
                }
                for item in sources[:8]
            ],
        })
    dense_scores = _normalize([item["dense_raw"] for item in details])
    lexical_scores = _normalize([item["lexical_raw"] for item in details])
    seed_scores = _normalize([item["seed_raw"] for item in details])
    for item, dense_score, lexical_score, seed_score in zip(details, dense_scores, lexical_scores, seed_scores):
        item["page_dense_score"] = round(dense_score, 8)
        item["page_lexical_score"] = round(lexical_score, 8)
        item["page_seed_score"] = round(seed_score, 8)
        item["page_score"] = round(
            0.72 * dense_score + 0.18 * lexical_score + 0.10 * seed_score,
            8,
        )
        anchor = _best_page_anchor(question, str(item.get("page_text") or ""))
        item["best_chunk"] = {"text": anchor}
    details.sort(key=lambda item: (-item["page_score"], item["best_seed_rank"], page_key(item)))
    for rank, item in enumerate(details, 1):
        item["page_candidate_rank"] = rank
    return details, {
        "neighbor_window": max(0, neighbor_window),
        "source_chunk_count": len(merged_chunks),
        "requested_page_count": len(unique_requested),
        "opened_page_count": len(opened),
        "page_candidate_count": len(details),
        "original_chunk_pages": [
            {"filename": page_key(item)[0], "page_number": page_key(item)[1], "merged_rank": item.get("merged_rank")}
            for item in merged_chunks
        ],
        "expanded_pages": [
            {"filename": page_key(item)[0], "page_number": page_key(item)[1], "expanded_from": item["expanded_from"]}
            for item in details
        ],
    }


def rank_page_documents(page_candidates: list[dict]) -> list[dict]:
    """Aggregate page evidence into document scores without domain rules."""
    grouped: dict[str, list[dict]] = {}
    for page in page_candidates:
        filename = str(page.get("filename") or "")
        if filename:
            grouped.setdefault(filename, []).append(page)
    documents = []
    for filename, pages in grouped.items():
        ordered = sorted(pages, key=lambda item: (-float(item.get("page_score") or 0.0), page_key(item)))
        scores = [float(item.get("page_score") or 0.0) for item in ordered[:3]]
        document_score = scores[0]
        if len(scores) > 1:
            document_score += 0.35 * scores[1]
        if len(scores) > 2:
            document_score += 0.15 * scores[2]
        documents.append({
            "filename": filename,
            "document_score": round(document_score, 8),
            "page_count": len(ordered),
            "pages": ordered,
        })
    documents.sort(key=lambda item: (-item["document_score"], item["filename"].casefold()))
    for rank, document in enumerate(documents, 1):
        document["document_rank"] = rank
    return documents


def select_document_first_pages(
    page_candidates: list[dict],
    *,
    final_page_k: int = 8,
    global_escape_pages: int = 1,
) -> tuple[list[dict], dict[str, Any]]:
    """Select mostly from the strongest document plus a global escape slot."""
    documents = rank_page_documents(page_candidates)
    if not documents or final_page_k <= 0:
        return [], {"document_scores": [], "primary_document": None, "global_escape_pages": []}
    escape_count = min(max(0, global_escape_pages), max(0, final_page_k - 1))
    main_count = final_page_k - escape_count
    primary = documents[0]
    selected = list(primary["pages"][:main_count])
    selected_keys = {page_key(item) for item in selected}
    escapes = []
    for page in page_candidates:
        if page_key(page) in selected_keys or page.get("filename") == primary["filename"]:
            continue
        escapes.append(page)
        selected_keys.add(page_key(page))
        if len(escapes) >= escape_count:
            break
    selected.extend(escapes)
    if len(selected) < final_page_k:
        for page in primary["pages"][main_count:]:
            if page_key(page) in selected_keys:
                continue
            selected.append(page)
            selected_keys.add(page_key(page))
            if len(selected) >= final_page_k:
                break
    for rank, page in enumerate(selected, 1):
        page["selected_rank"] = rank
    return selected, {
        "document_scores": [
            {
                "filename": item["filename"],
                "document_score": item["document_score"],
                "document_rank": item["document_rank"],
                "page_count": item["page_count"],
            }
            for item in documents
        ],
        "primary_document": primary["filename"],
        "primary_document_page_slots": main_count,
        "global_escape_pages": [
            {"filename": item.get("filename"), "page_number": item.get("page_number"), "page_score": item.get("page_score")}
            for item in escapes
        ],
    }


def score_candidate_documents(
    merged_chunks: list[dict],
    *,
    shortlist_k: int = 3,
) -> tuple[list[str], list[dict]]:
    """Rank documents using route ranks and structural support only."""
    grouped: dict[str, list[dict]] = {}
    for chunk in merged_chunks:
        filename = str(chunk.get("filename") or "").strip()
        if filename:
            grouped.setdefault(filename, []).append(chunk)
    documents = []
    for filename, chunks in grouped.items():
        dense_ranks = [int(item["dense_rank"]) for item in chunks if item.get("dense_rank") is not None]
        bm25_ranks = [int(item["bm25_rank"]) for item in chunks if item.get("bm25_rank") is not None]
        best_dense_rank = min(dense_ranks, default=None)
        best_bm25_rank = min(bm25_ranks, default=None)
        dense_rank_score = 1.0 / math.log2(best_dense_rank + 2.0) if best_dense_rank is not None else 0.0
        bm25_rank_score = 1.0 / math.log2(best_bm25_rank + 2.0) if best_bm25_rank is not None else 0.0
        chunk_support = min(1.0, math.log2(len(chunks) + 1.0) / math.log2(7.0))
        page_count = len({page_key(item)[1] for item in chunks})
        page_coverage = min(1.0, math.log2(page_count + 1.0) / math.log2(6.0))
        score = (
            0.55 * dense_rank_score
            + 0.20 * bm25_rank_score
            + 0.15 * chunk_support
            + 0.10 * page_coverage
        )
        documents.append({
            "filename": filename,
            "document_score": round(score, 8),
            "best_dense_rank": best_dense_rank,
            "best_bm25_rank": best_bm25_rank,
            "chunk_count": len(chunks),
            "page_count": page_count,
            "dense_rank_contribution": round(0.55 * dense_rank_score, 8),
            "bm25_rank_contribution": round(0.20 * bm25_rank_score, 8),
            "chunk_support_contribution": round(0.15 * chunk_support, 8),
            "page_coverage_contribution": round(0.10 * page_coverage, 8),
        })
    documents.sort(key=lambda item: (-item["document_score"], item["filename"].casefold()))
    for rank, document in enumerate(documents, 1):
        document["document_rank"] = rank
    return [item["filename"] for item in documents[: max(1, shortlist_k)]], documents


def _escape_milvus_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def retrieve_document_local_chunks(
    question: str,
    filenames: list[str],
    query_embedding: list[float],
    *,
    local_k: int = 30,
    dense_slots: int = 20,
    manager: MilvusManager | None = None,
) -> dict[str, Any]:
    """Run independent Dense/BM25 retrieval inside shortlisted documents."""
    manager = manager or MilvusManager()
    started = time.perf_counter()
    routes = []
    combined: list[dict] = []
    dense_calls = 0
    bm25_calls = 0
    for document_rank, filename in enumerate(filenames, 1):
        escaped = _escape_milvus_string(filename)
        filter_expr = f'({_LEAF_FILTER}) and filename == "{escaped}"'
        route_started = time.perf_counter()
        dense = manager.dense_retrieve(query_embedding, top_k=local_k, filter_expr=filter_expr)
        dense_calls += 1
        bm25 = manager.bm25_retrieve(question, top_k=local_k, filter_expr=filter_expr)
        bm25_calls += 1
        merged = merge_dense_primary(
            dense,
            bm25,
            dense_k=min(local_k, max(1, dense_slots)),
            bm25_k=local_k,
        )[:local_k]
        for item in merged:
            item["local_document_rank"] = document_rank
            item["local_dense_rank"] = item.pop("dense_rank", None)
            item["local_bm25_rank"] = item.pop("bm25_rank", None)
            item["local_chunk_rank"] = item.pop("merged_rank", None)
            item["candidate_source"] = "document_local"
        combined.extend(merged)
        routes.append({
            "filename": filename,
            "document_rank": document_rank,
            "dense_count": len(dense),
            "bm25_count": len(bm25),
            "local_chunk_count": len(merged),
            "dense_slots": min(local_k, max(1, dense_slots)),
            "bm25_supplement_slots": max(0, local_k - min(local_k, max(1, dense_slots))),
            "latency_ms": round((time.perf_counter() - route_started) * 1000, 2),
        })
    combined.sort(key=lambda item: (
        int(item.get("local_document_rank") or 10**9),
        int(item.get("local_chunk_rank") or 10**9),
    ))
    for rank, item in enumerate(combined, 1):
        item["merged_rank"] = rank
    return {
        "chunks": combined,
        "routes": routes,
        "dense_calls": dense_calls,
        "bm25_calls": bm25_calls,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def merge_global_local_chunks(global_chunks: list[dict], local_chunks: list[dict]) -> list[dict]:
    """Prefer local ranks, then append global chunks that add new evidence."""
    merged = []
    seen: set[tuple] = set()
    global_ranks = {chunk_key(item): rank for rank, item in enumerate(global_chunks, 1)}
    local_ranks = {chunk_key(item): rank for rank, item in enumerate(local_chunks, 1)}
    for source, items in (("local", local_chunks), ("global", global_chunks)):
        for item in items:
            key = chunk_key(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append({
                **item,
                "global_rank": global_ranks.get(key),
                "local_rank": local_ranks.get(key),
                "candidate_source": "both" if key in global_ranks and key in local_ranks else source,
            })
    for rank, item in enumerate(merged, 1):
        item["merged_rank"] = rank
    return merged
