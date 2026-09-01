"""Read-only retrieval ablations isolated from production RAG profiles.

The module exposes independent Dense and BM25 ranks, deterministic RRF, and a
single-call page-level rerank prototype.  It never uses benchmark gold data.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import defaultdict
from typing import Any

from document_page_store import DocumentPageStore
from embedding import embedding_service
from milvus_client import MilvusManager
from rag_utils import _rerank_documents
from table_store import TableStore


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%'-]*")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "which", "who", "with", "would",
}


def page_key(item: dict) -> tuple[str, int]:
    try:
        page_number = int(item.get("page_number") or 0)
    except (TypeError, ValueError):
        page_number = 0
    return str(item.get("filename") or "").strip(), page_number


def chunk_key(item: dict) -> tuple[str, str, int, int]:
    return (
        str(item.get("chunk_id") or item.get("id") or ""),
        *page_key(item),
        int(item.get("chunk_idx") or 0),
    )


def lexical_terms(text: object) -> set[str]:
    return {
        token.casefold().removesuffix("'s")
        for token in _TOKEN.findall(str(text or ""))
        if len(token) > 1 and token.casefold().removesuffix("'s") not in _STOPWORDS
    }


def rank_pages(documents: list[dict]) -> list[dict]:
    """Collapse ranked chunks to pages while preserving first/best rank."""
    pages: dict[tuple[str, int], dict] = {}
    for rank, document in enumerate(documents, 1):
        key = page_key(document)
        if not key[0]:
            continue
        page = pages.setdefault(key, {
            "filename": key[0],
            "page_number": key[1],
            "rank": rank,
            "best_chunk_rank": rank,
            "chunk_count": 0,
            "chunks": [],
        })
        page["chunk_count"] += 1
        page["chunks"].append(document)
    return sorted(pages.values(), key=lambda item: (item["rank"], item["filename"], item["page_number"]))


def rrf_fuse(dense: list[dict], bm25: list[dict], *, rrf_k: int = 60) -> list[dict]:
    fused: dict[tuple, dict] = {}
    for route, documents in (("dense", dense), ("bm25", bm25)):
        for rank, document in enumerate(documents, 1):
            key = chunk_key(document)
            item = fused.setdefault(key, {
                **document,
                "dense_rank": None,
                "bm25_rank": None,
                "rrf_score": 0.0,
            })
            item[f"{route}_rank"] = rank
            item["rrf_score"] += 1.0 / (max(1, rrf_k) + rank)
    result = sorted(
        fused.values(),
        key=lambda item: (
            -float(item["rrf_score"]),
            min(rank for rank in (item["dense_rank"], item["bm25_rank"]) if rank is not None),
            page_key(item),
        ),
    )
    for rank, item in enumerate(result, 1):
        item["rrf_rank"] = rank
        item["score"] = item["rrf_score"]
    return result


def retrieve_independent_routes(
    question: str,
    *,
    max_k: int = 120,
    rrf_k: int = 60,
    manager: MilvusManager | None = None,
    dense_vector: list[float] | None = None,
) -> dict[str, Any]:
    """Return independent Dense/BM25 rankings without changing production retrieval."""
    manager = manager or MilvusManager()
    filter_expr = '(evidence_type == "text_chunk" or evidence_type == "") and chunk_level == 3'
    started = time.perf_counter()
    embedding_started = time.perf_counter()
    dense_vector = dense_vector or embedding_service.get_embeddings([question])[0]
    embedding_ms = (time.perf_counter() - embedding_started) * 1000
    dense_started = time.perf_counter()
    dense = manager.dense_retrieve(dense_vector, top_k=max_k, filter_expr=filter_expr)
    dense_ms = (time.perf_counter() - dense_started) * 1000
    bm25_started = time.perf_counter()
    bm25 = manager.bm25_retrieve(question, top_k=max_k, filter_expr=filter_expr)
    bm25_ms = (time.perf_counter() - bm25_started) * 1000
    rrf_started = time.perf_counter()
    rrf = rrf_fuse(dense, bm25, rrf_k=rrf_k)
    return {
        "dense": dense,
        "bm25": bm25,
        "rrf": rrf,
        "dense_pages": rank_pages(dense),
        "bm25_pages": rank_pages(bm25),
        "rrf_pages": rank_pages(rrf),
        "latency_ms": {
            "embedding": round(embedding_ms, 2),
            "dense": round(dense_ms, 2),
            "bm25": round(bm25_ms, 2),
            "rrf": round((time.perf_counter() - rrf_started) * 1000, 2),
            "total": round((time.perf_counter() - started) * 1000, 2),
        },
    }


def _row_text(row: object) -> str:
    if isinstance(row, dict):
        return " | ".join(f"{key}: {value}" for key, value in row.items())
    if isinstance(row, list):
        return " | ".join(str(value) for value in row)
    return str(row or "")


def _relevant_table_text(question: str, tables: list[dict], max_chars: int) -> str:
    query_terms = lexical_terms(question)
    candidates = []
    for table in tables:
        title = str(table.get("title") or table.get("caption") or "").strip()
        columns = " | ".join(str(value) for value in (table.get("columns") or []))
        heading = f"Table: {title}\nColumns: {columns}".strip()
        for index, row in enumerate(table.get("rows") or []):
            text = _row_text(row)
            overlap = len(query_terms & lexical_terms(f"{heading} {text}"))
            candidates.append((overlap, -index, heading, text))
    candidates.sort(reverse=True)
    parts = []
    for _, _, heading, row in candidates[:4]:
        block = f"{heading}\nRow: {row}".strip()
        if len("\n".join(parts + [block])) > max_chars:
            break
        parts.append(block)
    return "\n".join(parts)


def build_page_candidates(
    question: str,
    rrf_chunks: list[dict],
    *,
    rrf_chunk_k: int = 100,
    page_candidate_k: int = 30,
    representation_chars: int = 1500,
    page_store: DocumentPageStore | None = None,
    table_store: TableStore | None = None,
) -> list[dict]:
    """Build compact, query-neutral page representations for a single rerank call."""
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for chunk in rrf_chunks[:rrf_chunk_k]:
        key = page_key(chunk)
        if key[0]:
            grouped[key].append(chunk)
    ranked_keys = sorted(
        grouped,
        key=lambda key: (
            min(int(item.get("rrf_rank") or 10**9) for item in grouped[key]),
            -min(3, len(grouped[key])),
            key,
        ),
    )[:page_candidate_k]
    page_store = page_store or DocumentPageStore()
    table_store = table_store or TableStore()
    opened = {
        page_key(page): page
        for page in page_store.get_pages_by_keys(ranked_keys)
    }
    tables_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for table in table_store.get_tables_by_page_keys(ranked_keys):
        tables_by_page[page_key(table)].append(table)

    pages = []
    for key in ranked_keys:
        chunks = sorted(grouped[key], key=lambda item: int(item.get("rrf_rank") or 10**9))[:3]
        page_text = str((opened.get(key) or {}).get("page_text") or "")
        best_chunks = "\n\n".join(str(item.get("text") or "").strip() for item in chunks)
        heading = page_text[:350].strip()
        table_text = _relevant_table_text(question, tables_by_page.get(key, []), max_chars=550)
        representation = (
            f"Document: {key[0]}\nPage: {key[1]}\n\n"
            f"Best matching chunks:\n{best_chunks}\n\n"
            f"Page heading / nearby context:\n{heading}"
        )
        if table_text:
            representation += f"\n\n{table_text}"
        representation = representation[:representation_chars]
        pages.append({
            "filename": key[0],
            "page_number": key[1],
            "text": representation,
            "content_hash": hashlib.sha256(representation.encode("utf-8")).hexdigest(),
            "source_chunks": chunks,
            "rrf_page_rank": len(pages) + 1,
            "representation_chars": len(representation),
            "table_representation_chars": len(table_text),
        })
    return pages


def page_level_jina_rerank(
    question: str,
    rrf_chunks: list[dict],
    *,
    rrf_chunk_k: int = 100,
    page_candidate_k: int = 30,
    final_page_k: int = 6,
    representation_chars: int = 1500,
) -> dict[str, Any]:
    pages = build_page_candidates(
        question,
        rrf_chunks,
        rrf_chunk_k=rrf_chunk_k,
        page_candidate_k=page_candidate_k,
        representation_chars=representation_chars,
    )
    started = time.perf_counter()
    reranked, meta = _rerank_documents(
        question,
        pages,
        top_k=len(pages),
        remote_candidate_k=len(pages),
        remote_max_chars=representation_chars,
    )
    return {
        "page_candidates": pages,
        "reranked_pages": reranked,
        "selected_pages": reranked[:final_page_k],
        "rerank_meta": meta,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
