"""Experimental Retrieval Core v4 primitives.

This module is isolated from the production Core v3 path. It contains retrieval
mechanics only; answer prompts, skills, validators, and Context Budget v3 are
not changed here.
"""

from __future__ import annotations

import time
from typing import Any

from embedding import embedding_service
from milvus_client import MilvusManager
from retrieval_ablation import chunk_key


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
