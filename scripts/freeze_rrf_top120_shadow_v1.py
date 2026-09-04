"""Explicit opt-in: run existing Milvus hybrid retrieval ONCE and freeze 30 queries.

No changes to retrieval implementation/configuration, no rerank or generation.
Do not use this to claim historical RRF ranks were preserved. This is a NEW
snapshot, to be shared unchanged by all shadow rerankers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_reranker_shadow_v1 import DEFAULT_INPUT, digest, fixture_rows, validate_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--allow-retrieval-snapshot", action="store_true", help="Explicitly authorize a fresh retrieval snapshot")
    args = parser.parse_args()
    if not args.allow_retrieval_snapshot:
        parser.error("Requires --allow-retrieval-snapshot; never silently retrieve during evaluation")
    if args.output.exists():
        parser.error("Snapshot already exists; refusing to overwrite")
    rows, groups = fixture_rows()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LANGCHAIN_TRACING_V2="false", LANGSMITH_TRACING="false")
    sys.path.insert(0, str(ROOT / "backend"))
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    from embedding import embedding_service
    from milvus_client import MilvusManager
    manager = MilvusManager()
    if not manager.uses_builtin_bm25:
        raise RuntimeError("Native BM25 required; no custom sparse fallback")
    expr = '(evidence_type == "text_chunk" or evidence_type == "") and chunk_level == 3'
    records = []
    for i, (key, group) in enumerate(groups.items(), 1):
        question = rows[key]["question"]
        print(f"[freeze {i:02d}/30] existing hybrid retrieval starting", flush=True)
        dense = embedding_service.get_embeddings([question])[0]
        chunks = manager.hybrid_retrieve(dense, None, top_k=120, rrf_k=60, filter_expr=expr, query_text=question)
        chunks = [{**c, "rrf_rank": rank} for rank, c in enumerate(chunks, 1)]
        records.append({"question_id": key, "group": group, "question": question,
                        "chunks": chunks, "candidate_sha256": digest(chunks)})
        print(f"[freeze {i:02d}/30] {len(chunks)} chunks", flush=True)
    payload = {"schema": "rrf_top120_shadow_v1", "retrieval": {
        "method": "MilvusManager.hybrid_retrieve", "top_k": 120, "rrf_k": 60,
        "route_search_limit": 240, "filter_expr": expr, "collection": manager.collection_name,
        "source": "fresh explicit snapshot, not reconstructed historical output",
        "implementation_sha256": hashlib.sha256((ROOT / "backend/milvus_client.py").read_bytes()).hexdigest()}, "records": records}
    validate_snapshot(payload, rows, groups)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: no overwriting a frozen input, even in concurrent runs.
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Frozen: {args.output}", flush=True)


if __name__ == "__main__":
    main()
