"""Build the isolated Evidence Block Retrieval v2 Milvus collection."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from database import SessionLocal  # noqa: E402
from document_page_store import DocumentPageStore  # noqa: E402
from evidence_block_retrieval_v2 import (  # noqa: E402
    DEFAULT_BLOCK_COLLECTION,
    EvidenceBlockMilvusManager,
    build_evidence_blocks_v2,
    index_document,
)
from milvus_client import MilvusManager  # noqa: E402
from models import DocumentPage, DocumentTable  # noqa: E402
from table_store import TableStore  # noqa: E402


CHUNK_FIELDS = [
    "text", "filename", "page_number", "chunk_id", "chunk_idx",
    "chunk_level", "content_hash",
]


def load_source_records() -> tuple[list[dict], list[dict], list[dict]]:
    """Read the existing stores without mutating them."""
    chunks = MilvusManager().query_all(output_fields=CHUNK_FIELDS)
    db = SessionLocal()
    try:
        pages = [DocumentPageStore._to_dict(item) for item in db.query(DocumentPage).all()]
        tables = [TableStore._to_dict(item) for item in db.query(DocumentTable).all()]
    finally:
        db.close()
    return chunks, pages, tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=os.getenv("EVIDENCE_BLOCK_MILVUS_COLLECTION", DEFAULT_BLOCK_COLLECTION))
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    chunks, pages, tables = load_source_records()
    blocks = build_evidence_blocks_v2(chunks, pages=pages, tables=tables)
    counts = {
        source_type: sum(block["source_type"] == source_type for block in blocks)
        for source_type in ("text", "table", "mixed")
    }
    print({
        "collection": args.collection,
        "source_chunks": len(chunks),
        "source_pages": len(pages),
        "source_tables": len(tables),
        "blocks": len(blocks),
        "block_types": counts,
    }, flush=True)
    if not args.execute:
        print("Dry run only. Add --execute --recreate to build the isolated block shadow collection.", flush=True)
        return

    from embedding import embedding_service

    manager = EvidenceBlockMilvusManager(args.collection)
    if args.recreate:
        manager.drop_collection()
    elif manager.has_collection() and manager.count() > 0:
        raise RuntimeError("block shadow collection is not empty; use --recreate to avoid duplicate rows")
    manager.init_collection(dense_dim=int(os.getenv("DENSE_EMBEDDING_DIM", "1024")))

    batch_size = max(1, args.batch_size)
    started = time.perf_counter()
    for offset in range(0, len(blocks), batch_size):
        source_batch = blocks[offset:offset + batch_size]
        vectors = embedding_service.get_embeddings([block["text"] for block in source_batch])
        manager.insert([index_document(block, vector) for block, vector in zip(source_batch, vectors)])
        print(f"[build] {min(offset + len(source_batch), len(blocks))}/{len(blocks)}", flush=True)
    manager.flush()
    print({
        "collection": args.collection,
        "entities": manager.count(),
        "seconds": round(time.perf_counter() - started, 2),
        "device": embedding_service._device,
    }, flush=True)


if __name__ == "__main__":
    main()
