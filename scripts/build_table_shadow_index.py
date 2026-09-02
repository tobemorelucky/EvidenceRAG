"""Build the independent table-aware shadow collection from PostgreSQL."""

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
from models import DocumentTable  # noqa: E402
from table_retrieval import DEFAULT_TABLE_COLLECTION, TableMilvusManager, build_table_document  # noqa: E402
from table_store import TableStore  # noqa: E402


def load_table_documents() -> tuple[list[dict], dict]:
    db = SessionLocal()
    try:
        tables = db.query(DocumentTable).order_by(
            DocumentTable.filename.asc(),
            DocumentTable.page_number.asc(),
            DocumentTable.table_index.asc(),
        ).all()
        documents = []
        skipped = []
        for table in tables:
            document = build_table_document(TableStore._to_dict(table))
            if document is None:
                skipped.append(table.table_id)
            else:
                documents.append(document)
        return documents, {
            "postgres_tables": len(tables),
            "indexable_tables": len(documents),
            "skipped_tables": len(skipped),
            "skipped_table_ids": skipped[:50],
        }
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default=os.getenv("TABLE_MILVUS_COLLECTION", DEFAULT_TABLE_COLLECTION))
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    documents, stats = load_table_documents()
    print({"collection": args.collection, **stats}, flush=True)
    if not args.execute:
        print("Dry run only. Add --execute --recreate to build the isolated table shadow collection.", flush=True)
        return

    from embedding import embedding_service

    manager = TableMilvusManager(args.collection)
    if args.recreate:
        manager.drop_collection()
    elif manager.has_collection() and manager.count() > 0:
        raise RuntimeError("table shadow collection is not empty; use --recreate to avoid duplicate rows")
    manager.init_collection(dense_dim=int(os.getenv("DENSE_EMBEDDING_DIM", "1024")))

    embedding_started = time.perf_counter()
    vectors = embedding_service.get_embeddings([item["search_text"] for item in documents])
    print(
        f"[embedding] tables={len(vectors)} device={embedding_service._device} "
        f"seconds={time.perf_counter() - embedding_started:.2f}",
        flush=True,
    )
    write_started = time.perf_counter()
    batch_size = max(1, args.batch_size)
    for offset in range(0, len(documents), batch_size):
        batch = []
        for document, vector in zip(documents[offset:offset + batch_size], vectors[offset:offset + batch_size]):
            batch.append({**document, "dense_embedding": vector})
        manager.insert(batch)
        print(f"[write] {min(offset + len(batch), len(documents))}/{len(documents)}", flush=True)
    manager.flush()
    print(
        f"[done] collection={args.collection} entities={manager.count()} "
        f"write_seconds={time.perf_counter() - write_started:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
