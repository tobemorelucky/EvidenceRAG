"""Rebuild the EvidenceRAG finance collection from the 40-document benchmark set.

The command is intentionally dry-run by default. Pass ``--execute`` after
reviewing the exact collection and source files printed by the dry run.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
CSV_PATH = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DOCUMENT_DIR = ROOT / "data" / "documents"


def benchmark_files() -> list[Path]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        names = sorted({(row.get("doc_name") or "").strip() for row in csv.DictReader(handle) if row.get("doc_name")})
    paths = [DOCUMENT_DIR / f"{name}.pdf" for name in names]
    missing = [path for path in paths if not path.is_file()]
    if len(paths) != 40 or missing:
        details = ", ".join(path.name for path in missing[:10])
        raise RuntimeError(f"Expected 40 benchmark PDFs, found {len(paths)} references; missing: {details or 'none'}")
    return paths


def _page_retrieval_text(page: dict) -> str:
    """Compact page representation for page discovery, retaining full page storage.

    Dense page retrieval needs enough narrative and table context to identify a
    page, not every token on a long 10-K page.  Capping this representation
    avoids quadratic attention cost while BM25 and subsequent chunk retrieval
    still see the complete page text.
    """
    max_chars = max(1200, int(os.getenv("PAGE_RETRIEVAL_MAX_CHARS", "3000")))
    page_text = page.get("page_text", "") or ""
    table_text = page.get("table_text", "") or ""
    if len(page_text) <= max_chars and not table_text:
        return page_text
    head_size = max_chars // 2
    tail_size = max_chars // 4
    table_size = max_chars - head_size - tail_size
    return "\n".join(
        part
        for part in (
            page_text[:head_size],
            table_text[:table_size],
            page_text[-tail_size:],
        )
        if part
    )


def _load_bundles(paths: list[Path], loader) -> tuple[list[dict], list[dict], list[tuple[str, int, int]]]:
    pages: list[dict] = []
    leaves: list[dict] = []
    stats: list[tuple[str, int, int]] = []
    for index, path in enumerate(paths, 1):
        bundle = loader.load_document_bundle(str(path), path.name)
        document_leaves = [chunk for chunk in bundle.get("chunks") or [] if int(chunk.get("chunk_level", 0)) == 3]
        if not document_leaves:
            raise RuntimeError(f"{path.name}: no retrievable chunks generated")
        document_pages = bundle.get("pages") or []
        pages.extend(document_pages)
        leaves.extend(document_leaves)
        stats.append((path.name, len(document_pages), len(document_leaves)))
        print(f"parsed [{index:02d}/40] {path.name}: pages={len(document_pages)} chunks={len(document_leaves)}", flush=True)
    return pages, leaves, stats


def rebuild(paths: list[Path]) -> None:
    os.environ.setdefault("MILVUS_SPARSE_MODE", "milvus_bm25")
    os.environ.setdefault("EMBEDDING_DEVICE", "cuda")
    # This benchmark path intentionally evaluates the page/chunk hybrid base.
    # Table parsing is an independently measured experimental branch, never an
    # accidental cost paid by the default rebuild because of a local .env.
    os.environ["TABLE_AWARE_INGESTION"] = "false"
    sys.path.insert(0, str(BACKEND))

    # Verify PostgreSQL before loading the embedding model, which can take time.
    from database import SessionLocal, init_db

    init_db()

    from document_loader import DocumentLoader
    from document_page_store import DocumentPageStore
    from embedding import embedding_service
    from milvus_client import MilvusManager
    from milvus_writer import MilvusWriter
    from models import DocumentPage, DocumentTable, ParentChunk
    manager = MilvusManager()
    if not manager.uses_builtin_bm25:
        raise RuntimeError("pymilvus does not expose Function/FunctionType; upgrade the rag environment before rebuilding")

    loader = DocumentLoader(chunk_size=800, chunk_overlap=128, include_parent_chunks=False)
    parse_started = time.perf_counter()
    pages, leaves, document_stats = _load_bundles(paths, loader)
    print(
        f"parse complete: documents={len(document_stats)} pages={len(pages)} chunks={len(leaves)} "
        f"seconds={time.perf_counter() - parse_started:.1f}",
        flush=True,
    )

    # Do not destroy the current derived index until every source PDF parsed.
    manager.drop_collection()
    manager.init_collection()
    db = SessionLocal()
    try:
        db.query(DocumentTable).delete(synchronize_session=False)
        db.query(DocumentPage).delete(synchronize_session=False)
        db.query(ParentChunk).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()

    page_store = DocumentPageStore()
    writer = MilvusWriter(embedding_service=embedding_service, milvus_manager=manager)

    embedding_started = time.perf_counter()
    vectors = embedding_service.get_embeddings([_page_retrieval_text(page) for page in pages] + [leaf["text"] for leaf in leaves])
    page_vectors = vectors[:len(pages)]
    leaf_vectors = vectors[len(pages):]
    print(
        f"embedding complete: vectors={len(vectors)} seconds={time.perf_counter() - embedding_started:.1f} "
        f"device={embedding_service._device}",
        flush=True,
    )

    write_started = time.perf_counter()
    page_store.insert_preembedded_pages(pages, page_vectors)
    writer.write_documents(leaves, batch_size=500, embeddings=leaf_vectors)
    print(f"write complete: seconds={time.perf_counter() - write_started:.1f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the 40-document EvidenceRAG finance index")
    parser.add_argument("--execute", action="store_true", help="perform the destructive rebuild")
    args = parser.parse_args()
    paths = benchmark_files()
    collection = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
    print(f"Collection: {collection}")
    print(f"Source: {DOCUMENT_DIR}")
    print(f"Validated PDFs: {len(paths)}")
    if not args.execute:
        print("Dry run only. Re-run with --execute to replace the collection and derived PostgreSQL records.")
        return
    rebuild(paths)
    print("FinanceBench index rebuild completed successfully.")


if __name__ == "__main__":
    main()
