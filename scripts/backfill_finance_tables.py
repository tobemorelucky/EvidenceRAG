"""Backfill PostgreSQL table records for the fixed FinanceBench document set.

This script does not write embeddings or modify Milvus. It is dry-run by
default; pass --execute to upsert accepted tables into the existing TableStore.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DOCUMENTS = ROOT / "data" / "documents"


def financebench_filenames(dataset: Path) -> set[str]:
    with dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        name if name.lower().endswith(".pdf") else f"{name}.pdf"
        for name in (str(row.get("doc_name") or "").strip() for row in rows)
        if name
    }


def select_pdf_paths(
    documents_dir: Path,
    dataset: Path,
    requested: list[str],
    *,
    all_pdfs: bool,
) -> list[Path]:
    available = {path.name.casefold(): path for path in documents_dir.glob("*.pdf")}
    if requested:
        names = [name if name.lower().endswith(".pdf") else f"{name}.pdf" for name in requested]
    elif all_pdfs:
        names = [path.name for path in available.values()]
    else:
        names = sorted(financebench_filenames(dataset))
    selected = []
    missing = []
    for name in names:
        path = available.get(name.casefold())
        if path is None:
            missing.append(name)
        elif path not in selected:
            selected.append(path)
    if missing:
        print(f"[warning] missing PDFs: {', '.join(missing[:10])}", flush=True)
    return sorted(selected, key=lambda path: path.name.casefold())


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill accepted financial tables without rebuilding vectors.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--documents-dir", type=Path, default=DOCUMENTS)
    parser.add_argument("--filename", action="append", default=[], help="Process only this PDF; repeat as needed.")
    parser.add_argument("--all-pdfs", action="store_true", help="Include non-FinanceBench PDFs in the documents directory.")
    parser.add_argument("--backend", choices=("pdfplumber", "pdfplumber_words"), default="pdfplumber_words")
    parser.add_argument("--max-pages", type=int, default=0, help="Parse at most N pages per PDF (0 means all).")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N selected PDFs (0 means all).")
    parser.add_argument("--execute", action="store_true", help="Persist accepted tables to PostgreSQL.")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "finance_table_backfill.json")
    args = parser.parse_args()

    paths = select_pdf_paths(
        args.documents_dir,
        args.dataset,
        args.filename,
        all_pdfs=args.all_pdfs,
    )
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit("No PDFs selected for table backfill.")

    os.environ["TABLE_AWARE_INGESTION"] = "true"
    sys.path.insert(0, str(BACKEND))
    from table_parser import TableAwareParser
    from evidence_frame import is_evidence_frame_eligible_table
    from table_store import TableStore

    table_parser = TableAwareParser()
    table_store = TableStore()
    records = []
    print(
        f"[setup] documents={len(paths)} backend={args.backend} execute={args.execute} "
        f"max_pages={args.max_pages or 'all'}",
        flush=True,
    )
    for index, path in enumerate(paths, 1):
        started = time.perf_counter()
        error = ""
        tables = []
        stored = 0
        try:
            candidates = table_parser.extract_tables(
                str(path),
                path.name,
                parser_backend=args.backend,
                max_pages=max(1, args.max_pages) if args.max_pages > 0 else None,
                include_rejected=True,
            )
            tables = [table for table in candidates if is_evidence_frame_eligible_table(table)]
            if args.execute:
                stored = table_store.upsert_tables(tables)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        record = {
            "filename": path.name,
            "accepted_tables": len(tables),
            "normalized_tables": sum(bool(table.get("normalized")) for table in tables),
            "quality_recovered_tables": sum(table.get("accepted", True) is False for table in tables),
            "stored_tables": stored,
            "error": error,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        records.append(record)
        print(
            f"[{index:02d}/{len(paths)}] {path.name}: accepted={len(tables)} "
            f"normalized={record['normalized_tables']} stored={stored}"
            f" recovered={record['quality_recovered_tables']}"
            + (f" error={error}" if error else ""),
            flush=True,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "execute": args.execute,
                    "backend": args.backend,
                    "documents": len(paths),
                    "records_completed": len(records),
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    summary = {
        "documents": len(records),
        "accepted_tables": sum(item["accepted_tables"] for item in records),
        "normalized_tables": sum(item["normalized_tables"] for item in records),
        "quality_recovered_tables": sum(item["quality_recovered_tables"] for item in records),
        "stored_tables": sum(item["stored_tables"] for item in records),
        "errors": sum(bool(item["error"]) for item in records),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"Report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
