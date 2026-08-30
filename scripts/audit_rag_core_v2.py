"""Read-only audit of the stores used by the RAG Core v2 experiment."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def _nonempty(value: object) -> bool:
    return bool(str(value or "").strip())


def _audit_baseline_compression(path: Path) -> dict:
    if not path.exists():
        return {"available": False, "source": str(path)}
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    traces = [record.get("rag_trace") or {} for record in records]
    original = [int(trace.get("answer_context_original_chars") or 0) for trace in traces]
    compact = [int(trace.get("answer_context_chars") or 0) for trace in traces]
    retained = [after / before for before, after in zip(original, compact) if before > 0]
    input_tokens = [
        int((record.get("usage") or {}).get("input_tokens") or (record.get("usage") or {}).get("prompt_tokens") or 0)
        for record in records
    ]
    return {
        "available": True,
        "source": str(path),
        "records": len(records),
        "strategy_counts": dict(Counter(str(trace.get("answer_context_strategy") or "unknown") for trace in traces)),
        "average_original_chars": round(statistics.fmean(original), 2) if original else 0,
        "average_compact_chars": round(statistics.fmean(compact), 2) if compact else 0,
        "average_retained_ratio": round(statistics.fmean(retained), 4) if retained else 0,
        "average_removed_ratio": round(1 - statistics.fmean(retained), 4) if retained else 0,
        "median_retained_ratio": round(statistics.median(retained), 4) if retained else 0,
        "average_evidence_units": round(
            statistics.fmean(int(trace.get("answer_context_unit_count") or 0) for trace in traces), 2
        ) if traces else 0,
        "average_prompt_tokens": round(statistics.fmean(input_tokens), 2) if input_tokens else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "rag_core_v2_store_audit.json")
    parser.add_argument(
        "--baseline-answers",
        type=Path,
        default=ROOT / "reports" / "evidencerag-clean-baseline-v1_answers.jsonl",
    )
    args = parser.parse_args()

    from database import SessionLocal
    from milvus_client import MilvusManager
    from models import DocumentPage, DocumentTable, ParentChunk
    from table_config import get_table_aware_config

    db = SessionLocal()
    try:
        parent_rows = db.query(ParentChunk).all()
        page_rows = db.query(DocumentPage).all()
        table_rows = db.query(DocumentTable).all()
    finally:
        db.close()

    manager = MilvusManager()
    milvus_rows = manager.query_all(
        output_fields=["text", "filename", "chunk_level", "evidence_type", "table_id"]
    )
    chunk_levels = Counter(int(row.get("chunk_level", 0) or 0) for row in milvus_rows)
    evidence_types = Counter(str(row.get("evidence_type") or "text_chunk") for row in milvus_rows)
    chunk_lengths = [len(str(row.get("text") or "")) for row in milvus_rows]
    page_lengths = [len(str(row.page_text or "")) for row in page_rows]
    table_config = get_table_aware_config()

    required_page_fields = {
        "filename": lambda row: _nonempty(row.filename),
        "page_number": lambda row: row.page_number is not None,
        "page_text": lambda row: _nonempty(row.page_text),
        "table_text": lambda row: _nonempty(row.table_text),
        "company": lambda row: _nonempty(row.company),
        "report_year": lambda row: int(row.report_year or 0) > 0,
        "financial_document_type": lambda row: _nonempty(row.financial_document_type),
    }
    field_counts = {
        name: sum(bool(predicate(row)) for row in page_rows)
        for name, predicate in required_page_fields.items()
    }
    table_documents = {row.filename for row in table_rows if _nonempty(row.filename)}
    table_pages = {(row.filename, row.page_number) for row in table_rows if _nonempty(row.filename)}
    payload = {
        "collection": manager.collection_name,
        "milvus": {
            "records": len(milvus_rows),
            "chunk_levels": dict(sorted(chunk_levels.items())),
            "evidence_types": dict(sorted(evidence_types.items())),
            "average_text_chars": round(statistics.fmean(chunk_lengths), 2) if chunk_lengths else 0,
            "median_text_chars": round(statistics.median(chunk_lengths), 2) if chunk_lengths else 0,
            "documents": len({str(row.get("filename") or "") for row in milvus_rows if row.get("filename")}),
        },
        "postgres_parent_chunks": {
            "records": len(parent_rows),
            "chunk_levels": dict(sorted(Counter(int(row.chunk_level or 0) for row in parent_rows).items())),
            "average_text_chars": round(statistics.fmean(len(row.text or "") for row in parent_rows), 2) if parent_rows else 0,
        },
        "document_pages": {
            "records": len(page_rows),
            "documents": len({row.filename for row in page_rows}),
            "average_page_chars": round(statistics.fmean(page_lengths), 2) if page_lengths else 0,
            "median_page_chars": round(statistics.median(page_lengths), 2) if page_lengths else 0,
            "field_nonempty_counts": field_counts,
            "field_nonempty_rates": {
                name: round(count / len(page_rows), 4) if page_rows else 0
                for name, count in field_counts.items()
            },
        },
        "table_store": {
            "accepted_records": len(table_rows),
            "documents_with_tables": len(table_documents),
            "pages_with_tables": len(table_pages),
            "rows_total": sum(len(row.rows or []) for row in table_rows),
            "tables_with_columns": sum(bool(row.columns) for row in table_rows),
            "tables_with_csv": sum(_nonempty(row.csv_text) for row in table_rows),
        },
        "configured_table_parser": {
            "ingestion_enabled": table_config.table_aware_ingestion,
            "retrieval_mode": table_config.table_aware_retrieval,
            "backend": table_config.table_parser_backend,
            "historical_backend_auditable_from_store": False,
        },
        "environment": {
            "candidate_k": int(os.getenv("FINANCE_RAG_CANDIDATE_K", "40")),
            "final_top_k": int(os.getenv("FINANCE_RAG_FINAL_TOP_K", "5")),
            "answer_max_units": int(os.getenv("RAG_ANSWER_MAX_EVIDENCE_UNITS", "10")),
            "answer_max_context_chars": int(os.getenv("RAG_ANSWER_MAX_CONTEXT_CHARS", "24000")),
            "answer_max_unit_chars": int(os.getenv("RAG_ANSWER_MAX_UNIT_CHARS", "2000")),
        },
        "clean_baseline_compression": _audit_baseline_compression(args.baseline_answers),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
