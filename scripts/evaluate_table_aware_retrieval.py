"""Retrieval-only A/B for frozen text pages plus a shadow table index."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

from sqlalchemy import func


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from database import SessionLocal  # noqa: E402
from models import DocumentTable  # noqa: E402
from table_retrieval import (  # noqa: E402
    DEFAULT_TABLE_COLLECTION,
    TableMilvusManager,
    merge_text_and_table_pages,
)


DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_FROZEN = ROOT / "reports" / "retrieval_document_local_diagnostic30.json"
DEFAULT_OUTPUT = ROOT / "reports" / "table_aware_retrieval_v1_diagnostic30.json"
KS = (5, 10, 20, 30)


def _filename(value: object) -> str:
    name = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return (name if name.casefold().endswith(".pdf") else f"{name}.pdf").casefold()


def _page_key(item: dict) -> tuple[str, int]:
    return _filename(item.get("filename") or item.get("doc_name")), int(item.get("page_number") or 0)


def _gold_pages(row: dict) -> set[tuple[str, int]]:
    """Use FinanceBench evidence pages directly as internal loader pages."""
    return {
        (_filename(item.get("doc_name")), int(item.get("evidence_page_num") or 0))
        for item in json.loads(row.get("evidence") or "[]")
    }


def _benchmark_gold_pages(row: dict) -> list[list]:
    return sorted([
        [_filename(item.get("doc_name")), int(item.get("evidence_page_num") or 0)]
        for item in json.loads(row.get("evidence") or "[]")
    ])


def _rank_pages(items: list[dict], gold_pages: set[tuple[str, int]]) -> int | None:
    for rank, item in enumerate(items, 1):
        if _page_key(item) in gold_pages:
            return rank
    return None


def _rank_tables(items: list[dict], gold_table_ids: set[str]) -> int | None:
    for rank, item in enumerate(items, 1):
        if str(item.get("table_id") or "") in gold_table_ids:
            return rank
    return None


def _mean(values) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 2) if usable else None


def _rate(values) -> float | None:
    usable = [value for value in values if value is not None]
    return round(sum(bool(value) for value in usable) / len(usable), 4) if usable else None


def _table_catalog(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    filenames = {key[0] for row in rows for key in _gold_pages(row)}
    db = SessionLocal()
    try:
        tables = db.query(DocumentTable).filter(func.lower(DocumentTable.filename).in_(filenames)).all()
        result: dict[tuple[str, int], list[dict]] = {}
        for table in tables:
            key = (_filename(table.filename), int(table.page_number or 0))
            result.setdefault(key, []).append({
                "table_id": table.table_id,
                "quality_score": float(table.quality_score or 0.0),
                "row_count": len(table.rows or []),
                "column_count": len(table.columns or []),
            })
        return result
    finally:
        db.close()


def summarize(records: list[dict]) -> dict:
    eligible = [item for item in records if item["gold_table_ids"]]
    return {
        "questions": len(records),
        "gold_table_eligible_questions": len(eligible),
        "candidate_hit": {
            "text": _rate([item["text"]["candidate_hit"] for item in records]),
            "text_plus_table": _rate([item["text_plus_table"]["candidate_hit"] for item in records]),
        },
        "gold_page_hit_at_k": {
            str(k): {
                "text": _rate([item["text"]["gold_page_rank"] is not None and item["text"]["gold_page_rank"] <= k for item in records]),
                "text_plus_table": _rate([
                    item["text_plus_table"]["gold_page_rank"] is not None
                    and item["text_plus_table"]["gold_page_rank"] <= k
                    for item in records
                ]),
            }
            for k in KS
        },
        "gold_table_hit_at_30": _rate([
            item["table"]["fused_gold_table_rank"] is not None
            and item["table"]["fused_gold_table_rank"] <= 30
            for item in eligible
        ]),
        "table_recall_at_k": {
            route: {
                str(k): _rate([
                    item["table"][f"{route}_gold_table_rank"] is not None
                    and item["table"][f"{route}_gold_table_rank"] <= k
                    for item in eligible
                ])
                for k in KS
            }
            for route in ("dense", "bm25", "fused")
        },
        "average_candidate_count": {
            "text": _mean([item["text"]["candidate_count"] for item in records]),
            "table": _mean([item["table"]["candidate_count"] for item in records]),
            "text_plus_table": _mean([item["text_plus_table"]["candidate_count"] for item in records]),
        },
        "average_latency_ms": {
            "frozen_text": _mean([item["latency_ms"]["frozen_text"] for item in records]),
            "table_query_embedding": _mean([item["latency_ms"]["table_query_embedding"] for item in records]),
            "table_dense": _mean([item["latency_ms"]["table_dense"] for item in records]),
            "table_bm25": _mean([item["latency_ms"]["table_bm25"] for item in records]),
            "table_total_excluding_embedding": _mean([
                item["latency_ms"]["table_total_excluding_embedding"] for item in records
            ]),
            "shadow_incremental_total": _mean([item["latency_ms"]["shadow_incremental_total"] for item in records]),
        },
        "recovered_candidate_ids": [
            item["financebench_id"] for item in records
            if not item["text"]["candidate_hit"] and item["text_plus_table"]["candidate_hit"]
        ],
        "page_rank_migration": {
            "improved": sum(
                item["text_plus_table"]["gold_page_rank"] is not None
                and (item["text"]["gold_page_rank"] is None
                     or item["text_plus_table"]["gold_page_rank"] < item["text"]["gold_page_rank"])
                for item in records
            ),
            "unchanged": sum(item["text_plus_table"]["gold_page_rank"] == item["text"]["gold_page_rank"] for item in records),
            "regressed": sum(
                item["text"]["gold_page_rank"] is not None
                and (item["text_plus_table"]["gold_page_rank"] is None
                     or item["text_plus_table"]["gold_page_rank"] > item["text"]["gold_page_rank"])
                for item in records
            ),
        },
        "table_extraction": {
            "gold_page_table_coverage": round(len(eligible) / max(1, len(records)), 4),
            "gold_pages_without_tables": sum(not item["gold_table_ids"] for item in records),
            "adjacent_table_present_when_gold_page_missing": sum(
                not item["gold_table_ids"] and item["gold_table_diagnostics"]["adjacent_table_count"] > 0
                for item in records
            ),
            "gold_table_count": sum(len(item["gold_table_ids"]) for item in records),
            "gold_tables_quality_at_least_065": sum(
                item["gold_table_diagnostics"]["quality_at_least_065"] for item in records
            ),
            "average_gold_table_quality": _mean([
                score for item in records for score in item["gold_table_diagnostics"]["quality_scores"]
            ]),
        },
    }


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Table-aware Retrieval v1 — Shadow A/B", "",
        "> Fixed diagnostic30; retrieval-only. Existing text retrieval is frozen; only local table Dense/BM25 is executed. Jina/LLM/Judge calls are zero.", "",
        "- FinanceBench `evidence_page_num` already follows the internal PDF-page contract and is used without offset conversion.",
        f"- Table collection: `{payload['table_collection']}`",
        f"- Questions: {summary['questions']}",
        f"- Gold-table eligible questions: {summary['gold_table_eligible_questions']}",
        f"- Candidate hit A/B: {summary['candidate_hit']['text']:.2%} / {summary['candidate_hit']['text_plus_table']:.2%}",
        f"- Gold table hit@30: {summary['gold_table_hit_at_30'] if summary['gold_table_hit_at_30'] is not None else 'n/a'}",
        "", "## Gold page hit", "",
        "| K | Text | Text + table |", "|---:|---:|---:|",
    ]
    for k in KS:
        item = summary["gold_page_hit_at_k"][str(k)]
        lines.append(f"| {k} | {item['text']:.2%} | {item['text_plus_table']:.2%} |")
    lines.extend(["", "## Table recall", "", "| Route | @5 | @10 | @20 | @30 |", "|---|---:|---:|---:|---:|"])
    for route in ("dense", "bm25", "fused"):
        values = summary["table_recall_at_k"][route]
        lines.append("| " + route + " | " + " | ".join(
            "n/a" if values[str(k)] is None else f"{values[str(k)]:.2%}" for k in KS
        ) + " |")
    lines.extend(["", "## Cost", ""])
    for key, value in summary["average_candidate_count"].items():
        lines.append(f"- Average candidate count `{key}`: {value}")
    for key, value in summary["average_latency_ms"].items():
        lines.append(f"- Average latency `{key}`: {value} ms")
    lines.extend(["", "## Table extraction diagnostics", ""])
    for key, value in summary["table_extraction"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append(f"- Page rank migration: {summary['page_rank_migration']}")
    lines.extend(["", "## Per question", "", "| ID | Gold pages | Gold tables | Text rank | Combined rank | Dense/BM25/Fused table rank | Candidates A/B |", "|---|---|---:|---:|---:|---|---:|"])
    for item in payload["records"]:
        pages = ", ".join(f"{name} p.{page}" for name, page in item["gold_pages"])
        table_ranks = "/".join(
            str(item["table"][f"{route}_gold_table_rank"] or "—") for route in ("dense", "bm25", "fused")
        )
        lines.append(
            f"| `{item['financebench_id']}` | {pages} | {len(item['gold_table_ids'])} | "
            f"{item['text']['gold_page_rank'] or '—'} | {item['text_plus_table']['gold_page_rank'] or '—'} | "
            f"{table_ranks} | {item['text']['candidate_count']} / {item['text_plus_table']['candidate_count']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    from embedding import embedding_service

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--frozen-report", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--variant", default="C_global_local_merge")
    parser.add_argument("--table-collection", default=DEFAULT_TABLE_COLLECTION)
    parser.add_argument("--table-k", type=int, default=30)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 30:
        raise SystemExit("--limit must be between 1 and 30")

    frozen = json.loads(args.frozen_report.read_text(encoding="utf-8"))
    frozen_records = frozen["records"][: args.limit]
    with args.dataset.open(encoding="utf-8-sig", newline="") as handle:
        by_id = {row["financebench_id"]: row for row in csv.DictReader(handle)}
    rows = [by_id[item["financebench_id"]] for item in frozen_records]
    table_catalog = _table_catalog(rows)
    manager = TableMilvusManager(args.table_collection)
    if not manager.has_collection() or manager.count() == 0:
        raise RuntimeError("table shadow collection is missing or empty; run build_table_shadow_index.py first")

    records = []
    for index, (row, frozen_record) in enumerate(zip(rows, frozen_records), 1):
        text_pages = frozen_record["page_candidate_traces"][args.variant].get("expanded_pages") or []
        gold_pages = _gold_pages(row)
        gold_tables = [table for key in gold_pages for table in table_catalog.get(key, [])]
        gold_table_ids = {table["table_id"] for table in gold_tables}
        adjacent_tables = [
            table
            for filename, page_number in gold_pages
            for adjacent_page in (page_number - 1, page_number + 1)
            if adjacent_page >= 0
            for table in table_catalog.get((filename, adjacent_page), [])
        ]
        embedding_started = time.perf_counter()
        query_embedding = embedding_service.get_embeddings([row["question"]])[0]
        embedding_ms = (time.perf_counter() - embedding_started) * 1000
        table_result = manager.retrieve(row["question"], query_embedding, top_k=args.table_k)
        combined_pages = merge_text_and_table_pages(text_pages, table_result["fused"])
        text_rank = _rank_pages(text_pages, gold_pages)
        combined_rank = _rank_pages(combined_pages, gold_pages)
        dense_table_rank = _rank_tables(table_result["dense"], gold_table_ids)
        bm25_table_rank = _rank_tables(table_result["bm25"], gold_table_ids)
        fused_table_rank = _rank_tables(table_result["fused"], gold_table_ids)
        text_keys = {_page_key(item) for item in text_pages}
        combined_keys = {_page_key(item) for item in combined_pages}
        record = {
            "financebench_id": row["financebench_id"],
            "question": row["question"],
            "benchmark_gold_pages": _benchmark_gold_pages(row),
            "gold_pages": sorted([list(item) for item in gold_pages]),
            "gold_table_ids": sorted(gold_table_ids),
            "gold_table_diagnostics": {
                "quality_scores": [table["quality_score"] for table in gold_tables],
                "quality_at_least_065": sum(table["quality_score"] >= 0.65 for table in gold_tables),
                "rows_nonempty": sum(table["row_count"] > 0 for table in gold_tables),
                "columns_nonempty": sum(table["column_count"] > 0 for table in gold_tables),
                "adjacent_table_count": len(adjacent_tables),
            },
            "text": {
                "candidate_hit": bool(gold_pages & text_keys),
                "gold_page_rank": text_rank,
                "candidate_count": len(text_keys),
            },
            "table": {
                "candidate_count": len(table_result["fused"]),
                "dense_gold_table_rank": dense_table_rank,
                "bm25_gold_table_rank": bm25_table_rank,
                "fused_gold_table_rank": fused_table_rank,
                "dense_results": table_result["dense"],
                "bm25_results": table_result["bm25"],
                "fused_results": table_result["fused"],
            },
            "text_plus_table": {
                "candidate_hit": bool(gold_pages & combined_keys),
                "gold_page_rank": combined_rank,
                "candidate_count": len(combined_keys),
            },
            "latency_ms": {
                "frozen_text": float(frozen_record.get("total_latency_ms") or 0.0),
                "table_query_embedding": round(embedding_ms, 2),
                "table_dense": table_result["latency_ms"]["dense"],
                "table_bm25": table_result["latency_ms"]["bm25"],
                "table_total_excluding_embedding": table_result["latency_ms"]["total"],
                "shadow_incremental_total": round(embedding_ms + table_result["latency_ms"]["total"], 2),
            },
        }
        records.append(record)
        print(
            f"[{index:02d}/{len(rows)}] {row['financebench_id']} page={text_rank}->{combined_rank} "
            f"table={dense_table_rank}/{bm25_table_rank}/{fused_table_rank}",
            flush=True,
        )

    payload = {
        "profile": "table_aware_retrieval_v1_shadow",
        "evaluation_scope": "fixed diagnostic30; retrieval-only; no Jina/LLM/Judge",
        "table_collection": args.table_collection,
        "frozen_report": str(args.frozen_report),
        "variant": args.variant,
        "config": {"table_k": args.table_k, "ks": KS},
        "summary": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {args.output}\nMarkdown: {markdown}", flush=True)


if __name__ == "__main__":
    main()
