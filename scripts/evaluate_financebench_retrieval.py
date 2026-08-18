"""FinanceBench evidence-retrieval evaluation.

This evaluates retrieval only.  It never calls the answer model, LangSmith, a
query planner, or agent.  Remote reranking is disabled by default and can be
enabled explicitly for a controlled retrieval-only comparison. Gold pages come
from the benchmark CSV's ``evidence`` field and results are written for failure
analysis.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_REPORT_DIR = ROOT / "reports"


def parse_gold_evidence(row: dict[str, str]) -> list[dict[str, Any]]:
    """Normalize FinanceBench evidence records without assuming one page per question."""
    raw = (row.get("evidence") or "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    records = payload if isinstance(payload, list) else [payload]
    result = []
    for record in records:
        if not isinstance(record, dict):
            continue
        document = str(record.get("doc_name") or row.get("doc_name") or "").strip()
        page = record.get("evidence_page_num")
        try:
            page = int(page)
        except (TypeError, ValueError):
            continue
        result.append(
            {
                "filename": document if document.lower().endswith(".pdf") else f"{document}.pdf",
                "page_number": page,
                "text": str(record.get("evidence_text") or ""),
            }
        )
    return result


def select_development_rows(rows: list[dict[str, str]], development_size: int = 20) -> list[dict[str, str]]:
    """Stable stratified development split by question type; remaining rows are holdout."""
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get("question_type") or "unknown", []).append(row)
    selected: list[dict[str, str]] = []
    ordered_groups = sorted(groups.items())
    while len(selected) < development_size:
        added = False
        for _, group in ordered_groups:
            group.sort(key=lambda item: item.get("financebench_id") or "")
            if group:
                selected.append(group.pop(0))
                added = True
                if len(selected) == development_size:
                    break
        if not added:
            break
    return selected


def is_development_row(row: dict[str, str], development_ids: set[str]) -> bool:
    return (row.get("financebench_id") or "") in development_ids


def pages_match(retrieved_page: int, gold_page: int, offset: int) -> bool:
    return retrieved_page == gold_page + offset


def page_hit_at_k(gold: list[dict[str, Any]], retrieved: list[dict[str, Any]], k: int, offset: int = 0) -> bool:
    return any(
        hit.get("filename") == item["filename"] and pages_match(int(hit.get("page_number", -1)), item["page_number"], offset)
        for hit in retrieved[:k]
        for item in gold
    )


def classify_failure(gold: list[dict[str, Any]], retrieved: list[dict[str, Any]]) -> str:
    if not gold:
        return "missing_gold_evidence"
    if not retrieved:
        return "empty_retrieval"
    gold_filenames = {item["filename"] for item in gold}
    document_hits = [item for item in retrieved if item.get("filename") in gold_filenames]
    if not document_hits:
        return "gold_document_not_retrieved"
    if page_hit_at_k(gold, document_hits, len(document_hits)):
        return "gold_page_retrieved"
    if page_hit_at_k(gold, document_hits, len(document_hits), offset=-1):
        return "gold_page_retrieved_offset_only"
    return "gold_page_not_retrieved"


def _retrieval_environment(*, enable_rerank: bool) -> None:
    """Disable non-retrieval branches before importing the RAG implementation."""
    os.environ["RAG_QUERY_PLANNER_ENABLED"] = "false"
    os.environ["RAG_ANCHOR_GUARD_ENABLED"] = "false"
    os.environ["RAG_COVER_FILTER_ENABLED"] = "false"
    os.environ["RAG_PAGE_FIRST_ENABLED"] = "true"
    os.environ["RAG_PAGE_NEIGHBOR_WINDOW"] = "0"
    os.environ["FINANCE_RAG_ENABLE_STEP_BACK"] = "false"
    if not enable_rerank:
        os.environ["RERANK_MODEL"] = ""
        os.environ["RERANK_BINDING_HOST"] = ""
        os.environ["RERANK_API_KEY"] = ""
    os.environ["TABLE_AWARE_RETRIEVAL"] = "off"
    os.environ["RAG_EVIDENCE_GROUPING_ENABLED"] = "false"


def evaluate(
    rows: list[dict[str, str]],
    candidate_k: int,
    final_k: int,
    *,
    enable_rerank: bool = False,
) -> list[dict[str, Any]]:
    _retrieval_environment(enable_rerank=enable_rerank)
    sys.path.insert(0, str(BACKEND))
    from rag_utils import retrieve_documents

    results = []
    for index, row in enumerate(rows, 1):
        gold = parse_gold_evidence(row)
        retrieval = retrieve_documents(row["question"], top_k=final_k, candidate_k=candidate_k, apply_page_merge=False)
        candidates = list(retrieval.get("candidate_docs") or [])
        final_docs = list(retrieval.get("final_retrieved_docs") or retrieval.get("docs") or [])

        def compact(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {
                    "filename": doc.get("filename", ""),
                    "page_number": int(doc.get("page_number", -1) if doc.get("page_number") is not None else -1),
                    "score": doc.get("rerank_score", doc.get("score")),
                    "chunk_id": doc.get("chunk_id", ""),
                }
                for doc in docs
            ]

        candidate_hits = compact(candidates)
        final_hits = compact(final_docs)
        record = {
            "financebench_id": row.get("financebench_id", ""),
            "question_type": row.get("question_type", ""),
            "question": row.get("question", ""),
            "gold": gold,
            "candidate_failure": classify_failure(gold, candidate_hits),
            "final_failure": classify_failure(gold, final_hits),
            "candidate_hits": candidate_hits[:candidate_k],
            "final_hits": final_hits[:final_k],
            "candidate_page_hit_at": {
                str(k): page_hit_at_k(gold, candidate_hits, k)
                for k in (1, 5, 10, candidate_k)
            },
            "candidate_page_hit_offset_minus_one_at": {
                str(k): page_hit_at_k(gold, candidate_hits, k, offset=-1)
                for k in (1, 5, 10, candidate_k)
            },
            "final_page_hit_at": {
                str(k): page_hit_at_k(gold, final_hits, k)
                for k in (1, final_k)
            },
            "page_first_trace": {
                "selected_documents": (retrieval.get("meta") or {}).get("page_first_selected_documents", []),
                "selected_pages": (retrieval.get("meta") or {}).get("page_first_selected_pages", []),
                "fallback": (retrieval.get("meta") or {}).get("page_first_fallback", ""),
            },
            "rerank": {
                "enabled": bool((retrieval.get("meta") or {}).get("rerank_enabled")),
                "applied": bool((retrieval.get("meta") or {}).get("rerank_applied")),
                "model": (retrieval.get("meta") or {}).get("rerank_model"),
                "error": (retrieval.get("meta") or {}).get("rerank_error"),
                "local_enabled": bool((retrieval.get("meta") or {}).get("local_rerank_enabled")),
                "local_applied": bool((retrieval.get("meta") or {}).get("local_rerank_applied")),
                "local_error": (retrieval.get("meta") or {}).get("local_rerank_error"),
            },
            "retrieval_ms": (retrieval.get("meta") or {}).get("latency_breakdown", {}).get("total_retrieval_ms"),
        }
        results.append(record)
        print(f"[{index:02d}/{len(rows)}] {record['financebench_id']}: {record['candidate_failure']}", flush=True)
    return results


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    counts = Counter(record["candidate_failure"] for record in records)
    final_counts = Counter(record["final_failure"] for record in records)
    candidate_depth = max((max(map(int, record["candidate_page_hit_at"].keys())) for record in records), default=0)
    return {
        "questions": total,
        "candidate_document_hit_at_10": round(
            sum(
                any(hit.get("filename") == gold["filename"] for hit in record["candidate_hits"][:10] for gold in record["gold"])
                for record in records
            ) / total,
            4,
        ) if total else 0,
        "candidate_page_hit_at_1": round(sum(record["candidate_page_hit_at"].get("1", False) for record in records) / total, 4) if total else 0,
        "candidate_page_hit_at_5": round(sum(record["candidate_page_hit_at"].get("5", False) for record in records) / total, 4) if total else 0,
        "candidate_page_hit_at_10": round(sum(record["candidate_page_hit_at"].get("10", False) for record in records) / total, 4) if total else 0,
        f"candidate_page_hit_at_{candidate_depth}": round(
            sum(record["candidate_page_hit_at"].get(str(candidate_depth), False) for record in records) / total,
            4,
        ) if total else 0,
        "candidate_page_hit_offset_minus_one_at_10": round(sum(record["candidate_page_hit_offset_minus_one_at"].get("10", False) for record in records) / total, 4) if total else 0,
        "final_page_hit_at_1": round(sum(record["final_page_hit_at"].get("1", False) for record in records) / total, 4) if total else 0,
        "final_page_hit_at_5": round(sum(record["final_page_hit_at"].get("5", False) for record in records) / total, 4) if total else 0,
        "candidate_failure_counts": dict(sorted(counts.items())),
        "final_failure_counts": dict(sorted(final_counts.items())),
        "note": "Primary page-hit metrics use exact benchmark page numbers. The offset metric is diagnostic only for detecting an index convention mismatch.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FinanceBench evidence retrieval without answer-model tokens")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N selected rows (0 means all).")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Enable the configured rerank service; the answer model remains disabled.",
    )
    args = parser.parse_args()

    with args.dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    dev_ids = {row.get("financebench_id") or "" for row in select_development_rows(all_rows)}
    if args.split == "dev":
        rows = [row for row in all_rows if is_development_row(row, dev_ids)]
    elif args.split == "holdout":
        rows = [row for row in all_rows if not is_development_row(row, dev_ids)]
    else:
        rows = all_rows
    if args.limit > 0:
        rows = rows[: args.limit]

    records = evaluate(
        rows,
        candidate_k=max(1, args.candidate_k),
        final_k=max(1, args.final_k),
        enable_rerank=args.enable_rerank,
    )
    summary = build_summary(records)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "split": args.split,
        "candidate_k": args.candidate_k,
        "final_k": args.final_k,
        "limit": args.limit,
        "rerank_enabled": args.enable_rerank,
        "summary": summary,
        "records": records,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    output = args.report_dir / f"financebench_retrieval_{args.split}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
