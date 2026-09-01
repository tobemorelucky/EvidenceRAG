"""Run the fixed diagnostic30 Retrieval Core v4 ablation profiles.

This is retrieval-only evaluation. It never calls the answer model or Judge and
never reads benchmark gold data inside the retrieval pipeline. Gold pages are
used only after retrieval to compute offline metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from document_page_store import DocumentPageStore  # noqa: E402
from rag_core_v3 import build_core_v3_evidence, merge_opened_pages, select_core_v3_pages  # noqa: E402
from rag_core_v4 import retrieve_dense_primary  # noqa: E402
from rag_utils import _rerank_documents  # noqa: E402
from runtime_profile import RETRIEVAL_DENSE_PRIMARY_PROFILE, apply_runtime_profile  # noqa: E402
from table_store import TableStore  # noqa: E402


DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json"
DEFAULT_BASELINE = ROOT / "reports" / "evidencerag-rag-core-v3-skills-all100-final-evidence-diagnostic.json"
PROFILES = {RETRIEVAL_DENSE_PRIMARY_PROFILE}


def _normalize_filename(value: object) -> str:
    name = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return name.casefold() if name.casefold().endswith(".pdf") else f"{name}.pdf".casefold()


def _page_key(item: dict) -> tuple[str, int]:
    try:
        page = int(item.get("page_number") or 0)
    except (TypeError, ValueError):
        page = 0
    return _normalize_filename(item.get("filename")), page


def _gold(row: dict) -> set[tuple[str, int]]:
    return {
        (_normalize_filename(item.get("doc_name")), int(item.get("evidence_page_num") or 0))
        for item in json.loads(row.get("evidence") or "[]")
    }


def _diagnostic_rows(dataset: Path, fixture: Path) -> tuple[list[dict], dict[str, str]]:
    groups = json.loads(fixture.read_text(encoding="utf-8"))
    ids: list[str] = []
    membership: dict[str, str] = {}
    for group in ("candidate_miss10", "selection_loss10", "correct_regression10"):
        for value in groups[group]:
            item = value if isinstance(value, str) else value["financebench_id"]
            ids.append(item)
            membership[item] = group
    with dataset.open(encoding="utf-8-sig", newline="") as handle:
        wanted = set(ids)
        by_id = {row["financebench_id"]: row for row in csv.DictReader(handle) if row["financebench_id"] in wanted}
    missing = [item for item in ids if item not in by_id]
    if missing:
        raise RuntimeError(f"diagnostic fixture IDs missing from dataset: {missing}")
    return [by_id[item] for item in ids], membership


def _first_page_rank(items: list[dict], gold: set[tuple[str, int]]) -> int | None:
    for rank, item in enumerate(items, 1):
        if _page_key(item) in gold:
            return rank
    return None


def _first_document_rank(items: list[dict], gold: set[tuple[str, int]]) -> int | None:
    gold_documents = {filename for filename, _ in gold}
    seen: set[str] = set()
    document_rank = 0
    for item in items:
        filename = _page_key(item)[0]
        if not filename or filename in seen:
            continue
        seen.add(filename)
        document_rank += 1
        if filename in gold_documents:
            return document_rank
    return None


def _load_baseline(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["financebench_id"]: item for item in payload.get("records") or []}


def _metrics(records: list[dict], baseline: dict[str, dict]) -> dict:
    count = max(1, len(records))
    recovered = []
    regressed = []
    for item in records:
        previous = baseline.get(item["financebench_id"], {})
        old = bool(previous.get("context_page_hit"))
        new = bool(item["context_hit"])
        if new and not old:
            recovered.append(item["financebench_id"])
        elif old and not new:
            regressed.append(item["financebench_id"])
    groups = {}
    for group in ("candidate_miss10", "selection_loss10", "correct_regression10"):
        subset = [item for item in records if item["group"] == group]
        groups[group] = {
            "questions": len(subset),
            "candidate_hit": round(sum(item["candidate_hit"] for item in subset) / max(1, len(subset)), 4),
            "selected_hit": round(sum(item["selected_hit"] for item in subset) / max(1, len(subset)), 4),
            "context_hit": round(sum(item["context_hit"] for item in subset) / max(1, len(subset)), 4),
        }
    return {
        "questions": len(records),
        "candidate_hit": round(sum(item["candidate_hit"] for item in records) / count, 4),
        "selected_hit": round(sum(item["selected_hit"] for item in records) / count, 4),
        "context_hit": round(sum(item["context_hit"] for item in records) / count, 4),
        "average_gold_page_rank": round(sum(item["gold_page_rank"] for item in records if item["gold_page_rank"]) / max(1, sum(item["gold_page_rank"] is not None for item in records)), 2),
        "average_gold_document_rank": round(sum(item["gold_document_rank"] for item in records if item["gold_document_rank"]) / max(1, sum(item["gold_document_rank"] is not None for item in records)), 2),
        "jina_calls": sum(item["jina_calls"] for item in records),
        "jina_chars": sum(item["jina_chars"] for item in records),
        "average_latency_ms": round(sum(item["latency_ms"] for item in records) / count, 2),
        "average_context_chars": round(sum(item["context_chars"] for item in records) / count, 2),
        "average_estimated_context_tokens": round(sum(item["estimated_context_tokens"] for item in records) / count, 2),
        "context_recovered_vs_core_v3": recovered,
        "context_regressed_vs_core_v3": regressed,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rerank-interval-seconds", type=float, default=8.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    apply_runtime_profile(args.profile)
    rows, membership = _diagnostic_rows(args.dataset, args.fixture)
    if args.limit:
        rows = rows[: args.limit]
    output = args.output or ROOT / "reports" / f"{args.profile}_diagnostic30.json"
    baseline = _load_baseline(args.baseline)
    page_store = DocumentPageStore()
    table_store = TableStore()
    records = []
    last_remote_request_at: float | None = None
    print(f"[setup] profile={args.profile} questions={len(rows)} dense_k=120 bm25_k=30", flush=True)

    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        retrieval = retrieve_dense_primary(row["question"], dense_k=120, bm25_k=30)
        candidates = retrieval["merged"]
        if last_remote_request_at is not None and args.rerank_interval_seconds > 0:
            wait = args.rerank_interval_seconds - (time.monotonic() - last_remote_request_at)
            if wait > 0:
                print(f"[rate-limit] waiting {wait:.1f}s", flush=True)
                time.sleep(wait)
        reranked, rerank_meta = _rerank_documents(
            row["question"], candidates, top_k=16, remote_candidate_k=18,
        )
        if rerank_meta.get("remote_success") and not rerank_meta.get("rerank_cache_hit"):
            last_remote_request_at = time.monotonic()
        selected, selector_trace = select_core_v3_pages(row["question"], candidates, reranked)
        selected_keys = [(item["filename"], int(item["page_number"])) for item in selected]
        opened = page_store.get_pages_by_keys(selected_keys)
        answer_docs = merge_opened_pages(selected, opened)
        tables = table_store.get_tables_by_page_keys(selected_keys)
        evidence, context_meta = build_core_v3_evidence(row["question"], answer_docs, tables)
        gold = _gold(row)
        candidate_keys = {_page_key(item) for item in candidates}
        selected_keys_normalized = {_page_key(item) for item in selected}
        context_keys = {
            (_normalize_filename(item.get("filename")), int(item.get("page_number") or 0))
            for item in context_meta.get("answer_context_pages") or []
        }
        jina_chars = int(rerank_meta.get("remote_rerank_input_chars") or 0)
        record = {
            "financebench_id": row["financebench_id"],
            "group": membership[row["financebench_id"]],
            "candidate_hit": bool(gold & candidate_keys),
            "selected_hit": bool(gold & selected_keys_normalized),
            "context_hit": bool(gold & context_keys),
            "gold_page_rank": _first_page_rank(candidates, gold),
            "gold_document_rank": _first_document_rank(candidates, gold),
            "dense_gold_page_rank": _first_page_rank(retrieval["dense"], gold),
            "bm25_gold_page_rank": _first_page_rank(retrieval["bm25"], gold),
            "merged_rank_trace": [
                {
                    "filename": item.get("filename"),
                    "page_number": item.get("page_number"),
                    "dense_rank": item.get("dense_rank"),
                    "bm25_rank": item.get("bm25_rank"),
                    "merged_rank": item.get("merged_rank"),
                }
                for item in candidates
            ],
            "selected_pages": selector_trace.get("selected_pages") or [],
            "jina_calls": int(bool(rerank_meta.get("remote_success") or rerank_meta.get("rerank_cache_hit"))),
            "jina_chars": jina_chars,
            "rerank_provider": rerank_meta.get("rerank_provider") or "none",
            "context_chars": len(evidence),
            "estimated_context_tokens": math.ceil(len(evidence) / 4),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "latency_breakdown_ms": retrieval["latency_ms"],
        }
        records.append(record)
        print(
            f"[{index:02d}/{len(rows)}] {row['financebench_id']} "
            f"candidate={record['candidate_hit']} selected={record['selected_hit']} "
            f"context={record['context_hit']} page_rank={record['gold_page_rank']}",
            flush=True,
        )

    metrics = _metrics(records, baseline)
    payload = {
        "profile": args.profile,
        "evaluation_scope": "fixed diagnostic30; retrieval-only",
        "metrics": metrics,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text("\n".join([
        f"# Retrieval Core v4 — {args.profile}", "",
        "> Fixed diagnostic30; retrieval-only. No answer model or Judge was called.", "",
        f"- Candidate hit: {metrics['candidate_hit']:.2%}",
        f"- Selected hit: {metrics['selected_hit']:.2%}",
        f"- Context hit: {metrics['context_hit']:.2%}",
        f"- Average gold page/document rank: {metrics['average_gold_page_rank']} / {metrics['average_gold_document_rank']}",
        f"- Jina calls/chars: {metrics['jina_calls']} / {metrics['jina_chars']}",
        f"- Average latency: {metrics['average_latency_ms']:.2f} ms",
        f"- Average context chars / estimated tokens: {metrics['average_context_chars']:.0f} / {metrics['average_estimated_context_tokens']:.0f}",
        f"- Context recovered vs Core v3: {', '.join(metrics['context_recovered_vs_core_v3']) or 'none'}",
        f"- Context regressed vs Core v3: {', '.join(metrics['context_regressed_vs_core_v3']) or 'none'}", "",
    ]), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {output}\nMarkdown: {markdown}", flush=True)


if __name__ == "__main__":
    main()
