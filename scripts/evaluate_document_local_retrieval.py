"""Evaluate Retrieval Core v4 document-local retrieval on diagnostic30.

Variants share one global discovery and one document-local retrieval pass:
A = current global page selection, B = local-only pages, C = local+global pages.
The script never calls Jina, the answer model, or Judge.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from document_page_store import DocumentPageStore  # noqa: E402
from rag_core_v3 import build_core_v3_evidence  # noqa: E402
from rag_core_v4 import (  # noqa: E402
    expand_and_rank_pages,
    merge_global_local_chunks,
    retrieve_dense_primary,
    retrieve_document_local_chunks,
    score_candidate_documents,
    select_document_first_pages,
)
from runtime_profile import RETRIEVAL_DOCUMENT_LOCAL_PROFILE, apply_runtime_profile  # noqa: E402
from table_store import TableStore  # noqa: E402


DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json"
VARIANTS = ("A_global", "B_document_local", "C_global_local_merge")


def _filename(value: object) -> str:
    name = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return (name if name.casefold().endswith(".pdf") else f"{name}.pdf").casefold()


def _page_key(item: dict) -> tuple[str, int]:
    try:
        page = int(item.get("page_number") or 0)
    except (TypeError, ValueError):
        page = 0
    return _filename(item.get("filename")), page


def _gold(row: dict) -> set[tuple[str, int]]:
    return {
        (_filename(item.get("doc_name")), int(item.get("evidence_page_num") or 0))
        for item in json.loads(row.get("evidence") or "[]")
    }


def _load_rows(dataset: Path, fixture: Path) -> tuple[list[dict], dict[str, str]]:
    groups = json.loads(fixture.read_text(encoding="utf-8"))
    ids, membership = [], {}
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
        raise RuntimeError(f"missing diagnostic IDs: {missing}")
    return [by_id[item] for item in ids], membership


def _rank(items: list[dict], gold: set[tuple[str, int]]) -> int | None:
    for rank, item in enumerate(items, 1):
        if _page_key(item) in gold:
            return rank
    return None


def _document_rank(trace: list[dict], gold: set[tuple[str, int]]) -> int | None:
    gold_documents = {item[0] for item in gold}
    for item in trace:
        if _filename(item.get("filename")) in gold_documents:
            return int(item.get("document_rank") or 0) or None
    return None


def _variant_result(
    question: str,
    page_candidates: list[dict],
    gold: set[tuple[str, int]],
    table_store: TableStore,
) -> dict:
    started = time.perf_counter()
    selected, selection_trace = select_document_first_pages(
        page_candidates, final_page_k=8, global_escape_pages=1,
    )
    selected_keys = [(str(item.get("filename") or ""), int(item.get("page_number") or 0)) for item in selected]
    tables = table_store.get_tables_by_page_keys(selected_keys)
    evidence, context_meta = build_core_v3_evidence(question, selected, tables)
    candidate_keys = {_page_key(item) for item in page_candidates}
    selected_keys_normalized = {_page_key(item) for item in selected}
    context_keys = {
        (_filename(item.get("filename")), int(item.get("page_number") or 0))
        for item in context_meta.get("answer_context_pages") or []
    }
    return {
        "candidate_hit": bool(gold & candidate_keys),
        "selected_hit": bool(gold & selected_keys_normalized),
        "context_hit": bool(gold & context_keys),
        "gold_page_rank": _rank(page_candidates, gold),
        "candidate_page_count": len(page_candidates),
        "selected_pages": [
            {
                "filename": item.get("filename"),
                "page_number": item.get("page_number"),
                "page_candidate_rank": item.get("page_candidate_rank"),
                "page_score": item.get("page_score"),
            }
            for item in selected
        ],
        "context_chars": len(evidence),
        "variant_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "selection_trace": selection_trace,
    }


def _safe_mean(values: list[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 2) if usable else None


def _summarize(records: list[dict], variant: str) -> dict:
    count = max(1, len(records))
    values = [item["variants"][variant] for item in records]
    baseline_values = [item["variants"]["A_global"] for item in records]
    recovered = [
        item["financebench_id"] for item, current, base in zip(records, values, baseline_values)
        if current["context_hit"] and not base["context_hit"]
    ]
    regressed = [
        item["financebench_id"] for item, current, base in zip(records, values, baseline_values)
        if base["context_hit"] and not current["context_hit"]
    ]
    groups = {}
    for group in ("candidate_miss10", "selection_loss10", "correct_regression10"):
        subset = [item["variants"][variant] for item in records if item["group"] == group]
        groups[group] = {
            "candidate_hit": round(sum(item["candidate_hit"] for item in subset) / max(1, len(subset)), 4),
            "selected_hit": round(sum(item["selected_hit"] for item in subset) / max(1, len(subset)), 4),
            "context_hit": round(sum(item["context_hit"] for item in subset) / max(1, len(subset)), 4),
        }
    return {
        "questions": len(records),
        "candidate_hit": round(sum(item["candidate_hit"] for item in values) / count, 4),
        "selected_hit": round(sum(item["selected_hit"] for item in values) / count, 4),
        "context_hit": round(sum(item["context_hit"] for item in values) / count, 4),
        "average_gold_page_rank": _safe_mean([item["gold_page_rank"] for item in values]),
        "average_candidate_pages": _safe_mean([item["candidate_page_count"] for item in values]),
        "average_context_chars": _safe_mean([item["context_chars"] for item in values]),
        "average_variant_latency_ms": _safe_mean([item["variant_latency_ms"] for item in values]),
        "recovered_vs_global": recovered,
        "regressed_vs_global": regressed,
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--document-shortlist-k", type=int, default=3)
    parser.add_argument("--local-k", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "retrieval_document_local_diagnostic30.json")
    args = parser.parse_args()
    apply_runtime_profile(RETRIEVAL_DOCUMENT_LOCAL_PROFILE)
    rows, membership = _load_rows(args.dataset, args.fixture)
    if args.limit:
        rows = rows[: args.limit]
    page_store, table_store = DocumentPageStore(), TableStore()
    records = []
    print(
        f"[setup] profile={RETRIEVAL_DOCUMENT_LOCAL_PROFILE} questions={len(rows)} "
        f"shortlist_k={args.document_shortlist_k} local_k={args.local_k} jina=false",
        flush=True,
    )
    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        global_result = retrieve_dense_primary(row["question"], dense_k=120, bm25_k=30)
        shortlist, document_trace = score_candidate_documents(
            global_result["merged"], shortlist_k=args.document_shortlist_k,
        )
        local_result = retrieve_document_local_chunks(
            row["question"], shortlist, global_result["query_embedding"],
            local_k=args.local_k,
        )
        combined_chunks = merge_global_local_chunks(global_result["merged"], local_result["chunks"])
        global_pages, global_page_trace = expand_and_rank_pages(
            row["question"], global_result["merged"], global_result["query_embedding"],
            neighbor_window=1, page_store=page_store,
        )
        local_pages, local_page_trace = expand_and_rank_pages(
            row["question"], local_result["chunks"], global_result["query_embedding"],
            neighbor_window=0, page_store=page_store,
        )
        combined_pages, combined_page_trace = expand_and_rank_pages(
            row["question"], combined_chunks, global_result["query_embedding"],
            neighbor_window=1, page_store=page_store,
        )
        gold = _gold(row)
        variants = {
            "A_global": _variant_result(row["question"], global_pages, gold, table_store),
            "B_document_local": _variant_result(row["question"], local_pages, gold, table_store),
            "C_global_local_merge": _variant_result(row["question"], combined_pages, gold, table_store),
        }
        record = {
            "financebench_id": row["financebench_id"],
            "group": membership[row["financebench_id"]],
            "gold_document_rank_before_local": _document_rank(document_trace, gold),
            "gold_page_rank_before_local": variants["A_global"]["gold_page_rank"],
            "gold_page_rank_after_local": variants["B_document_local"]["gold_page_rank"],
            "gold_page_rank_after_merge": variants["C_global_local_merge"]["gold_page_rank"],
            "candidate_documents": document_trace,
            "document_shortlist": shortlist,
            "dense_calls": 1 + local_result["dense_calls"],
            "bm25_calls": 1 + local_result["bm25_calls"],
            "global_retrieval_latency_ms": global_result["latency_ms"]["total"],
            "document_local_latency_ms": local_result["latency_ms"],
            "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "local_routes": local_result["routes"],
            "variants": variants,
            "page_candidate_traces": {
                "A_global": global_page_trace,
                "B_document_local": local_page_trace,
                "C_global_local_merge": combined_page_trace,
            },
        }
        records.append(record)
        print(
            f"[{index:02d}/{len(rows)}] {row['financebench_id']} "
            f"doc_rank={record['gold_document_rank_before_local']} "
            f"page_rank={record['gold_page_rank_before_local']}->{record['gold_page_rank_after_local']} "
            + " ".join(
                f"{name[0]}:{int(value['candidate_hit'])}/{int(value['selected_hit'])}/{int(value['context_hit'])}"
                for name, value in variants.items()
            ),
            flush=True,
        )
    metrics = {variant: _summarize(records, variant) for variant in VARIANTS}
    cost = {
        "dense_calls": sum(item["dense_calls"] for item in records),
        "bm25_calls": sum(item["bm25_calls"] for item in records),
        "jina_calls": 0,
        "average_global_retrieval_latency_ms": _safe_mean([item["global_retrieval_latency_ms"] for item in records]),
        "average_document_local_latency_ms": _safe_mean([item["document_local_latency_ms"] for item in records]),
        "average_total_latency_ms": _safe_mean([item["total_latency_ms"] for item in records]),
    }
    ranks = {
        "average_gold_document_rank_before_local": _safe_mean([item["gold_document_rank_before_local"] for item in records]),
        "average_gold_page_rank_before_local": _safe_mean([item["gold_page_rank_before_local"] for item in records]),
        "average_gold_page_rank_after_local": _safe_mean([item["gold_page_rank_after_local"] for item in records]),
        "average_gold_page_rank_after_merge": _safe_mean([item["gold_page_rank_after_merge"] for item in records]),
    }
    payload = {
        "profile": RETRIEVAL_DOCUMENT_LOCAL_PROFILE,
        "evaluation_scope": "fixed diagnostic30; retrieval-only; no Jina/answer/Judge",
        "config": {"document_shortlist_k": args.document_shortlist_k, "local_k": args.local_k},
        "metrics": metrics,
        "ranks": ranks,
        "cost": cost,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    lines = [
        "# Retrieval Core v4 Phase 3 — Document-local Retrieval", "",
        "> Fixed diagnostic30; retrieval-only. No Jina, answer model, or Judge calls.", "",
        "| Variant | Candidate | Selected | Context | Avg gold page rank | Avg pages | Avg context chars |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = metrics[variant]
        lines.append(
            f"| {variant} | {item['candidate_hit']:.2%} | {item['selected_hit']:.2%} | "
            f"{item['context_hit']:.2%} | {item['average_gold_page_rank']} | "
            f"{item['average_candidate_pages']} | {item['average_context_chars']} |"
        )
    lines.extend(["", "## Rank migration", ""])
    lines.extend(f"- {key}: {value}" for key, value in ranks.items())
    lines.extend(["", "## Cost", ""])
    lines.extend(f"- {key}: {value}" for key, value in cost.items())
    lines.extend(["", "## Regressions relative to A", ""])
    for variant in ("B_document_local", "C_global_local_merge"):
        lines.append(f"- {variant} recovered: {', '.join(metrics[variant]['recovered_vs_global']) or 'none'}")
        lines.append(f"- {variant} regressed: {', '.join(metrics[variant]['regressed_vs_global']) or 'none'}")
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": metrics, "ranks": ranks, "cost": cost}, ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {args.output}\nMarkdown: {markdown}", flush=True)


if __name__ == "__main__":
    main()
