"""Offline A/B for Page Selector v1 on the fixed diagnostic30 set.

The script reuses the current local retrieval candidates and context builder.
It never calls Jina, an answer model, or a Judge, and it does not alter the
production pipeline.
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
from page_selector_v1 import select_pages_v1  # noqa: E402
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
DEFAULT_OUTPUT = ROOT / "reports" / "page_selector_v1_diagnostic30.json"
GROUPS = ("candidate_miss10", "selection_loss10", "correct_regression10")


def _filename(value: object) -> str:
    name = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return (name if name.casefold().endswith(".pdf") else f"{name}.pdf").casefold()


def _page_key(item: dict) -> tuple[str, int]:
    try:
        page_number = int(item.get("page_number") or 0)
    except (TypeError, ValueError):
        page_number = 0
    return _filename(item.get("filename") or item.get("doc_name")), page_number


def _gold_pages(row: dict) -> set[tuple[str, int]]:
    return {
        (_filename(item.get("doc_name")), int(item.get("evidence_page_num") or 0))
        for item in json.loads(row.get("evidence") or "[]")
    }


def _load_rows(dataset: Path, fixture: Path) -> list[dict]:
    groups = json.loads(fixture.read_text(encoding="utf-8"))
    ordered_ids: list[str] = []
    membership: dict[str, str] = {}
    for group in GROUPS:
        for value in groups[group]:
            financebench_id = value if isinstance(value, str) else value["financebench_id"]
            ordered_ids.append(financebench_id)
            membership[financebench_id] = group
    with dataset.open(encoding="utf-8-sig", newline="") as handle:
        wanted = set(ordered_ids)
        by_id = {row["financebench_id"]: row for row in csv.DictReader(handle) if row["financebench_id"] in wanted}
    missing = [financebench_id for financebench_id in ordered_ids if financebench_id not in by_id]
    if missing:
        raise RuntimeError(f"missing diagnostic IDs: {missing}")
    return [{**by_id[financebench_id], "diagnostic_group": membership[financebench_id]} for financebench_id in ordered_ids]


def _rank(items: list[dict], gold: set[tuple[str, int]]) -> int | None:
    for rank, item in enumerate(items, 1):
        if _page_key(item) in gold:
            return rank
    return None


def _trace_rank(items: list[dict], gold: set[tuple[str, int]]) -> int | None:
    for item in items:
        if _page_key(item) in gold:
            return int(item.get("rank") or 0) or None
    return None


def _evaluate_selection(
    question: str,
    selected: list[dict],
    gold: set[tuple[str, int]],
    table_store: TableStore,
) -> dict:
    page_keys = [(str(item.get("filename") or ""), int(item.get("page_number") or 0)) for item in selected]
    tables = table_store.get_tables_by_page_keys(page_keys)
    evidence, context_meta = build_core_v3_evidence(question, selected, tables)
    selected_keys = {_page_key(item) for item in selected}
    context_keys = {_page_key(item) for item in context_meta.get("answer_context_pages") or []}
    return {
        "selected_hit": bool(gold & selected_keys),
        "context_hit": bool(gold & context_keys),
        "selected_pages": [
            {
                "document_id": item.get("document_id"),
                "page_id": item.get("page_id"),
                "filename": item.get("filename"),
                "page_number": item.get("page_number"),
            }
            for item in selected
        ],
        "context_pages": [
            {"filename": item[0], "page_number": item[1]}
            for item in sorted(context_keys)
        ],
        "context_chars": len(evidence),
    }


def _rate(records: list[dict], route: str, field: str) -> float:
    return round(sum(bool(item[route][field]) for item in records) / max(1, len(records)), 4)


def _mean(values: list[int | float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 2) if usable else None


def _group_summary(records: list[dict]) -> dict:
    baseline_context = _rate(records, "baseline", "context_hit")
    selector_context = _rate(records, "page_selector_v1", "context_hit")
    return {
        "questions": len(records),
        "candidate_hit": round(sum(item["candidate_hit"] for item in records) / max(1, len(records)), 4),
        "baseline": {
            "selected_hit": _rate(records, "baseline", "selected_hit"),
            "context_hit": baseline_context,
            "average_gold_page_rank": _mean([item["baseline"]["gold_page_rank"] for item in records]),
        },
        "page_selector_v1": {
            "selected_hit": _rate(records, "page_selector_v1", "selected_hit"),
            "context_hit": selector_context,
            "average_gold_page_rank": _mean([item["page_selector_v1"]["gold_page_rank"] for item in records]),
        },
        "context_hit_delta": round(selector_context - baseline_context, 4),
        "recovered": [
            item["financebench_id"] for item in records
            if item["page_selector_v1"]["context_hit"] and not item["baseline"]["context_hit"]
        ],
        "regressed": [
            item["financebench_id"] for item in records
            if item["baseline"]["context_hit"] and not item["page_selector_v1"]["context_hit"]
        ],
    }


def summarize(records: list[dict]) -> dict:
    overall = _group_summary(records)
    groups = {
        group: _group_summary([item for item in records if item["group"] == group])
        for group in GROUPS
    }
    selection = groups["selection_loss10"]
    regression = groups["correct_regression10"]
    overall["groups"] = groups
    overall["acceptance"] = {
        "selection_loss_context_improved": selection["context_hit_delta"] > 0,
        "correct_regression_not_decreased": regression["context_hit_delta"] >= 0,
        "passed": selection["context_hit_delta"] > 0 and regression["context_hit_delta"] >= 0,
    }
    overall["average_selector_latency_ms"] = _mean([item["selector_latency_ms"] for item in records])
    overall["external_calls"] = {"jina": 0, "llm": 0, "judge": 0}
    return overall


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def _percentage_points(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    baseline = summary["baseline"]
    selector = summary["page_selector_v1"]
    lines = [
        "# Page Selector v1 — Offline diagnostic30",
        "",
        "- Scope: fixed diagnostic30; local retrieval and deterministic context assembly only.",
        "- External calls: Jina=0, LLM=0, Judge=0.",
        "- Production pipeline: unchanged.",
        f"- Questions: {summary['questions']}",
        "",
        "## Overall",
        "",
        "| Metric | Existing document-first | Page Selector v1 | Delta |",
        "|---|---:|---:|---:|",
        f"| Candidate gold-page hit | {_percent(summary['candidate_hit'])} | {_percent(summary['candidate_hit'])} | 0.00 pp |",
        f"| Selected gold-page hit | {_percent(baseline['selected_hit'])} | {_percent(selector['selected_hit'])} | {_percentage_points(selector['selected_hit'] - baseline['selected_hit'])} |",
        f"| Context gold-page hit | {_percent(baseline['context_hit'])} | {_percent(selector['context_hit'])} | {_percentage_points(summary['context_hit_delta'])} |",
        f"| Average gold-page rank | {baseline['average_gold_page_rank']} | {selector['average_gold_page_rank']} | — |",
        "",
        f"- Average selector latency: {summary['average_selector_latency_ms']} ms/question",
        f"- Acceptance passed: `{summary['acceptance']['passed']}`",
        "",
        "## Diagnostic groups",
        "",
        "| Group | N | Candidate | Selected old/new | Context old/new | Recovered | Regressed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        item = summary["groups"][group]
        lines.append(
            f"| {group} | {item['questions']} | {_percent(item['candidate_hit'])} | "
            f"{_percent(item['baseline']['selected_hit'])} / {_percent(item['page_selector_v1']['selected_hit'])} | "
            f"{_percent(item['baseline']['context_hit'])} / {_percent(item['page_selector_v1']['context_hit'])} | "
            f"{len(item['recovered'])} | {len(item['regressed'])} |"
        )
    lines.extend(["", "## Per question", ""])
    for index, item in enumerate(payload["records"], 1):
        old = item["baseline"]
        new = item["page_selector_v1"]
        top_trace = ", ".join(
            f"{entry['filename']} p.{entry['page_number']}={entry['score']:.4f}"
            for entry in item["page_score_trace"][:8]
        )
        lines.extend([
            f"### {index}. {item['financebench_id']}",
            "",
            f"- Group: `{item['group']}`",
            f"- Question: {item['question']}",
            f"- Candidate hit: {item['candidate_hit']}",
            f"- Gold-page rank existing/v1: {old['gold_page_rank']} / {new['gold_page_rank']}",
            f"- Selected hit existing/v1: {old['selected_hit']} / {new['selected_hit']}",
            f"- Context hit existing/v1: {old['context_hit']} / {new['context_hit']}",
            f"- Top page-score trace: {top_trace}",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--document-shortlist-k", type=int, default=3)
    parser.add_argument("--local-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    apply_runtime_profile(RETRIEVAL_DOCUMENT_LOCAL_PROFILE)
    rows = _load_rows(args.dataset, args.fixture)
    if args.limit:
        rows = rows[: args.limit]
    page_store, table_store = DocumentPageStore(), TableStore()
    records = []
    print(
        f"[setup] questions={len(rows)} top_k={args.top_k} shortlist={args.document_shortlist_k} "
        f"local_k={args.local_k} jina=false llm=false judge=false",
        flush=True,
    )
    for index, row in enumerate(rows, 1):
        global_result = retrieve_dense_primary(row["question"], dense_k=120, bm25_k=30)
        shortlist, _ = score_candidate_documents(global_result["merged"], shortlist_k=args.document_shortlist_k)
        local_result = retrieve_document_local_chunks(
            row["question"], shortlist, global_result["query_embedding"], local_k=args.local_k,
        )
        combined_chunks = merge_global_local_chunks(global_result["merged"], local_result["chunks"])
        candidate_pages, _ = expand_and_rank_pages(
            row["question"], combined_chunks, global_result["query_embedding"],
            neighbor_window=1, page_store=page_store,
        )
        candidate_page_keys = [
            (str(item.get("filename") or ""), int(item.get("page_number") or 0))
            for item in candidate_pages
        ]
        candidate_tables = table_store.get_tables_by_page_keys(candidate_page_keys)
        baseline_selected, _ = select_document_first_pages(
            candidate_pages, final_page_k=args.top_k, global_escape_pages=1,
        )
        selector_started = time.perf_counter()
        selector_selected, selector_trace = select_pages_v1(
            row["question"], combined_chunks, page_records=candidate_pages,
            table_metadata=candidate_tables, top_k=args.top_k,
        )
        selector_latency_ms = round((time.perf_counter() - selector_started) * 1000, 2)
        gold = _gold_pages(row)
        candidate_hit = bool(gold & {_page_key(item) for item in candidate_pages})
        baseline = _evaluate_selection(row["question"], baseline_selected, gold, table_store)
        selector = _evaluate_selection(row["question"], selector_selected, gold, table_store)
        baseline["gold_page_rank"] = _rank(candidate_pages, gold)
        selector["gold_page_rank"] = _trace_rank(selector_trace["page_scores"], gold)
        records.append({
            "financebench_id": row["financebench_id"],
            "group": row["diagnostic_group"],
            "question": row["question"],
            "gold_pages": [{"filename": item[0], "page_number": item[1]} for item in sorted(gold)],
            "candidate_hit": candidate_hit,
            "candidate_page_count": len(candidate_pages),
            "baseline": baseline,
            "page_selector_v1": selector,
            "selector_latency_ms": selector_latency_ms,
            "page_score_trace": selector_trace["page_scores"],
        })
        print(
            f"[{index:02d}/{len(rows)}] {row['financebench_id']} group={row['diagnostic_group']} "
            f"candidate={int(candidate_hit)} selected={int(baseline['selected_hit'])}->{int(selector['selected_hit'])} "
            f"context={int(baseline['context_hit'])}->{int(selector['context_hit'])} "
            f"rank={baseline['gold_page_rank']}->{selector['gold_page_rank']}",
            flush=True,
        )
    payload = {
        "evaluation": "page_selector_v1_diagnostic30",
        "scope": "fixed diagnostic30; no Jina/LLM/Judge; production pipeline unchanged",
        "config": {
            "profile": RETRIEVAL_DOCUMENT_LOCAL_PROFILE,
            "document_shortlist_k": args.document_shortlist_k,
            "local_k": args.local_k,
            "top_k": args.top_k,
        },
        "summary": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {args.output}\nMarkdown: {markdown}", flush=True)


if __name__ == "__main__":
    main()
