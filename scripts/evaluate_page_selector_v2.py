"""Offline comparison of production selection, Page Selector v1, and v2.

The fixed diagnostic30 retrieval is run locally. No Jina, answer model, or
Judge is called, and neither shadow selector is connected to production.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from document_page_store import DocumentPageStore  # noqa: E402
from page_selector_v1 import select_pages_v1  # noqa: E402
from page_selector_v2 import select_page_groups_v2  # noqa: E402
from rag_core_v4 import (  # noqa: E402
    expand_and_rank_pages,
    merge_global_local_chunks,
    retrieve_dense_primary,
    retrieve_document_local_chunks,
    score_candidate_documents,
    select_document_first_pages,
)
from runtime_profile import RETRIEVAL_DOCUMENT_LOCAL_PROFILE, apply_runtime_profile  # noqa: E402
from scripts.evaluate_page_selector_v1 import (  # noqa: E402
    GROUPS,
    _evaluate_selection,
    _gold_pages,
    _load_rows,
    _page_key,
    _rank,
)
from table_store import TableStore  # noqa: E402


DEFAULT_OUTPUT = ROOT / "reports" / "page_selector_v2_diagnostic30.json"
ROUTES = ("baseline", "page_selector_v1", "page_selector_v2")


def _selected_rank(items: list[dict], gold: set[tuple[str, int]]) -> int | None:
    return _rank(items, gold)


def _group_contains_gold(group: dict, gold: set[tuple[str, int]]) -> bool:
    return any(_page_key(page) in gold for page in group.get("pages") or [])


def _page_set(items: list[dict]) -> set[tuple[str, int]]:
    return {_page_key(item) for item in items}


def _rate(records: list[dict], route: str, field: str) -> float:
    return round(sum(bool(item[route][field]) for item in records) / max(1, len(records)), 4)


def _mean(values: list[int | float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 2) if usable else None


def _route_summary(records: list[dict], route: str) -> dict:
    return {
        "selected_hit": _rate(records, route, "selected_hit"),
        "context_hit": _rate(records, route, "context_hit"),
        "average_selected_gold_page_rank": _mean([item[route]["gold_page_rank"] for item in records]),
    }


def _group_summary(records: list[dict]) -> dict:
    routes = {route: _route_summary(records, route) for route in ROUTES}
    baseline = routes["baseline"]
    v2 = routes["page_selector_v2"]
    return {
        "questions": len(records),
        "candidate_hit": round(sum(item["candidate_hit"] for item in records) / max(1, len(records)), 4),
        **routes,
        "v2_gold_group_hit": _rate(records, "page_selector_v2", "gold_group_hit"),
        "v2_context_delta_vs_baseline": round(v2["context_hit"] - baseline["context_hit"], 4),
        "v2_recovered_vs_baseline": [
            item["financebench_id"] for item in records
            if item["page_selector_v2"]["context_hit"] and not item["baseline"]["context_hit"]
        ],
        "v2_regressed_vs_baseline": [
            item["financebench_id"] for item in records
            if item["baseline"]["context_hit"] and not item["page_selector_v2"]["context_hit"]
        ],
        "v2_changed_vs_v1": [
            item["financebench_id"] for item in records
            if item["page_selector_v1"]["context_hit"] != item["page_selector_v2"]["context_hit"]
        ],
    }


def summarize(records: list[dict]) -> dict:
    summary = _group_summary(records)
    summary["groups"] = {
        group: _group_summary([item for item in records if item["group"] == group])
        for group in GROUPS
    }
    selection = summary["groups"]["selection_loss10"]
    regression = summary["groups"]["correct_regression10"]
    summary["acceptance"] = {
        "selection_loss_context_improved": selection["v2_context_delta_vs_baseline"] > 0,
        "correct_regression_not_decreased": regression["v2_context_delta_vs_baseline"] >= 0,
        "passed": selection["v2_context_delta_vs_baseline"] > 0
        and regression["v2_context_delta_vs_baseline"] >= 0,
    }
    summary["average_v2_latency_ms"] = _mean([item["page_selector_v2"]["latency_ms"] for item in records])
    summary["external_calls"] = {"jina": 0, "llm": 0, "judge": 0}
    return summary


def _percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Page Selection v2 shadow — diagnostic30",
        "",
        "- Retrieval candidates: current local Core v4 flow.",
        "- Comparison: production document-first vs Page Selector v1 vs coverage-group v2.",
        "- External calls: Jina=0, LLM=0, Judge=0.",
        "- Production pipeline: unchanged.",
        "",
        "## Overall",
        "",
        "| Metric | Production | v1 | v2 |",
        "|---|---:|---:|---:|",
        f"| Candidate hit | {_percent(summary['candidate_hit'])} | {_percent(summary['candidate_hit'])} | {_percent(summary['candidate_hit'])} |",
        f"| Selected hit | {_percent(summary['baseline']['selected_hit'])} | {_percent(summary['page_selector_v1']['selected_hit'])} | {_percent(summary['page_selector_v2']['selected_hit'])} |",
        f"| Context hit | {_percent(summary['baseline']['context_hit'])} | {_percent(summary['page_selector_v1']['context_hit'])} | {_percent(summary['page_selector_v2']['context_hit'])} |",
        f"| Average selected gold-page rank | {summary['baseline']['average_selected_gold_page_rank']} | {summary['page_selector_v1']['average_selected_gold_page_rank']} | {summary['page_selector_v2']['average_selected_gold_page_rank']} |",
        f"| Gold group hit | — | — | {_percent(summary['v2_gold_group_hit'])} |",
        "",
        f"- Acceptance passed: `{summary['acceptance']['passed']}`",
        f"- Average v2 latency: {summary['average_v2_latency_ms']} ms/question",
        "",
        "## Groups",
        "",
        "| Group | Candidate | Context production/v1/v2 | v2 recovered/regressed vs production |",
        "|---|---:|---:|---:|",
    ]
    for group in GROUPS:
        item = summary["groups"][group]
        lines.append(
            f"| {group} | {_percent(item['candidate_hit'])} | "
            f"{_percent(item['baseline']['context_hit'])} / {_percent(item['page_selector_v1']['context_hit'])} / {_percent(item['page_selector_v2']['context_hit'])} | "
            f"{len(item['v2_recovered_vs_baseline'])} / {len(item['v2_regressed_vs_baseline'])} |"
        )
    lines.extend(["", "## Per question", ""])
    for index, item in enumerate(payload["records"], 1):
        base, v1, v2 = item["baseline"], item["page_selector_v1"], item["page_selector_v2"]
        lines.extend([
            f"### {index}. {item['financebench_id']}",
            "",
            f"- Group: `{item['group']}`",
            f"- Question: {item['question']}",
            f"- Candidate hit/rank: {item['candidate_hit']} / {item['candidate_gold_page_rank']}",
            f"- Selected hit production/v1/v2: {base['selected_hit']} / {v1['selected_hit']} / {v2['selected_hit']}",
            f"- Context hit production/v1/v2: {base['context_hit']} / {v1['context_hit']} / {v2['context_hit']}",
            f"- Selected gold-page rank production/v1/v2: {base['gold_page_rank']} / {v1['gold_page_rank']} / {v2['gold_page_rank']}",
            f"- Candidate/selected gold-group hit: {v2['candidate_gold_group_hit']} / {v2['gold_group_hit']}",
            f"- v1→v2 added pages: {v2['difference_vs_v1']['added_pages']}",
            f"- v1→v2 removed pages: {v2['difference_vs_v1']['removed_pages']}",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--document-shortlist-k", type=int, default=3)
    parser.add_argument("--local-k", type=int, default=30)
    parser.add_argument("--page-budget", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    apply_runtime_profile(RETRIEVAL_DOCUMENT_LOCAL_PROFILE)
    rows = _load_rows(args.dataset, args.fixture)
    if args.limit:
        rows = rows[: args.limit]
    page_store, table_store = DocumentPageStore(), TableStore()
    records = []
    print(
        f"[setup] questions={len(rows)} page_budget={args.page_budget} shortlist={args.document_shortlist_k} "
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
        page_keys = [(str(item.get("filename") or ""), int(item.get("page_number") or 0)) for item in candidate_pages]
        candidate_tables = table_store.get_tables_by_page_keys(page_keys)
        baseline_pages, _ = select_document_first_pages(
            candidate_pages, final_page_k=args.page_budget, global_escape_pages=1,
        )
        v1_pages, _ = select_pages_v1(
            row["question"], combined_chunks, page_records=candidate_pages,
            table_metadata=candidate_tables, top_k=args.page_budget,
        )
        started = time.perf_counter()
        v2_pages, v2_trace = select_page_groups_v2(
            row["question"], combined_chunks, page_records=candidate_pages,
            table_metadata=candidate_tables, page_budget=args.page_budget,
        )
        v2_latency = round((time.perf_counter() - started) * 1000, 2)
        gold = _gold_pages(row)
        candidate_hit = bool(gold & _page_set(candidate_pages))
        baseline = _evaluate_selection(row["question"], baseline_pages, gold, table_store)
        v1 = _evaluate_selection(row["question"], v1_pages, gold, table_store)
        v2 = _evaluate_selection(row["question"], v2_pages, gold, table_store)
        baseline["gold_page_rank"] = _selected_rank(baseline_pages, gold)
        v1["gold_page_rank"] = _selected_rank(v1_pages, gold)
        v2["gold_page_rank"] = _selected_rank(v2_pages, gold)
        v2["candidate_gold_group_hit"] = any(_group_contains_gold(group, gold) for group in v2_trace["evidence_groups"])
        v2["gold_group_hit"] = any(_group_contains_gold(group, gold) for group in v2_trace["selected_groups"])
        v2["latency_ms"] = v2_latency
        v1_keys, v2_keys = _page_set(v1_pages), _page_set(v2_pages)
        v2["difference_vs_v1"] = {
            "added_pages": [list(item) for item in sorted(v2_keys - v1_keys)],
            "removed_pages": [list(item) for item in sorted(v1_keys - v2_keys)],
            "selected_hit_changed": v1["selected_hit"] != v2["selected_hit"],
            "context_hit_changed": v1["context_hit"] != v2["context_hit"],
        }
        records.append({
            "financebench_id": row["financebench_id"],
            "group": row["diagnostic_group"],
            "question": row["question"],
            "gold_pages": [{"filename": item[0], "page_number": item[1]} for item in sorted(gold)],
            "candidate_hit": candidate_hit,
            "candidate_gold_page_rank": _rank(candidate_pages, gold),
            "baseline": baseline,
            "page_selector_v1": v1,
            "page_selector_v2": v2,
            "evidence_group_trace": v2_trace,
        })
        print(
            f"[{index:02d}/{len(rows)}] {row['financebench_id']} group={row['diagnostic_group']} "
            f"candidate={int(candidate_hit)} context={int(baseline['context_hit'])}/{int(v1['context_hit'])}/{int(v2['context_hit'])} "
            f"gold_group={int(v2['gold_group_hit'])}",
            flush=True,
        )
    payload = {
        "evaluation": "page_selector_v2_diagnostic30",
        "scope": "fixed diagnostic30; production/v1/v2; no Jina/LLM/Judge; production unchanged",
        "config": {
            "profile": RETRIEVAL_DOCUMENT_LOCAL_PROFILE,
            "document_shortlist_k": args.document_shortlist_k,
            "local_k": args.local_k,
            "page_budget": args.page_budget,
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
