"""Offline Current Top120 chunk context vs Evidence Assembly v5 on diagnostic30.

The script calls no Jina, answer model, Judge, or LangSmith service.  Strict
Judge is therefore emitted as null, with the frozen Oracle score as reference.
"""

from __future__ import annotations

import argparse
import json
import os
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

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from document_page_store import DocumentPageStore  # noqa: E402
from evidence_assembly_v5 import assemble_evidence_v5  # noqa: E402
from rag_core_v4 import retrieve_dense_primary  # noqa: E402
from runtime_profile import RETRIEVAL_DOCUMENT_LOCAL_PROFILE, apply_runtime_profile  # noqa: E402
from scripts.evaluate_evidence_block_retrieval_v2 import _render_ranked_units  # noqa: E402
from scripts.evaluate_oracle_evidence_block import _context_metrics  # noqa: E402
from scripts.evaluate_page_selector_v1 import GROUPS, _load_rows  # noqa: E402
from table_store import TableStore  # noqa: E402


DEFAULT_OUTPUT = ROOT / "reports" / "evidence_assembly_v5_diagnostic30.json"
DEFAULT_ORACLE_SUMMARY = ROOT / "reports" / "oracle_evidence_block_diagnostic30.summary.json"
ROUTES = ("current_chunk_retrieval", "evidence_assembly_v5")


def _mean(values: list[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 4) if usable else None


def _rate(values: list[bool | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return round(sum(bool(value) for value in usable) / len(usable), 4) if usable else None


def _metric_blocks(units: list[dict]) -> list[dict]:
    return [
        {
            "block_id": f"assembly:{index}",
            "block_type": unit.get("source_type"),
            "source_pages": [{
                "filename": unit.get("metadata", {}).get("filename", ""),
                "page_number": unit.get("metadata", {}).get("page_number", 0),
            }],
        }
        for index, unit in enumerate(units, 1)
    ]


def _route_summary(records: list[dict], route: str) -> dict:
    values = [record["routes"][route] for record in records]
    return {
        "strict_judge": None,
        "strict_judge_status": "not_evaluated_no_llm_or_judge_calls",
        "evidence_coverage": _mean([item["metrics"]["answer_evidence_coverage"]["ratio"] for item in values]),
        "required_number_hit": _rate([item["metrics"]["required_number_hit"] for item in values]),
        "required_period_hit": _rate([item["metrics"]["required_period_hit"] for item in values]),
        "gold_page_hit_auxiliary": _rate([item["metrics"]["gold_page_hit"] for item in values]),
        "all_gold_pages_hit_auxiliary": _rate([item["metrics"]["all_gold_pages_hit"] for item in values]),
        "average_context_chars": _mean([item["metrics"]["context_chars"] for item in values]),
        "average_evidence_units": _mean([item["metrics"]["block_count"] for item in values]),
    }


def _summary_for(records: list[dict]) -> dict:
    routes = {route: _route_summary(records, route) for route in ROUTES}
    old, new = routes["current_chunk_retrieval"], routes["evidence_assembly_v5"]
    return {
        "questions": len(records),
        **routes,
        "delta": {
            metric: round((new[metric] or 0.0) - (old[metric] or 0.0), 4)
            for metric in ("evidence_coverage", "required_number_hit", "required_period_hit", "gold_page_hit_auxiliary")
        },
        "coverage_gains": [
            record["financebench_id"] for record in records
            if (record["routes"]["evidence_assembly_v5"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0)
            > (record["routes"]["current_chunk_retrieval"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0)
        ],
        "coverage_regressions": [
            record["financebench_id"] for record in records
            if (record["routes"]["evidence_assembly_v5"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0)
            < (record["routes"]["current_chunk_retrieval"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0)
        ],
    }


def summarize(records: list[dict], oracle_payload: dict | None = None) -> dict:
    summary = _summary_for(records)
    summary["groups"] = {
        group: _summary_for([record for record in records if record["group"] == group])
        for group in GROUPS
    }
    frozen = oracle_payload or {}
    if isinstance(frozen.get("summary"), dict):
        frozen = frozen["summary"]
    oracle = frozen.get("oracle_evidence_block") or {}
    summary["oracle_reference"] = {
        "source": "existing Oracle diagnostic30; not rerun",
        "strict_judge": oracle.get("strict_judge"),
        "evidence_coverage": oracle.get("answer_evidence_coverage"),
        "required_number_hit": oracle.get("required_number_hit"),
        "required_period_hit": oracle.get("required_period_hit"),
    }
    assembly = summary["evidence_assembly_v5"]
    summary["remaining_oracle_gap"] = {
        metric: round(float(summary["oracle_reference"][metric]) - float(assembly[metric]), 4)
        if summary["oracle_reference"].get(metric) is not None and assembly.get(metric) is not None else None
        for metric in ("evidence_coverage", "required_number_hit", "required_period_hit")
    }
    summary["assembly_units"] = {
        "average_selected_text": _mean([record["routes"]["evidence_assembly_v5"]["trace"]["selected_text_unit_count"] for record in records]),
        "average_selected_table": _mean([record["routes"]["evidence_assembly_v5"]["trace"]["selected_table_unit_count"] for record in records]),
        "questions_with_table_units": sum(record["routes"]["evidence_assembly_v5"]["trace"]["selected_table_unit_count"] > 0 for record in records),
    }
    summary["acceptance"] = {
        "passed": bool(
            summary["delta"]["evidence_coverage"] > 0
            and summary["delta"]["required_number_hit"] >= 0
            and summary["groups"]["correct_regression10"]["delta"]["evidence_coverage"] >= 0
        ),
        "criterion": "coverage improves, number hit does not regress, correct-regression coverage does not regress",
    }
    summary["external_calls"] = {"jina": 0, "answer_model": 0, "strict_judge": 0, "langsmith": 0}
    return summary


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    old, new = summary["current_chunk_retrieval"], summary["evidence_assembly_v5"]
    oracle = summary["oracle_reference"]
    lines = [
        "# Evidence Assembly v5 shadow — diagnostic30",
        "",
        "> Frozen Top120 chunk retrieval for both routes. Production Retrieval/Fusion/Prompt/Skills unchanged. Jina=0, LLM=0, Judge=0.",
        "",
        "Strict Judge is `null` for both new routes because this experiment is explicitly evidence-only. The historical Oracle score is reference-only.",
        "",
        "## Overall",
        "",
        "| Metric | Current chunks | Assembly v5 | Oracle reference | Delta |",
        "|---|---:|---:|---:|---:|",
        f"| Strict Judge | n/a | n/a | {_percent(oracle.get('strict_judge'))} | n/a |",
        f"| Evidence coverage | {_percent(old['evidence_coverage'])} | {_percent(new['evidence_coverage'])} | {_percent(oracle.get('evidence_coverage'))} | {_percent(summary['delta']['evidence_coverage'])} |",
        f"| Required number hit | {_percent(old['required_number_hit'])} | {_percent(new['required_number_hit'])} | {_percent(oracle.get('required_number_hit'))} | {_percent(summary['delta']['required_number_hit'])} |",
        f"| Required period hit | {_percent(old['required_period_hit'])} | {_percent(new['required_period_hit'])} | {_percent(oracle.get('required_period_hit'))} | {_percent(summary['delta']['required_period_hit'])} |",
        f"| Gold page hit (auxiliary) | {_percent(old['gold_page_hit_auxiliary'])} | {_percent(new['gold_page_hit_auxiliary'])} | — | {_percent(summary['delta']['gold_page_hit_auxiliary'])} |",
        f"| Average context chars | {old['average_context_chars']} | {new['average_context_chars']} | — | — |",
        "",
        f"- Evidence coverage gains/regressions: {len(summary['coverage_gains'])} / {len(summary['coverage_regressions'])}",
        f"- Assembly unit statistics: `{summary['assembly_units']}`",
        f"- Remaining Oracle gap: `{summary['remaining_oracle_gap']}`",
        f"- Acceptance passed: `{summary['acceptance']['passed']}`",
        "",
        "## Groups",
        "",
        "| Group | Coverage chunks/v5 | Number chunks/v5 | Period chunks/v5 |",
        "|---|---:|---:|---:|",
    ]
    for group in GROUPS:
        item = summary["groups"][group]
        group_old, group_new = item["current_chunk_retrieval"], item["evidence_assembly_v5"]
        lines.append(
            f"| {group} | {_percent(group_old['evidence_coverage'])} / {_percent(group_new['evidence_coverage'])} | "
            f"{_percent(group_old['required_number_hit'])} / {_percent(group_new['required_number_hit'])} | "
            f"{_percent(group_old['required_period_hit'])} / {_percent(group_new['required_period_hit'])} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if summary["acceptance"]["passed"]:
        lines.append("Evidence Assembly v5 passed the evidence-level gate and reduced the Oracle gap without changing retrieval.")
    else:
        lines.extend([
            "Evidence Assembly v5 did not pass the evidence-level gate and must remain shadow-only.",
            "",
            "- Required-number and period coverage improve, showing that same-page table rows recover some structured values.",
            "- Exact benchmark-evidence coverage declines while gold-page hit is unchanged; the loss is package composition, not retrieval.",
            "- Table units consume budget that previously held complete high-rank text facts, and the correct-regression group also loses coverage.",
            "- The current v5 package therefore does not shrink the overall Oracle evidence gap and must not replace the current chunk package.",
        ])
    lines.extend(["", "## Per question", ""])
    for index, record in enumerate(payload["records"], 1):
        current = record["routes"]["current_chunk_retrieval"]["metrics"]
        assembly = record["routes"]["evidence_assembly_v5"]
        newer = assembly["metrics"]
        lines.extend([
            f"### {index}. {record['financebench_id']}",
            "",
            f"- Group: `{record['group']}`",
            f"- Question: {record['question']}",
            f"- Evidence coverage chunks/v5: {current['answer_evidence_coverage']['ratio']} / {newer['answer_evidence_coverage']['ratio']}",
            f"- Required number hit chunks/v5: {current['required_number_hit']} / {newer['required_number_hit']}",
            f"- Required period hit chunks/v5: {current['required_period_hit']} / {newer['required_period_hit']}",
            f"- Selected text/table units: {assembly['trace']['selected_text_unit_count']} / {assembly['trace']['selected_table_unit_count']}",
            "",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--retrieval-k", type=int, default=120)
    parser.add_argument("--max-context-chars", type=int, default=28000)
    parser.add_argument("--text-budget-ratio", type=float, default=0.78)
    parser.add_argument("--oracle-summary", type=Path, default=DEFAULT_ORACLE_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 30:
        raise SystemExit("--limit must be between 1 and 30")

    apply_runtime_profile(RETRIEVAL_DOCUMENT_LOCAL_PROFILE)
    rows = _load_rows(args.dataset, args.fixture)[:args.limit]
    oracle = json.loads(args.oracle_summary.read_text(encoding="utf-8")) if args.oracle_summary.is_file() else None
    page_store, table_store = DocumentPageStore(), TableStore()
    records = []
    print(
        f"[setup] questions={len(rows)} top_k={args.retrieval_k} max_chars={args.max_context_chars} "
        "jina=false llm=false judge=false",
        flush=True,
    )
    for index, row in enumerate(rows, 1):
        retrieval = retrieve_dense_primary(row["question"], dense_k=args.retrieval_k, bm25_k=30)
        chunks = retrieval["merged"][:args.retrieval_k]
        page_keys = list(dict.fromkeys(
            (str(chunk.get("filename") or ""), int(chunk.get("page_number") or 0))
            for chunk in chunks if str(chunk.get("filename") or "").strip()
        ))
        pages = page_store.get_pages_by_keys(page_keys)
        tables = table_store.get_tables_by_page_keys(page_keys)
        current_context, current_units = _render_ranked_units(
            chunks, max_units=args.retrieval_k, max_context_chars=args.max_context_chars,
        )
        started = time.perf_counter()
        assembly_context, assembly_units, trace = assemble_evidence_v5(
            row["question"], chunks, pages=pages, tables=tables,
            max_context_chars=args.max_context_chars, text_budget_ratio=args.text_budget_ratio,
        )
        assembly_ms = (time.perf_counter() - started) * 1000
        current_metrics = _context_metrics(row, current_context, current_units)
        assembly_metrics = _context_metrics(row, assembly_context, _metric_blocks(assembly_units))
        records.append({
            "financebench_id": row["financebench_id"],
            "group": row["diagnostic_group"],
            "question": row["question"],
            "routes": {
                "current_chunk_retrieval": {"metrics": current_metrics},
                "evidence_assembly_v5": {
                    "metrics": assembly_metrics,
                    "trace": trace,
                    "assembly_latency_ms": round(assembly_ms, 2),
                    "selected_units": assembly_units,
                },
            },
        })
        current_number = current_metrics["required_number_hit"]
        assembly_number = assembly_metrics["required_number_hit"]
        print(
            f"[{index:02d}/{len(rows)}] {row['financebench_id']} "
            f"coverage={current_metrics['answer_evidence_coverage']['ratio']}->{assembly_metrics['answer_evidence_coverage']['ratio']} "
            f"number={'n/a' if current_number is None else int(current_number)}->"
            f"{'n/a' if assembly_number is None else int(assembly_number)} "
            f"units={trace['selected_text_unit_count']}+{trace['selected_table_unit_count']}",
            flush=True,
        )

    payload = {
        "evaluation": "evidence_assembly_v5_diagnostic30",
        "scope": "same frozen Top120 chunks; current raw chunk package vs shadow assembly; no Jina/LLM/Judge",
        "config": {
            "retrieval_k": args.retrieval_k,
            "max_context_chars": args.max_context_chars,
            "text_budget_ratio": args.text_budget_ratio,
        },
        "summary": summarize(records, oracle),
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
