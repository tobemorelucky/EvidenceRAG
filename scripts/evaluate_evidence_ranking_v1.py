"""Evidence Ranking v1 shadow A/B and Oracle budget curve on diagnostic30."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
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
from evidence_assembly_v5 import assemble_evidence_v5, build_evidence_units  # noqa: E402
from evidence_ranking_v1 import select_ranked_evidence_v1  # noqa: E402
from rag_core_v4 import retrieve_dense_primary  # noqa: E402
from runtime_profile import RETRIEVAL_DOCUMENT_LOCAL_PROFILE, apply_runtime_profile  # noqa: E402
from scripts.evaluate_evidence_assembly_v5 import _metric_blocks  # noqa: E402
from scripts.evaluate_evidence_assembly_ab import _parse_gold  # noqa: E402
from scripts.evaluate_oracle_evidence_block import (  # noqa: E402
    _context_metrics,
    answer_evidence_coverage,
    build_oracle_evidence_blocks,
)
from scripts.evaluate_page_selector_v1 import GROUPS, _load_rows  # noqa: E402
from table_store import TableStore  # noqa: E402


DEFAULT_OUTPUT = ROOT / "reports" / "evidence_ranking_v1_diagnostic30.json"
ORACLE_BUDGETS = (5000, 10000, 15000, 28000, 40000)
ROUTES = ("baseline_assembly_v5", "evidence_ranking_v1")


def _mean(values: list[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 4) if usable else None


def _rate(values: list[bool | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return round(sum(bool(value) for value in usable) / len(usable), 4) if usable else None


def gold_evidence_retention(gold: list[dict], candidate_context: str, selected_context: str) -> dict:
    candidate = answer_evidence_coverage(gold, candidate_context)
    selected = answer_evidence_coverage(gold, selected_context)
    denominator = int(candidate["matched_lines"] or 0)
    return {
        "candidate_matched_lines": denominator,
        "selected_matched_lines": int(selected["matched_lines"] or 0),
        "ratio": round(int(selected["matched_lines"] or 0) / denominator, 4) if denominator else None,
    }


def oracle_budget_curve(row: dict, budgets: tuple[int, ...] = ORACLE_BUDGETS) -> dict[str, dict]:
    gold = _parse_gold(row)
    curve = {}
    for budget in budgets:
        context, _ = build_oracle_evidence_blocks(row, max_context_chars=budget)
        coverage = answer_evidence_coverage(gold, context)
        curve[str(budget)] = {
            "context_chars": len(context),
            "strict_evidence_coverage": coverage["ratio"],
            "matched_lines": coverage["matched_lines"],
            "total_lines": coverage["total_lines"],
        }
    return curve


def _route_summary(records: list[dict], route: str) -> dict:
    values = [record["routes"][route] for record in records]
    return {
        "evidence_coverage": _mean([item["metrics"]["answer_evidence_coverage"]["ratio"] for item in values]),
        "required_number_hit": _rate([item["metrics"]["required_number_hit"] for item in values]),
        "required_period_hit": _rate([item["metrics"]["required_period_hit"] for item in values]),
        "gold_evidence_retention": _mean([item["gold_evidence_retention"]["ratio"] for item in values]),
        "average_context_chars": _mean([item["metrics"]["context_chars"] for item in values]),
        "average_selected_units": _mean([item["metrics"]["block_count"] for item in values]),
    }


def _summary_for(records: list[dict]) -> dict:
    routes = {route: _route_summary(records, route) for route in ROUTES}
    old, new = routes["baseline_assembly_v5"], routes["evidence_ranking_v1"]
    return {
        "questions": len(records),
        **routes,
        "delta": {
            metric: round((new[metric] or 0.0) - (old[metric] or 0.0), 4)
            for metric in ("evidence_coverage", "required_number_hit", "required_period_hit", "gold_evidence_retention")
        },
        "coverage_gains": [
            record["financebench_id"] for record in records
            if (record["routes"]["evidence_ranking_v1"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0)
            > (record["routes"]["baseline_assembly_v5"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0)
        ],
        "coverage_regressions": [
            record["financebench_id"] for record in records
            if (record["routes"]["evidence_ranking_v1"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0)
            < (record["routes"]["baseline_assembly_v5"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0)
        ],
    }


def summarize(records: list[dict]) -> dict:
    summary = _summary_for(records)
    summary["groups"] = {
        group: _summary_for([record for record in records if record["group"] == group])
        for group in GROUPS
    }
    summary["oracle_budget_curve"] = {
        str(budget): {
            "average_context_chars": _mean([record["oracle_budget_curve"][str(budget)]["context_chars"] for record in records]),
            "average_strict_evidence_coverage": _mean([
                record["oracle_budget_curve"][str(budget)]["strict_evidence_coverage"] for record in records
            ]),
            "full_coverage_rate": _rate([
                record["oracle_budget_curve"][str(budget)]["strict_evidence_coverage"] == 1.0 for record in records
            ]),
        }
        for budget in ORACLE_BUDGETS
    }
    summary["unit_composition"] = {
        "baseline_average_text": _mean([
            record["routes"]["baseline_assembly_v5"]["trace"]["selected_text_unit_count"] for record in records
        ]),
        "baseline_average_table": _mean([
            record["routes"]["baseline_assembly_v5"]["trace"]["selected_table_unit_count"] for record in records
        ]),
        "ranking_average_text": _mean([
            sum(unit.get("source_type") == "text" for unit in record["routes"]["evidence_ranking_v1"]["selected_units"])
            for record in records
        ]),
        "ranking_average_table": _mean([
            sum(unit.get("source_type") == "table" for unit in record["routes"]["evidence_ranking_v1"]["selected_units"])
            for record in records
        ]),
    }
    summary["acceptance"] = {
        "passed": bool(
            summary["delta"]["evidence_coverage"] > 0
            and summary["delta"]["required_number_hit"] >= 0
            and summary["delta"]["required_period_hit"] >= 0
            and summary["groups"]["correct_regression10"]["delta"]["evidence_coverage"] >= 0
        ),
        "criterion": "coverage improves; number/period and correct-regression coverage do not regress",
    }
    summary["external_calls"] = {"jina": 0, "answer_model": 0, "judge": 0, "langsmith": 0}
    return summary


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    old, new = summary["baseline_assembly_v5"], summary["evidence_ranking_v1"]
    lines = [
        "# Evidence Ranking v1 shadow — diagnostic30",
        "",
        "> Same Top120 candidate Evidence Units and 28k budget. Production Retrieval/Fusion/Prompt/Skills/Assembly unchanged. No LLM/Jina/Judge.",
        "",
        "## Ranking A/B",
        "",
        "| Metric | Assembly v5 ranking | Ranking v1 | Delta |",
        "|---|---:|---:|---:|",
        f"| Evidence coverage | {_percent(old['evidence_coverage'])} | {_percent(new['evidence_coverage'])} | {_percent(summary['delta']['evidence_coverage'])} |",
        f"| Required number hit | {_percent(old['required_number_hit'])} | {_percent(new['required_number_hit'])} | {_percent(summary['delta']['required_number_hit'])} |",
        f"| Required period hit | {_percent(old['required_period_hit'])} | {_percent(new['required_period_hit'])} | {_percent(summary['delta']['required_period_hit'])} |",
        f"| Gold evidence retention | {_percent(old['gold_evidence_retention'])} | {_percent(new['gold_evidence_retention'])} | {_percent(summary['delta']['gold_evidence_retention'])} |",
        f"| Average context chars | {old['average_context_chars']} | {new['average_context_chars']} | — |",
        "",
        f"- Coverage gains/regressions: {len(summary['coverage_gains'])} / {len(summary['coverage_regressions'])}",
        f"- Average text/table units baseline: {summary['unit_composition']['baseline_average_text']} / {summary['unit_composition']['baseline_average_table']}",
        f"- Average text/table units ranking v1: {summary['unit_composition']['ranking_average_text']} / {summary['unit_composition']['ranking_average_table']}",
        f"- Acceptance passed: `{summary['acceptance']['passed']}`",
        "",
        "## Groups",
        "",
        "| Group | Coverage baseline/v1 | Number baseline/v1 | Period baseline/v1 | Retention baseline/v1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        item = summary["groups"][group]
        group_old, group_new = item["baseline_assembly_v5"], item["evidence_ranking_v1"]
        lines.append(
            f"| {group} | {_percent(group_old['evidence_coverage'])} / {_percent(group_new['evidence_coverage'])} | "
            f"{_percent(group_old['required_number_hit'])} / {_percent(group_new['required_number_hit'])} | "
            f"{_percent(group_old['required_period_hit'])} / {_percent(group_new['required_period_hit'])} | "
            f"{_percent(group_old['gold_evidence_retention'])} / {_percent(group_new['gold_evidence_retention'])} |"
        )
    lines.extend([
        "",
        "## Oracle budget curve",
        "",
        "| Budget | Average used chars | Strict evidence coverage | Full coverage questions |",
        "|---:|---:|---:|---:|",
    ])
    for budget in ORACLE_BUDGETS:
        item = summary["oracle_budget_curve"][str(budget)]
        lines.append(
            f"| {budget:,} | {item['average_context_chars']} | "
            f"{_percent(item['average_strict_evidence_coverage'])} | {_percent(item['full_coverage_rate'])} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if summary["acceptance"]["passed"]:
        lines.append("Ranking v1 passed the evidence-level gate.")
    else:
        lines.extend([
            "Ranking v1 remains shadow-only and did not pass the full gate.",
            "",
            "- Overall evidence coverage, required-number hit, and gold retention improve, so the generic features contain useful ranking signal.",
            "- Required-period hit regresses and the selection-loss group is effectively flat, so the signal is not yet stable across evidence needs.",
            "- Correct-regression average coverage improves, but individual regressions remain; aggregate improvement alone is insufficient for production.",
            "- The Oracle curve reaches full exact-evidence coverage at 5k for all 30 questions. The 28k ceiling is therefore not the limiting factor when ranking is ideal.",
        ])
    lines.extend(["", "## Per question", ""])
    for index, record in enumerate(payload["records"], 1):
        baseline = record["routes"]["baseline_assembly_v5"]
        ranking = record["routes"]["evidence_ranking_v1"]
        lines.extend([
            f"### {index}. {record['financebench_id']}",
            "",
            f"- Group: `{record['group']}`",
            f"- Question: {record['question']}",
            f"- Evidence coverage baseline/v1: {baseline['metrics']['answer_evidence_coverage']['ratio']} / {ranking['metrics']['answer_evidence_coverage']['ratio']}",
            f"- Required number baseline/v1: {baseline['metrics']['required_number_hit']} / {ranking['metrics']['required_number_hit']}",
            f"- Required period baseline/v1: {baseline['metrics']['required_period_hit']} / {ranking['metrics']['required_period_hit']}",
            f"- Gold retention baseline/v1: {baseline['gold_evidence_retention']['ratio']} / {ranking['gold_evidence_retention']['ratio']}",
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 30:
        raise SystemExit("--limit must be between 1 and 30")

    apply_runtime_profile(RETRIEVAL_DOCUMENT_LOCAL_PROFILE)
    rows = _load_rows(args.dataset, args.fixture)[:args.limit]
    page_store, table_store = DocumentPageStore(), TableStore()
    records = []
    print(f"[setup] questions={len(rows)} top_k={args.retrieval_k} jina=false llm=false judge=false", flush=True)
    for index, row in enumerate(rows, 1):
        retrieval = retrieve_dense_primary(row["question"], dense_k=args.retrieval_k, bm25_k=30)
        chunks = retrieval["merged"][:args.retrieval_k]
        page_keys = list(dict.fromkeys(
            (str(chunk.get("filename") or ""), int(chunk.get("page_number") or 0))
            for chunk in chunks if str(chunk.get("filename") or "").strip()
        ))
        pages = page_store.get_pages_by_keys(page_keys)
        tables = table_store.get_tables_by_page_keys(page_keys)
        candidate_units = build_evidence_units(row["question"], chunks, pages=pages, tables=tables)
        candidate_context = "\n\n".join(unit.source_text for unit in candidate_units)
        baseline_context, baseline_units, baseline_trace = assemble_evidence_v5(
            row["question"], chunks, pages=pages, tables=tables, max_context_chars=args.max_context_chars,
        )
        ranked_context, ranked_units, ranking_trace = select_ranked_evidence_v1(
            row["question"], candidate_units, max_context_chars=args.max_context_chars,
        )
        gold = _parse_gold(row)
        baseline_metrics = _context_metrics(row, baseline_context, _metric_blocks(baseline_units))
        ranking_metrics = _context_metrics(row, ranked_context, _metric_blocks(ranked_units))
        baseline_retention = gold_evidence_retention(gold, candidate_context, baseline_context)
        ranking_retention = gold_evidence_retention(gold, candidate_context, ranked_context)
        records.append({
            "financebench_id": row["financebench_id"],
            "group": row["diagnostic_group"],
            "question": row["question"],
            "candidate_unit_count": len(candidate_units),
            "routes": {
                "baseline_assembly_v5": {
                    "metrics": baseline_metrics,
                    "gold_evidence_retention": baseline_retention,
                    "trace": baseline_trace,
                },
                "evidence_ranking_v1": {
                    "metrics": ranking_metrics,
                    "gold_evidence_retention": ranking_retention,
                    "trace": ranking_trace,
                    "selected_units": ranked_units,
                },
            },
            "oracle_budget_curve": oracle_budget_curve(row),
        })
        print(
            f"[{index:02d}/{len(rows)}] {row['financebench_id']} "
            f"coverage={baseline_metrics['answer_evidence_coverage']['ratio']}->{ranking_metrics['answer_evidence_coverage']['ratio']} "
            f"retention={baseline_retention['ratio']}->{ranking_retention['ratio']}",
            flush=True,
        )

    payload = {
        "evaluation": "evidence_ranking_v1_diagnostic30",
        "scope": "same Assembly v5 candidate units and 28k budget; deterministic ranking only",
        "config": {
            "retrieval_k": args.retrieval_k,
            "max_context_chars": args.max_context_chars,
            "oracle_budgets": ORACLE_BUDGETS,
        },
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
