"""Offline A/B: current first-fit packing vs utility Packing v1."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evidence_packing_v1 import select_evidence_packing_v1  # noqa: E402
from scripts.evaluate_evidence_metadata_counterfactual_v1 import (  # noqa: E402
    _context_metrics,
    _gold,
    _gold_retention,
    _load_dataset,
)


DEFAULT_INPUT = ROOT / "reports" / "evidence_metadata_counterfactual_v1.json"
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_JSON = ROOT / "reports" / "evidence_packing_optimization_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "evidence_packing_optimization_v1.md"
GROUPS = ("selection_loss10", "correct_regression10", "candidate_miss10")
ROUTES = ("sequential_first_fit", "packing_optimization_v1")


def _mean(values: list[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 4) if usable else None


def _rate(values: list[bool | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return round(sum(bool(value) for value in usable) / len(usable), 4) if usable else None


def _route_summary(records: list[dict], route: str) -> dict:
    values = [record["routes"][route] for record in records]
    return {
        "evidence_coverage": _mean([value["metrics"]["answer_evidence_coverage"]["ratio"] for value in values]),
        "gold_evidence_retention": _mean([value["gold_evidence_retention"]["ratio"] for value in values]),
        "required_number_hit": _rate([value["metrics"]["required_number_hit"] for value in values]),
        "required_period_hit": _rate([value["metrics"]["required_period_hit"] for value in values]),
        "context_characters": _mean([value["metrics"]["context_chars"] for value in values]),
        "selected_gold_page_hit": _rate([value["metrics"]["selected_gold_page_hit"] for value in values]),
        "average_selected_units": _mean([value["metrics"]["selected_unit_count"] for value in values]),
    }


def _summary_for(records: list[dict]) -> dict:
    routes = {route: _route_summary(records, route) for route in ROUTES}
    old, new = routes[ROUTES[0]], routes[ROUTES[1]]
    metrics = (
        "evidence_coverage", "gold_evidence_retention", "required_number_hit",
        "required_period_hit", "selected_gold_page_hit",
    )
    return {
        "questions": len(records),
        **routes,
        "delta": {metric: round((new[metric] or 0.0) - (old[metric] or 0.0), 4) for metric in metrics},
        "coverage_gains": [record["financebench_id"] for record in records if (
            record["routes"][ROUTES[1]]["metrics"]["answer_evidence_coverage"]["ratio"] or 0
        ) > (
            record["routes"][ROUTES[0]]["metrics"]["answer_evidence_coverage"]["ratio"] or 0
        )],
        "coverage_regressions": [record["financebench_id"] for record in records if (
            record["routes"][ROUTES[1]]["metrics"]["answer_evidence_coverage"]["ratio"] or 0
        ) < (
            record["routes"][ROUTES[0]]["metrics"]["answer_evidence_coverage"]["ratio"] or 0
        )],
    }


def summarize(records: list[dict]) -> dict:
    result = _summary_for(records)
    result["groups"] = {
        group: _summary_for([record for record in records if record["group"] == group])
        for group in GROUPS
    }
    selection = result["groups"]["selection_loss10"]
    correct = result["groups"]["correct_regression10"]
    strong_signal = bool(
        selection["delta"]["evidence_coverage"] >= 0.02
        and selection["delta"]["gold_evidence_retention"] >= 0
        and correct["delta"]["evidence_coverage"] >= 0
        and result["delta"]["required_number_hit"] >= 0
        and result["delta"]["required_period_hit"] >= 0
        and result["packing_optimization_v1"]["context_characters"] <= 28000
    )
    regression_rate = round(len(result["coverage_regressions"]) / max(1, result["questions"]), 4)
    safe_for_direct_integration = bool(
        strong_signal
        and not correct["coverage_regressions"]
        and regression_rate <= 0.1
    )
    replacement_counts = [
        record["routes"][ROUTES[1]]["packing_trace"]["replacement_count"] for record in records
    ]
    result["packing"] = {
        "replacement_count": sum(replacement_counts),
        "average_replacements_per_question": _mean(replacement_counts),
        "max_replacements_per_question": max(replacement_counts, default=0),
        "gold_used_for_selection": False,
        "candidate_set_unchanged": all(
            record["candidate_unit_count"] == record["routes"][ROUTES[1]]["packing_trace"]["candidate_unit_count"]
            for record in records
        ),
        "budget_respected": all(record["routes"][ROUTES[1]]["metrics"]["context_chars"] <= 28000 for record in records),
    }
    result["decision"] = {
        "strong_shadow_signal": strong_signal,
        "safe_for_direct_production_integration": safe_for_direct_integration,
        "criterion": "strong aggregate gate plus zero correct-regression question regressions and <=10% overall coverage-regression rate",
        "overall_coverage_regression_rate": regression_rate,
        "correct_regression_question_regressions": correct["coverage_regressions"],
        "next_focus": "packing_v1_regression_guard" if strong_signal else "ranking_or_evidence_block",
        "reason": (
            "Aggregate gains are strong, but per-question regressions exceed the direct-integration safety gate."
            if strong_signal and not safe_for_direct_integration else
            "Packing v1 satisfies the aggregate and stability gates."
            if safe_for_direct_integration else
            "Packing v1 does not produce a sufficiently stable evidence-level gain."
        ),
    }
    result["external_calls"] = {"retrieval": 0, "llm": 0, "jina": 0, "judge": 0, "langsmith": 0}
    return result


def evaluate_record(record: dict, row: dict) -> dict:
    candidates = list(record.get("candidate_units") or [])
    context, selected, packing_trace = select_evidence_packing_v1(
        record["question"], candidates, max_context_chars=28000,
    )
    candidate_context = "\n\n".join(str(unit.get("source_text") or "") for unit in candidates)
    baseline = record["routes"]["current_ranking"]
    return {
        "financebench_id": record["financebench_id"],
        "group": record["group"],
        "question": record["question"],
        "candidate_unit_count": len(candidates),
        "routes": {
            "sequential_first_fit": baseline,
            "packing_optimization_v1": {
                "metrics": _context_metrics(row, context, selected),
                "gold_evidence_retention": _gold_retention(_gold(row), candidate_context, context),
                "selected_unit_ranks": [int((unit.get("current_ranking") or {}).get("rank") or 0) for unit in selected],
                "packing_trace": packing_trace,
            },
        },
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Packing Optimization v1 shadow",
        "",
        "> Same frozen candidate units, Ranking v1 scores, and 28K character budget. Generic utility contains no gold evidence. Retrieval=0, LLM=0, Jina=0, Judge=0, LangSmith=0.",
        "",
        "## Constraints",
        "",
        f"- Candidate set unchanged: `{summary['packing']['candidate_set_unchanged']}`",
        f"- Budget respected: `{summary['packing']['budget_respected']}`",
        f"- Gold used for selection: `{summary['packing']['gold_used_for_selection']}`",
        f"- Accepted single-unit replacements: `{summary['packing']['replacement_count']}`",
        f"- Average/max replacements per question: `{summary['packing']['average_replacements_per_question']}` / `{summary['packing']['max_replacements_per_question']}`",
        "",
        "## A/B metrics",
        "",
        "| Group | Route | Coverage | Gold retention | Number hit | Period hit | Gold page hit | Context chars |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("overall", *GROUPS):
        item = summary if group == "overall" else summary["groups"][group]
        for route in ROUTES:
            value = item[route]
            lines.append(
                f"| {group} | {route} | {_pct(value['evidence_coverage'])} | "
                f"{_pct(value['gold_evidence_retention'])} | {_pct(value['required_number_hit'])} | "
                f"{_pct(value['required_period_hit'])} | {_pct(value['selected_gold_page_hit'])} | "
                f"{value['context_characters']} |"
            )
    decision = summary["decision"]
    lines.extend([
        "",
        "## Decision",
        "",
        f"- Strong shadow signal: `{decision['strong_shadow_signal']}`",
        f"- Safe for direct production integration: `{decision['safe_for_direct_production_integration']}`",
        f"- Criterion: {decision['criterion']}",
        f"- Overall coverage regression rate: `{decision['overall_coverage_regression_rate']}`",
        f"- Correct-regression question regressions: `{decision['correct_regression_question_regressions']}`",
        f"- Reason: {decision['reason']}",
        f"- Next focus: **{decision['next_focus']}**",
        "",
        "## Per question",
        "",
    ])
    for index, record in enumerate(payload["records"], 1):
        old = record["routes"][ROUTES[0]]
        new = record["routes"][ROUTES[1]]
        lines.extend([
            f"### {index}. {record['financebench_id']} — {record['group']}",
            "",
            f"- Question: {record['question']}",
            f"- Coverage: `{old['metrics']['answer_evidence_coverage']['ratio']}` → `{new['metrics']['answer_evidence_coverage']['ratio']}`",
            f"- Gold retention: `{old['gold_evidence_retention']['ratio']}` → `{new['gold_evidence_retention']['ratio']}`",
            f"- Required number: `{old['metrics']['required_number_hit']}` → `{new['metrics']['required_number_hit']}`",
            f"- Required period: `{old['metrics']['required_period_hit']}` → `{new['metrics']['required_period_hit']}`",
            f"- Context chars: `{old['metrics']['context_chars']}` → `{new['metrics']['context_chars']}`",
            f"- Selected units / replacements: `{new['metrics']['selected_unit_count']}` / `{new['packing_trace']['replacement_count']}`",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    source = json.loads(args.input_json.read_text(encoding="utf-8"))
    rows = _load_dataset(args.dataset)
    records = []
    for index, record in enumerate(source.get("records") or [], 1):
        result = evaluate_record(record, rows[record["financebench_id"]])
        records.append(result)
        old = result["routes"][ROUTES[0]]["metrics"]["answer_evidence_coverage"]["ratio"]
        new = result["routes"][ROUTES[1]]["metrics"]["answer_evidence_coverage"]["ratio"]
        print(f"[{index:02d}/30] {record['financebench_id']} coverage={old}->{new}", flush=True)
    payload = {
        "evaluation": "evidence_packing_optimization_v1_shadow",
        "scope": "same frozen candidate units, Ranking v1 scores, 28K budget; gold-free utility",
        "summary": summarize(records),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {args.output_json}\nMarkdown: {args.output_markdown}", flush=True)


if __name__ == "__main__":
    main()
