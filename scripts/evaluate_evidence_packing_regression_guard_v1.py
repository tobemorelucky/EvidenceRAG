"""Offline grid evaluation for conservative Evidence Packing v1 guards."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evidence_packing_guard_v1 import select_evidence_packing_guard_v1  # noqa: E402
from scripts.evaluate_evidence_metadata_counterfactual_v1 import (  # noqa: E402
    _context_metrics,
    _gold,
    _gold_retention,
    _load_dataset,
)


DEFAULT_CANDIDATES = ROOT / "reports" / "evidence_metadata_counterfactual_v1.json"
DEFAULT_PACKING_V1 = ROOT / "reports" / "evidence_packing_optimization_v1.json"
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_JSON = ROOT / "reports" / "evidence_packing_regression_guard_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "evidence_packing_regression_guard_v1.md"
GROUPS = ("selection_loss10", "correct_regression10", "candidate_miss10")
BASELINE_ROUTES = ("sequential_first_fit", "packing_optimization_v1")
GUARD_CONFIGS = tuple(
    {
        "route": f"guard_t{int(threshold * 100):03d}_{'unlimited' if limit is None else f'max{limit}'}",
        "replacement_threshold": threshold,
        "max_replacements": limit,
        "anchor_top_n": 5,
        "anchor_min_query_relevance": 0.35,
    }
    for threshold in (1.05, 1.10)
    for limit in (None, 5, 10)
)
ROUTES = BASELINE_ROUTES + tuple(config["route"] for config in GUARD_CONFIGS)


def _mean(values):
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 4) if usable else None


def _rate(values):
    usable = [value for value in values if value is not None]
    return round(sum(bool(value) for value in usable) / len(usable), 4) if usable else None


def _route_summary(records: list[dict], route: str) -> dict:
    values = [record["routes"][route] for record in records]
    replacements = [
        (value.get("packing") or value.get("packing_trace") or {}).get("replacement_count", 0)
        for value in values
    ]
    distribution = dict(sorted(Counter(replacements).items()))
    return {
        "evidence_coverage": _mean([value["metrics"]["answer_evidence_coverage"]["ratio"] for value in values]),
        "gold_evidence_retention": _mean([value["gold_evidence_retention"]["ratio"] for value in values]),
        "required_number_hit": _rate([value["metrics"]["required_number_hit"] for value in values]),
        "required_period_hit": _rate([value["metrics"]["required_period_hit"] for value in values]),
        "selected_gold_page_hit": _rate([value["metrics"]["selected_gold_page_hit"] for value in values]),
        "context_characters": _mean([value["metrics"]["context_chars"] for value in values]),
        "average_replacements": _mean(replacements),
        "max_replacements": max(replacements, default=0),
        "replacement_distribution": distribution,
    }


def _coverage_regressions(records: list[dict], route: str) -> list[str]:
    return [
        record["financebench_id"] for record in records
        if record["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"]
        < record["routes"]["sequential_first_fit"]["metrics"]["answer_evidence_coverage"]["ratio"]
    ]


def summarize(records: list[dict]) -> dict:
    result = {
        "questions": len(records),
        "routes": {route: _route_summary(records, route) for route in ROUTES},
        "groups": {},
    }
    for group in GROUPS:
        group_records = [record for record in records if record["group"] == group]
        result["groups"][group] = {
            "questions": len(group_records),
            "routes": {route: _route_summary(group_records, route) for route in ROUTES},
            "coverage_regressions_vs_sequential": {
                route: _coverage_regressions(group_records, route) for route in ROUTES[1:]
            },
        }
    result["coverage_regressions_vs_sequential"] = {
        route: _coverage_regressions(records, route) for route in ROUTES[1:]
    }

    packing_gain = (
        result["routes"]["packing_optimization_v1"]["evidence_coverage"]
        - result["routes"]["sequential_first_fit"]["evidence_coverage"]
    )
    candidates = []
    for config in GUARD_CONFIGS:
        route = config["route"]
        values = result["routes"][route]
        gain = values["evidence_coverage"] - result["routes"]["sequential_first_fit"]["evidence_coverage"]
        regressions = result["coverage_regressions_vs_sequential"][route]
        correct_regressions = result["groups"]["correct_regression10"]["coverage_regressions_vs_sequential"][route]
        preserves_gain = gain >= packing_gain * 0.75
        candidates.append({
            **config,
            "coverage_gain_vs_sequential": round(gain, 4),
            "preserves_at_least_75pct_of_packing_v1_gain": preserves_gain,
            "overall_regression_count": len(regressions),
            "correct_regression_count": len(correct_regressions),
        })
    packing_regressions = result["coverage_regressions_vs_sequential"]["packing_optimization_v1"]
    packing_correct_regressions = result["groups"]["correct_regression10"]["coverage_regressions_vs_sequential"]["packing_optimization_v1"]
    stable = [
        candidate for candidate in candidates
        if candidate["preserves_at_least_75pct_of_packing_v1_gain"]
        and candidate["overall_regression_count"] <= len(packing_regressions)
        and candidate["correct_regression_count"] <= len(packing_correct_regressions)
        and (
            candidate["overall_regression_count"] < len(packing_regressions)
            or candidate["correct_regression_count"] < len(packing_correct_regressions)
        )
    ]
    best_guard = min(
        candidates,
        key=lambda candidate: (
            candidate["correct_regression_count"],
            candidate["overall_regression_count"],
            -result["groups"]["selection_loss10"]["routes"][candidate["route"]]["evidence_coverage"],
            -result["routes"][candidate["route"]]["evidence_coverage"],
            candidate["max_replacements"] is None,
            result["routes"][candidate["route"]]["average_replacements"],
        ),
    )
    result["guard_grid"] = candidates
    result["decision"] = {
        "stable_guard_found": bool(stable),
        "stable_route": min(stable, key=lambda item: (item["correct_regression_count"], item["overall_regression_count"])) if stable else None,
        "best_guard_candidate": best_guard,
        "packing_v1_overall_regression_count": len(packing_regressions),
        "packing_v1_correct_regression_count": len(packing_correct_regressions),
        "selection_rule": "preserve >=75% of Packing v1 coverage gain and reduce at least one regression count without worsening the other",
        "reason": (
            "At least one guard reduces regression risk while preserving most aggregate coverage gain."
            if stable else
            "No tested guard reduces Packing v1 regression counts; do not treat a grid winner as stable."
        ),
    }
    result["constraints"] = {
        "candidate_set_unchanged": all(record["candidate_set_unchanged"] for record in records),
        "budget_respected": all(
            record["routes"][route]["metrics"]["context_chars"] <= 28000
            for record in records for route in ROUTES
        ),
        "gold_used_for_selection": False,
        "external_calls": {"retrieval": 0, "llm": 0, "jina": 0, "judge": 0, "langsmith": 0},
    }
    return result


def _compact_packing(trace: dict) -> dict:
    reasons = Counter(item["selection_reason"] for item in trace["trace"])
    return {
        key: trace[key] for key in (
            "packing", "replacement_count", "replacement_threshold", "max_replacements",
            "protected_anchor_count", "anchor_top_n", "anchor_min_query_relevance",
            "candidate_unit_count", "selected_unit_count", "context_chars",
            "max_context_chars", "gold_used_for_selection",
        )
    } | {"selection_reason_counts": dict(sorted(reasons.items()))}


def evaluate_record(source: dict, prior: dict, row: dict) -> dict:
    candidates = list(source.get("candidate_units") or [])
    candidate_context = "\n\n".join(str(unit.get("source_text") or "") for unit in candidates)
    routes = {
        "sequential_first_fit": prior["routes"]["sequential_first_fit"],
        "packing_optimization_v1": prior["routes"]["packing_optimization_v1"],
    }
    for config in GUARD_CONFIGS:
        context, selected, trace = select_evidence_packing_guard_v1(
            source["question"],
            candidates,
            max_context_chars=28000,
            replacement_threshold=config["replacement_threshold"],
            max_replacements=config["max_replacements"],
            anchor_top_n=config["anchor_top_n"],
            anchor_min_query_relevance=config["anchor_min_query_relevance"],
        )
        routes[config["route"]] = {
            "metrics": _context_metrics(row, context, selected),
            "gold_evidence_retention": _gold_retention(_gold(row), candidate_context, context),
            "selected_unit_ranks": [int((unit.get("current_ranking") or {}).get("rank") or 0) for unit in selected],
            "packing": _compact_packing(trace),
        }
    return {
        "financebench_id": source["financebench_id"],
        "group": source["group"],
        "question": source["question"],
        "candidate_unit_count": len(candidates),
        "candidate_set_unchanged": len(candidates) == prior["candidate_unit_count"],
        "routes": routes,
    }


def _pct(value):
    return "n/a" if value is None else f"{value:.2%}"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Packing Regression Guard v1 shadow",
        "",
        "> Frozen 30-question candidates, Ranking v1 scores, and 28K budget. No Retrieval, LLM, Jina, Judge, or LangSmith calls.",
        "",
        "## Stable configuration",
        "",
        f"- Stable guard found: `{summary['decision']['stable_guard_found']}`",
        f"- Stable route: `{summary['decision']['stable_route']}`",
        f"- Best diagnostic candidate: `{summary['decision']['best_guard_candidate']['route']}`",
        f"- Packing v1 overall/correct-regression regressions: `{summary['decision']['packing_v1_overall_regression_count']}` / `{summary['decision']['packing_v1_correct_regression_count']}`",
        f"- Best guard overall/correct-regression regressions: `{summary['decision']['best_guard_candidate']['overall_regression_count']}` / `{summary['decision']['best_guard_candidate']['correct_regression_count']}`",
        f"- Selection rule: {summary['decision']['selection_rule']}",
        f"- Conclusion: {summary['decision']['reason']}",
        "",
        "## Overall A/B",
        "",
        "| Route | Coverage | Gold retention | Number hit | Period hit | Gold page hit | Context chars | Avg/max replacements |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route in ROUTES:
        value = summary["routes"][route]
        lines.append(
            f"| {route} | {_pct(value['evidence_coverage'])} | {_pct(value['gold_evidence_retention'])} | "
            f"{_pct(value['required_number_hit'])} | {_pct(value['required_period_hit'])} | "
            f"{_pct(value['selected_gold_page_hit'])} | {value['context_characters']} | "
            f"{value['average_replacements']} / {value['max_replacements']} |"
        )
    lines.extend(["", "## Group comparison", ""])
    for group in ("selection_loss10", "correct_regression10"):
        lines.extend([
            f"### {group}", "",
            "| Route | Coverage | Gold retention | Number hit | Period hit | Gold page hit | Regressions vs sequential |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        item = summary["groups"][group]
        for route in ROUTES:
            value = item["routes"][route]
            regressions = 0 if route == "sequential_first_fit" else len(item["coverage_regressions_vs_sequential"][route])
            lines.append(
                f"| {route} | {_pct(value['evidence_coverage'])} | {_pct(value['gold_evidence_retention'])} | "
                f"{_pct(value['required_number_hit'])} | {_pct(value['required_period_hit'])} | "
                f"{_pct(value['selected_gold_page_hit'])} | {regressions} |"
            )
        lines.append("")
    lines.extend(["## Replacement distributions", ""])
    for route in ROUTES[1:]:
        lines.append(f"- `{route}`: `{summary['routes'][route]['replacement_distribution']}`")
    lines.extend(["", "## Per-question stable-route changes", ""])
    stable = (
        summary["decision"]["stable_route"] or summary["decision"]["best_guard_candidate"]
    )
    stable = stable["route"]
    for record in payload["records"]:
        baseline = record["routes"]["sequential_first_fit"]["metrics"]["answer_evidence_coverage"]["ratio"]
        packing = record["routes"]["packing_optimization_v1"]["metrics"]["answer_evidence_coverage"]["ratio"]
        guarded = record["routes"][stable]["metrics"]["answer_evidence_coverage"]["ratio"]
        replacements = record["routes"][stable]["packing"]["replacement_count"]
        lines.append(
            f"- `{record['financebench_id']}` ({record['group']}): coverage `{baseline} → {packing} → {guarded}`, replacements `{replacements}`"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--packing-v1-json", type=Path, default=DEFAULT_PACKING_V1)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    candidates = json.loads(args.candidate_json.read_text(encoding="utf-8"))["records"]
    prior_records = json.loads(args.packing_v1_json.read_text(encoding="utf-8"))["records"]
    prior_by_id = {record["financebench_id"]: record for record in prior_records}
    rows = _load_dataset(args.dataset)
    records = []
    for index, source in enumerate(candidates, 1):
        result = evaluate_record(source, prior_by_id[source["financebench_id"]], rows[source["financebench_id"]])
        records.append(result)
        print(f"[{index:02d}/30] {source['financebench_id']} guard grid complete", flush=True)
    payload = {
        "evaluation": "evidence_packing_regression_guard_v1_shadow",
        "guard_configs": GUARD_CONFIGS,
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
