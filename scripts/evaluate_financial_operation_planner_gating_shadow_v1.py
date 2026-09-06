"""Offline gating replay over frozen Financial Operation Planner Shadow v2 results."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/financial_operation_planner_shadow_v2"
FINAL100 = ROOT / "reports/final100"
OUTPUT = ROOT / "reports/financial_operation_planner_gating_shadow_v1"

STRATEGIES = {
    "A_all_planner": None,
    "B_calculation_ratio_margin_turnover_growth": {"calculation", "ratio", "margin", "turnover", "growth"},
    "C_calculation_ratio": {"calculation", "ratio"},
    "D_explicit_formula": "formula",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gate_task(record: dict) -> str:
    text = f"{record.get('question', '')} {(record.get('planner') or {}).get('metric', '')}".lower()
    for label, pattern in (
        ("ratio", r"\bratio\b"), ("margin", r"\bmargin\b"),
        ("turnover", r"\bturnover\b"), ("growth", r"\bgrowth\b"),
    ):
        if re.search(pattern, text):
            return label
    task_type = str((record.get("planner") or {}).get("task_type") or "lookup").lower()
    return task_type if task_type in {"calculation", "comparison", "trend", "judgment", "lookup"} else "other"


def strategy_enabled(name: str, record: dict) -> bool:
    if not record.get("planner_triggered"):
        return False
    rule = STRATEGIES[name]
    if rule is None:
        return True
    if rule == "formula":
        return bool(str((record.get("planner") or {}).get("formula") or "").strip())
    return gate_task(record) in rule


def evaluate_strategy(
    name: str,
    results: dict[str, dict],
    baseline: dict[str, dict],
    planner_judges: dict[str, dict],
) -> tuple[dict, list[dict]]:
    migrations = Counter()
    task_outcomes: dict[str, Counter] = {}
    records = []
    for question_id, result in results.items():
        baseline_score = int(baseline[question_id]["judge"]["score"])
        enabled = strategy_enabled(name, result)
        has_frozen_judge = question_id in planner_judges
        planner_score = int(planner_judges[question_id]["judge"]["score"]) if enabled and has_frozen_judge else baseline_score
        if baseline_score == 0 and planner_score == 1:
            outcome = "gained"
        elif baseline_score == 1 and planner_score == 0:
            outcome = "regressed"
        elif baseline_score == 0:
            outcome = "maintained_incorrect"
        else:
            outcome = "maintained_correct"
        task = gate_task(result)
        migrations[outcome] += 1
        task_outcomes.setdefault(task, Counter())[outcome] += 1
        records.append({
            "id": question_id, "strategy": name, "gate_task": task,
            "planner_enabled": enabled, "frozen_planner_judge_available": has_frozen_judge,
            "baseline_score": baseline_score, "strategy_score": planner_score, "outcome": outcome,
        })
    baseline_correct = sum(int(record["judge"]["score"]) for record in baseline.values())
    strategy_correct = sum(record["strategy_score"] for record in records)
    gained, regressed = migrations["gained"], migrations["regressed"]
    summary = {
        "strategy": name,
        "planner_enabled": sum(record["planner_enabled"] for record in records),
        "planner_enabled_with_frozen_judge": sum(record["planner_enabled"] and record["frozen_planner_judge_available"] for record in records),
        "baseline_strict_judge": {"correct": baseline_correct, "total": 100, "accuracy": baseline_correct / 100},
        "strategy_strict_judge": {"correct": strategy_correct, "total": 100, "accuracy": strategy_correct / 100},
        "newly_correct": gained, "regressed": regressed, "net_gain": gained - regressed,
        "delta_accuracy": (strategy_correct - baseline_correct) / 100,
        "outcomes": dict(migrations),
        "task_outcomes": {task: dict(counts) for task, counts in sorted(task_outcomes.items())},
        "gained_ids": [record["id"] for record in records if record["outcome"] == "gained"],
        "regressed_ids": [record["id"] for record in records if record["outcome"] == "regressed"],
    }
    return summary, records


def main() -> None:
    results = {record["id"]: record for record in load_jsonl(SOURCE / "results.jsonl")}
    selection = json.loads((SOURCE / "judge_selection.json").read_text(encoding="utf-8"))
    planner_judges = {record["id"]: record for record in load_jsonl(SOURCE / "judge/planner_judge_results.jsonl") if record.get("status") == "ok"}
    baseline = {record["question_id"]: record for record in load_jsonl(FINAL100 / "judge_results.jsonl")}
    if len(results) != 100 or len(baseline) != 100 or len(selection) != 60 or len(planner_judges) != 60:
        raise ValueError("Expected frozen v2 inputs: results=100, selection=60, planner judges=60, baseline judges=100")
    if {record["id"] for record in selection} != set(planner_judges):
        raise ValueError("Frozen Judge selection and results do not match")

    summaries, replay = {}, []
    for name in STRATEGIES:
        summary, records = evaluate_strategy(name, results, baseline, planner_judges)
        summaries[name] = summary
        replay.extend(records)
    payload = {
        "baseline_correct": 68,
        "strategies": summaries,
        "method": {
            "models_called": 0,
            "enabled_case": "Use the frozen Planner Judge score when available; otherwise retain the original final100 Judge score.",
            "gate_task": "Explicit ratio/margin/turnover/growth terms take precedence; remaining records use the frozen planner task_type.",
            "gold_or_reference_used_for_gating": False,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "results.json").write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Financial Operation Planner Gating Shadow v1", "", "## Method", "",
        "This is a fully offline counterfactual replay. An enabled case uses its frozen Planner v2 Judge result; a disabled case uses the original final100 Judge result. No model or pipeline stage was called.", "",
        "Specific metric-family labels (`ratio`, `margin`, `turnover`, `growth`) are assigned before the general frozen planner `task_type`, so the nested gates remain distinct.", "",
        "## Strategy comparison", "", "| Strategy | Enabled | Strict Judge | Newly correct | Regressed | Net gain | Delta |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, summary in summaries.items():
        lines.append(f"| {name} | {summary['planner_enabled']} | {summary['strategy_strict_judge']['correct']}/100 | {summary['newly_correct']} | {summary['regressed']} | {summary['net_gain']:+d} | {summary['delta_accuracy']:+.1%} |")
    lines += ["", "## Task-type gains and regressions", ""]
    for name, summary in summaries.items():
        lines += [f"### {name}", "", "| Gate task | Gains | Regressions | Net |", "|---|---:|---:|---:|"]
        for task, counts in summary["task_outcomes"].items():
            gain, regression = counts.get("gained", 0), counts.get("regressed", 0)
            if gain or regression:
                lines.append(f"| {task} | {gain} | {regression} | {gain - regression:+d} |")
        lines += ["", f"- Gained IDs: {', '.join(summary['gained_ids']) or 'None'}", f"- Regressed IDs: {', '.join(summary['regressed_ids']) or 'None'}", ""]
    best = max(summaries.values(), key=lambda item: (item["net_gain"], item["strategy_strict_judge"]["correct"], -item["planner_enabled"]))
    lines += ["## Offline conclusion", "", f"Best observed frozen gate: **{best['strategy']}**, with **{best['strategy_strict_judge']['correct']}/100**, net gain **{best['net_gain']:+d}**, and {best['planner_enabled']} enabled questions.", "",
        "This result estimates gating safety on the same frozen outputs; it does not prove generalization to new Planner generations.", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
