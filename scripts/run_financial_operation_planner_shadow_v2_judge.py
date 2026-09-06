"""Judge the frozen 60-case Financial Operation Planner Shadow v2 selection."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "reports/financial_operation_planner_shadow_v2"
FINAL100 = ROOT / "reports/final100"
DATASET = ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_OUTPUT = SOURCE / "judge"
JUDGE_MODEL = "deepseek-v4-pro-ga-260813"
EXPECTED_SELECTION = 60


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    temporary.replace(path)


def load_inputs() -> tuple[list[dict], dict[str, dict], dict[str, dict], dict[str, dict]]:
    results = {record["id"]: record for record in load_jsonl(SOURCE / "results.jsonl")}
    selected = json.loads((SOURCE / "judge_selection.json").read_text(encoding="utf-8"))
    baseline_judges = {record["question_id"]: record for record in load_jsonl(FINAL100 / "judge_results.jsonl")}
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["financebench_id"]: row for row in csv.DictReader(stream)}
    selected_ids = [record["id"] for record in selected]
    if len(results) != 100 or len(baseline_judges) != 100 or len(rows) != 100:
        raise ValueError("Expected complete 100-question frozen inputs")
    if len(selected) != EXPECTED_SELECTION or len(set(selected_ids)) != EXPECTED_SELECTION:
        raise ValueError(f"Expected exactly {EXPECTED_SELECTION} unique frozen Judge cases")
    if any(question_id not in results or results[question_id].get("status") != "ok" for question_id in selected_ids):
        raise ValueError("Judge selection contains a missing or failed planner answer")
    if any(results[question_id].get("planner_answer") != record.get("planner_answer") for question_id, record in zip(selected_ids, selected)):
        raise ValueError("judge_selection.json has drifted from results.jsonl")
    return selected, results, baseline_judges, rows


def classify_change(record: dict, judge_reason: str, *, regression: bool) -> str:
    """Assign a broad, question-agnostic diagnosis from the plan and Judge reason."""
    plan = record.get("planner") or {}
    task = str(plan.get("task_type") or "").lower()
    reason = str(judge_reason or "").lower()
    combined = f"{record.get('question', '')}\n{record.get('planner_answer', '')}\n{reason}".lower()
    if re.search(r"cannot (?:answer|determine|calculate)|insufficient (?:evidence|information|data)|refus", combined):
        return "refusal"
    if re.search(r"wrong (?:company|entity|segment|scope|period|year)|entity|segment|scope|period", reason):
        return "scope"
    if task in {"trend", "comparison"} or re.search(r"increase|decrease|direction|trend|higher|lower", reason):
        return "trend/direction"
    if re.search(r"metric|definition|margin|ratio|ebit|ebitda|mapping|proxy", reason):
        return "metric mapping"
    if task == "calculation" or re.search(r"calculat|formula|arithmetic|operand|round", reason):
        return "calculation"
    return "other planner error" if regression else "other"


def run_judge(output: Path, selected: list[dict], rows: dict[str, dict]) -> None:
    if not (os.getenv("JUDGE_API_KEY") or os.getenv("ARK_API_KEY")):
        raise ValueError("JUDGE_API_KEY or ARK_API_KEY is required")
    from scripts.run_jina_full_baseline_v1 import judge_answer

    path = output / "planner_judge_results.jsonl"
    saved = {record["id"]: record for record in load_jsonl(path)}
    holder: list = []
    config = {"model": JUDGE_MODEL, "max_completion_tokens": 512, "timeout_seconds": 60}
    ordered_ids = [record["id"] for record in selected]
    for index, source in enumerate(selected, 1):
        question_id = source["id"]
        existing = saved.get(question_id, {})
        if existing.get("status") == "ok":
            print(f"[{index}/{EXPECTED_SELECTION}] {question_id} cached", flush=True)
            continue
        started = time.perf_counter()
        record = {"id": question_id, "baseline_judge_score": source["baseline_judge_score"]}
        try:
            row = rows[question_id]
            judged = judge_answer(row["question"], row["answer"], source["planner_answer"], config, holder)
            record.update(status="ok", judge=judged)
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__, error=str(exc)[:500])
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        saved[question_id] = record
        atomic_jsonl(path, [saved[item] for item in ordered_ids if item in saved])
        print(f"[{index}/{EXPECTED_SELECTION}] {question_id} {record['status']}", flush=True)


def summarize(
    selected: list[dict],
    all_results: dict[str, dict],
    baseline_judges: dict[str, dict],
    planner_judges: list[dict],
) -> dict:
    judged = {record["id"]: record for record in planner_judges if record.get("status") == "ok"}
    selected_ids = {record["id"] for record in selected}
    if len(judged) != EXPECTED_SELECTION:
        raise ValueError(f"Report requires {EXPECTED_SELECTION} successful Planner Judge results; found {len(judged)}")
    outcomes = Counter()
    outcome_ids: dict[str, list[str]] = {
        "baseline_incorrect_planner_correct": [], "baseline_correct_planner_incorrect": [],
        "maintained_incorrect": [], "maintained_correct": [],
    }
    recovery_types: Counter = Counter()
    regression_types: Counter = Counter()
    for source in selected:
        question_id = source["id"]
        baseline_score = int(baseline_judges[question_id]["judge"]["score"])
        planner_score = int(judged[question_id]["judge"]["score"])
        if baseline_score == 0 and planner_score == 1:
            name = "baseline_incorrect_planner_correct"
            recovery_types[classify_change(source, judged[question_id]["judge"].get("reason", ""), regression=False)] += 1
        elif baseline_score == 1 and planner_score == 0:
            name = "baseline_correct_planner_incorrect"
            regression_types[classify_change(source, judged[question_id]["judge"].get("reason", ""), regression=True)] += 1
        elif baseline_score == 0:
            name = "maintained_incorrect"
        else:
            name = "maintained_correct"
        outcomes[name] += 1
        outcome_ids[name].append(question_id)

    baseline_correct = sum(int(record["judge"]["score"]) for record in baseline_judges.values())
    planner_correct = 0
    for question_id in all_results:
        if question_id in selected_ids:
            planner_correct += int(judged[question_id]["judge"]["score"])
        else:
            planner_correct += int(baseline_judges[question_id]["judge"]["score"])
    recovered = outcomes["baseline_incorrect_planner_correct"]
    regressed = outcomes["baseline_correct_planner_incorrect"]
    usage = Counter()
    latencies = []
    for record in judged.values():
        usage.update({key: int(record["judge"].get("usage", {}).get(key) or 0) for key in ("input_tokens", "output_tokens", "total_tokens")})
        latencies.append(float(record.get("latency_ms") or 0))
    return {
        "total": 100,
        "judged_selection": EXPECTED_SELECTION,
        "baseline_strict_judge": {"correct": baseline_correct, "total": 100, "accuracy": baseline_correct / 100},
        "planner_strict_judge": {"correct": planner_correct, "total": 100, "accuracy": planner_correct / 100},
        "delta_accuracy": (planner_correct - baseline_correct) / 100,
        "net_gain_questions": recovered - regressed,
        "meets_net_gain_at_least_5": recovered - regressed >= 5,
        "outcomes": dict(outcomes),
        "ids": outcome_ids,
        "recovery_types": dict(sorted(recovery_types.items())),
        "regression_types": dict(sorted(regression_types.items())),
        "judge_cost": {
            "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "latency_ms_per_judged_question": sum(latencies) / len(latencies) if latencies else None,
        },
        "method": "The frozen 60 changed/risk cases use new Pro Judge results; the other 40 reuse final100 Judge results.",
    }


def write_report(output: Path, selected: list[dict], all_results: dict[str, dict], baseline: dict[str, dict]) -> dict:
    planner_judges = load_jsonl(output / "planner_judge_results.jsonl")
    summary = summarize(selected, all_results, baseline, planner_judges)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Financial Operation Planner Shadow v2 — Judge", "", "## Result", "",
        f"- Baseline Strict Judge: **{summary['baseline_strict_judge']['correct']}/100 ({summary['baseline_strict_judge']['accuracy']:.1%})**",
        f"- Planner Strict Judge: **{summary['planner_strict_judge']['correct']}/100 ({summary['planner_strict_judge']['accuracy']:.1%})**",
        f"- Delta accuracy: **{summary['delta_accuracy']:+.1%}**",
        f"- Net gain: **{summary['net_gain_questions']:+d} questions**",
        f"- Acceptance (net gain >=5): **{'passed' if summary['meets_net_gain_at_least_5'] else 'failed'}**", "",
        "## Outcome migration", "", "| Outcome | Count |", "|---|---:|",
        f"| Baseline incorrect → Planner correct | {summary['outcomes'].get('baseline_incorrect_planner_correct', 0)} |",
        f"| Baseline correct → Planner incorrect | {summary['outcomes'].get('baseline_correct_planner_incorrect', 0)} |",
        f"| Maintained incorrect | {summary['outcomes'].get('maintained_incorrect', 0)} |",
        f"| Maintained correct | {summary['outcomes'].get('maintained_correct', 0)} |", "",
        "## Attribution", "",
        f"- Recovery types: `{json.dumps(summary['recovery_types'], ensure_ascii=False)}`",
        f"- Regression types: `{json.dumps(summary['regression_types'], ensure_ascii=False)}`", "",
        "Attribution is deterministic and broad: it uses the frozen planner task plus the new Judge reason; it is diagnostic, not a second evaluation.", "",
        "## Judge cost", "",
        f"- Tokens: **{summary['judge_cost']['total_tokens']:,}** (input {summary['judge_cost']['input_tokens']:,}, output {summary['judge_cost']['output_tokens']:,})",
        f"- Mean latency: **{summary['judge_cost']['latency_ms_per_judged_question'] or 0:.0f} ms/question**", "",
        "## Changed cases", "",
    ]
    by_id = {record["id"]: record for record in planner_judges}
    for outcome in ("baseline_incorrect_planner_correct", "baseline_correct_planner_incorrect"):
        lines += [f"### {outcome}", ""]
        for question_id in summary["ids"][outcome]:
            source = all_results[question_id]
            reason = by_id[question_id]["judge"].get("reason", "")
            lines.append(f"- `{question_id}` — {classify_change(source, reason, regression=outcome.endswith('planner_incorrect'))}: {reason}")
        if not summary["ids"][outcome]:
            lines.append("- None")
        lines.append("")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("judge", "report", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected, all_results, baseline, rows = load_inputs()
    if args.stage in {"judge", "all"}:
        run_judge(args.output_dir, selected, rows)
    if args.stage in {"report", "all"}:
        print(json.dumps(write_report(args.output_dir, selected, all_results, baseline), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
