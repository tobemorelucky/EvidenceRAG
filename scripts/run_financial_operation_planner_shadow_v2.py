"""Full final100 answer-only safety shadow for the v2 operation planner candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from financial_operation_planner_v2 import attach_plan, create_planner_model, plan_question, planner_trigger  # noqa: E402


FINAL100 = ROOT / "reports/final100"
DATASET = ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_OUTPUT = ROOT / "reports/financial_operation_planner_shadow_v2"
ANSWER_MODEL = "deepseek-v4-flash-ga-260731"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    temporary.replace(path)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_inputs() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    answers = {record["question_id"]: record for record in load_jsonl(FINAL100 / "answers.jsonl")}
    judges = {record["question_id"]: record for record in load_jsonl(FINAL100 / "judge_results.jsonl")}
    if len(rows) != 100 or len(answers) != 100 or len(judges) != 100:
        raise ValueError("Expected complete final100 inputs")
    if any(answers.get(row["financebench_id"], {}).get("status") != "ok" for row in rows):
        raise ValueError("Frozen baseline answers are incomplete")
    return rows, answers, judges


def configure_answer() -> None:
    from runtime_profile import apply_runtime_profile

    apply_runtime_profile("clean_baseline")
    os.environ.update(
        MODEL=ANSWER_MODEL, ANSWER_PROMPT_MODE="baseline", ANSWER_TEMPERATURE="0.1",
        ANSWER_MAX_COMPLETION_TOKENS="1024", ANSWER_THINKING_MODE="disabled",
        ANSWER_TIMEOUT_SECONDS="60", ANSWER_MAX_RETRIES="0",
    )


def run(output: Path, rows: list[dict], baseline: dict[str, dict], judges: dict[str, dict]) -> None:
    if not os.getenv("ARK_API_KEY"):
        raise ValueError("ARK_API_KEY is required")
    path = output / "results.jsonl"
    saved = {record["id"]: record for record in load_jsonl(path)}
    configure_answer()
    from answer_generator import generate_answer

    planner_model = None
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        frozen = baseline[question_id]
        context_hash = digest(frozen["evidence"])
        triggered, trigger_reason = planner_trigger(row["question"])
        existing = saved.get(question_id, {})
        if existing.get("status") == "ok":
            if existing.get("context_sha256") != context_hash or existing.get("planner_triggered") != triggered:
                raise ValueError(f"Frozen input drift for {question_id}")
            continue
        record = {
            "id": question_id, "question": row["question"], "baseline_answer": frozen["answer"],
            "baseline_judge_score": judges[question_id]["judge"]["score"],
            "context_sha256": context_hash, "context_chars": len(frozen["evidence"]),
            "planner_enabled": True, "planner_triggered": triggered, "trigger_reason": trigger_reason,
        }
        if not triggered:
            record.update(
                status="ok", planner_answer=frozen["answer"], planner=None,
                planner_usage={}, planner_latency_ms=0, answer_usage={}, answer_latency_ms=0,
                text_changed=False, material_changed=False, planner_added_calculation=False,
            )
        else:
            if planner_model is None:
                planner_model = create_planner_model()
            planner_started = time.perf_counter()
            try:
                plan, planner_usage, raw = plan_question(row["question"], planner_model)
                record.update(planner=plan, planner_raw=raw, planner_usage=planner_usage)
                record["planner_latency_ms"] = (time.perf_counter() - planner_started) * 1000
                answer_started = time.perf_counter()
                guided_context = attach_plan(frozen["evidence"], plan)
                answer, answer_usage = generate_answer(row["question"], guided_context, [], "", "clean_baseline", "baseline")
                if not str(answer).strip():
                    raise RuntimeError("Empty planner answer")
                record.update(status="ok", planner_answer=answer, answer_usage=answer_usage,
                              answer_latency_ms=(time.perf_counter() - answer_started) * 1000,
                              guidance_chars=len(guided_context) - len(frozen["evidence"]))
                baseline_signature = answer_signature(frozen["answer"])
                planner_signature = answer_signature(answer)
                record["text_changed"] = normalize(frozen["answer"]) != normalize(answer)
                record["material_changed"] = baseline_signature != planner_signature
                record["planner_added_calculation"] = added_calculation(frozen["answer"], answer)
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__)
                record.setdefault("planner_latency_ms", (time.perf_counter() - planner_started) * 1000)
        record["planner_tokens"] = int((record.get("planner_usage") or {}).get("total_tokens") or 0)
        saved[question_id] = record
        atomic_jsonl(path, [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved])
        print(f"[{index}/100] {question_id} triggered={triggered} {record['status']}", flush=True)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def answer_signature(text: str) -> dict:
    lowered = normalize(text)
    numbers = re.findall(r"(?<![a-z])[-+]?\(?\d[\d,]*(?:\.\d+)?\)?%?", lowered)
    conclusions = sorted(set(re.findall(
        r"\b(?:yes|no|increase(?:d)?|decrease(?:d)?|improv(?:e|ed)|declin(?:e|ed)|higher|lower|highest|lowest|most|least|accelerat(?:e|ed)|decelerat(?:e|ed))\b",
        lowered,
    )))
    return {"numbers": numbers, "conclusions": conclusions}


def added_calculation(baseline: str, planner: str) -> bool:
    formula = re.compile(r"(?:\d[\d,.()%$ ]*)\s*[+\-*/÷]\s*(?:\d|\$)|\b(?:formula|calculation)\b", re.IGNORECASE)
    return not bool(formula.search(baseline or "")) and bool(formula.search(planner or ""))


def usage_sum(records: list[dict], field: str) -> dict:
    values = [record.get(field) or {} for record in records]
    return {key: sum(int(value.get(key) or 0) for value in values) for key in ("input_tokens", "output_tokens", "total_tokens")}


def build_summary(records: list[dict]) -> dict:
    triggered = [record for record in records if record.get("planner_triggered")]
    changed = [record for record in records if record.get("material_changed")]
    added = [record for record in records if record.get("planner_added_calculation")]
    judge_candidates = [
        record for record in records
        if record.get("status") == "ok" and (
            record.get("material_changed") or record.get("baseline_judge_score") == 0 or record.get("planner_added_calculation")
        )
    ]
    estimated_gain = [record["id"] for record in judge_candidates if record.get("baseline_judge_score") == 0]
    regressions = [record["id"] for record in judge_candidates if record.get("baseline_judge_score") == 1]
    planner_cost, answer_cost = usage_sum(records, "planner_usage"), usage_sum(records, "answer_usage")
    planner_latency = [record["planner_latency_ms"] for record in triggered if record.get("status") == "ok"]
    answer_latency = [record["answer_latency_ms"] for record in triggered if record.get("status") == "ok"]
    return {
        "total": len(records),
        "planner_triggered": len(triggered),
        "planner_trigger_rate": len(triggered) / len(records) if records else None,
        "changed_answers": len(changed),
        "text_changed_answers": sum(bool(record.get("text_changed")) for record in records),
        "planner_added_calculation": len(added),
        "estimated_gain": len(estimated_gain),
        "estimated_gain_note": "Baseline-incorrect Judge candidates only; not measured accuracy gain.",
        "regression_candidates": len(regressions),
        "judge_candidate_count": len(judge_candidates),
        "ids": {
            "estimated_gain_candidates": estimated_gain,
            "regression_candidates": regressions,
            "judge_candidates": [record["id"] for record in judge_candidates],
        },
        "failures": sum(record.get("status") != "ok" for record in records),
        "cost": {
            "planner": {**planner_cost, "latency_ms_per_triggered": sum(planner_latency) / len(planner_latency) if planner_latency else None},
            "answer": {**answer_cost, "latency_ms_per_triggered": sum(answer_latency) / len(answer_latency) if answer_latency else None},
        },
    }


def write_outputs(output: Path) -> dict:
    records = load_jsonl(output / "results.jsonl")
    if len(records) != 100:
        raise ValueError("Summary requires 100 records")
    summary = build_summary(records)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    selected = [record for record in records if record["id"] in set(summary["ids"]["judge_candidates"])]
    (output / "judge_selection.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Financial Operation Planner Production Candidate Shadow v2", "", "## Answer-only result", "",
        f"- Total: **{summary['total']}**", f"- Planner triggered: **{summary['planner_triggered']} ({summary['planner_trigger_rate']:.2%})**",
        f"- Materially changed answers: **{summary['changed_answers']}**", f"- Added calculations: **{summary['planner_added_calculation']}**",
        f"- Judge candidates: **{summary['judge_candidate_count']}**", f"- Estimated gain candidates: **{summary['estimated_gain']}**",
        f"- Regression candidates: **{summary['regression_candidates']}**", f"- Execution failures: **{summary['failures']}**", "",
        "`estimated_gain` is a candidate count, not a measured score. No new Judge call was made.", "", "## Cost", "",
        "| Stage | Input tokens | Output tokens | Total tokens | Latency/triggered |", "|---|---:|---:|---:|---:|"]
    for stage in ("planner", "answer"):
        cost = summary["cost"][stage]
        lines.append(f"| {stage} | {cost['input_tokens']:,} | {cost['output_tokens']:,} | {cost['total_tokens']:,} | {cost['latency_ms_per_triggered'] or 0:.0f} ms |")
    lines += ["", "## Per-question", "", "| ID | Triggered | Reason | Baseline Judge | Material change | Added calculation |", "|---|---:|---|---:|---:|---:|"]
    for record in records:
        lines.append(f"| {record['id']} | {record['planner_triggered']} | {record['trigger_reason']} | {record['baseline_judge_score']} | {record.get('material_changed', False)} | {record.get('planner_added_calculation', False)} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("answer", "report", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, baseline, judges = load_inputs()
    if args.stage in {"answer", "all"}:
        run(args.output_dir, rows, baseline, judges)
    if args.stage in {"report", "all"}:
        print(json.dumps(write_outputs(args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
