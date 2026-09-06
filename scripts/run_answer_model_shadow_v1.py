"""Compare DeepSeek-V4-Pro answers with the frozen final100 Flash baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_jina_full_baseline_v1 import judge_answer  # noqa: E402


DATASET = ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"
FINAL100 = ROOT / "reports/final100"
DEFAULT_OUTPUT = ROOT / "reports/answer_model_shadow_v1"
ANSWER_MODEL = "deepseek-v4-pro-ga-260813"
JUDGE_MODEL = "deepseek-v4-pro-ga-260813"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_inputs() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    baseline_answers = {record["question_id"]: record for record in load_jsonl(FINAL100 / "answers.jsonl")}
    baseline_judges = {record["question_id"]: record for record in load_jsonl(FINAL100 / "judge_results.jsonl")}
    if len(rows) != 100 or len(baseline_answers) != 100 or len(baseline_judges) != 100:
        raise ValueError("Expected exactly 100 dataset, final100 answer, and final100 judge records")
    if any(baseline_answers.get(row["financebench_id"], {}).get("status") != "ok" for row in rows):
        raise ValueError("The frozen final100 answers are incomplete")
    if any(baseline_judges.get(row["financebench_id"], {}).get("status") != "ok" for row in rows):
        raise ValueError("The frozen final100 judges are incomplete")
    return rows, baseline_answers, baseline_judges


def configure_answer() -> None:
    from runtime_profile import apply_runtime_profile

    apply_runtime_profile("clean_baseline")
    os.environ.update(
        MODEL=ANSWER_MODEL,
        ANSWER_PROMPT_MODE="baseline",
        ANSWER_TEMPERATURE="0.1",
        ANSWER_MAX_COMPLETION_TOKENS="1024",
        ANSWER_THINKING_MODE="disabled",
        ANSWER_TIMEOUT_SECONDS="60",
        ANSWER_MAX_RETRIES="0",
    )


def run_answers(output: Path, rows: list[dict], baseline: dict[str, dict]) -> None:
    if not os.getenv("ARK_API_KEY"):
        raise ValueError("ARK_API_KEY is required")
    configure_answer()
    from answer_generator import generate_answer

    path = output / "answers.jsonl"
    saved = {record["question_id"]: record for record in load_jsonl(path)}
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        frozen = baseline[question_id]
        context_hash = digest(frozen["evidence"])
        existing = saved.get(question_id, {})
        if existing.get("answer_status") == "ok":
            if existing.get("context_sha256") != context_hash:
                raise ValueError(f"Frozen context drift for {question_id}")
            continue
        started = time.perf_counter()
        record = {
            **existing,
            "question_id": question_id,
            "question": row["question"],
            "reference_answer": row["answer"],
            "answer_model": ANSWER_MODEL,
            "prompt_mode": "baseline",
            "context_sha256": context_hash,
            "context_chars": len(frozen["evidence"]),
        }
        try:
            answer, usage = generate_answer(row["question"], frozen["evidence"], [], "", "clean_baseline", "baseline")
            if not str(answer).strip():
                raise RuntimeError("Empty answer")
            record.update(answer_status="ok", answer=answer, answer_usage=usage)
            record.pop("answer_error", None)
        except Exception as exc:
            record.update(answer_status="error", answer_error=type(exc).__name__)
        record["answer_latency_ms"] = (time.perf_counter() - started) * 1000
        saved[question_id] = record
        atomic_jsonl(path, [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved])
        print(f"[answer {index}/100] {question_id} {record['answer_status']}", flush=True)


def run_judges(output: Path, rows: list[dict]) -> None:
    if not (os.getenv("JUDGE_API_KEY") or os.getenv("ARK_API_KEY")):
        raise ValueError("JUDGE_API_KEY or ARK_API_KEY is required")
    path = output / "answers.jsonl"
    saved = {record["question_id"]: record for record in load_jsonl(path)}
    holder: list = []
    config = {"model": JUDGE_MODEL, "max_completion_tokens": 512, "timeout_seconds": 60}
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        record = saved.get(question_id)
        if not record or record.get("answer_status") != "ok":
            continue
        if record.get("judge_status") == "ok":
            continue
        started = time.perf_counter()
        try:
            result = judge_answer(row["question"], row["answer"], record["answer"], config, holder)
            record.update(judge_status="ok", judge=result)
            record.pop("judge_error", None)
        except Exception as exc:
            record.update(judge_status="error", judge_error=type(exc).__name__)
        record["judge_latency_ms"] = (time.perf_counter() - started) * 1000
        atomic_jsonl(path, [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved])
        print(f"[judge {index}/100] {question_id} {record['judge_status']}", flush=True)


def _usage(records: list[dict], field: str) -> dict:
    usages = [record.get(field) or {} for record in records]
    return {
        key: sum(int(usage.get(key) or 0) for usage in usages)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }


def comparison_counts(
    rows: list[dict], baseline_judges: dict[str, dict], pro_records: dict[str, dict]
) -> tuple[dict, dict[str, list[str]]]:
    groups = {key: [] for key in ("newly_correct", "regressed", "still_incorrect", "still_correct")}
    for row in rows:
        question_id = row["financebench_id"]
        flash_correct = baseline_judges[question_id]["judge"]["score"] == 1
        pro = pro_records.get(question_id, {})
        pro_correct = pro.get("judge_status") == "ok" and pro.get("judge", {}).get("score") == 1
        key = (
            "still_correct" if flash_correct and pro_correct
            else "regressed" if flash_correct
            else "newly_correct" if pro_correct
            else "still_incorrect"
        )
        groups[key].append(question_id)
    return {key: len(value) for key, value in groups.items()}, groups


def write_report(output: Path, rows: list[dict], baseline_judges: dict[str, dict]) -> dict:
    records = load_jsonl(output / "answers.jsonl")
    by_id = {record["question_id"]: record for record in records}
    counts, groups = comparison_counts(rows, baseline_judges, by_id)
    judged = [record for record in records if record.get("judge_status") == "ok"]
    correct = sum(record["judge"]["score"] == 1 for record in judged)
    answer_usage, judge_usage = _usage(records, "answer_usage"), _usage(judged, "judge_usage")
    # judge usage is nested in the shared judge result
    judge_usage = _usage([{"usage": record.get("judge", {}).get("usage", {})} for record in judged], "usage")
    answer_latencies = [record["answer_latency_ms"] for record in records if record.get("answer_status") == "ok"]
    judge_latencies = [record["judge_latency_ms"] for record in judged]
    summary = {
        "questions": 100,
        "models": {"flash_baseline": "deepseek-v4-flash-ga-260731", "pro_answer": ANSWER_MODEL, "judge": JUDGE_MODEL},
        "strict_judge": {"correct": correct, "total": len(judged), "accuracy": correct / len(judged) if judged else None},
        "flash_strict_judge": {"correct": sum(record["judge"]["score"] == 1 for record in baseline_judges.values()), "total": 100},
        "comparison": counts,
        "question_ids": groups,
        "failures": {
            "answer": sum(record.get("answer_status") != "ok" for record in records),
            "judge": sum(record.get("answer_status") == "ok" and record.get("judge_status") != "ok" for record in records),
        },
        "cost": {
            "answer": {**answer_usage, "latency_ms_total": sum(answer_latencies), "latency_ms_per_question": sum(answer_latencies) / len(answer_latencies) if answer_latencies else None},
            "judge": {**judge_usage, "latency_ms_total": sum(judge_latencies), "latency_ms_per_question": sum(judge_latencies) / len(judge_latencies) if judge_latencies else None},
        },
    }
    atomic_json(output / "summary.json", summary)
    lines = [
        "# DeepSeek Answer Model Shadow v1", "",
        "## Result", "",
        f"- Flash baseline Strict Judge: **{summary['flash_strict_judge']['correct']}/100**",
        f"- Pro answer Strict Judge: **{correct}/{len(judged)} ({(summary['strict_judge']['accuracy'] or 0):.2%})**",
        f"- Newly correct / regressed / still incorrect / still correct: **{counts['newly_correct']} / {counts['regressed']} / {counts['still_incorrect']} / {counts['still_correct']}**",
        "", "## Changed question IDs", "",
        f"### Flash incorrect -> Pro correct ({counts['newly_correct']})", "",
        "- " + ("\n- ".join(groups["newly_correct"]) if groups["newly_correct"] else "None"), "",
        f"### Flash correct -> Pro incorrect ({counts['regressed']})", "",
        "- " + ("\n- ".join(groups["regressed"]) if groups["regressed"] else "None"), "",
        "## Cost", "",
        "| Stage | Input tokens | Output tokens | Total tokens | Latency/question |", "|---|---:|---:|---:|---:|",
        f"| Pro answer | {answer_usage['input_tokens']:,} | {answer_usage['output_tokens']:,} | {answer_usage['total_tokens']:,} | {summary['cost']['answer']['latency_ms_per_question'] or 0:.0f} ms |",
        f"| Pro judge | {judge_usage['input_tokens']:,} | {judge_usage['output_tokens']:,} | {judge_usage['total_tokens']:,} | {summary['cost']['judge']['latency_ms_per_question'] or 0:.0f} ms |",
        "", "## Per-question comparison", "",
        "| ID | Flash | Pro | Change |", "|---|---:|---:|---|",
    ]
    for row in rows:
        question_id = row["financebench_id"]
        flash = baseline_judges[question_id]["judge"]["score"]
        pro = by_id.get(question_id, {}).get("judge", {}).get("score")
        change = next(key for key, ids in groups.items() if question_id in ids)
        lines.append(f"| {question_id} | {flash} | {pro if pro is not None else 'error'} | {change} |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("answer", "judge", "report", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, baseline_answers, baseline_judges = load_inputs()
    if args.stage in {"answer", "all"}:
        run_answers(args.output_dir, rows, baseline_answers)
    if args.stage in {"judge", "all"}:
        run_judges(args.output_dir, rows)
    if args.stage in {"report", "all"}:
        summary = write_report(args.output_dir, rows, baseline_judges)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
