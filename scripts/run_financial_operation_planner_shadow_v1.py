"""Question-only operation planning shadow over 22 failures and 15 controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_jina_full_baseline_v1 import judge_answer  # noqa: E402


FINAL100 = ROOT / "reports/final100"
DATASET = ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_OUTPUT = ROOT / "reports/financial_operation_planner_shadow_v1"
PLANNER_MODEL = ANSWER_MODEL = "deepseek-v4-flash-ga-260731"
JUDGE_MODEL = "deepseek-v4-pro-ga-260813"
CONTROL_SEED = 20260905
TASK_TYPES = {"calculation", "comparison", "trend", "lookup", "judgment"}
REQUIRED_PLAN_KEYS = {"task_type", "entity", "period", "metric", "required_facts", "operation"}
OPTIONAL_PLAN_KEYS = {"formula", "direction"}

PLANNER_PROMPT = """You are a financial operation planner. Analyze only the question below.
Do not answer it, do not use external knowledge, and do not infer any value from a reference answer or gold evidence.
Return exactly one JSON object with no Markdown and no extra keys:
{{
  "task_type": "calculation|comparison|trend|lookup|judgment",
  "entity": "entity or scope requested by the question, or unknown",
  "period": "period requested by the question, or unknown",
  "metric": "financial metric or fact requested by the question",
  "required_facts": ["facts or operands that evidence must supply"],
  "operation": "concise operation to perform",
  "formula": "formula when explicitly required or null",
  "direction": "comparison/selection direction when required or null"
}}
The plan is procedural metadata, not factual evidence.

Question:
{question}"""

PLAN_GUIDANCE = """Operation plan (procedure guidance only; not factual evidence):
{plan}

Follow this plan only when its required facts are supported by the Evidence below. Never treat the plan as a source of values.

Original Evidence (unchanged):
{evidence}"""


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


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text") or "") if isinstance(item, dict) else str(item) for item in content)
    return str(content or "")


def parse_plan(text: str) -> dict:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict) or set(payload) - REQUIRED_PLAN_KEYS - OPTIONAL_PLAN_KEYS:
        raise ValueError("Planner output must be one JSON object with only schema keys")
    if not REQUIRED_PLAN_KEYS.issubset(payload):
        raise ValueError("Planner output is missing required keys")
    if payload["task_type"] not in TASK_TYPES:
        raise ValueError("Invalid task_type")
    if not all(isinstance(payload[key], str) and payload[key].strip() for key in ("entity", "period", "metric", "operation")):
        raise ValueError("Required scalar plan fields must be non-empty strings")
    if not isinstance(payload["required_facts"], list) or not all(isinstance(item, str) and item.strip() for item in payload["required_facts"]):
        raise ValueError("required_facts must be a list of non-empty strings")
    for key in OPTIONAL_PLAN_KEYS:
        if key in payload and payload[key] is not None and not isinstance(payload[key], str):
            raise ValueError(f"{key} must be a string or null")
    return payload


def select_cohorts(answers: dict[str, dict], judges: dict[str, dict]) -> tuple[list[str], list[str]]:
    failures = [
        question_id for question_id, answer in answers.items()
        if answer.get("status") == "ok"
        and answer.get("context_metrics", {}).get("candidate_gold_page_hit") is True
        and judges.get(question_id, {}).get("status") == "ok"
        and judges[question_id].get("judge", {}).get("score") == 0
    ]
    eligible_controls = [
        question_id for question_id, answer in answers.items()
        if answer.get("status") == "ok"
        and answer.get("context_metrics", {}).get("candidate_gold_page_hit") is True
        and judges.get(question_id, {}).get("status") == "ok"
        and judges[question_id].get("judge", {}).get("score") == 1
    ]
    controls = random.Random(CONTROL_SEED).sample(eligible_controls, 15)
    if len(failures) != 22:
        raise ValueError(f"Expected 22 failures, found {len(failures)}")
    return failures, controls


def load_inputs() -> tuple[list[dict], dict[str, dict], dict[str, dict], dict[str, str]]:
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        rows_by_id = {row["financebench_id"]: row for row in csv.DictReader(stream)}
    answers = {record["question_id"]: record for record in load_jsonl(FINAL100 / "answers.jsonl")}
    judges = {record["question_id"]: record for record in load_jsonl(FINAL100 / "judge_results.jsonl")}
    if not all(len(value) == 100 for value in (rows_by_id, answers, judges)):
        raise ValueError("Expected complete final100 inputs")
    failures, controls = select_cohorts(answers, judges)
    cohorts = {question_id: "failure" for question_id in failures} | {question_id: "control" for question_id in controls}
    ordered_ids = failures + controls
    return [rows_by_id[question_id] for question_id in ordered_ids], answers, judges, cohorts


def create_model(max_tokens: int, temperature: float):
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=ANSWER_MODEL,
        model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=temperature,
        max_completion_tokens=max_tokens,
        timeout=60,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )


def run_planner(output: Path, rows: list[dict], cohorts: dict[str, str]) -> None:
    from langchain_core.messages import HumanMessage, SystemMessage

    path = output / "planner_results.jsonl"
    saved = {record["id"]: record for record in load_jsonl(path)}
    model = create_model(512, 0)
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        question_hash = digest(row["question"])
        existing = saved.get(question_id, {})
        if existing.get("status") == "ok":
            if existing.get("question_sha256") != question_hash:
                raise ValueError(f"Question drift for {question_id}")
            continue
        started = time.perf_counter()
        record = {"id": question_id, "cohort": cohorts[question_id], "question_sha256": question_hash, "model": PLANNER_MODEL}
        try:
            response = model.invoke([
                SystemMessage(content="Return strict JSON only."),
                HumanMessage(content=PLANNER_PROMPT.format(question=row["question"])),
            ])
            raw = content_text(response.content).strip()
            record.update(status="ok", plan=parse_plan(raw), raw=raw, usage=dict(getattr(response, "usage_metadata", {}) or {}))
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__)
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        saved[question_id] = record
        atomic_jsonl(path, [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved])
        print(f"[planner {index}/37] {question_id} {record['status']}", flush=True)


def configure_answer() -> None:
    from runtime_profile import apply_runtime_profile

    apply_runtime_profile("clean_baseline")
    os.environ.update(
        MODEL=ANSWER_MODEL, ANSWER_PROMPT_MODE="baseline", ANSWER_TEMPERATURE="0.1",
        ANSWER_MAX_COMPLETION_TOKENS="1024", ANSWER_THINKING_MODE="disabled",
        ANSWER_TIMEOUT_SECONDS="60", ANSWER_MAX_RETRIES="0",
    )


def run_answers(output: Path, rows: list[dict], baseline: dict[str, dict], cohorts: dict[str, str]) -> None:
    plans = {record["id"]: record for record in load_jsonl(output / "planner_results.jsonl")}
    path = output / "answers.jsonl"
    saved = {record["id"]: record for record in load_jsonl(path)}
    configure_answer()
    from answer_generator import generate_answer

    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        context = baseline[question_id]["evidence"]
        plan_record = plans.get(question_id, {})
        plan_hash = digest(json.dumps(plan_record.get("plan"), ensure_ascii=False, sort_keys=True))
        existing = saved.get(question_id, {})
        if existing.get("status") == "ok":
            if existing.get("context_sha256") != digest(context) or existing.get("plan_sha256") != plan_hash:
                raise ValueError(f"Frozen input drift for {question_id}")
            continue
        started = time.perf_counter()
        record = {
            "id": question_id, "cohort": cohorts[question_id], "model": ANSWER_MODEL,
            "context_sha256": digest(context), "context_chars": len(context), "plan_sha256": plan_hash,
            "original_answer": baseline[question_id]["answer"],
        }
        if plan_record.get("status") != "ok":
            record.update(status="error", error_type="planner_unavailable")
        else:
            try:
                plan_json = json.dumps(plan_record["plan"], ensure_ascii=False, separators=(",", ":"))
                guided_evidence = PLAN_GUIDANCE.format(plan=plan_json, evidence=context)
                answer, usage = generate_answer(row["question"], guided_evidence, [], "", "clean_baseline", "baseline")
                if not str(answer).strip():
                    raise RuntimeError("Empty answer")
                record.update(status="ok", answer=answer, usage=usage, guidance_chars=len(guided_evidence) - len(context))
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__)
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        saved[question_id] = record
        atomic_jsonl(path, [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved])
        print(f"[answer {index}/37] {question_id} {record['status']}", flush=True)


def run_judges(output: Path, rows: list[dict], cohorts: dict[str, str]) -> None:
    answers = {record["id"]: record for record in load_jsonl(output / "answers.jsonl")}
    path = output / "judge_results.jsonl"
    saved = {record["id"]: record for record in load_jsonl(path)}
    holder: list = []
    config = {"model": JUDGE_MODEL, "max_completion_tokens": 512, "timeout_seconds": 60}
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        answer = answers.get(question_id, {})
        existing = saved.get(question_id, {})
        if existing.get("status") == "ok":
            continue
        started = time.perf_counter()
        record = {"id": question_id, "cohort": cohorts[question_id]}
        if answer.get("status") != "ok":
            record.update(status="error", error_type="answer_unavailable")
        else:
            try:
                result = judge_answer(row["question"], row["answer"], answer["answer"], config, holder)
                record.update(status="ok", judge=result)
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__)
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        saved[question_id] = record
        atomic_jsonl(path, [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved])
        print(f"[judge {index}/37] {question_id} {record['status']}", flush=True)


def usage(records: list[dict], getter) -> dict:
    values = [getter(record) or {} for record in records]
    return {key: sum(int(value.get(key) or 0) for value in values) for key in ("input_tokens", "output_tokens", "total_tokens")}


def summarize(plans: list[dict], answers: list[dict], judges: list[dict]) -> dict:
    judged = {record["id"]: record for record in judges if record.get("status") == "ok"}
    recovered = [question_id for question_id, record in judged.items() if record["cohort"] == "failure" and record["judge"]["score"] == 1]
    regressions = [question_id for question_id, record in judged.items() if record["cohort"] == "control" and record["judge"]["score"] == 0]
    maintained_errors = [question_id for question_id, record in judged.items() if record["cohort"] == "failure" and record["judge"]["score"] == 0]
    maintained_correct = [question_id for question_id, record in judged.items() if record["cohort"] == "control" and record["judge"]["score"] == 1]
    stage_records = {"planner": plans, "answer": answers, "judge": list(judged.values())}
    summary = {
        "cohorts": {"failures": 22, "correct_controls": 15, "control_seed": CONTROL_SEED},
        "outcomes": {
            "recovered": len(recovered), "regressed": len(regressions),
            "maintained_incorrect": len(maintained_errors), "maintained_correct": len(maintained_correct),
        },
        "ids": {
            "recovered": recovered, "regressed": regressions,
            "maintained_incorrect": maintained_errors, "maintained_correct": maintained_correct,
        },
        "failures": {
            "planner": sum(record.get("status") != "ok" for record in plans),
            "answer": sum(record.get("status") != "ok" for record in answers),
            "judge": 37 - len(judged),
        },
        "acceptance": {"required_recoveries": 5, "passed": len(recovered) >= 5, "stop_answer_optimization": len(recovered) < 5},
        "cost": {},
    }
    for stage, records in stage_records.items():
        stage_usage = usage(records, (lambda record: record.get("judge", {}).get("usage")) if stage == "judge" else (lambda record: record.get("usage")))
        latencies = [record["latency_ms"] for record in records if record.get("status") == "ok"]
        summary["cost"][stage] = {**stage_usage, "latency_ms_per_question": sum(latencies) / len(latencies) if latencies else None}
    return summary


def write_report(output: Path, rows: list[dict], baseline_judges: dict[str, dict]) -> dict:
    plans, answers, judges = (load_jsonl(output / name) for name in ("planner_results.jsonl", "answers.jsonl", "judge_results.jsonl"))
    if not all(len(records) == 37 for records in (plans, answers, judges)):
        raise ValueError("Report requires 37 complete records per stage")
    summary = summarize(plans, answers, judges)
    plan_by_id, answer_by_id, judge_by_id = ({record["id"]: record for record in records} for records in (plans, answers, judges))
    lines = ["# Financial Operation Planner Shadow v1", "", "## Summary", "",
        f"- Recovered failures: **{summary['outcomes']['recovered']}/22**",
        f"- Regressed correct controls: **{summary['outcomes']['regressed']}/15**",
        f"- Maintained incorrect/correct: **{summary['outcomes']['maintained_incorrect']} / {summary['outcomes']['maintained_correct']}**",
        f"- Acceptance >=5 recoveries: **{'passed' if summary['acceptance']['passed'] else 'failed'}**", "",
        "## Cost", "", "| Stage | Input tokens | Output tokens | Total tokens | Latency/question |", "|---|---:|---:|---:|---:|"]
    for stage in ("planner", "answer", "judge"):
        cost = summary["cost"][stage]
        lines.append(f"| {stage} | {cost['input_tokens']:,} | {cost['output_tokens']:,} | {cost['total_tokens']:,} | {cost['latency_ms_per_question'] or 0:.0f} ms |")
    lines += ["", "## IDs", "", f"- Recovered: {', '.join(summary['ids']['recovered']) or 'None'}",
        f"- Regressed: {', '.join(summary['ids']['regressed']) or 'None'}", "", "## Per-case results", ""]
    for row in rows:
        question_id = row["financebench_id"]
        plan, answer, judged = plan_by_id[question_id], answer_by_id[question_id], judge_by_id[question_id]
        lines += [f"### {question_id} — {plan['cohort']} — {judged.get('judge', {}).get('verdict', 'error')}", "",
            f"**Question:** {row['question']}", "", f"**Planner JSON:** `{json.dumps(plan.get('plan'), ensure_ascii=False)}`", "",
            f"**Original answer:** {answer.get('original_answer', '')}", "", f"**Planned answer:** {answer.get('answer', '')}", "",
            f"**Original Judge:** {baseline_judges[question_id]['judge']['reason']}", "",
            f"**New Judge:** {judged.get('judge', {}).get('reason', judged.get('error_type', ''))}", ""]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("planner", "answer", "judge", "report", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    if args.stage != "report" and not os.getenv("ARK_API_KEY"):
        raise ValueError("ARK_API_KEY is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, baseline_answers, baseline_judges, cohorts = load_inputs()
    if args.stage in {"planner", "all"}:
        run_planner(args.output_dir, rows, cohorts)
    if args.stage in {"answer", "all"}:
        run_answers(args.output_dir, rows, baseline_answers, cohorts)
    if args.stage in {"judge", "all"}:
        run_judges(args.output_dir, rows, cohorts)
    if args.stage in {"report", "all"}:
        summary = write_report(args.output_dir, rows, baseline_judges)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
