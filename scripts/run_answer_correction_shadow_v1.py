"""Run a 22-case financial answer correction shadow over frozen final100 contexts."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from run_jina_full_baseline_v1 import judge_answer  # noqa: E402


FINAL100 = ROOT / "reports/final100"
DATASET = ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_OUTPUT = ROOT / "reports/answer_correction_shadow_v1"
ANSWER_MODEL = "deepseek-v4-flash-ga-260731"
JUDGE_MODEL = "deepseek-v4-pro-ga-260813"

CRITIC_SYSTEM_PROMPT = """You are a financial answer critic and reviser.
Use only the supplied evidence. Recheck the original answer only for:
1. calculation: formula, operands, arithmetic, signs, and rounding;
2. direction: increase/decrease, higher/lower, and trend conclusions;
3. metric: consistency with the financial metric requested by the question;
4. scope: entity, segment, fiscal period, quarter/year, and unit consistency.

Correct errors supported by the evidence. Do not add outside facts, do not mention a reference answer,
and do not expose analysis or an internal critique. Preserve useful source citations. Return only the
complete revised answer that should replace the original answer."""

CRITIC_USER_TEMPLATE = """Question:
{question}

Original answer:
{original_answer}

Evidence:
{evidence}

Revised answer:"""


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


def select_failures(answers: dict[str, dict], judges: dict[str, dict]) -> list[str]:
    return [
        question_id
        for question_id, answer in answers.items()
        if answer.get("status") == "ok"
        and answer.get("context_metrics", {}).get("candidate_gold_page_hit") is True
        and judges.get(question_id, {}).get("status") == "ok"
        and judges[question_id].get("judge", {}).get("score") == 0
    ]


def load_inputs() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        rows_by_id = {row["financebench_id"]: row for row in csv.DictReader(stream)}
    answers = {record["question_id"]: record for record in load_jsonl(FINAL100 / "answers.jsonl")}
    judges = {record["question_id"]: record for record in load_jsonl(FINAL100 / "judge_results.jsonl")}
    ids = select_failures(answers, judges)
    if len(rows_by_id) != 100 or len(answers) != 100 or len(judges) != 100 or len(ids) != 22:
        raise ValueError("Expected a complete final100 baseline and exactly 22 context-hit answer failures")
    return [rows_by_id[question_id] for question_id in ids], answers, judges


def create_critic_model():
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=ANSWER_MODEL,
        model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.1,
        max_completion_tokens=1024,
        timeout=60,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )


def content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text") or "") if isinstance(item, dict) else str(item) for item in content)
    return str(content or "")


def run_revisions(output: Path, rows: list[dict], baseline_answers: dict[str, dict]) -> None:
    if not os.getenv("ARK_API_KEY"):
        raise ValueError("ARK_API_KEY is required")
    path = output / "revised_answers.jsonl"
    saved = {record["id"]: record for record in load_jsonl(path)}
    model = create_critic_model()
    from langchain_core.messages import HumanMessage, SystemMessage

    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        baseline = baseline_answers[question_id]
        context_hash = digest(baseline["evidence"])
        existing = saved.get(question_id, {})
        if existing.get("status") == "ok":
            if existing.get("context_sha256") != context_hash:
                raise ValueError(f"Frozen context drift for {question_id}")
            continue
        started = time.perf_counter()
        record = {
            "id": question_id,
            "question": row["question"],
            "original_answer": baseline["answer"],
            "context_sha256": context_hash,
            "context_chars": len(baseline["evidence"]),
            "model": ANSWER_MODEL,
        }
        try:
            response = model.invoke([
                SystemMessage(content=CRITIC_SYSTEM_PROMPT),
                HumanMessage(content=CRITIC_USER_TEMPLATE.format(
                    question=row["question"], original_answer=baseline["answer"], evidence=baseline["evidence"]
                )),
            ])
            revised = content_text(response.content).strip()
            if not revised:
                raise RuntimeError("Empty revised answer")
            record.update(
                status="ok",
                revised_answer=revised,
                usage=dict(getattr(response, "usage_metadata", {}) or {}),
            )
        except Exception as exc:
            record.update(status="error", error_type=type(exc).__name__)
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        saved[question_id] = record
        atomic_jsonl(path, [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved])
        print(f"[revise {index}/22] {question_id} {record['status']}", flush=True)


def run_judges(output: Path, rows: list[dict]) -> None:
    if not (os.getenv("JUDGE_API_KEY") or os.getenv("ARK_API_KEY")):
        raise ValueError("JUDGE_API_KEY or ARK_API_KEY is required")
    revisions = {record["id"]: record for record in load_jsonl(output / "revised_answers.jsonl")}
    path = output / "judge_results.jsonl"
    saved = {record["id"]: record for record in load_jsonl(path)}
    holder: list = []
    config = {"model": JUDGE_MODEL, "max_completion_tokens": 512, "timeout_seconds": 60}
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        revision = revisions.get(question_id, {})
        if saved.get(question_id, {}).get("status") == "ok":
            continue
        started = time.perf_counter()
        record = {"id": question_id}
        if revision.get("status") != "ok":
            record.update(status="error", error_type="revision_unavailable")
        else:
            try:
                result = judge_answer(row["question"], row["answer"], revision["revised_answer"], config, holder)
                record.update(status="ok", judge=result)
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__)
        record["latency_ms"] = (time.perf_counter() - started) * 1000
        saved[question_id] = record
        atomic_jsonl(path, [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved])
        print(f"[judge {index}/22] {question_id} {record['status']}", flush=True)


def usage_sum(records: list[dict], getter) -> dict:
    usages = [getter(record) or {} for record in records]
    return {key: sum(int(usage.get(key) or 0) for usage in usages) for key in ("input_tokens", "output_tokens", "total_tokens")}


def build_summary(revisions: list[dict], judges: list[dict]) -> dict:
    valid_judges = [record for record in judges if record.get("status") == "ok"]
    recovered = [record["id"] for record in valid_judges if record["judge"]["score"] == 1]
    still_wrong = [record["id"] for record in valid_judges if record["judge"]["score"] == 0]
    revision_usage = usage_sum(revisions, lambda record: record.get("usage"))
    judge_usage = usage_sum(valid_judges, lambda record: record.get("judge", {}).get("usage"))
    revision_latency = [record["latency_ms"] for record in revisions if record.get("status") == "ok"]
    judge_latency = [record["latency_ms"] for record in valid_judges]
    return {
        "cohort": 22,
        "original_incorrect": 22,
        "recovered": len(recovered),
        "still_incorrect": len(still_wrong),
        "newly_incorrect": 0,
        "newly_incorrect_note": "Not observable: the cohort contains no originally-correct controls.",
        "recovered_ids": recovered,
        "still_incorrect_ids": still_wrong,
        "failures": {
            "revision": sum(record.get("status") != "ok" for record in revisions),
            "judge": sum(record.get("status") != "ok" for record in judges),
        },
        "acceptance": {"threshold": 5, "passed": len(recovered) >= 5},
        "cost": {
            "revision": {**revision_usage, "latency_ms_per_question": sum(revision_latency) / len(revision_latency) if revision_latency else None},
            "judge": {**judge_usage, "latency_ms_per_question": sum(judge_latency) / len(judge_latency) if judge_latency else None},
        },
    }


def write_report(output: Path, rows: list[dict], baseline_judges: dict[str, dict]) -> dict:
    revisions = load_jsonl(output / "revised_answers.jsonl")
    judges = load_jsonl(output / "judge_results.jsonl")
    if len(revisions) != 22 or len(judges) != 22:
        raise ValueError("Report requires 22 revision and Judge records")
    summary = build_summary(revisions, judges)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    revisions_by_id = {record["id"]: record for record in revisions}
    judges_by_id = {record["id"]: record for record in judges}
    lines = ["# Answer Correction Shadow v1", "", "## Summary", "",
        f"- Original incorrect: **22/22**", f"- Recovered: **{summary['recovered']}/22**",
        f"- Still incorrect: **{summary['still_incorrect']}/22**",
        "- Newly incorrect: **0 (not observable because this cohort has no originally-correct controls)**",
        f"- Acceptance threshold >=5 recovered: **{'passed' if summary['acceptance']['passed'] else 'failed'}**", "",
        "## Cost", "", "| Stage | Input tokens | Output tokens | Total tokens | Latency/question |",
        "|---|---:|---:|---:|---:|",
        f"| Flash critic/revision | {summary['cost']['revision']['input_tokens']:,} | {summary['cost']['revision']['output_tokens']:,} | {summary['cost']['revision']['total_tokens']:,} | {summary['cost']['revision']['latency_ms_per_question'] or 0:.0f} ms |",
        f"| Pro Judge | {summary['cost']['judge']['input_tokens']:,} | {summary['cost']['judge']['output_tokens']:,} | {summary['cost']['judge']['total_tokens']:,} | {summary['cost']['judge']['latency_ms_per_question'] or 0:.0f} ms |",
        "", "## Per-case results", ""]
    for row in rows:
        question_id = row["financebench_id"]
        revision, judged = revisions_by_id[question_id], judges_by_id[question_id]
        verdict = judged.get("judge", {}).get("verdict", "error")
        lines += [f"### {question_id} — {verdict}", "", f"**Question:** {row['question']}", "",
            f"**Original Judge:** {baseline_judges[question_id]['judge']['reason']}", "",
            f"**Original answer:** {revision.get('original_answer', '')}", "",
            f"**Revised answer:** {revision.get('revised_answer', '')}", "",
            f"**Revision Judge:** {judged.get('judge', {}).get('reason', judged.get('error_type', ''))}", ""]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("revise", "judge", "report", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, baseline_answers, baseline_judges = load_inputs()
    if args.stage in {"revise", "all"}:
        run_revisions(args.output_dir, rows, baseline_answers)
    if args.stage in {"judge", "all"}:
        run_judges(args.output_dir, rows)
    if args.stage in {"report", "all"}:
        print(json.dumps(build_summary(
            load_jsonl(args.output_dir / "revised_answers.jsonl"),
            load_jsonl(args.output_dir / "judge_results.jsonl"),
        ), ensure_ascii=False, indent=2))
        write_report(args.output_dir, rows, baseline_judges)


if __name__ == "__main__":
    main()
