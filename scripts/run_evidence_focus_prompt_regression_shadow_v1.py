"""Answer-only regression for the finance evidence-focus prompt.

This runner deliberately reads the frozen final100 answer contexts.  It never
imports or invokes retrieval, query rewriting, reranking, or context builders.
Baseline answers/judgements are reused; only evidence-focus answers and their
judgements are newly requested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
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
ATTRIBUTION = ROOT / "reports/answer_failure_attribution_v2/failure_cases.json"
DEFAULT_OUTPUT = ROOT / "reports/evidence_focus_prompt_regression_shadow_v1"
SEED = 20260908
ANSWER_MODEL = "deepseek-v4-flash-ga-260731"
JUDGE_MODEL = "deepseek-v4-pro-ga-260813"
PROMPT_MODE = "finance_evidence_focus_v1"
REFUSAL_RE = re.compile(
    r"(?:无法(?:计算|确定|回答|判断)|证据不足|信息不足|不能得出|cannot\s+(?:determine|calculate|answer)|"
    r"insufficient\s+(?:evidence|information))",
    re.IGNORECASE,
)


def load_local_environment() -> None:
    """Load the same repository-local credentials used by existing eval runners."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    os.environ.update(
        LANGSMITH_TRACING="false",
        LANGSMITH_TRACING_V2="false",
        LANGCHAIN_TRACING_V2="false",
    )


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _judge_score(record: dict) -> int:
    return int(record.get("status") == "ok" and record.get("judge", {}).get("score") == 1)


def load_frozen_inputs() -> tuple[list[dict], dict[str, dict], dict[str, dict], dict[str, dict]]:
    required = [
        DATASET,
        FINAL100 / "retrieval_snapshot.json",
        FINAL100 / "jina_results.json",
        FINAL100 / "answers.jsonl",
        FINAL100 / "judge_results.jsonl",
        ATTRIBUTION,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    answers = {record["question_id"]: record for record in load_jsonl(FINAL100 / "answers.jsonl")}
    judges = {record["question_id"]: record for record in load_jsonl(FINAL100 / "judge_results.jsonl")}
    attribution_rows = json.loads(ATTRIBUTION.read_text(encoding="utf-8"))
    attribution = {record["id"]: record for record in attribution_rows}
    if len(rows) != 100 or len(answers) != 100 or len(judges) != 100:
        raise ValueError("Expected complete 100-row final100 dataset, answers, and judges")
    return rows, answers, judges, attribution


def select_experiment_rows(
    rows: list[dict], answers: dict[str, dict], judges: dict[str, dict], seed: int = SEED
) -> list[dict]:
    failures = []
    correct = []
    for row in rows:
        question_id = row["financebench_id"]
        context_hit = answers[question_id].get("context_metrics", {}).get("context_hit") is True
        score = _judge_score(judges[question_id])
        if not score and context_hit:
            failures.append(row)
        elif score:
            correct.append(row)
    if len(failures) != 22:
        raise ValueError(f"Expected 22 frozen context-hit answer failures, found {len(failures)}")
    regression = random.Random(seed).sample(correct, 10)
    selected = [dict(row, selection_group="answer_failure22") for row in failures]
    selected.extend(dict(row, selection_group="correct_regression10") for row in regression)
    return selected


def prepare(output: Path, seed: int = SEED) -> list[dict]:
    rows, answers, judges, attribution = load_frozen_inputs()
    selected = select_experiment_rows(rows, answers, judges, seed)
    manifest = {
        "experiment": "evidence_focus_prompt_regression_shadow_v1",
        "seed": seed,
        "questions": len(selected),
        "selection": {"answer_failure_context_hit": 22, "strict_correct_regression": 10},
        "frozen_inputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
            for name, path in {
                "retrieval_snapshot": FINAL100 / "retrieval_snapshot.json",
                "jina_results": FINAL100 / "jina_results.json",
                "answers": FINAL100 / "answers.jsonl",
                "judges": FINAL100 / "judge_results.jsonl",
            }.items()
        },
        "question_ids": [row["financebench_id"] for row in selected],
    }
    atomic_json(output / "manifest.json", manifest)
    records = []
    for row in selected:
        question_id = row["financebench_id"]
        frozen_answer = answers[question_id]
        frozen_judge = judges[question_id]
        failure = attribution.get(question_id, {})
        records.append(
            {
                "question_id": question_id,
                "selection_group": row["selection_group"],
                "question": row["question"],
                "reference_answer": row["answer"],
                "frozen_context_sha256": text_sha256(frozen_answer["evidence"]),
                "frozen_context_chars": len(frozen_answer["evidence"]),
                "context_hit": frozen_answer.get("context_metrics", {}).get("context_hit"),
                "baseline": {
                    "prompt_mode": "clean_baseline_v1",
                    "answer": frozen_answer["answer"],
                    "usage": frozen_answer.get("usage", {}),
                    "latency_ms": frozen_answer.get("latency_ms"),
                    "judge": frozen_judge.get("judge", {}),
                    "refusal_language": bool(REFUSAL_RE.search(frozen_answer["answer"])),
                },
                "baseline_failure_category": failure.get("category"),
                "baseline_failure_category_name": failure.get("category_name"),
            }
        )
    atomic_jsonl(output / "results.jsonl", records)
    return selected


def _ordered_records(output: Path) -> list[dict]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    records = {record["question_id"]: record for record in load_jsonl(output / "results.jsonl")}
    return [records[question_id] for question_id in manifest["question_ids"]]


def _assert_frozen_context(record: dict, answers: dict[str, dict]) -> str:
    evidence = answers[record["question_id"]]["evidence"]
    if text_sha256(evidence) != record["frozen_context_sha256"]:
        raise ValueError(f"Frozen context drift for {record['question_id']}")
    return evidence


def configure_answer() -> None:
    os.environ.update(
        ANSWER_PROMPT_MODE=PROMPT_MODE,
        ANSWER_TEMPERATURE="0.1",
        ANSWER_MAX_COMPLETION_TOKENS="1024",
        ANSWER_THINKING_MODE="disabled",
        ANSWER_TIMEOUT_SECONDS="90",
        ANSWER_MAX_RETRIES="1",
    )


def run_answers(output: Path) -> None:
    load_local_environment()
    if not os.getenv("ARK_API_KEY"):
        raise ValueError("ARK_API_KEY is required for the answer stage")
    _, frozen_answers, _, _ = load_frozen_inputs()
    configure_answer()
    from answer_generator import generate_answer

    records = _ordered_records(output)
    for index, record in enumerate(records, 1):
        focus = record.get("evidence_focus", {})
        if focus.get("answer_status") == "ok":
            _assert_frozen_context(record, frozen_answers)
            continue
        evidence = _assert_frozen_context(record, frozen_answers)
        started = time.perf_counter()
        try:
            answer, usage = generate_answer(
                record["question"], evidence, [], "", "finance", PROMPT_MODE
            )
            if not str(answer).strip():
                raise RuntimeError("Empty answer")
            record["evidence_focus"] = {
                "prompt_mode": PROMPT_MODE,
                "answer_model": ANSWER_MODEL,
                "answer_status": "ok",
                "answer": answer,
                "usage": usage,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "refusal_language": bool(REFUSAL_RE.search(answer)),
            }
        except Exception as exc:
            record["evidence_focus"] = {
                "prompt_mode": PROMPT_MODE,
                "answer_model": ANSWER_MODEL,
                "answer_status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        atomic_jsonl(output / "results.jsonl", records)
        print(f"[answer {index:02d}/32] {record['question_id']} {record['evidence_focus']['answer_status']}", flush=True)


def run_judges(output: Path) -> None:
    load_local_environment()
    if not (os.getenv("JUDGE_API_KEY") or os.getenv("ARK_API_KEY")):
        raise ValueError("JUDGE_API_KEY or ARK_API_KEY is required for the judge stage")
    records = _ordered_records(output)
    holder: list = []
    config = {"model": JUDGE_MODEL, "max_completion_tokens": 512, "timeout_seconds": 90}
    for index, record in enumerate(records, 1):
        focus = record.get("evidence_focus", {})
        if focus.get("answer_status") != "ok" or focus.get("judge_status") == "ok":
            continue
        started = time.perf_counter()
        try:
            result = judge_answer(
                record["question"], record["reference_answer"], focus["answer"], config, holder
            )
            focus.update(
                judge_status="ok",
                judge=result,
                judge_latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            focus.update(
                judge_status="error",
                judge_error=f"{type(exc).__name__}: {exc}",
                judge_latency_ms=(time.perf_counter() - started) * 1000,
            )
        atomic_jsonl(output / "results.jsonl", records)
        print(f"[judge {index:02d}/32] {record['question_id']} {focus.get('judge_status')}", flush=True)


def _usage_total(value: dict) -> int:
    if value.get("total_tokens") is not None:
        return int(value["total_tokens"])
    return int(value.get("input_tokens") or 0) + int(value.get("output_tokens") or 0)


def summarize(records: list[dict]) -> dict:
    transitions = {key: [] for key in ("newly_correct", "regressed", "still_incorrect", "still_correct")}
    for record in records:
        baseline_correct = int(record["baseline"].get("judge", {}).get("score") == 1)
        focus_correct = int(record.get("evidence_focus", {}).get("judge", {}).get("score") == 1)
        key = (
            "still_correct" if baseline_correct and focus_correct
            else "regressed" if baseline_correct
            else "newly_correct" if focus_correct
            else "still_incorrect"
        )
        transitions[key].append(record["question_id"])
        record["comparison"] = {
            "transition": key,
            "answer_changed": record["baseline"]["answer"].strip() != record.get("evidence_focus", {}).get("answer", "").strip(),
            "refusal_change": [
                record["baseline"].get("refusal_language", False),
                record.get("evidence_focus", {}).get("refusal_language", False),
            ],
        }

    judged = [r for r in records if r.get("evidence_focus", {}).get("judge_status") == "ok"]
    baseline_correct = sum(r["baseline"].get("judge", {}).get("score") == 1 for r in records)
    focus_correct = sum(r["evidence_focus"]["judge"].get("score") == 1 for r in judged)
    baseline_tokens = [_usage_total(r["baseline"].get("usage", {})) for r in records]
    focus_tokens = [_usage_total(r.get("evidence_focus", {}).get("usage", {})) for r in records if r.get("evidence_focus", {}).get("answer_status") == "ok"]
    baseline_refusals = sum(r["baseline"].get("refusal_language", False) and r.get("context_hit") is True for r in records)
    focus_refusals = sum(r.get("evidence_focus", {}).get("refusal_language", False) and r.get("context_hit") is True for r in records)
    category_stats = {}
    for code, label in (("E", "calculation_execution_error"), ("F", "reasoning_direction_error")):
        category_records = [r for r in records if r.get("baseline_failure_category") == code]
        recovered = sum(r["question_id"] in transitions["newly_correct"] for r in category_records)
        category_stats[label] = {
            "baseline_failures": len(category_records),
            "recovered": recovered,
            "remaining": len(category_records) - recovered,
        }
    newly_correct = len(transitions["newly_correct"])
    regressed = len(transitions["regressed"])
    average_delta = (
        sum(focus_tokens) / len(focus_tokens) - sum(baseline_tokens) / len(baseline_tokens)
        if focus_tokens and baseline_tokens else None
    )
    gates = {
        "net_improvement_at_least_5": newly_correct - regressed >= 5,
        "regressions_at_most_2": regressed <= 2,
        "average_token_increase_at_most_500": average_delta is not None and average_delta <= 500,
    }
    return {
        "experiment": "evidence_focus_prompt_regression_shadow_v1",
        "questions": len(records),
        "judged": len(judged),
        "groups": {group: sum(r["selection_group"] == group for r in records) for group in ("answer_failure22", "correct_regression10")},
        "strict_judge": {
            "baseline_correct": baseline_correct,
            "evidence_focus_correct": focus_correct,
            "delta_correct": focus_correct - baseline_correct,
            "baseline_accuracy": baseline_correct / len(records),
            "evidence_focus_accuracy": focus_correct / len(judged) if judged else None,
        },
        "transitions": {key: {"count": len(ids), "question_ids": ids} for key, ids in transitions.items()},
        "unnecessary_refusal": {"baseline": baseline_refusals, "evidence_focus": focus_refusals, "delta": focus_refusals - baseline_refusals},
        "failure_type_change": category_stats,
        "answer_tokens": {
            "baseline_total": sum(baseline_tokens),
            "evidence_focus_total": sum(focus_tokens),
            "baseline_average": sum(baseline_tokens) / len(baseline_tokens) if baseline_tokens else None,
            "evidence_focus_average": sum(focus_tokens) / len(focus_tokens) if focus_tokens else None,
            "average_delta": average_delta,
        },
        "acceptance": {
            **gates,
            "passed": all(gates.values()),
            "recommendation": "eligible_for_production_review" if all(gates.values()) else "keep_shadow_do_not_change_finance_online_v2",
        },
        "failures": {
            "answer": sum(r.get("evidence_focus", {}).get("answer_status") != "ok" for r in records),
            "judge": sum(r.get("evidence_focus", {}).get("answer_status") == "ok" and r.get("evidence_focus", {}).get("judge_status") != "ok" for r in records),
        },
    }


def write_report(output: Path) -> dict:
    records = _ordered_records(output)
    summary = summarize(records)
    atomic_jsonl(output / "results.jsonl", records)
    atomic_json(output / "summary.json", summary)
    strict = summary["strict_judge"]
    transitions = summary["transitions"]
    token = summary["answer_tokens"]
    lines = [
        "# Evidence Focus Prompt Regression Shadow v1", "",
        "这是严格的 Answer-only 实验。Retrieval snapshot、Jina 结果和最终 context 均来自 `reports/final100`，未重新检索、重排或构建上下文。", "",
        "## 汇总", "",
        f"- 样本：22 条 context-hit answer failure + 10 条固定随机正确回归样本，共 **{summary['questions']}** 题。",
        f"- Clean baseline：**{strict['baseline_correct']}/{summary['questions']}**。",
        f"- Evidence focus：**{strict['evidence_focus_correct']}/{summary['judged']}**。",
        f"- 新增正确 / 回退 / 保持错误 / 保持正确：**{transitions['newly_correct']['count']} / {transitions['regressed']['count']} / {transitions['still_incorrect']['count']} / {transitions['still_correct']['count']}**。",
        f"- 净变化：**{strict['delta_correct']:+d} 题**。",
        f"- 不必要拒答：**{summary['unnecessary_refusal']['baseline']} → {summary['unnecessary_refusal']['evidence_focus']}**。",
        f"- 平均回答 token：**{token['baseline_average']:.1f} → {token['evidence_focus_average']:.1f}**（{token['average_delta']:+.1f}）。" if token["average_delta"] is not None else "- Token 数据不完整。",
        "", "## 错误类型变化", "",
    ]
    for label, values in summary["failure_type_change"].items():
        lines.append(f"- `{label}`：原 {values['baseline_failures']}，恢复 {values['recovered']}，仍错误 {values['remaining']}。")
    acceptance = summary["acceptance"]
    lines.extend([
        "", "## 验收", "",
        f"- 净提升 ≥5：{'通过' if acceptance['net_improvement_at_least_5'] else '未通过'}",
        f"- 回退 ≤2：{'通过' if acceptance['regressions_at_most_2'] else '未通过'}",
        f"- 平均 token 增量 ≤500：{'通过' if acceptance['average_token_increase_at_most_500'] else '未通过'}",
        f"- 结论：**{'可进入生产评审' if acceptance['passed'] else '保持 shadow，不修改 finance_online_v2 默认模式'}**。",
        "", "## 逐题结果", "",
    ])
    for index, record in enumerate(records, 1):
        baseline = record["baseline"]
        focus = record.get("evidence_focus", {})
        comparison = record.get("comparison", {})
        lines.extend([
            f"### {index}. {record['question_id']}", "",
            f"- 分组：`{record['selection_group']}`",
            f"- 原错误类型：{record.get('baseline_failure_category_name') or '正确回归样本'}",
            f"- 变化：`{comparison.get('transition', 'not_judged')}`",
            f"- Baseline Judge：{baseline.get('judge', {}).get('verdict')} — {baseline.get('judge', {}).get('reason', '')}",
            f"- Evidence Focus Judge：{focus.get('judge', {}).get('verdict')} — {focus.get('judge', {}).get('reason', focus.get('judge_error', ''))}",
            "", "**问题**", "", record["question"], "",
            "**参考答案**", "", record["reference_answer"], "",
            "**Clean baseline answer**", "", baseline["answer"], "",
            "**Evidence focus answer**", "", focus.get("answer", focus.get("error", "未生成")), "",
        ])
    (output / "answers.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "answer", "judge", "report", "all"), default="prepare")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--allow-paid", action="store_true", help="Required for answer/judge model calls")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in {"prepare", "all"} or not (output / "manifest.json").exists():
        selected = prepare(output, args.seed)
        print(f"[prepare] frozen questions={len(selected)} failures=22 regression=10 seed={args.seed}", flush=True)
    if args.stage in {"answer", "judge", "all"} and not args.allow_paid:
        raise ValueError("Model calls require explicit --allow-paid")
    if args.stage in {"answer", "all"}:
        run_answers(output)
    if args.stage in {"judge", "all"}:
        run_judges(output)
    if args.stage in {"report", "all"}:
        summary = write_report(output)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
