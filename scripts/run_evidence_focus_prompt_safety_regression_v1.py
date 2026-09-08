"""Answer-only safety regression for ``finance_evidence_focus_v1``.

The experiment uses the 58 frozen final100 baseline-correct cases not sampled
by the preceding 10-case regression guard. It never imports retrieval, Jina,
query-rewrite, or context-construction code.
"""

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
sys.path.insert(0, str(ROOT / "scripts"))

from financebench_judge_common import JUDGE_PROMPT, _content_text, _parse_verdict  # noqa: E402


DATASET = ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"
FINAL100 = ROOT / "reports/final100"
PRIOR_MANIFEST = ROOT / "reports/evidence_focus_prompt_regression_shadow_v1/manifest.json"
DEFAULT_OUTPUT = ROOT / "reports/evidence_focus_prompt_safety_regression_v1"
ANSWER_MODEL = "deepseek-v4-flash-ga-260731"
JUDGE_MODEL = "deepseek-v4-pro-ga-260813"
PROMPT_MODE = "finance_evidence_focus_v1"
REFUSAL_RE = re.compile(
    r"(?:无法(?:计算|确定|回答|判断)|证据不足|信息不足|不能得出|cannot\s+(?:determine|calculate|answer)|"
    r"insufficient\s+(?:evidence|information))",
    re.IGNORECASE,
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_local_environment() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    os.environ.update(
        LANGSMITH_TRACING="false",
        LANGSMITH_TRACING_V2="false",
        LANGCHAIN_TRACING_V2="false",
    )


def load_inputs() -> tuple[list[dict], dict[str, dict], dict[str, dict], set[str]]:
    required = [
        DATASET,
        FINAL100 / "retrieval_snapshot.json",
        FINAL100 / "jina_results.json",
        FINAL100 / "answers.jsonl",
        FINAL100 / "judge_results.jsonl",
        PRIOR_MANIFEST,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    answers = {record["question_id"]: record for record in load_jsonl(FINAL100 / "answers.jsonl")}
    judges = {record["question_id"]: record for record in load_jsonl(FINAL100 / "judge_results.jsonl")}
    prior = json.loads(PRIOR_MANIFEST.read_text(encoding="utf-8"))
    prior_regression = set(prior["question_ids"][-10:])
    if len(rows) != 100 or len(answers) != 100 or len(judges) != 100 or len(prior_regression) != 10:
        raise ValueError("Frozen final100 or prior 10-case regression manifest is incomplete")
    return rows, answers, judges, prior_regression


def select_safety_rows(rows: list[dict], judges: dict[str, dict], excluded: set[str]) -> list[dict]:
    selected = [
        row
        for row in rows
        if judges[row["financebench_id"]].get("status") == "ok"
        and judges[row["financebench_id"]].get("judge", {}).get("score") == 1
        and row["financebench_id"] not in excluded
    ]
    if len(selected) != 58:
        raise ValueError(f"Expected 58 remaining baseline-correct cases, found {len(selected)}")
    return selected


def prepare(output: Path) -> None:
    rows, answers, judges, excluded = load_inputs()
    selected = select_safety_rows(rows, judges, excluded)
    frozen_paths = {
        "retrieval_snapshot": FINAL100 / "retrieval_snapshot.json",
        "jina_results": FINAL100 / "jina_results.json",
        "answers": FINAL100 / "answers.jsonl",
        "judges": FINAL100 / "judge_results.jsonl",
    }
    atomic_json(
        output / "manifest.json",
        {
            "experiment": "evidence_focus_prompt_safety_regression_v1",
            "questions": 58,
            "selection": "all final100 strict-correct cases excluding the prior seeded regression10",
            "excluded_prior_regression_ids": sorted(excluded),
            "frozen_inputs": {
                name: {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
                for name, path in frozen_paths.items()
            },
            "question_ids": [row["financebench_id"] for row in selected],
        },
    )
    records = []
    for row in selected:
        question_id = row["financebench_id"]
        frozen = answers[question_id]
        records.append(
            {
                "question_id": question_id,
                "question": row["question"],
                "reference_answer": row["answer"],
                "frozen_context_sha256": text_sha256(frozen["evidence"]),
                "frozen_context_chars": len(frozen["evidence"]),
                "baseline": {
                    "prompt_mode": "clean_baseline_v1",
                    "answer": frozen["answer"],
                    "usage": frozen.get("usage", {}),
                    "latency_ms": frozen.get("latency_ms"),
                    "judge": judges[question_id]["judge"],
                },
            }
        )
    atomic_jsonl(output / "results.jsonl", records)
    print("[prepare] frozen safety cases=58; prior regression10 excluded", flush=True)


def ordered_records(output: Path) -> list[dict]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    records = {record["question_id"]: record for record in load_jsonl(output / "results.jsonl")}
    return [records[question_id] for question_id in manifest["question_ids"]]


def frozen_context(record: dict, answers: dict[str, dict]) -> str:
    evidence = answers[record["question_id"]]["evidence"]
    if text_sha256(evidence) != record["frozen_context_sha256"]:
        raise ValueError(f"Frozen context drift for {record['question_id']}")
    return evidence


def run_answers(output: Path) -> None:
    load_local_environment()
    if not os.getenv("ARK_API_KEY"):
        raise ValueError("ARK_API_KEY is required for answer generation")
    _, answers, _, _ = load_inputs()
    os.environ.update(
        ANSWER_PROMPT_MODE=PROMPT_MODE,
        ANSWER_TEMPERATURE="0.1",
        ANSWER_MAX_COMPLETION_TOKENS="1024",
        ANSWER_THINKING_MODE="disabled",
        ANSWER_TIMEOUT_SECONDS="90",
        ANSWER_MAX_RETRIES="1",
    )
    from answer_generator import generate_answer

    records = ordered_records(output)
    for index, record in enumerate(records, 1):
        focus = record.get("evidence_focus", {})
        evidence = frozen_context(record, answers)
        if focus.get("answer_status") == "ok":
            continue
        started = time.perf_counter()
        try:
            answer, usage = generate_answer(record["question"], evidence, [], "", "finance", PROMPT_MODE)
            if not str(answer).strip():
                raise RuntimeError("Empty answer")
            record["evidence_focus"] = {
                "prompt_mode": PROMPT_MODE,
                "answer_model": ANSWER_MODEL,
                "answer_status": "ok",
                "answer": answer,
                "usage": usage,
                "latency_ms": (time.perf_counter() - started) * 1000,
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
        print(f"[answer {index:02d}/58] {record['question_id']} {record['evidence_focus']['answer_status']}", flush=True)


def create_judge_model():
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=JUDGE_MODEL,
        model_provider="openai",
        api_key=os.getenv("JUDGE_API_KEY") or os.getenv("ARK_API_KEY"),
        base_url=os.getenv("JUDGE_BASE_URL") or os.getenv("BASE_URL"),
        temperature=0,
        max_completion_tokens=512,
        timeout=90,
        max_retries=1,
        extra_body={"thinking": {"type": "disabled"}},
    )


def run_judges(output: Path) -> None:
    load_local_environment()
    if not (os.getenv("JUDGE_API_KEY") or os.getenv("ARK_API_KEY")):
        raise ValueError("JUDGE_API_KEY or ARK_API_KEY is required for judging")
    model = create_judge_model()
    records = ordered_records(output)
    for index, record in enumerate(records, 1):
        focus = record.get("evidence_focus", {})
        if focus.get("answer_status") != "ok" or focus.get("judge_status") == "ok":
            continue
        started = time.perf_counter()
        try:
            response = model.invoke(
                JUDGE_PROMPT.format(
                    question=record["question"],
                    reference=record["reference_answer"],
                    answer=focus["answer"],
                )
            )
            verdict = _parse_verdict(_content_text(getattr(response, "content", response)))
            if verdict["verdict"] not in {"correct", "incorrect"}:
                raise RuntimeError("Invalid Judge output")
            focus.update(
                judge_status="ok",
                judge={
                    **verdict,
                    "judge_model": JUDGE_MODEL,
                    "usage": dict(getattr(response, "usage_metadata", {}) or {}),
                },
                judge_latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            focus.update(
                judge_status="error",
                judge_error=f"{type(exc).__name__}: {exc}",
                judge_latency_ms=(time.perf_counter() - started) * 1000,
            )
        atomic_jsonl(output / "results.jsonl", records)
        print(f"[judge {index:02d}/58] {record['question_id']} {focus.get('judge_status')}", flush=True)


def classify_regression(answer: str, judge_reason: str) -> str:
    """Deterministic diagnosis based only on the new answer and Judge reason."""
    combined = f"{answer}\n{judge_reason}".lower()
    reason = judge_reason.lower()
    if REFUSAL_RE.search(combined):
        return "unnecessary_refusal"
    if re.search(r"direction|trend|increase|decrease|higher|lower|improv|declin|方向|趋势|增加|下降", reason):
        return "wrong_direction"
    if re.search(r"wrong metric|different metric|metric mismatch|指标|口径", reason):
        return "wrong_metric"
    if re.search(
        r"calculat|arithmetic|formula|ratio|percentage|computed|rounded|rounding|combined total|"
        r"does not match the reference answer|numeric|算术|公式|计算|合计|四舍五入",
        reason,
    ):
        return "wrong_calculation"
    return "hallucination"


def token_total(usage: dict) -> int:
    return int(usage.get("total_tokens") or (int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)))


def write_report(output: Path) -> dict:
    records = ordered_records(output)
    regressed = []
    kept = []
    error_types = {name: [] for name in ("hallucination", "wrong_calculation", "wrong_direction", "wrong_metric", "unnecessary_refusal")}
    for record in records:
        focus = record.get("evidence_focus", {})
        focus_correct = focus.get("judge_status") == "ok" and focus.get("judge", {}).get("score") == 1
        if focus_correct:
            kept.append(record["question_id"])
            transition = "still_correct"
            diagnosis = None
        else:
            regressed.append(record["question_id"])
            transition = "regressed"
            diagnosis = classify_regression(focus.get("answer", ""), focus.get("judge", {}).get("reason", ""))
            error_types[diagnosis].append(record["question_id"])
        record["comparison"] = {
            "transition": transition,
            "answer_changed": record["baseline"]["answer"].strip() != focus.get("answer", "").strip(),
            "regression_type": diagnosis,
        }
    baseline_tokens = [token_total(record["baseline"].get("usage", {})) for record in records]
    focus_tokens = [token_total(record.get("evidence_focus", {}).get("usage", {})) for record in records if record.get("evidence_focus", {}).get("answer_status") == "ok"]
    average_delta = sum(focus_tokens) / len(focus_tokens) - sum(baseline_tokens) / len(baseline_tokens) if focus_tokens else None
    summary = {
        "experiment": "evidence_focus_prompt_safety_regression_v1",
        "questions": len(records),
        "judged": sum(record.get("evidence_focus", {}).get("judge_status") == "ok" for record in records),
        "baseline_correct_to_focus_wrong": {"count": len(regressed), "question_ids": regressed},
        "baseline_correct_to_focus_correct": {"count": len(kept), "question_ids": kept},
        "regression_types": {name: {"count": len(ids), "question_ids": ids} for name, ids in error_types.items()},
        "answer_tokens": {
            "baseline_total": sum(baseline_tokens),
            "focus_total": sum(focus_tokens),
            "baseline_average": sum(baseline_tokens) / len(baseline_tokens),
            "focus_average": sum(focus_tokens) / len(focus_tokens) if focus_tokens else None,
            "average_delta": average_delta,
        },
        "failures": {
            "answer": sum(record.get("evidence_focus", {}).get("answer_status") != "ok" for record in records),
            "judge": sum(record.get("evidence_focus", {}).get("answer_status") == "ok" and record.get("evidence_focus", {}).get("judge_status") != "ok" for record in records),
        },
        "acceptance": {
            "regressions_at_most_2": len(regressed) <= 2,
            "passed": len(regressed) <= 2,
            "recommendation": "generate_integration_recommendation_keep_default_unchanged" if len(regressed) <= 2 else "keep_shadow",
        },
    }
    atomic_jsonl(output / "results.jsonl", records)
    atomic_json(output / "summary.json", summary)
    lines = [
        "# Evidence Focus Prompt Safety Regression v1", "",
        "本实验只复用 `reports/final100/answers.jsonl` 中冻结的 context；未重新执行 Retrieval、Query Rewrite、Jina 或 context build。", "",
        "## 汇总", "",
        f"- 测试：剩余全部 **{len(records)}** 条 baseline Strict Judge 正确样本。",
        f"- 保持正确：**{len(kept)}/{len(records)}**。",
        f"- 新增回退：**{len(regressed)}**。",
        f"- 平均回答 token：**{summary['answer_tokens']['baseline_average']:.1f} → {summary['answer_tokens']['focus_average']:.1f}**（{average_delta:+.1f}）。",
        f"- 验收：**{'通过' if summary['acceptance']['passed'] else '未通过'}**（要求回退 ≤2）。", "",
        "## 回退类型", "",
    ]
    for name, values in summary["regression_types"].items():
        lines.append(f"- `{name}`：{values['count']} — {', '.join(values['question_ids']) or '无'}")
    lines.extend(["", "## 逐题结果", ""])
    for index, record in enumerate(records, 1):
        baseline = record["baseline"]
        focus = record.get("evidence_focus", {})
        comparison = record["comparison"]
        lines.extend([
            f"### {index}. {record['question_id']}", "",
            f"- 变化：`{comparison['transition']}`",
            f"- 回退类型：{comparison.get('regression_type') or '无'}",
            f"- Baseline Judge：{baseline['judge'].get('verdict')} — {baseline['judge'].get('reason', '')}",
            f"- Focus Judge：{focus.get('judge', {}).get('verdict')} — {focus.get('judge', {}).get('reason', focus.get('judge_error', ''))}",
            f"- Answer changed：{comparison['answer_changed']}", "",
            "**问题**", "", record["question"], "",
            "**参考答案**", "", record["reference_answer"], "",
            "**Baseline answer**", "", baseline["answer"], "",
            "**Evidence Focus answer**", "", focus.get("answer", focus.get("error", "未生成")), "",
        ])
    (output / "answers.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "answer", "judge", "report", "all"), default="prepare")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.stage in {"prepare", "all"} or not (output / "manifest.json").exists():
        prepare(output)
    if args.stage in {"answer", "judge", "all"} and not args.allow_paid:
        raise ValueError("Model calls require explicit --allow-paid")
    if args.stage in {"answer", "all"}:
        run_answers(output)
    if args.stage in {"judge", "all"}:
        run_judges(output)
    if args.stage in {"report", "all"}:
        print(json.dumps(write_report(output), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
