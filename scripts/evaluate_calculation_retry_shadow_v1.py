"""Run one targeted calculation retry on the frozen 15 answer failures.

Retrieval, Jina, Judge, and LangSmith are not called.  The existing evidence
context is reused byte-for-byte.  Reference answers are used only after
generation for deterministic offline diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from calculation_retry_shadow_v1 import (  # noqa: E402
    RETRY_SYSTEM_INSTRUCTION,
    create_shadow_model,
    invoke_retry,
    refusal_detected,
    retry_eligibility,
)


DEFAULT_ANSWERS = ROOT / "reports/jina_full_baseline_input120_all100/answers.jsonl"
DEFAULT_AUDIT = ROOT / "reports/answer_failure_audit_v1.json"
DEFAULT_VERIFIER = ROOT / "reports/answer_verifier_shadow_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/calculation_retry_shadow_v1"
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\(?-?\$?\d[\d,]*(?:\.\d+)?%?")


def normalized_numbers(text: str) -> set[str]:
    values: set[str] = set()
    for raw in _NUMBER_RE.findall(str(text or "")):
        cleaned = raw.replace("$", "").replace(",", "").replace("(", "-").rstrip("%")
        try:
            number = float(cleaned)
        except ValueError:
            continue
        if number.is_integer() and 1900 <= abs(number) <= 2100:
            continue
        values.add(f"{number:.8f}".rstrip("0").rstrip("."))
    return values


def answer_diagnostics(reference: str, answer: str) -> dict:
    reference_numbers = normalized_numbers(reference)
    answer_numbers = normalized_numbers(answer)
    matched = reference_numbers & answer_numbers
    return {
        "refusal_detected": refusal_detected(answer),
        "reference_numbers": sorted(reference_numbers),
        "matched_reference_numbers": sorted(matched),
        "reference_number_recall": round(len(matched) / len(reference_numbers), 4) if reference_numbers else None,
        "any_reference_number_hit": bool(matched),
    }


def retry_comparison(reference: str, baseline: str, retry: str) -> dict:
    before = answer_diagnostics(reference, baseline)
    after = answer_diagnostics(reference, retry)
    before_recall = before["reference_number_recall"]
    after_recall = after["reference_number_recall"]
    numeric_gain = (
        before_recall is not None and after_recall is not None and after_recall > before_recall
    )
    numeric_regression = (
        before_recall is not None and after_recall is not None and after_recall < before_recall
    )
    refusal_reduced = before["refusal_detected"] and not after["refusal_detected"]
    refusal_regression = not before["refusal_detected"] and after["refusal_detected"]
    return {
        "baseline": before,
        "retry": after,
        "reference_number_recovered": numeric_gain,
        "refusal_reduced": refusal_reduced,
        "regression": bool(numeric_regression or refusal_regression),
        "regression_reasons": [
            reason for condition, reason in (
                (numeric_regression, "reference_number_recall_decreased"),
                (refusal_regression, "new_refusal"),
            ) if condition
        ],
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def fingerprint(record: dict) -> str:
    content = "\n".join((record["question"], record["evidence"], record["answer"], RETRY_SYSTEM_INSTRUCTION))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def summarize(records: list[dict]) -> dict:
    selected = [record for record in records if record["eligibility"]["eligible"]]
    completed = [record for record in selected if record.get("status") == "ok"]
    comparisons = [record["comparison"] for record in completed]
    gains = sum(item["reference_number_recovered"] for item in comparisons)
    refusals_reduced = sum(item["refusal_reduced"] for item in comparisons)
    regressions = sum(item["regression"] for item in comparisons)
    clearly_effective = (gains + refusals_reduced) >= 2 and regressions == 0
    return {
        "audit_questions": len(records),
        "retry_candidates": len(selected),
        "retry_completed": len(completed),
        "reference_number_recovered": gains,
        "refusal_reduced": refusals_reduced,
        "regressions": regressions,
        "total_retry_input_tokens": sum(int(record.get("usage", {}).get("input_tokens") or 0) for record in completed),
        "total_retry_output_tokens": sum(int(record.get("usage", {}).get("output_tokens") or 0) for record in completed),
        "average_retry_latency_ms": round(
            sum(float(record.get("latency_ms") or 0) for record in completed) / len(completed), 2
        ) if completed else None,
        "clearly_effective": clearly_effective,
        "recommend_full100": clearly_effective,
    }


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Calculation Retry Shadow v1",
        "",
        "固定复用 Jina Full Baseline 的15题及其原始 Evidence；只对满足通用计算词与 verifier warning 双重条件的题调用一次 DeepSeek-V4-Flash。未调用 Retrieval、Jina、Judge 或 LangSmith。参考答案只用于生成后的离线数字对齐。",
        "",
        "## 汇总",
        "",
        f"- 审计题数：{summary['audit_questions']}",
        f"- Retry 候选/完成：{summary['retry_candidates']}/{summary['retry_completed']}",
        f"- 恢复参考关键数字：{summary['reference_number_recovered']}",
        f"- 减少拒答：{summary['refusal_reduced']}",
        f"- 确定性回退：{summary['regressions']}",
        f"- Retry 输入/输出 token：{summary['total_retry_input_tokens']}/{summary['total_retry_output_tokens']}",
        f"- 平均 Retry 延迟：{summary['average_retry_latency_ms']} ms",
        f"- 是否达到明显有效标准：`{summary['clearly_effective']}`",
        f"- 是否建议进入100题：`{summary['recommend_full100']}`",
        "",
        "这些指标不是官方 Judge 分数；文本语义正确性仍需人工复核。",
        "",
        "## 逐题结果",
        "",
    ]
    for index, record in enumerate(payload["records"], 1):
        lines.extend([
            f"### {index}. {record['question_id']} — `{record['failure_type']}`",
            "",
            f"**问题：** {record['question']}",
            "",
            f"**筛选：** `{record['status']}`；term=`{record['eligibility']['question_type_term']}`；warnings=`{record['eligibility']['eligible_warnings']}`",
            "",
            f"**参考答案（仅离线评估）：** {record['reference_answer']}",
            "",
            f"**Baseline：** {record['baseline_answer']}",
            "",
        ])
        if record.get("status") == "ok":
            comparison = record["comparison"]
            lines.extend([
                f"**Retry：** {record['retry_answer']}",
                "",
                f"- 数字恢复：`{comparison['reference_number_recovered']}`",
                f"- 拒答减少：`{comparison['refusal_reduced']}`",
                f"- 回退：`{comparison['regression']}` {comparison['regression_reasons']}",
                f"- Token：`{json.dumps(record.get('usage', {}), ensure_ascii=False)}`",
                f"- 延迟：`{record.get('latency_ms')} ms`",
                "",
            ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--verifier", type=Path, default=DEFAULT_VERIFIER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Select candidates without calling the answer model")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)

    answers = {row["financebench_id"]: row for row in load_jsonl(args.answers)}
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    verifier = json.loads(args.verifier.read_text(encoding="utf-8"))
    verification_by_id = {row["question_id"]: row["verification"] for row in verifier["failure_records"]}
    if len(audit.get("items", [])) != 15:
        raise ValueError("Expected the exact 15-item answer_failure_audit_v1 set")

    state_path = args.output_dir / "state.json"
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"records": []}
    cached = {row["question_id"]: row for row in previous.get("records", [])}
    records: list[dict] = []
    model = None
    for index, item in enumerate(audit["items"], 1):
        question_id = item["question_id"]
        source = answers[question_id]
        verification = verification_by_id[question_id]
        eligibility = retry_eligibility(source["question"], verification.get("warnings", []))
        record = {
            "question_id": question_id,
            "failure_type": item["category"],
            "question": source["question"],
            "reference_answer": item["reference_answer"],
            "baseline_answer": source["answer"],
            "evidence_sha256": hashlib.sha256(source["evidence"].encode("utf-8")).hexdigest(),
            "evidence_chars": len(source["evidence"]),
            "eligibility": eligibility,
            "verifier_warnings": verification.get("warnings", []),
            "fingerprint": fingerprint(source),
            "status": "not_selected" if not eligibility["eligible"] else "pending",
        }
        old = cached.get(question_id, {})
        if eligibility["eligible"] and old.get("status") == "ok" and old.get("fingerprint") == record["fingerprint"]:
            record.update({key: old[key] for key in ("status", "retry_answer", "usage", "latency_ms", "comparison")})
        elif eligibility["eligible"] and not args.dry_run:
            model = model or create_shadow_model()
            started = time.perf_counter()
            retry_answer, usage = invoke_retry(model, source["question"], source["evidence"])
            if not retry_answer.strip():
                raise RuntimeError(f"Empty retry answer for {question_id}")
            record.update(
                status="ok",
                retry_answer=retry_answer,
                usage=usage,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                comparison=retry_comparison(item["reference_answer"], source["answer"], retry_answer),
            )
        records.append(record)
        payload = {
            "schema": "calculation_retry_shadow_v1",
            "external_calls": {"retrieval": 0, "jina": 0, "judge": 0, "langsmith": 0, "answer_llm": sum(r.get("status") == "ok" for r in records)},
            "retry_instruction": RETRY_SYSTEM_INSTRUCTION,
            "summary": summarize(records),
            "records": records,
        }
        atomic_write(state_path, payload)
        print(f"[{index:02d}/15] {question_id}: {record['status']}", flush=True)

    payload["summary"] = summarize(records)
    atomic_write(state_path, payload)
    (args.output_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    (args.output_dir / "results.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
