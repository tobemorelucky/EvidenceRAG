"""Run a frozen-context answer A/B with Financial Evidence Summary v1.

Route A reuses the existing DeepSeek-V4-Flash baseline answer because its
question, context, profile, and default prompt are already frozen.  Route B
performs one new Flash call with the same original context plus a deterministic
summary.  Retrieval, Jina, Judge, and LangSmith are never called.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_finance_reasoning_prompt_v1_1_shadow import diagnose_resolution  # noqa: E402


DEFAULT_ANSWERS = ROOT / "reports/jina_full_baseline_input120_all100/answers.jsonl"
DEFAULT_AUDIT = ROOT / "reports/answer_failure_audit_v1.json"
DEFAULT_SUMMARIES = ROOT / "reports/financial_evidence_summary_shadow_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/financial_evidence_summary_answer_shadow_v1"
PROFILE = ROOT / "configs/experiments/jina_full_baseline_input120_v1.json"
_REFUSAL_RE = re.compile(
    r"\b(?:cannot|can't|unable to|insufficient|not enough|does not provide|"
    r"do not provide|no information|cannot be determined|cannot determine|"
    r"cannot calculate|not possible to calculate)\b",
    re.I,
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def format_summary_context(facts: list[dict[str, Any]]) -> str:
    """Serialize extracted facts while retaining every exact source span."""
    lines = [
        "Financial Evidence Summary (deterministically derived from the evidence above):",
        "This summary is not an independent source. Use its cited source spans to interpret the original evidence.",
    ]
    for index, fact in enumerate(facts, 1):
        source = fact["source_span"]
        lines.extend([
            f"Fact {index}:",
            f"  Entity: {fact.get('entity') or 'unknown'}",
            f"  Period: {fact.get('period') or 'unknown'}",
            f"  Metric: {fact.get('metric') or 'unknown'}",
            f"  Value: {fact.get('value') if fact.get('value') is not None else 'not explicitly numeric'}",
            f"  Unit: {fact.get('unit') or 'unknown'}",
            f"  Source: {source['document']} | Page: {source['page']} | Chunk: {source['chunk_id']} | Line: {source['line_number']}",
            f"  Source span: {source['text']}",
            f"  Ambiguity flags: {', '.join(fact.get('ambiguity_flags') or []) or 'none'}",
        ])
    return "\n".join(lines)


def combine_context(original_context: str, facts: list[dict[str, Any]]) -> str:
    """Append, never replace or truncate, the frozen context."""
    if not facts:
        return original_context
    return f"{original_context}\n\n{format_summary_context(facts)}"


def _numeric_recall(diagnostic: dict[str, Any]) -> float | None:
    value = diagnostic.get("numeric_coverage")
    return float(value) if value is not None else None


def compare_answers(failure_type: str, reference: str, baseline: str, summary_answer: str) -> dict[str, Any]:
    """Conservative post-generation diagnostic; it is not an LLM Judge."""
    baseline_diag = diagnose_resolution(failure_type, reference, baseline)
    summary_diag = diagnose_resolution(failure_type, reference, summary_answer)
    baseline_numeric = _numeric_recall(baseline_diag)
    summary_numeric = _numeric_recall(summary_diag)
    regression_reasons = []
    if not _REFUSAL_RE.search(baseline or "") and _REFUSAL_RE.search(summary_answer or ""):
        regression_reasons.append("new_refusal")
    if baseline_numeric is not None and summary_numeric is not None and summary_numeric < baseline_numeric:
        regression_reasons.append("reference_number_recall_decreased")
    for field in ("polarity_match", "direction_match", "selection_match", "critical_fact_match", "proper_name_match"):
        if baseline_diag.get(field) is True and summary_diag.get(field) is False:
            regression_reasons.append(f"{field}_regressed")
    recovered = summary_diag.get("status") == "likely_resolved"
    return {
        "status": "likely_recovered" if recovered else "not_recovered",
        "diagnostic": summary_diag,
        "baseline_diagnostic": baseline_diag,
        "recovered": recovered,
        "regression": bool(regression_reasons),
        "regression_reasons": regression_reasons,
        "note": "Deterministic reference-alignment diagnostic; not an official Judge result.",
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "state.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "state.json")
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in payload["records"]),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "ok"]
    return {
        "questions": len(records),
        "summary_calls_completed": len(completed),
        "likely_recovered": sum(record.get("recovered") is True for record in completed),
        "deterministic_regressions": sum(record.get("regression") is True for record in completed),
        "baseline_reused": len(records),
        "deepseek_flash_calls": len(completed),
        "retrieval_calls": 0,
        "jina_calls": 0,
        "judge_calls": 0,
        "langsmith_calls": 0,
        "total_input_tokens": sum(int(record.get("usage", {}).get("input_tokens") or 0) for record in completed),
        "total_output_tokens": sum(int(record.get("usage", {}).get("output_tokens") or 0) for record in completed),
        "average_latency_ms": round(
            sum(float(record.get("latency_ms") or 0) for record in completed) / len(completed), 2
        ) if completed else None,
        "result_contract": "deterministic diagnostic only; no Judge",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = summarize(payload["records"])
    lines = [
        "# Financial Evidence Summary Answer Shadow v1", "",
        "A 组复用已冻结 Jina Full Baseline 的 DeepSeek-V4-Flash 答案；B 组保留原 context，并追加纯规则 Financial Evidence Summary 后调用同一 Flash 模型和默认 prompt。参考答案仅用于生成后的确定性诊断，不进入模型上下文。未调用 Retrieval、Jina、Judge 或 LangSmith。", "",
        "## 汇总", "",
        f"- 题目：{summary['questions']}",
        f"- B 组完成：{summary['summary_calls_completed']}/{summary['questions']}",
        f"- 可能恢复：{summary['likely_recovered']}",
        f"- 确定性回退：{summary['deterministic_regressions']}",
        f"- DeepSeek Flash 新调用：{summary['deepseek_flash_calls']}",
        f"- 输入/输出 token：{summary['total_input_tokens']}/{summary['total_output_tokens']}",
        f"- 平均延迟：{summary['average_latency_ms']} ms", "",
        "> `likely_recovered` 和 `regression` 是参考答案对齐的确定性诊断，不是 Strict Judge 结果。", "",
        "## 逐题", "",
    ]
    for index, record in enumerate(payload["records"], 1):
        result = record.get("summary_result") or {}
        lines.extend([
            f"### {index}. {record['id']} — `{record['baseline_failure_type']}`", "",
            f"**问题：** {record['question']}", "",
            f"**Baseline answer：** {record['baseline_answer']}", "",
            f"**Summary answer：** {record.get('summary_answer') or '尚未完成'}", "",
            f"- Summary result：`{result.get('status', record.get('status'))}`",
            f"- Recovered：`{record.get('recovered')}`",
            f"- Regression：`{record.get('regression')}` {result.get('regression_reasons', [])}",
            f"- Summary facts：`{record['summary_fact_count']}`；新增字符：`{record['summary_chars']}`",
            f"- Token：`{json.dumps(record.get('usage', {}), ensure_ascii=False)}`；延迟：`{record.get('latency_ms')} ms`", "",
        ])
    return "\n".join(lines)


def _prepare_records(answers: dict[str, dict], audit: dict, summaries: dict) -> list[dict[str, Any]]:
    summary_by_id = {record["question_id"]: record for record in summaries["records"]}
    records = []
    for item in audit["items"]:
        source = answers[item["question_id"]]
        facts = summary_by_id[item["question_id"]]["financial_evidence_summary"]
        summary_text = format_summary_context(facts)
        records.append({
            "id": item["question_id"], "question": source["question"],
            "baseline_answer": source["answer"], "summary_answer": None,
            "baseline_failure_type": item["category"], "summary_result": None,
            "recovered": None, "regression": None, "status": "pending",
            "summary_fact_count": len(facts), "summary_chars": len(summary_text),
            "evidence_sha256": digest(source["evidence"]), "summary_sha256": digest(summary_text),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--summaries", type=Path, default=DEFAULT_SUMMARIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true", help="Validate and write the experiment without model calls")
    args = parser.parse_args()

    answers = {row["financebench_id"]: row for row in load_jsonl(args.answers)}
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    summaries = json.loads(args.summaries.read_text(encoding="utf-8"))
    if len(audit.get("items") or []) != 15 or len(summaries.get("records") or []) != 15:
        raise ValueError("Expected the exact 15-item answer failure and summary artifacts")
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    if profile["answer"]["model"] != "deepseek-v4-flash-ga-260731":
        raise ValueError("Shadow experiment is restricted to DeepSeek-V4-Flash")

    manifest = {
        "experiment": "financial_evidence_summary_answer_shadow_v1",
        "model": profile["answer"]["model"], "prompt_mode": "baseline",
        "profile": profile["answer"]["profile"], "baseline_route": "reused_frozen_answer",
        "summary_route": "frozen_context_plus_financial_evidence_summary_v1",
        "reference_in_model_context": False, "original_context_removed": False,
        "answers_sha256": hashlib.sha256(args.answers.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        "summaries_sha256": hashlib.sha256(args.summaries.read_bytes()).hexdigest(),
    }
    payload = {"manifest": manifest, "records": _prepare_records(answers, audit, summaries), "complete": False}
    state_path = args.output_dir / "state.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("manifest") != manifest:
            raise ValueError("Checkpoint inputs drifted; use a new output directory")
        payload = previous
    write_outputs(args.output_dir, payload)
    if args.dry_run:
        print(json.dumps(summarize(payload["records"]), ensure_ascii=False, indent=2))
        print(f"Dry run: {args.output_dir / 'report.md'}")
        return

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    from runtime_profile import apply_runtime_profile
    apply_runtime_profile(profile["answer"]["profile"])
    os.environ.update({
        "MODEL": profile["answer"]["model"],
        "ANSWER_TEMPERATURE": str(profile["answer"]["temperature"]),
        "ANSWER_MAX_COMPLETION_TOKENS": str(profile["answer"]["max_completion_tokens"]),
        "ANSWER_THINKING_MODE": profile["answer"]["thinking"],
        "ANSWER_TIMEOUT_SECONDS": str(profile["answer"]["timeout_seconds"]),
        "ANSWER_MAX_RETRIES": "0",
        "ANSWER_PROMPT_MODE": "baseline",
        "LANGSMITH_TRACING": "false", "LANGSMITH_TRACING_V2": "false", "LANGCHAIN_TRACING_V2": "false",
    })
    from answer_generator import generate_answer

    audit_by_id = {item["question_id"]: item for item in audit["items"]}
    summary_by_id = {record["question_id"]: record for record in summaries["records"]}
    for index, record in enumerate(payload["records"], 1):
        if record.get("status") == "ok":
            continue
        source = answers[record["id"]]
        facts = summary_by_id[record["id"]]["financial_evidence_summary"]
        summary_text = format_summary_context(facts)
        if digest(source["evidence"]) != record["evidence_sha256"] or digest(summary_text) != record["summary_sha256"]:
            raise ValueError(f"Frozen input drift for {record['id']}")
        combined = combine_context(source["evidence"], facts)
        if not combined.startswith(source["evidence"]):
            raise AssertionError("Original context was not preserved")
        started = time.perf_counter()
        answer, usage = generate_answer(source["question"], combined, [], "", profile["answer"]["profile"], "baseline")
        if not answer.strip():
            raise RuntimeError(f"Empty answer for {record['id']}")
        comparison = compare_answers(
            record["baseline_failure_type"], audit_by_id[record["id"]]["reference_answer"],
            record["baseline_answer"], answer,
        )
        record.update({
            "summary_answer": answer, "summary_result": comparison,
            "recovered": comparison["recovered"], "regression": comparison["regression"],
            "usage": usage, "latency_ms": round((time.perf_counter() - started) * 1000, 2), "status": "ok",
        })
        write_outputs(args.output_dir, payload)
        print(f"[{index:02d}/15] {record['id']}: {comparison['status']}", flush=True)
    payload["complete"] = True
    payload["summary"] = summarize(payload["records"])
    write_outputs(args.output_dir, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
