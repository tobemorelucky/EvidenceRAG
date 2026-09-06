"""Compare frozen Jina answers with compressed-evidence DeepSeek Flash answers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from evidence_compression_shadow_v1 import compress_financial_evidence_v1  # noqa: E402
from run_financial_evidence_summary_answer_shadow_v1 import compare_answers, digest, load_jsonl  # noqa: E402


DEFAULT_ANSWERS = ROOT / "reports/jina_full_baseline_input120_all100/answers.jsonl"
DEFAULT_AUDIT = ROOT / "reports/answer_failure_audit_v1.json"
DEFAULT_SUMMARIES = ROOT / "reports/financial_evidence_summary_shadow_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/evidence_compression_shadow_v1"
PROFILE = ROOT / "configs/experiments/jina_full_baseline_input120_v1.json"


def _activity_choice(value: str | None) -> str | None:
    lowered = str(value or "").casefold()
    if lowered.startswith("operat"):
        return "operations"
    if lowered.startswith("invest"):
        return "investing"
    if lowered.startswith("financ"):
        return "financing"
    return lowered or None


def compare_compressed_answers(failure_type: str, reference: str, baseline: str, answer: str) -> dict[str, Any]:
    """Apply the shared diagnostic plus generic cash-flow activity normalization."""
    result = compare_answers(failure_type, reference, baseline, answer)
    diagnostic = result["diagnostic"]
    reference_choice = _activity_choice(diagnostic.get("reference_selection_choice"))
    answer_choice = _activity_choice(diagnostic.get("answer_selection_choice"))
    if reference_choice and answer_choice and reference_choice == answer_choice:
        diagnostic["selection_match"] = True
    ignored_activity_terms = {"operations", "operating", "investing", "financing", "activities"}
    required_names = set(diagnostic.get("required_proper_names") or []) - ignored_activity_terms
    if required_names:
        diagnostic["proper_name_match"] = all(name in answer.casefold() for name in required_names)
    elif diagnostic.get("required_proper_names"):
        diagnostic["proper_name_match"] = True
    if failure_type == "reasoning_failure" and not diagnostic.get("refusal_detected"):
        numeric = diagnostic.get("numeric_coverage")
        number_ok = numeric is None or numeric >= 0.8
        checks = (
            diagnostic.get("polarity_match"), diagnostic.get("direction_match"),
            diagnostic.get("selection_match"), diagnostic.get("critical_fact_match"),
            diagnostic.get("proper_name_match"), number_ok,
            float(diagnostic.get("leading_reference_term_coverage") or 0) >= 0.4,
        )
        if all(checks):
            diagnostic["status"] = "likely_resolved"
            result["status"] = "likely_recovered"
            result["recovered"] = True
    return result


def _prepare_records(answers: dict[str, dict], audit: dict, summaries: dict) -> list[dict[str, Any]]:
    summary_by_id = {record["question_id"]: record for record in summaries["records"]}
    records = []
    for item in audit["items"]:
        source = answers[item["question_id"]]
        compressed = compress_financial_evidence_v1(source["question"], summary_by_id[item["question_id"]])
        records.append({
            "id": item["question_id"], "question": source["question"],
            "baseline_failure_type": item["category"], "baseline_answer": source["answer"],
            "compressed_answer": None, "compression_result": {
                "original_context_chars": len(source["evidence"]),
                "compressed_context_chars": len(compressed["text"]),
                "character_change": len(compressed["text"]) - len(source["evidence"]),
                "kept_facts": len(compressed["kept_facts"]),
                "kept_spans": compressed["kept_span_count"],
                "dropped_facts": len(compressed["dropped_facts"]),
                "drop_reasons": compressed["drop_reasons"],
            },
            "failure_recovery": None, "recovered": None, "regression": None,
            "baseline_usage": source.get("usage") or {}, "compressed_usage": {},
            "token_change": None, "latency_ms": None, "status": "pending",
            "original_context_sha256": digest(source["evidence"]),
            "compressed_context_sha256": digest(compressed["text"]),
        })
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "ok"]
    baseline_inputs = [int(record["baseline_usage"].get("input_tokens") or 0) for record in completed]
    compressed_inputs = [int(record["compressed_usage"].get("input_tokens") or 0) for record in completed]
    return {
        "questions": len(records), "compressed_calls_completed": len(completed),
        "likely_recovered": sum(record.get("recovered") is True for record in completed),
        "deterministic_regressions": sum(record.get("regression") is True for record in completed),
        "baseline_reused": len(records), "deepseek_flash_calls": len(completed),
        "retrieval_calls": 0, "jina_calls": 0, "judge_calls": 0, "langsmith_calls": 0,
        "average_original_context_chars": round(sum(record["compression_result"]["original_context_chars"] for record in records) / len(records), 2),
        "average_compressed_context_chars": round(sum(record["compression_result"]["compressed_context_chars"] for record in records) / len(records), 2),
        "total_baseline_input_tokens": sum(baseline_inputs),
        "total_compressed_input_tokens": sum(compressed_inputs),
        "input_token_change": sum(compressed_inputs) - sum(baseline_inputs),
        "input_token_change_percent": round((sum(compressed_inputs) / sum(baseline_inputs) - 1) * 100, 2) if sum(baseline_inputs) else None,
        "total_compressed_output_tokens": sum(int(record["compressed_usage"].get("output_tokens") or 0) for record in completed),
        "average_latency_ms": round(sum(float(record.get("latency_ms") or 0) for record in completed) / len(completed), 2) if completed else None,
        "result_contract": "deterministic reference-alignment diagnostic only; no Judge",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = summarize(payload["records"])
    lines = [
        "# Evidence Compression Shadow v1", "",
        "A 组复用冻结 Jina context 的原 DeepSeek-V4-Flash baseline；B 组只向相同默认 prompt 提供由 Financial Evidence Summary v1 过滤出的压缩证据。参考答案只在回答后参与确定性诊断，未进入压缩器或模型上下文。未调用 Retrieval、Jina、Judge 或 LangSmith。", "",
        "## 汇总", "",
        f"- 题目/B组完成：{summary['questions']}/{summary['compressed_calls_completed']}",
        f"- 可能恢复：{summary['likely_recovered']}",
        f"- 确定性回退：{summary['deterministic_regressions']}",
        f"- 平均原始/压缩 context 字符：{summary['average_original_context_chars']} / {summary['average_compressed_context_chars']}",
        f"- Baseline/压缩输入 token：{summary['total_baseline_input_tokens']} / {summary['total_compressed_input_tokens']}",
        f"- 输入 token 变化：{summary['input_token_change']}（{summary['input_token_change_percent']}%）",
        f"- 压缩回答输出 token：{summary['total_compressed_output_tokens']}",
        f"- 平均延迟：{summary['average_latency_ms']} ms", "",
        "> `likely_recovered` 和 `regression` 是确定性参考答案对齐诊断，不是 Strict Judge。", "",
        "## 逐题", "",
    ]
    for index, record in enumerate(payload["records"], 1):
        recovery = record.get("failure_recovery") or {}
        comp = record["compression_result"]
        lines.extend([
            f"### {index}. {record['id']} — `{record['baseline_failure_type']}`", "",
            f"**问题：** {record['question']}", "",
            f"**A — Baseline：** {record['baseline_answer']}", "",
            f"**B — Compressed：** {record.get('compressed_answer') or '尚未完成'}", "",
            f"- Recovery：`{recovery.get('status', record['status'])}`；recovered=`{record.get('recovered')}`",
            f"- Regression：`{record.get('regression')}` {recovery.get('regression_reasons', [])}",
            f"- Context chars：`{comp['original_context_chars']} → {comp['compressed_context_chars']}`",
            f"- Facts kept/dropped：`{comp['kept_facts']}/{comp['dropped_facts']}`；reasons=`{comp['drop_reasons']}`",
            f"- Input token：`{record['baseline_usage'].get('input_tokens')} → {record['compressed_usage'].get('input_tokens')}`；变化=`{record.get('token_change')}`",
            f"- B延迟：`{record.get('latency_ms')} ms`", "",
        ])
    return "\n".join(lines)


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "state.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "state.json")
    (output_dir / "results.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in payload["records"]), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--summaries", type=Path, default=DEFAULT_SUMMARIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    answers = {row["financebench_id"]: row for row in load_jsonl(args.answers)}
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    summaries = json.loads(args.summaries.read_text(encoding="utf-8"))
    if len(audit.get("items") or []) != 15 or len(summaries.get("records") or []) != 15:
        raise ValueError("Expected the exact 15-item frozen artifacts")
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    if profile["answer"]["model"] != "deepseek-v4-flash-ga-260731":
        raise ValueError("Only DeepSeek-V4-Flash is allowed")
    manifest = {
        "experiment": "evidence_compression_shadow_v1", "model": profile["answer"]["model"],
        "prompt_mode": "baseline", "baseline_route": "reused_frozen_jina_context_answer",
        "compressed_route": "financial_evidence_summary_v1_filtered_facts_only",
        "reference_in_compression_or_model_context": False,
        "answers_sha256": hashlib.sha256(args.answers.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        "summaries_sha256": hashlib.sha256(args.summaries.read_bytes()).hexdigest(),
    }
    payload = {"manifest": manifest, "records": _prepare_records(answers, audit, summaries), "complete": False}
    state_path = args.output_dir / "state.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("manifest") != manifest:
            raise ValueError("Checkpoint input drift; use a new output directory")
        payload = previous
    audit_by_id = {item["question_id"]: item for item in audit["items"]}
    for record in payload["records"]:
        if record.get("status") == "ok":
            recovery = compare_compressed_answers(
                record["baseline_failure_type"], audit_by_id[record["id"]]["reference_answer"],
                record["baseline_answer"], record["compressed_answer"],
            )
            record.update(failure_recovery=recovery, recovered=recovery["recovered"], regression=recovery["regression"])
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
        "MODEL": profile["answer"]["model"], "ANSWER_TEMPERATURE": str(profile["answer"]["temperature"]),
        "ANSWER_MAX_COMPLETION_TOKENS": str(profile["answer"]["max_completion_tokens"]),
        "ANSWER_THINKING_MODE": profile["answer"]["thinking"],
        "ANSWER_TIMEOUT_SECONDS": str(profile["answer"]["timeout_seconds"]), "ANSWER_MAX_RETRIES": "0",
        "ANSWER_PROMPT_MODE": "baseline", "LANGSMITH_TRACING": "false",
        "LANGSMITH_TRACING_V2": "false", "LANGCHAIN_TRACING_V2": "false",
    })
    from answer_generator import generate_answer

    summary_by_id = {record["question_id"]: record for record in summaries["records"]}
    for index, record in enumerate(payload["records"], 1):
        if record.get("status") == "ok":
            continue
        source = answers[record["id"]]
        compressed = compress_financial_evidence_v1(source["question"], summary_by_id[record["id"]])
        if digest(source["evidence"]) != record["original_context_sha256"] or digest(compressed["text"]) != record["compressed_context_sha256"]:
            raise ValueError(f"Frozen input drift for {record['id']}")
        started = time.perf_counter()
        answer, usage = generate_answer(
            source["question"], compressed["text"], [], "", profile["answer"]["profile"], "baseline"
        )
        if not answer.strip():
            raise RuntimeError(f"Empty answer for {record['id']}")
        recovery = compare_compressed_answers(
            record["baseline_failure_type"], audit_by_id[record["id"]]["reference_answer"],
            record["baseline_answer"], answer,
        )
        baseline_input = int(record["baseline_usage"].get("input_tokens") or 0)
        compressed_input = int(usage.get("input_tokens") or 0)
        record.update({
            "compressed_answer": answer, "failure_recovery": recovery,
            "recovered": recovery["recovered"], "regression": recovery["regression"],
            "compressed_usage": usage, "token_change": compressed_input - baseline_input,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2), "status": "ok",
        })
        write_outputs(args.output_dir, payload)
        print(f"[{index:02d}/15] {record['id']}: {recovery['status']}", flush=True)
    payload["complete"] = True
    payload["summary"] = summarize(payload["records"])
    write_outputs(args.output_dir, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
