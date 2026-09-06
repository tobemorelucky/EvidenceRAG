"""Run Evidence Guidance v1 on the frozen 15 answer failures."""

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

from evidence_guidance_shadow_v1 import build_evidence_guidance_v1  # noqa: E402
from run_evidence_compression_shadow_v1 import compare_compressed_answers  # noqa: E402
from run_financial_evidence_summary_answer_shadow_v1 import digest, load_jsonl  # noqa: E402


DEFAULT_ANSWERS = ROOT / "reports/jina_full_baseline_input120_all100/answers.jsonl"
DEFAULT_AUDIT = ROOT / "reports/answer_failure_audit_v1.json"
DEFAULT_SUMMARIES = ROOT / "reports/financial_evidence_summary_shadow_v1.json"
DEFAULT_FRAMES = ROOT / "reports/evidence_frame_shadow_v1.json"
DEFAULT_SCHEMAS = ROOT / "reports/financial_operation_schema_shadow_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/evidence_guidance_shadow_v1"
PROFILE = ROOT / "configs/experiments/jina_full_baseline_input120_v1.json"


def _index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {record["question_id"]: record for record in payload["records"]}


def _prepare_records(answers: dict, audit: dict, summaries: dict, frames: dict, schemas: dict) -> list[dict[str, Any]]:
    summary_by_id, frame_by_id, schema_by_id = _index(summaries), _index(frames), _index(schemas)
    records = []
    for item in audit["items"]:
        question_id = item["question_id"]
        source = answers[question_id]
        guidance = build_evidence_guidance_v1(
            source["question"], summary_by_id[question_id], frame_by_id[question_id], schema_by_id[question_id]
        )
        records.append({
            "id": question_id, "question": source["question"], "baseline_failure_type": item["category"],
            "baseline_answer": source["answer"], "guidance_answer": None, "guidance_result": None,
            "recovered": None, "regression": None, "status": "pending",
            "guidance": {
                "characters": guidance["characters"], "estimated_tokens": guidance["estimated_tokens"],
                "selected_fact_count": len(guidance["selected_facts"]),
                "available_relevant_fact_count": guidance["available_relevant_facts"],
                "target": guidance["target"], "forbidden_confusion": guidance["forbidden_confusion"],
                "formula_included": guidance["formula_included"], "text": guidance["text"],
            },
            "baseline_usage": source.get("usage") or {}, "guidance_usage": {}, "token_increase": None,
            "latency_ms": None, "evidence_sha256": digest(source["evidence"]),
            "guidance_sha256": digest(guidance["text"]),
        })
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("status") == "ok"]
    token_increases = [int(record.get("token_increase") or 0) for record in completed]
    recoveries = [record["id"] for record in completed if record.get("recovered")]
    regressions = [record["id"] for record in completed if record.get("regression")]
    stop_condition_passed = len(recoveries) >= 3 and len(regressions) <= 2
    return {
        "questions": len(records), "guidance_calls_completed": len(completed),
        "confirmed_recoveries": len(recoveries), "recovered_ids": recoveries,
        "deterministic_regressions": len(regressions), "regression_ids": regressions,
        "average_guidance_estimated_tokens": round(sum(record["guidance"]["estimated_tokens"] for record in records) / len(records), 2),
        "maximum_guidance_estimated_tokens": max(record["guidance"]["estimated_tokens"] for record in records),
        "total_input_token_increase": sum(token_increases),
        "average_input_token_increase": round(sum(token_increases) / len(token_increases), 2) if token_increases else None,
        "average_latency_ms": round(sum(float(record.get("latency_ms") or 0) for record in completed) / len(completed), 2) if completed else None,
        "deepseek_flash_calls": len(completed), "retrieval_calls": 0, "jina_calls": 0,
        "judge_calls": 0, "langsmith_calls": 0,
        "stop_condition": "continue only if recoveries >=3 and regressions <=2",
        "stop_condition_passed": stop_condition_passed,
        "decision": "continue" if stop_condition_passed else "stop",
        "result_contract": "deterministic reference-alignment diagnostic only; no Judge",
    }


def render_markdown(payload: dict[str, Any]) -> str:
    summary = summarize(payload["records"])
    lines = [
        "# Evidence Guidance Shadow v1", "",
        "A 组复用冻结 Jina context 的 DeepSeek-V4-Flash baseline；B 组完整保留原 context，仅追加不超过800估算token的本地 guidance，并继续使用 baseline prompt。Gold/reference 不参与 guidance 构建或模型输入。未调用 Retrieval、Jina、Judge 或 LangSmith。", "",
        "## 汇总", "",
        f"- 题目/B组完成：{summary['questions']}/{summary['guidance_calls_completed']}",
        f"- 确定性恢复：{summary['confirmed_recoveries']} — `{summary['recovered_ids']}`",
        f"- 确定性回退：{summary['deterministic_regressions']} — `{summary['regression_ids']}`",
        f"- Guidance平均/最大估算token：{summary['average_guidance_estimated_tokens']} / {summary['maximum_guidance_estimated_tokens']}",
        f"- 平均/总输入token增加：{summary['average_input_token_increase']} / {summary['total_input_token_increase']}",
        f"- 平均B组延迟：{summary['average_latency_ms']} ms",
        f"- 停止条件是否通过：`{summary['stop_condition_passed']}`；决定：`{summary['decision']}`", "",
        "> 恢复与回退来自确定性参考答案对齐，不是 Strict Judge。", "",
        "## 典型错误变化", "",
    ]
    changed = [record for record in payload["records"] if record.get("recovered") or record.get("regression")]
    if not changed:
        lines.append("无确定性变化。")
    for record in changed:
        lines.extend([
            f"### {record['id']} — `{record['baseline_failure_type']}`", "",
            f"- Recovered：`{record['recovered']}`；Regression：`{record['regression']}`",
            f"- 变化原因：`{(record.get('guidance_result') or {}).get('regression_reasons', [])}`",
            f"- Guidance target：`{record['guidance']['target']}`", "",
            f"**Baseline：** {record['baseline_answer']}", "",
            f"**Guidance answer：** {record.get('guidance_answer') or '尚未完成'}", "",
        ])
    lines.extend(["## 全部逐题", ""])
    for index, record in enumerate(payload["records"], 1):
        result = record.get("guidance_result") or {}
        lines.extend([
            f"### {index}. {record['id']} — `{record['baseline_failure_type']}`", "",
            f"**问题：** {record['question']}", "",
            f"**Guidance：**\n```text\n{record['guidance']['text']}\n```", "",
            f"**A：** {record['baseline_answer']}", "",
            f"**B：** {record.get('guidance_answer') or '尚未完成'}", "",
            f"- Result：`{result.get('status', record['status'])}`；recovered=`{record.get('recovered')}`；regression=`{record.get('regression')}`",
            f"- Token increase：`{record.get('token_increase')}`；latency=`{record.get('latency_ms')} ms`", "",
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
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--schemas", type=Path, default=DEFAULT_SCHEMAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    answers = {row["financebench_id"]: row for row in load_jsonl(args.answers)}
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    summaries = json.loads(args.summaries.read_text(encoding="utf-8"))
    frames = json.loads(args.frames.read_text(encoding="utf-8"))
    schemas = json.loads(args.schemas.read_text(encoding="utf-8"))
    if any(len(payload.get("records") or []) != 15 for payload in (summaries, frames, schemas)) or len(audit.get("items") or []) != 15:
        raise ValueError("Expected the exact frozen 15-item artifacts")
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    if profile["answer"]["model"] != "deepseek-v4-flash-ga-260731":
        raise ValueError("Only DeepSeek-V4-Flash is allowed")
    inputs = (args.answers, args.audit, args.summaries, args.frames, args.schemas)
    manifest = {
        "experiment": "evidence_guidance_shadow_v1", "model": profile["answer"]["model"],
        "prompt_mode": "baseline", "original_context_preserved": True,
        "reference_in_guidance_or_model_context": False, "max_guidance_estimated_tokens": 800,
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
    }
    payload = {"manifest": manifest, "records": _prepare_records(answers, audit, summaries, frames, schemas), "complete": False}
    state_path = args.output_dir / "state.json"
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8"))
        if previous.get("manifest") != manifest:
            raise ValueError("Checkpoint input drift; use a new output directory")
        payload = previous
    audit_by_id = {item["question_id"]: item for item in audit["items"]}
    for record in payload["records"]:
        if record.get("status") == "ok":
            result = compare_compressed_answers(
                record["baseline_failure_type"], audit_by_id[record["id"]]["reference_answer"],
                record["baseline_answer"], record["guidance_answer"],
            )
            record.update(guidance_result=result, recovered=result["recovered"], regression=result["regression"])
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

    for index, record in enumerate(payload["records"], 1):
        if record.get("status") == "ok":
            continue
        source = answers[record["id"]]
        guidance = record["guidance"]["text"]
        if digest(source["evidence"]) != record["evidence_sha256"] or digest(guidance) != record["guidance_sha256"]:
            raise ValueError(f"Frozen input drift for {record['id']}")
        combined = f"{source['evidence']}\n\n{guidance}"
        if not combined.startswith(source["evidence"]):
            raise AssertionError("Original context was not preserved")
        started = time.perf_counter()
        answer, usage = generate_answer(
            source["question"], combined, [], "", profile["answer"]["profile"], "baseline"
        )
        result = compare_compressed_answers(
            record["baseline_failure_type"], audit_by_id[record["id"]]["reference_answer"],
            record["baseline_answer"], answer,
        )
        baseline_input = int(record["baseline_usage"].get("input_tokens") or 0)
        guidance_input = int(usage.get("input_tokens") or 0)
        record.update({
            "guidance_answer": answer, "guidance_result": result, "recovered": result["recovered"],
            "regression": result["regression"], "guidance_usage": usage,
            "token_increase": guidance_input - baseline_input,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2), "status": "ok",
        })
        write_outputs(args.output_dir, payload)
        print(f"[{index:02d}/15] {record['id']}: {result['status']}", flush=True)
    payload["complete"] = True
    payload["summary"] = summarize(payload["records"])
    write_outputs(args.output_dir, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()

