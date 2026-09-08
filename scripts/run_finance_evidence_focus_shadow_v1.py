"""Evaluate Evidence Focus Prompt v1 over the frozen three-question context.

The clean-baseline and terminology-alignment answers are reused from the
completed predecessor experiment. Only three new answer-model calls are made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

PREDECESSOR_STATE = ROOT / "reports" / "finance_terminology_alignment_shadow_v1" / "state.json"
DEFAULT_OUTPUT = ROOT / "reports" / "finance_evidence_focus_shadow_v1"
PROMPT_MODES = (
    "clean_baseline_v1",
    "finance_terminology_alignment_v1",
    "finance_evidence_focus_v1",
)

from scripts.run_finance_terminology_alignment_shadow_v1 import assess_answer


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_predecessor(path: Path = PREDECESSOR_STATE) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("complete") or len(payload.get("records") or []) != 3:
        raise ValueError("Expected a completed three-question terminology-alignment checkpoint")
    for record in payload["records"]:
        if not record.get("evidence") or not record.get("evidence_sha256"):
            raise ValueError(f"Frozen evidence missing for {record.get('question_id')}")
        if hashlib.sha256(record["evidence"].encode("utf-8")).hexdigest() != record["evidence_sha256"]:
            raise ValueError(f"Frozen evidence hash mismatch for {record.get('question_id')}")
        for mode in PROMPT_MODES[:2]:
            if (record.get("results", {}).get(mode) or {}).get("status") != "ok":
                raise ValueError(f"Missing predecessor result {record.get('question_id')} / {mode}")
    return payload


def initialize_payload(predecessor_path: Path) -> dict:
    predecessor = load_predecessor(predecessor_path)
    records = []
    for source in predecessor["records"]:
        records.append({
            "question_id": source["question_id"],
            "question": source["question"],
            "required_number_groups": source["required_number_groups"],
            "direction_terms": source["direction_terms"],
            "evidence": source["evidence"],
            "evidence_sha256": source["evidence_sha256"],
            "source_context_metrics": source.get("source_context_metrics") or {},
            "results": {
                mode: {**source["results"][mode], "reused_from_predecessor": True}
                for mode in PROMPT_MODES[:2]
            },
        })
    return {
        "manifest": {
            "experiment": "finance_evidence_focus_shadow_v1",
            "predecessor": str(predecessor_path.relative_to(ROOT)).replace("\\", "/"),
            "predecessor_sha256": _sha256(predecessor_path),
            "question_ids": [record["question_id"] for record in records],
            "prompt_modes": list(PROMPT_MODES),
            "retrieval_calls": 0,
            "query_rewrite_calls": 0,
            "jina_calls": 0,
            "judge_calls": 0,
            "langsmith_calls": 0,
            "answer_calls_reused": 6,
            "answer_calls_expected": 3,
        },
        "records": records,
        "complete": False,
    }


def _total_tokens(result: dict) -> int:
    usage = result.get("usage") or {}
    return int(usage.get("total_tokens") or usage.get("total_token_count") or 0)


def build_summary(payload: dict) -> dict:
    modes = {}
    for mode in PROMPT_MODES:
        results = [record["results"].get(mode) or {} for record in payload["records"]]
        modes[mode] = {
            "correct": sum(bool((result.get("assessment") or {}).get("acceptance_pass")) for result in results),
            "unnecessary_refusals": sum(
                bool((result.get("assessment") or {}).get("unnecessary_refusal")) for result in results
            ),
            "total_tokens": sum(_total_tokens(result) for result in results),
            "average_tokens": round(sum(_total_tokens(result) for result in results) / len(results), 2),
            "average_latency_ms": round(
                sum(float(result.get("latency_ms") or 0) for result in results) / len(results), 2
            ),
        }
    comparisons = []
    for record in payload["records"]:
        baseline_ok = bool((record["results"]["clean_baseline_v1"].get("assessment") or {}).get("acceptance_pass"))
        focus = record["results"].get("finance_evidence_focus_v1") or {}
        focus_ok = bool((focus.get("assessment") or {}).get("acceptance_pass"))
        comparisons.append({
            "question_id": record["question_id"],
            "baseline_correct": baseline_ok,
            "focus_correct": focus_ok,
            "recovered_vs_baseline": not baseline_ok and focus_ok,
            "regression_vs_baseline": baseline_ok and not focus_ok,
            "focus_unnecessary_refusal": bool((focus.get("assessment") or {}).get("unnecessary_refusal")),
        })
    baseline_tokens = modes["clean_baseline_v1"]["average_tokens"]
    focus_tokens = modes["finance_evidence_focus_v1"]["average_tokens"]
    return {
        "questions": len(payload["records"]),
        "modes": modes,
        "recovered_vs_baseline": sum(item["recovered_vs_baseline"] for item in comparisons),
        "regressions_vs_baseline": sum(item["regression_vs_baseline"] for item in comparisons),
        "focus_average_token_delta_vs_baseline": round(focus_tokens - baseline_tokens, 2),
        "acceptance": {
            "fy2015_remains_0_66": comparisons[0]["focus_correct"],
            "fy2017_remains_0_83": comparisons[1]["focus_correct"],
            "fy2022_uses_adobe_evidence_and_correct_direction": comparisons[2]["focus_correct"],
            "no_first_two_regression": not any(item["regression_vs_baseline"] for item in comparisons[:2]),
            "overall_pass": all(item["focus_correct"] for item in comparisons),
        },
        "comparisons": comparisons,
    }


def write_outputs(output_dir: Path, payload: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output_dir / "state.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "state.json")
    summary = build_summary(payload)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    lines = [
        "# Evidence Focus Prompt Shadow v1",
        "",
        "三种 Prompt 使用相同的 final100 frozen evidence。前两组答案复用上一轮结果，本轮只生成 Evidence Focus 三题答案。",
        "",
        "## 汇总",
        "",
    ]
    for mode in PROMPT_MODES:
        stats = summary["modes"][mode]
        lines.append(
            f"- `{mode}`：{stats['correct']}/3；不必要拒答 {stats['unnecessary_refusals']}；"
            f"平均 token {stats['average_tokens']}；平均延迟 {stats['average_latency_ms']} ms"
        )
    lines.extend([
        f"- Evidence Focus 相对 baseline：恢复 {summary['recovered_vs_baseline']}，回退 {summary['regressions_vs_baseline']}",
        f"- 平均 token 变化：{summary['focus_average_token_delta_vs_baseline']}",
        "",
    ])
    for index, record in enumerate(payload["records"], 1):
        lines.extend([f"## {index}. {record['question_id']}", "", record["question"], ""])
        for mode in PROMPT_MODES:
            result = record["results"].get(mode) or {}
            reused = "（复用）" if result.get("reused_from_predecessor") else ""
            lines.extend([
                f"### {mode}{reused}", "", result.get("answer") or f"状态：{result.get('status', 'pending')}", "",
                f"- assessment: `{json.dumps(result.get('assessment') or {}, ensure_ascii=False)}`",
                f"- usage: `{json.dumps(result.get('usage') or {}, ensure_ascii=False)}`",
                f"- latency_ms: `{result.get('latency_ms')}`", "",
            ])
    (output_dir / "answers.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predecessor-state", type=Path, default=PREDECESSOR_STATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    predecessor_path = args.predecessor_state.resolve()
    expected = initialize_payload(predecessor_path)
    state_path = args.output_dir / "state.json"
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if payload.get("manifest") != expected["manifest"]:
            raise ValueError("Existing checkpoint does not match the frozen predecessor")
    else:
        payload = expected
    write_outputs(args.output_dir, payload)
    if args.prepare_only:
        print(f"Prepared {len(payload['records'])} frozen cases; new answer calls=3")
        return

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    os.environ.update({
        "LANGSMITH_TRACING": "false",
        "LANGSMITH_TRACING_V2": "false",
        "LANGCHAIN_TRACING_V2": "false",
    })
    from answer_generator import generate_answer

    for record in payload["records"]:
        mode = "finance_evidence_focus_v1"
        if (record["results"].get(mode) or {}).get("status") == "ok":
            continue
        started = time.perf_counter()
        try:
            answer, usage = generate_answer(
                record["question"], record["evidence"], history=[], task_policy="",
                profile="finance", prompt_mode=mode,
            )
            result = {
                "status": "ok",
                "answer": answer,
                "assessment": assess_answer(answer, record),
                "usage": usage,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            result = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        record["results"][mode] = result
        write_outputs(args.output_dir, payload)
        print(f"[{record['question_id']}] {mode}: {result['status']}", flush=True)
    payload["complete"] = all(
        (record["results"].get(mode) or {}).get("status") == "ok"
        for record in payload["records"] for mode in PROMPT_MODES
    )
    write_outputs(args.output_dir, payload)
    print(f"Complete={payload['complete']} report={args.output_dir / 'answers.md'}")


if __name__ == "__main__":
    main()
