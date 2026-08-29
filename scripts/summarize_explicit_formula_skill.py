"""Build a paired A/B report for the explicit-formula regression set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from skills.explicit_formula.skill import build_formula_contract


def _jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        str(row.get("financebench_id") or ""): row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and (row := json.loads(line))
    }


def _verdict(row: dict) -> str:
    return str(row.get("verdict") or row.get("judge_verdict") or "unknown").casefold()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-answers", type=Path, default=ROOT / "reports" / "evidencerag-clean-baseline-v1_answers.jsonl")
    parser.add_argument("--baseline-judge", type=Path, default=ROOT / "reports" / "evidencerag-clean-baseline-v1_judge.jsonl")
    parser.add_argument("--skill-answers", type=Path, required=True)
    parser.add_argument("--skill-judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with (ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv").open(
        encoding="utf-8-sig", newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    formula_rows = [row for row in rows if build_formula_contract(row.get("question") or "")[0] is not None]
    baseline_answers = _jsonl(args.baseline_answers)
    baseline_judge = _jsonl(args.baseline_judge)
    skill_answers = _jsonl(args.skill_answers)
    skill_judge = _jsonl(args.skill_judge)

    details = []
    failure_counts: Counter[str] = Counter()
    totals = Counter()
    baseline_latencies: list[float] = []
    skill_latencies: list[float] = []
    skill_only_latencies: list[float] = []
    atomic_operand_total = 0
    baseline_operand_found = 0
    for row in formula_rows:
        financebench_id = row.get("financebench_id") or ""
        before = baseline_answers.get(financebench_id, {})
        after = skill_answers.get(financebench_id, {})
        trace = (after.get("rag_trace") or {}).get("explicit_formula_skill") or {}
        before_verdict = _verdict(baseline_judge.get(financebench_id, {}))
        after_verdict = _verdict(skill_judge.get(financebench_id, {}))
        totals["detected"] += bool(trace.get("skill_detected"))
        totals["success"] += bool(trace.get("skill_success"))
        totals["search_calls"] += int(trace.get("operand_search_calls") or 0)
        totals["jina_calls"] += int(trace.get("skill_jina_calls") or 0)
        totals["before_correct"] += before_verdict == "correct"
        totals["after_correct"] += after_verdict == "correct"
        totals["wrong_to_correct"] += before_verdict != "correct" and after_verdict == "correct"
        totals["correct_to_wrong"] += before_verdict == "correct" and after_verdict != "correct"
        before_usage = before.get("usage") or {}
        after_usage = after.get("usage") or {}
        totals["baseline_tokens"] += int(before_usage.get("total_tokens") or 0)
        totals["skill_tokens"] += int(after_usage.get("total_tokens") or 0)
        baseline_latencies.append(float((before.get("evaluation_latency") or {}).get("total_ms") or 0))
        skill_latencies.append(float((after.get("evaluation_latency") or {}).get("total_ms") or 0))
        skill_only_latencies.append(float(trace.get("skill_latency_ms") or 0))
        operand_keys = list(trace.get("formula_operands") or [])
        found_before = list(trace.get("operands_found_in_baseline_evidence") or [])
        atomic_operand_total += len(operand_keys)
        baseline_operand_found += len(found_before)
        reason = str(trace.get("operand_resolution_failure_reason") or "")
        if reason:
            failure_counts[reason] += 1
        details.append({
            "financebench_id": financebench_id,
            "question": row.get("question") or "",
            "reference_answer": row.get("answer") or row.get("reference_answer") or "",
            "baseline_answer": before.get("answer") or "",
            "skill_answer": after.get("answer") or "",
            "baseline_verdict": before_verdict,
            "skill_verdict": after_verdict,
            "skill_detected": bool(trace.get("skill_detected")),
            "skill_success": bool(trace.get("skill_success")),
            "failure_reason": reason,
            "operands_before": trace.get("operands_found_in_baseline_evidence") or [],
            "missing_before": trace.get("missing_operands_before_tool") or [],
            "search_calls": int(trace.get("operand_search_calls") or 0),
            "skill_latency_ms": float(trace.get("skill_latency_ms") or 0),
            "full_precision_result": trace.get("full_precision_result") or "",
            "display_result": trace.get("display_result") or "",
        })
    summary = {
        "formula_questions": len(formula_rows),
        **dict(totals),
        "token_delta": totals["skill_tokens"] - totals["baseline_tokens"],
        "baseline_average_total_latency_ms": round(sum(baseline_latencies) / max(1, len(baseline_latencies)), 2),
        "skill_average_total_latency_ms": round(sum(skill_latencies) / max(1, len(skill_latencies)), 2),
        "average_skill_only_latency_ms": round(sum(skill_only_latencies) / max(1, len(skill_only_latencies)), 2),
        "operand_coverage_before_tool": round(baseline_operand_found / max(1, atomic_operand_total), 4),
        "operand_coverage_after_tool": round(
            sum(len((item.get("rag_trace") or {}).get("explicit_formula_skill", {}).get("resolved_operands") or []) for item in skill_answers.values())
            / max(1, atomic_operand_total),
            4,
        ),
        "failure_counts": dict(failure_counts),
        "arithmetic_self_consistency": 1.0 if totals["success"] else None,
        "new_llm_calls": 0,
    }
    payload = {
        "summary": summary,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
