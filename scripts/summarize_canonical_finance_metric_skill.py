"""Build a paired explicit-skill versus finance-skills report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from skills.canonical_finance_metric.skill import detect_metric_alias


def _jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row.get("financebench_id") or "")] = row
    return result


def _correct(row: dict) -> bool:
    return str(row.get("verdict") or row.get("judge_verdict") or "").casefold() == "correct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-answers", type=Path, default=ROOT / "reports" / "evidencerag-skill-explicit-formula-v1_answers.jsonl")
    parser.add_argument("--baseline-judge", type=Path, default=ROOT / "reports" / "evidencerag-skill-explicit-formula-v1_judge.jsonl")
    parser.add_argument("--skill-answers", type=Path, required=True)
    parser.add_argument("--skill-judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with (ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv").open(
        encoding="utf-8-sig", newline="",
    ) as handle:
        source = list(csv.DictReader(handle))
    targets = [row for row in source if detect_metric_alias(row.get("question") or "")[0]]
    before_answers, before_judges = _jsonl(args.baseline_answers), _jsonl(args.baseline_judge)
    after_answers, after_judges = _jsonl(args.skill_answers), _jsonl(args.skill_judge)
    totals = Counter()
    failures = Counter()
    per_metric: dict[str, Counter] = defaultdict(Counter)
    details = []
    latencies: list[float] = []
    for source_row in targets:
        financebench_id = str(source_row.get("financebench_id") or "")
        before, after = before_answers.get(financebench_id, {}), after_answers.get(financebench_id, {})
        before_judge, after_judge = before_judges.get(financebench_id, {}), after_judges.get(financebench_id, {})
        metric, alias = detect_metric_alias(source_row.get("question") or "")
        trace = (after.get("rag_trace") or {}).get("canonical_finance_metric_skill") or {}
        detected, success = bool(trace.get("skill_detected")), bool(trace.get("skill_success"))
        authoritative_numeric = bool(trace.get("authoritative_numeric"))
        authoritative_answer = bool(trace.get("authoritative_answer"))
        before_correct, after_correct = _correct(before_judge), _correct(after_judge)
        for counter in (totals, per_metric[metric]):
            counter["questions"] += 1
            counter["detected"] += detected
            counter["success"] += success
            counter["authoritative_numeric"] += authoritative_numeric
            counter["authoritative_answer"] += authoritative_answer
            counter["before_correct"] += before_correct
            counter["after_correct"] += after_correct
            counter["wrong_to_correct"] += not before_correct and after_correct
            counter["correct_to_wrong"] += before_correct and not after_correct
            counter["dense_bm25_calls"] += int(trace.get("skill_dense_bm25_calls") or 0)
            counter["jina_calls"] += int(trace.get("skill_jina_calls") or 0)
            counter["llm_calls"] += int(trace.get("skill_llm_calls") or 0)
        reason = str(trace.get("operand_resolution_failure_reason") or trace.get("fallback_reason") or "")
        if detected and not success:
            failures[reason or "unspecified"] += 1
        totals["before_tokens"] += int((before.get("usage") or {}).get("total_tokens") or 0)
        totals["after_tokens"] += int((after.get("usage") or {}).get("total_tokens") or 0)
        latencies.append(float(trace.get("skill_latency_ms") or 0))
        details.append({
            "financebench_id": financebench_id, "metric": metric, "matched_alias": alias,
            "question": source_row.get("question") or "", "reference_answer": source_row.get("answer") or source_row.get("reference_answer") or "",
            "baseline_answer": before.get("answer") or "", "skill_answer": after.get("answer") or "",
            "baseline_correct": before_correct, "skill_correct": after_correct,
            "detected": detected, "success": success, "authoritative_numeric": authoritative_numeric,
            "authoritative_answer": authoritative_answer, "failure_reason": reason,
            "display_result": trace.get("metric_display_result") or "",
            "search_calls": int(trace.get("skill_dense_bm25_calls") or 0),
            "skill_latency_ms": float(trace.get("skill_latency_ms") or 0),
        })
    payload = {
        "summary": {
            **dict(totals),
            "net_gain": totals["after_correct"] - totals["before_correct"],
            "token_delta": totals["after_tokens"] - totals["before_tokens"],
            "average_skill_latency_ms": round(sum(latencies) / max(1, len(latencies)), 2),
            "failure_reasons": dict(failures),
            "per_metric": {name: dict(values) for name, values in sorted(per_metric.items())},
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
