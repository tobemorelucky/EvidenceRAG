"""Measure QuerySpec-to-EvidenceFrame utilization without answer-model calls."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"


def _summarize(records: list[dict]) -> dict:
    return {
        "questions": len(records),
        "evidence_frame_questions": sum(int(item.get("evidence_frame_count") or 0) > 0 for item in records),
        "queryspec_related_frame_questions": sum(int(item.get("relevant_frame_count") or 0) > 0 for item in records),
        "structured_answerable_questions": sum(bool((item.get("evidence_coverage") or {}).get("structured_answerable")) for item in records),
        "structured_execution_ready_questions": sum(bool((item.get("evidence_coverage") or {}).get("structured_execution_ready")) for item in records),
        "structured_executions": sum(int(item.get("frames_used_for_execution") or 0) > 0 for item in records),
        "operand_resolution_failures": {
            reason: sum(str(item.get("operand_resolution_failure_reason") or "") == reason for item in records)
            for reason in sorted({str(item.get("operand_resolution_failure_reason") or "") for item in records} - {""})
        },
        "supplemental_triggered": sum(bool(item.get("supplemental_triggered")) for item in records),
        "supplemental_effective": sum(bool(item.get("supplemental_effective")) for item in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--enable-rerank", action="store_true")
    parser.add_argument("--enable-supplemental", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    from run_financebench_langsmith_experiment import _configure_static_baseline, _development_ids

    dev_ids = _development_ids(list(rows))
    if args.split != "all":
        rows = [row for row in rows if ((row.get("financebench_id") or "") in dev_ids) == (args.split == "dev")]
    if args.question_id:
        requested = set(args.question_id)
        rows = [row for row in rows if (row.get("financebench_id") or "") in requested]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No FinanceBench rows selected.")

    _configure_static_baseline(
        512, "disabled", False, True, args.enable_rerank, False, False,
        True, True, True, True, True, True, True, args.enable_supplemental,
    )
    sys.path.insert(0, str(BACKEND))
    from rag_orchestrator import prepare_rag_response

    records = []
    for index, row in enumerate(rows, start=1):
        prepared = prepare_rag_response(str(row.get("question") or ""), profile="finance", mode="static")
        trace = prepared.get("rag_trace") or {}
        record = {
            "financebench_id": row.get("financebench_id"),
            "task_type": trace.get("task_type"),
            "evidence_frame_count": trace.get("evidence_frame_count", 0),
            "relevant_frame_count": trace.get("relevant_frame_count", 0),
            "frame_match_method": trace.get("frame_match_method"),
            "frame_match_score": trace.get("frame_match_score"),
            "operand_resolution_failure_reason": trace.get("operand_resolution_failure_reason"),
            "evidence_coverage": trace.get("evidence_coverage") or {},
            "frames_used_for_execution": trace.get("frames_used_for_execution", 0),
            "supplemental_triggered": trace.get("supplemental_triggered", False),
            "supplemental_effective": trace.get("supplemental_effective", False),
        }
        records.append(record)
        print(
            f"[{index:02d}/{len(rows)}] {record['financebench_id']} "
            f"frames={record['evidence_frame_count']} relevant={record['relevant_frame_count']} "
            f"ready={bool(record['evidence_coverage'].get('structured_execution_ready'))}",
            flush=True,
        )
    payload = {"summary": _summarize(records), "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
