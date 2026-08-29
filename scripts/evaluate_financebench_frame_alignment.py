"""Measure QuerySpec-to-EvidenceFrame utilization without answer-model calls."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"


def _summarize(records: list[dict]) -> dict:
    gate_names = (
        "frame_matched",
        "measure_validated",
        "period_validated",
        "metadata_validated",
        "operand_unique",
        "operation_validated",
    )
    return {
        "questions": len(records),
        "evidence_frame_questions": sum(int(item.get("evidence_frame_count") or 0) > 0 for item in records),
        "queryspec_related_frame_questions": sum(int(item.get("relevant_frame_count") or 0) > 0 for item in records),
        "structured_answerable_questions": sum(bool((item.get("evidence_coverage") or {}).get("structured_answerable")) for item in records),
        "structured_execution_ready_questions": sum(bool((item.get("evidence_coverage") or {}).get("structured_execution_ready")) for item in records),
        "structured_executions": sum(int(item.get("frames_used_for_execution") or 0) > 0 for item in records),
        "execution_gate_funnel": {
            gate: sum(bool((item.get("structured_gate_trace") or {}).get(gate)) for item in records)
            for gate in gate_names
        },
        "operand_resolution_failures": {
            reason: sum(str(item.get("operand_resolution_failure_reason") or "") == reason for item in records)
            for reason in sorted({str(item.get("operand_resolution_failure_reason") or "") for item in records} - {""})
        },
        "supplemental_triggered": sum(bool(item.get("supplemental_triggered")) for item in records),
        "supplemental_effective": sum(bool(item.get("supplemental_effective")) for item in records),
        "candidate_coverage_complete": sum((item.get("candidate_coverage") or {}).get("status") == "complete" for item in records),
        "selected_page_coverage_complete": sum((item.get("selected_page_coverage") or {}).get("status") == "complete" for item in records),
        "compact_context_coverage_complete": sum((item.get("compact_context_coverage") or {}).get("status") == "complete" for item in records),
        "candidate_complete_selected_incomplete": sum(
            (item.get("candidate_coverage") or {}).get("status") == "complete"
            and (item.get("selected_page_coverage") or {}).get("status") != "complete"
            for item in records
        ),
        "candidate_complete_selected_incomplete_before_protection": sum(
            (item.get("candidate_coverage") or {}).get("status") == "complete"
            and (item.get("protected_page_coverage_before") or {}).get("status") != "complete"
            for item in records
        ),
        "protected_page_questions": sum(int(item.get("protected_page_count") or 0) > 0 for item in records),
        "protected_page_count": sum(int(item.get("protected_page_count") or 0) for item in records),
        "protected_page_budget_violations": sum(
            int(item.get("selected_page_count_after_protection") or 0)
            > int(item.get("selected_page_count_before_protection") or 0)
            for item in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--enable-rerank", action="store_true")
    parser.add_argument("--enable-supplemental", action="store_true")
    parser.add_argument(
        "--enable-langsmith-tracing",
        action="store_true",
        help="Opt in to LangSmith tracing; local diagnostics disable it by default.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    from run_financebench_langsmith_experiment import _configure_static_baseline, _development_ids

    if not args.enable_langsmith_tracing:
        # The experiment helper loads .env with override=True. Reset tracing
        # afterwards so this no-LLM local diagnostic cannot consume trace quota.
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGSMITH_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

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
        True, True, True, True, True, True, True,
        True, True, True, True, True, args.enable_supplemental,
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
            "structured_gate_trace": (trace.get("evidence_coverage") or {}).get("structured_gate_trace") or {},
            "query_spec": prepared.get("query_spec") or {},
            "execution_contract": trace.get("execution_contract") or {},
            "structured_authoritative": bool(trace.get("structured_authoritative")),
            "calculation": prepared.get("calculation"),
            "frames_used_for_execution": trace.get("frames_used_for_execution", 0),
            "supplemental_triggered": trace.get("supplemental_triggered", False),
            "supplemental_effective": trace.get("supplemental_effective", False),
            "supplemental_query": trace.get("supplemental_query") or "",
            "supplemental_skip_reason": trace.get("supplemental_skip_reason") or "",
            "candidate_missing_operands": trace.get("candidate_missing_operands") or [],
            "supplemental_new_pages": trace.get("new_pages") or [],
            "supplemental_requirement_improvements": trace.get("supplemental_requirement_improvements") or [],
            "candidate_coverage": trace.get("candidate_coverage") or {},
            "selected_page_coverage": trace.get("selected_page_coverage") or {},
            "compact_context_coverage": trace.get("compact_context_coverage") or {},
            "evidence_flow_stage": trace.get("evidence_flow_stage") or "",
            "protected_pages": trace.get("protected_pages") or [],
            "protected_page_replacements": trace.get("protected_page_replacements") or [],
            "protected_page_coverage_before": trace.get("protected_page_coverage_before") or {},
            "protected_page_coverage_after": trace.get("protected_page_coverage_after") or {},
            "protected_page_count": trace.get("protected_page_count", 0),
            "selected_page_count_before_protection": trace.get("selected_page_count_before_protection", 0),
            "selected_page_count_after_protection": trace.get("selected_page_count_after_protection", 0),
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
