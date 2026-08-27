"""Evaluate the current answer pipeline with benchmark-provided evidence.

This is a diagnostic upper-bound evaluator, not a retrieval benchmark. It skips
Dense/BM25, RRF, page selection, and reranking. FinanceBench is treated as a
fixed, previously seen regression set; results must not be described as unseen
generalization.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
load_dotenv(ROOT / ".env", override=True)


def _clean_text(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def parse_gold_pages(row: dict[str, str], mode: str = "gold_page") -> list[dict[str, Any]]:
    """Build source-preserving documents from FinanceBench gold evidence."""
    try:
        payload = json.loads(str(row.get("evidence") or ""))
    except (TypeError, json.JSONDecodeError):
        return []
    records = payload if isinstance(payload, list) else [payload]
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    snippets: dict[tuple[str, int], list[str]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        document = _clean_text(record.get("doc_name") or row.get("doc_name"))
        try:
            page_number = int(float(record.get("evidence_page_num")))
        except (TypeError, ValueError):
            continue
        if not document:
            continue
        filename = document if document.lower().endswith(".pdf") else f"{document}.pdf"
        key = (filename, page_number)
        snippet = _clean_text(record.get("evidence_text"))
        full_page = _clean_text(record.get("evidence_text_full_page"))
        if snippet:
            snippets.setdefault(key, [])
            if snippet not in snippets[key]:
                snippets[key].append(snippet)
        text = full_page if mode == "gold_page" and full_page else snippet
        existing = grouped.get(key)
        if existing is None or len(text) > len(str(existing.get("text") or "")):
            grouped[key] = {
                "filename": filename,
                "doc_name": document.removesuffix(".pdf"),
                "page_number": page_number,
                "text": text,
                "page_text": text,
                "type": "oracle_gold_page" if full_page and mode == "gold_page" else "oracle_gold_evidence",
                "full_page_available": bool(full_page),
            }
    for key, document in grouped.items():
        if not str(document.get("text") or "").strip():
            document["text"] = "\n".join(snippets.get(key) or [])
            document["page_text"] = document["text"]
    return [document for document in grouped.values() if str(document.get("text") or "").strip()]


def select_development_rows(rows: list[dict[str, str]], size: int = 20) -> list[dict[str, str]]:
    """Preserve the historical 20/80 ID split for regression comparison only."""
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get("question_type") or "unknown", []).append(row)
    selected: list[dict[str, str]] = []
    ordered = sorted(groups.items())
    while len(selected) < size:
        added = False
        for _, group in ordered:
            group.sort(key=lambda item: item.get("financebench_id") or "")
            if group:
                selected.append(group.pop(0))
                added = True
                if len(selected) == size:
                    break
        if not added:
            break
    return selected


def _format_evidence(documents: list[dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(
        f"Source: {doc.get('filename') or 'Unknown'} | Page: {doc.get('page_number', 'N/A')}\n"
        f"{doc.get('text') or doc.get('page_text') or ''}"
        for doc in documents
    )


def _input_tokens(usage: dict[str, Any]) -> int:
    return int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)


def classify_failure(
    *,
    documents: list[dict[str, Any]],
    evidence: str,
    task_spec: dict[str, Any],
    coverage: dict[str, Any],
    calculation: dict[str, Any] | None,
    judge: dict[str, Any],
    error: str,
) -> str:
    if error:
        return "oracle_pipeline_error"
    if not documents:
        return "missing_gold_evidence"
    if not evidence.strip():
        return "empty_oracle_context"
    if judge.get("verdict") == "invalid_judge_output":
        return "invalid_judge_output"
    if int(judge.get("score") or 0) == 1:
        return "none"
    if not judge:
        return "not_judged"
    if task_spec.get("task_type") == "calculation" and calculation is None:
        return "gold_page_calculation_not_executed"
    if coverage.get("status") != "complete":
        return "gold_page_evidence_not_fully_interpreted"
    return "gold_page_answer_incorrect"


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    judged = [record for record in records if record.get("judge_result")]
    correct = sum(int((record.get("judge_result") or {}).get("score") or 0) for record in judged)
    return {
        "questions": len(records),
        "judged": len(judged),
        "correct": correct,
        "accuracy": round(correct / len(judged), 4) if judged else None,
        "task_types": dict(sorted(Counter(record.get("task_type") or "unknown" for record in records).items())),
        "failure_types": dict(sorted(Counter(record.get("failure_type") or "unknown" for record in records).items())),
        "calculation_questions": sum(bool(record.get("is_calculation")) for record in records),
        "calculations_executed": sum(bool(record.get("calculation")) for record in records),
        "evidence_frames": sum(int(record.get("evidence_frame_count") or 0) for record in records),
        "structured_calculations": sum(int(record.get("frames_used_for_execution") or 0) > 0 for record in records),
        "answerable": sum((record.get("coverage_dimensions") or {}).get("answerable") is True for record in records),
        "answer_input_tokens": sum(int(record.get("answer_input_tokens") or 0) for record in records),
        "answer_total_tokens": sum(int((record.get("usage") or {}).get("total_tokens") or 0) for record in records),
        "average_latency_ms": round(
            sum(float(record.get("latency_ms") or 0) for record in records) / len(records), 2
        ) if records else 0.0,
        "benchmark_status": "fixed_seen_regression",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EvidenceRAG answers using FinanceBench gold pages.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--mode", choices=("gold_page", "gold_evidence"), default="gold_page")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--judge-interval-seconds", type=float, default=0.0)
    parser.add_argument("--thinking", choices=("enabled", "disabled", "auto"), default="disabled")
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--evidence-frame", action="store_true", help="Load existing table records for gold pages.")
    parser.add_argument("--structured-executor", action="store_true", help="Use validated gold-page EvidenceFrames before text-row fallback.")
    parser.add_argument("--structured-coverage", action="store_true", help="Apply structured coverage dimensions to oracle evidence.")
    args = parser.parse_args()

    if args.structured_executor and not args.evidence_frame:
        parser.error("--structured-executor requires --evidence-frame.")
    if args.structured_coverage and not args.evidence_frame:
        parser.error("--structured-coverage requires --evidence-frame.")

    os.environ.update(
        {
            "FINANCE_POLICY_ENABLED": "false",
            "ANSWER_THINKING_MODE": args.thinking,
            "ANSWER_MAX_COMPLETION_TOKENS": str(max(64, args.max_completion_tokens)),
            "ANSWER_TEMPERATURE": "0",
            "EVIDENCE_FRAME_ENABLED": "true" if args.evidence_frame else "false",
            "STRUCTURED_EXECUTOR_ENABLED": "true" if args.structured_executor else "false",
            "STRUCTURED_COVERAGE_ENABLED": "true" if args.structured_coverage else "false",
        }
    )
    with args.dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    dev_ids = {row.get("financebench_id") or "" for row in select_development_rows(list(all_rows))}
    if args.split == "dev":
        rows = [row for row in all_rows if (row.get("financebench_id") or "") in dev_ids]
    elif args.split == "holdout":
        rows = [row for row in all_rows if (row.get("financebench_id") or "") not in dev_ids]
    else:
        rows = all_rows
    if args.question_id:
        requested = set(args.question_id)
        rows = [row for row in rows if (row.get("financebench_id") or "") in requested]
        missing = requested - {row.get("financebench_id") or "" for row in rows}
        if missing:
            parser.error(f"question IDs not found in selected split: {', '.join(sorted(missing))}")
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No FinanceBench rows selected.")

    output = args.output or ROOT / "reports" / f"financebench_oracle_{args.mode}_{args.split}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.resume and output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                completed.add(str(json.loads(line).get("financebench_id") or ""))
            except json.JSONDecodeError:
                continue
        rows = [row for row in rows if (row.get("financebench_id") or "") not in completed]

    sys.path.insert(0, str(BACKEND))
    from answer_generator import generate_answer
    from calculation_service import build_calculation_result, format_calculation_evidence
    from evidence_context import build_compact_evidence
    from evidence_coverage import assess_structured_coverage
    from query_parser import assess_required_field_coverage, build_answer_directives, parse_query
    from rag_orchestrator import _build_evidence_frames_for_documents

    judge_model = None
    judge_prompt = None
    judge_with_retry = None
    if not args.skip_judge:
        from judge_financebench_langsmith_experiment import JUDGE_PROMPT, _judge_model, _judge_with_retry

        judge_model = _judge_model()
        judge_prompt = JUDGE_PROMPT
        judge_with_retry = _judge_with_retry

    print(
        f"[setup] oracle={args.mode} split={args.split} questions={len(rows)} "
        f"judge={not args.skip_judge} benchmark=fixed_seen_regression",
        flush=True,
    )
    mode = "a" if args.resume else "w"
    with output.open(mode, encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            started = time.perf_counter()
            question = str(row.get("question") or "")
            reference = str(row.get("answer") or "")
            documents = parse_gold_pages(row, args.mode)
            task_spec = parse_query(question)
            coverage = assess_required_field_coverage(task_spec, documents)
            evidence_frames: list[dict[str, Any]] = []
            frame_trace: dict[str, Any] = {}
            if args.evidence_frame:
                evidence_frames, frame_trace = _build_evidence_frames_for_documents(
                    documents,
                    str(task_spec.get("company") or ""),
                )
            if args.structured_coverage:
                coverage = assess_structured_coverage(task_spec, documents, evidence_frames, coverage)
            calculation = build_calculation_result(
                task_spec,
                coverage,
                documents,
                evidence_frames=evidence_frames,
            )
            evidence, compression = build_compact_evidence(question, documents, task_spec, calculation)
            if not evidence:
                evidence = _format_evidence(documents)
            directives = build_answer_directives(question, task_spec)
            if directives:
                directive_text = "\n".join(f"- {directive}" for directive in directives)
                evidence = f"Question-specific answer contract (instructions, not evidence):\n{directive_text}\n\n---\n\n{evidence}"
            calculation_evidence = format_calculation_evidence(calculation)
            if calculation_evidence:
                evidence = f"{evidence}\n\n---\n\n{calculation_evidence}"

            answer = ""
            usage: dict[str, Any] = {}
            error = ""
            try:
                answer, usage = generate_answer(question, evidence, [], "")
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            judge_result: dict[str, Any] = {}
            if judge_model is not None and judge_with_retry is not None and answer:
                try:
                    judge_result = judge_with_retry(
                        judge_model,
                        judge_prompt.format(question=question, reference=reference, answer=answer),
                    )
                except Exception as exc:
                    judge_result = {
                        "score": 0,
                        "verdict": "invalid_judge_output",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
            failure_type = classify_failure(
                documents=documents,
                evidence=evidence,
                task_spec=task_spec,
                coverage=coverage,
                calculation=calculation,
                judge=judge_result,
                error=error,
            )
            record = {
                "financebench_id": str(row.get("financebench_id") or ""),
                "question": question,
                "reference_answer": reference,
                "answer": answer,
                "oracle_mode": args.mode,
                "gold_pages": [
                    {"filename": doc["filename"], "page_number": doc["page_number"]}
                    for doc in documents
                ],
                "task_type": str(task_spec.get("task_type") or "lookup"),
                "is_calculation": task_spec.get("task_type") == "calculation",
                "calculation": calculation,
                "oracle_path": "gold_page_evidence_frame" if args.evidence_frame else "gold_page_current_pipeline",
                "evidence_frame_count": len(evidence_frames),
                "frames_used_for_execution": len((calculation or {}).get("operand_evidence_ids") or []),
                "evidence_frame_trace": frame_trace,
                "coverage_status": coverage.get("status"),
                "coverage_dimensions": {
                    key: coverage.get(key)
                    for key in (
                        "page_supported", "row_supported", "period_supported", "unit_scale_supported",
                        "scope_supported", "operands_validated", "answerable",
                    )
                },
                "coverage_missing_fields": coverage.get("missing_fields") or [],
                "compression": compression,
                "answer_input_tokens": _input_tokens(usage),
                "usage": usage,
                "judge_result": judge_result,
                "failure_type": failure_type,
                "error": error,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            verdict = judge_result.get("verdict") or ("generated" if answer else "error")
            print(f"[{index:02d}/{len(rows)}] {record['financebench_id']}: {verdict} ({failure_type})", flush=True)
            if judge_model is not None and args.judge_interval_seconds > 0 and index < len(rows):
                time.sleep(max(0.0, args.judge_interval_seconds))

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = build_summary(records)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "dataset": str(args.dataset),
                "oracle_mode": args.mode,
                "split": args.split,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {output}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
