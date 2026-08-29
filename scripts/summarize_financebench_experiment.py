"""Create a compact, reproducible FinanceBench summary from answer and judge JSONL files."""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSONL file: {path}") from exc


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _normalize_document(value: object) -> str:
    return Path(str(value or "")).stem.casefold().strip()


def _gold_pages(dataset: Path) -> dict[str, set[tuple[str, int]]]:
    result: dict[str, set[tuple[str, int]]] = {}
    with dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            pages: set[tuple[str, int]] = set()
            try:
                evidence = json.loads(row.get("evidence") or "[]")
            except json.JSONDecodeError:
                evidence = []
            for item in evidence if isinstance(evidence, list) else [evidence]:
                try:
                    pages.add((_normalize_document(item.get("doc_name")), int(float(item.get("evidence_page_num")))))
                except (AttributeError, TypeError, ValueError):
                    continue
            result[str(row.get("financebench_id") or "")] = pages
    return result


def _page_keys(items: list[dict]) -> set[tuple[str, int]]:
    result = set()
    for item in items or []:
        try:
            result.add((_normalize_document(item.get("filename")), int(item.get("page_number"))))
        except (AttributeError, TypeError, ValueError):
            continue
    return result


def _summarize_split(
    label: str,
    answers_path: Path,
    judge_path: Path,
    gold_by_id: dict[str, set[tuple[str, int]]],
) -> tuple[dict, set[str]]:
    answers = _read_jsonl(answers_path)
    judges = _read_jsonl(judge_path)
    judges_by_run = {str(item.get("run_id") or ""): item for item in judges}
    judges_by_id = {str(item.get("financebench_id") or ""): item for item in judges}
    answer_ids = [str(item.get("financebench_id") or "") for item in answers]
    if not all(answer_ids) or len(answer_ids) != len(set(answer_ids)):
        raise SystemExit(f"{label}: missing or duplicate financebench_id in answer file.")
    matched = [
        judges_by_run.get(str(item.get("langsmith_trace_id") or item.get("evaluation_run_id") or ""))
        or judges_by_id.get(str(item.get("financebench_id") or ""))
        for item in answers
    ]
    if any(item is None for item in matched):
        raise SystemExit(f"{label}: not every answer has a matching Judge run_id or financebench_id.")

    traces = [item.get("rag_trace") or {} for item in answers]
    usages = [item.get("usage") or {} for item in answers]
    rerank_fallback_reasons = Counter(
        str(trace.get("fallback_reason") or (trace.get("rerank_trace") or {}).get("fallback_reason") or "unknown")
        for trace in traces
        if trace.get("rerank_fallback_used")
    )
    correct = sum(int(item.get("score") or 0) for item in matched if item)
    task_totals: Counter = Counter()
    task_correct: Counter = Counter()
    candidate_page_hits = context_page_hits = candidate_context_losses = 0
    citation_page_hits = 0
    for answer, judge, trace in zip(answers, matched, traces):
        task_type = str(trace.get("task_type") or (trace.get("evidence_coverage") or {}).get("task_type") or "unknown")
        task_totals[task_type] += 1
        task_correct[task_type] += int((judge or {}).get("score") or 0)
        gold = gold_by_id.get(str(answer.get("financebench_id") or ""), set())
        candidates = _page_keys(trace.get("page_first_selected_pages") or trace.get("page_stage_candidates") or [])
        context = _page_keys(trace.get("answer_context_pages") or answer.get("citations") or [])
        citations = _page_keys(answer.get("citations") or [])
        candidate_hit = bool(gold & candidates)
        context_hit = bool(gold & context)
        candidate_page_hits += candidate_hit
        context_page_hits += context_hit
        citation_page_hits += bool(gold & citations)
        candidate_context_losses += candidate_hit and not context_hit

    task_metrics = {
        task: {
            "questions": task_totals[task],
            "correct": task_correct[task],
            "accuracy": round(task_correct[task] / task_totals[task], 4),
        }
        for task in sorted(task_totals)
    }
    latencies = [float((item.get("evaluation_latency") or {}).get("total_ms") or 0) for item in answers]
    gate_names = (
        "frame_matched", "measure_validated", "period_validated",
        "metadata_validated", "operand_unique", "operation_validated",
    )
    flow_stages = Counter(str(trace.get("evidence_flow_stage") or "unknown") for trace in traces)
    return (
        {
            "answers": len(answers),
            "judge_records": len(judges),
            "correct": correct,
            "accuracy": round(correct / len(answers), 4) if answers else 0.0,
            "task_types": task_metrics,
            "invalid_judge_outputs": sum(item.get("verdict") == "invalid_judge_output" for item in matched if item),
            "empty_retrievals": sum(trace.get("rrf_fused_candidate_count") == 0 for trace in traces),
            "rerank_providers": dict(sorted(Counter(trace.get("rerank_provider") or "unknown" for trace in traces).items())),
            "remote_rerank_successes": sum(trace.get("rerank_provider") == "remote" for trace in traces),
            "remote_rerank_result_uses": sum(bool(trace.get("remote_success")) for trace in traces),
            "remote_rerank_attempts": sum(int(trace.get("remote_attempt_count") or 0) for trace in traces),
            "rerank_cache_hits": sum(bool(trace.get("rerank_cache_hit")) for trace in traces),
            "local_fallbacks": sum(bool(trace.get("rerank_fallback_used")) for trace in traces),
            "rerank_fallback_reasons": dict(sorted(rerank_fallback_reasons.items())),
            "remote_rerank_input_chars": sum(int(trace.get("remote_rerank_input_chars") or 0) for trace in traces),
            "answer_input_tokens": sum(int(usage.get("input_tokens") or 0) for usage in usages),
            "answer_output_tokens": sum(int(usage.get("output_tokens") or 0) for usage in usages),
            "answer_total_tokens": sum(int(usage.get("total_tokens") or 0) for usage in usages),
            "average_answer_tokens": round(sum(int(usage.get("total_tokens") or 0) for usage in usages) / len(answers), 2) if answers else 0.0,
            "average_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "candidate_gold_page_hits": candidate_page_hits,
            "candidate_gold_page_hit_rate": round(candidate_page_hits / len(answers), 4) if answers else 0.0,
            "context_gold_page_hits": context_page_hits,
            "context_gold_page_hit_rate": round(context_page_hits / len(answers), 4) if answers else 0.0,
            "citation_gold_page_hits": citation_page_hits,
            "candidate_to_context_losses": candidate_context_losses,
            "evidence_frame_questions": sum(int(trace.get("evidence_frame_count") or 0) > 0 for trace in traces),
            "queryspec_related_frame_questions": sum(int(trace.get("relevant_frame_count") or 0) > 0 for trace in traces),
            "structured_answerable_questions": sum(bool((trace.get("evidence_coverage") or {}).get("structured_answerable")) for trace in traces),
            "structured_execution_ready_questions": sum(bool((trace.get("evidence_coverage") or {}).get("structured_execution_ready")) for trace in traces),
            "structured_executions": sum(int(trace.get("frames_used_for_execution") or 0) > 0 for trace in traces),
            "execution_gate_funnel": {
                gate: sum(bool(((trace.get("evidence_coverage") or {}).get("structured_gate_trace") or {}).get(gate)) for trace in traces)
                for gate in gate_names
            },
            "answer_consistency_checked": sum(bool((trace.get("answer_consistency") or {}).get("checked")) for trace in traces),
            "answer_consistency_repaired": sum(bool((trace.get("answer_consistency") or {}).get("repaired")) for trace in traces),
            "candidate_coverage_complete": sum((trace.get("candidate_coverage") or {}).get("status") == "complete" for trace in traces),
            "selected_page_coverage_complete": sum((trace.get("selected_page_coverage") or {}).get("status") == "complete" for trace in traces),
            "compact_context_coverage_complete": sum((trace.get("compact_context_coverage") or {}).get("status") == "complete" for trace in traces),
            "candidate_complete_selected_incomplete": sum(
                (trace.get("candidate_coverage") or {}).get("status") == "complete"
                and (trace.get("selected_page_coverage") or {}).get("status") != "complete"
                for trace in traces
            ),
            "candidate_complete_selected_incomplete_before_protection": sum(
                (trace.get("candidate_coverage") or {}).get("status") == "complete"
                and (trace.get("protected_page_coverage_before") or {}).get("status") != "complete"
                for trace in traces
            ),
            "evidence_flow_stages": dict(sorted(flow_stages.items())),
            "protected_page_questions": sum(int(trace.get("protected_page_count") or 0) > 0 for trace in traces),
            "protected_page_count": sum(int(trace.get("protected_page_count") or 0) for trace in traces),
            "protected_page_budget_violations": sum(
                int(trace.get("selected_page_count_after_protection") or 0)
                > int(trace.get("selected_page_count_before_protection") or 0)
                for trace in traces
            ),
            "numeric_display_eligible": sum(bool((trace.get("numeric_display_validation") or {}).get("eligible")) for trace in traces),
            "numeric_display_repaired": sum(bool((trace.get("numeric_display_validation") or {}).get("repaired")) for trace in traces),
            "answer_facet_contract_questions": sum(bool((trace.get("answer_facet_validation") or {}).get("required_facets")) for trace in traces),
            "answer_facet_omission_questions": sum(bool((trace.get("answer_facet_validation") or {}).get("missing_facets")) for trace in traces),
            "protected_evidence_questions": sum(bool(trace.get("answer_context_protected_evidence")) for trace in traces),
            "dropped_protected_evidence": sum(len(trace.get("answer_context_dropped_protected_evidence") or []) for trace in traces),
            "supplemental_triggered": sum(bool(trace.get("supplemental_triggered")) for trace in traces),
            "supplemental_effective": sum(bool(trace.get("supplemental_effective")) for trace in traces),
            "supplemental_recovered": sum(
                bool(trace.get("supplemental_triggered"))
                and not bool((trace.get("coverage_before") or {}).get("answerable"))
                and bool((trace.get("coverage_after") or {}).get("answerable"))
                for trace in traces
            ),
            "answer_file": _display_path(answers_path),
            "judge_file": _display_path(judge_path),
        },
        set(answer_ids),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed FinanceBench answer and Judge JSONL files.")
    parser.add_argument(
        "--split",
        action="append",
        nargs=3,
        metavar=("LABEL", "ANSWERS_JSONL", "JUDGE_JSONL"),
        required=True,
        help="Repeat for each split to combine.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--oracle-summary", type=Path, help="Optional Oracle summary JSON for oracle-gap reporting.")
    parser.add_argument("--baseline-summary", type=Path, help="Optional prior summary JSON (for example v14) for deltas.")
    parser.add_argument("--markdown-output", type=Path, help="Defaults to the JSON output path with .md suffix.")
    args = parser.parse_args()

    splits: dict[str, dict] = {}
    all_ids: set[str] = set()
    gold_by_id = _gold_pages(args.dataset)
    for label, answers_name, judge_name in args.split:
        if label in splits:
            raise SystemExit(f"Duplicate split label: {label}")
        summary, ids = _summarize_split(label, ROOT / answers_name, ROOT / judge_name, gold_by_id)
        overlap = all_ids & ids
        if overlap:
            raise SystemExit(f"Duplicate FinanceBench IDs across splits: {sorted(overlap)[:3]}")
        splits[label] = summary
        all_ids.update(ids)

    total = sum(item["answers"] for item in splits.values())
    correct = sum(item["correct"] for item in splits.values())
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_status": "fixed_seen_regression",
        "splits": splits,
        "combined": {
            "questions": total,
            "unique_financebench_ids": len(all_ids),
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "empty_retrievals": sum(item["empty_retrievals"] for item in splits.values()),
            "remote_rerank_input_chars": sum(item["remote_rerank_input_chars"] for item in splits.values()),
            "remote_rerank_successes": sum(item["remote_rerank_successes"] for item in splits.values()),
            "remote_rerank_result_uses": sum(item["remote_rerank_result_uses"] for item in splits.values()),
            "remote_rerank_attempts": sum(item["remote_rerank_attempts"] for item in splits.values()),
            "rerank_cache_hits": sum(item["rerank_cache_hits"] for item in splits.values()),
            "local_fallbacks": sum(item["local_fallbacks"] for item in splits.values()),
            "answer_input_tokens": sum(item["answer_input_tokens"] for item in splits.values()),
            "answer_output_tokens": sum(item["answer_output_tokens"] for item in splits.values()),
            "answer_total_tokens": sum(item["answer_total_tokens"] for item in splits.values()),
            "candidate_gold_page_hits": sum(item["candidate_gold_page_hits"] for item in splits.values()),
            "context_gold_page_hits": sum(item["context_gold_page_hits"] for item in splits.values()),
            "citation_gold_page_hits": sum(item["citation_gold_page_hits"] for item in splits.values()),
            "candidate_to_context_losses": sum(item["candidate_to_context_losses"] for item in splits.values()),
            "evidence_frame_questions": sum(item["evidence_frame_questions"] for item in splits.values()),
            "queryspec_related_frame_questions": sum(item["queryspec_related_frame_questions"] for item in splits.values()),
            "structured_answerable_questions": sum(item["structured_answerable_questions"] for item in splits.values()),
            "structured_execution_ready_questions": sum(item["structured_execution_ready_questions"] for item in splits.values()),
            "structured_executions": sum(item["structured_executions"] for item in splits.values()),
            "answer_consistency_checked": sum(item["answer_consistency_checked"] for item in splits.values()),
            "answer_consistency_repaired": sum(item["answer_consistency_repaired"] for item in splits.values()),
            "candidate_coverage_complete": sum(item["candidate_coverage_complete"] for item in splits.values()),
            "selected_page_coverage_complete": sum(item["selected_page_coverage_complete"] for item in splits.values()),
            "compact_context_coverage_complete": sum(item["compact_context_coverage_complete"] for item in splits.values()),
            "candidate_complete_selected_incomplete": sum(item["candidate_complete_selected_incomplete"] for item in splits.values()),
            "candidate_complete_selected_incomplete_before_protection": sum(
                item["candidate_complete_selected_incomplete_before_protection"] for item in splits.values()
            ),
            "protected_page_questions": sum(item["protected_page_questions"] for item in splits.values()),
            "protected_page_count": sum(item["protected_page_count"] for item in splits.values()),
            "protected_page_budget_violations": sum(item["protected_page_budget_violations"] for item in splits.values()),
            "numeric_display_eligible": sum(item["numeric_display_eligible"] for item in splits.values()),
            "numeric_display_repaired": sum(item["numeric_display_repaired"] for item in splits.values()),
            "answer_facet_contract_questions": sum(item["answer_facet_contract_questions"] for item in splits.values()),
            "answer_facet_omission_questions": sum(item["answer_facet_omission_questions"] for item in splits.values()),
            "protected_evidence_questions": sum(item["protected_evidence_questions"] for item in splits.values()),
            "dropped_protected_evidence": sum(item["dropped_protected_evidence"] for item in splits.values()),
            "supplemental_triggered": sum(item["supplemental_triggered"] for item in splits.values()),
            "supplemental_effective": sum(item["supplemental_effective"] for item in splits.values()),
            "supplemental_recovered": sum(item["supplemental_recovered"] for item in splits.values()),
        },
    }
    combined = payload["combined"]
    combined["candidate_gold_page_hit_rate"] = round(combined["candidate_gold_page_hits"] / total, 4) if total else 0.0
    combined["context_gold_page_hit_rate"] = round(combined["context_gold_page_hits"] / total, 4) if total else 0.0
    combined["average_answer_tokens"] = round(combined["answer_total_tokens"] / total, 2) if total else 0.0
    combined["average_latency_ms"] = round(
        sum(item["average_latency_ms"] * item["answers"] for item in splits.values()) / total,
        2,
    ) if total else 0.0
    if args.oracle_summary:
        oracle_payload = json.loads(args.oracle_summary.read_text(encoding="utf-8"))
        oracle = oracle_payload.get("summary", oracle_payload)
        combined["oracle_accuracy"] = oracle.get("accuracy")
        combined["oracle_gap"] = (
            round(float(oracle["accuracy"]) - combined["accuracy"], 4)
            if oracle.get("accuracy") is not None else None
        )
    if args.baseline_summary:
        baseline_payload = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
        baseline = baseline_payload.get("combined", baseline_payload)
        baseline_questions = int(baseline.get("questions") or baseline.get("unique_financebench_ids") or 0)
        comparable = not baseline_questions or baseline_questions == total
        payload["baseline_comparison"] = {
            "baseline_file": str(args.baseline_summary),
            "comparable_question_count": comparable,
            "baseline_questions": baseline_questions or None,
            "accuracy_delta": round(combined["accuracy"] - float(baseline.get("accuracy") or 0), 4) if comparable else None,
            "answer_tokens_delta": combined["answer_total_tokens"] - int(baseline.get("answer_total_tokens") or 0) if comparable else None,
            "average_latency_ms_delta": (
                round(combined["average_latency_ms"] - float(baseline.get("average_latency_ms") or 0), 2)
                if comparable and baseline.get("average_latency_ms") is not None else None
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_output = args.markdown_output or args.output.with_suffix(".md")
    lines = [
        "# EvidenceRAG FinanceBench 固定回归报告",
        "",
        "> 本数据集 100 题均已被查看；本报告只表示 fixed seen regression，不代表未见数据泛化能力。",
        "",
        f"- Questions: {total}",
        f"- Accuracy: {combined['accuracy']:.2%}",
        f"- Candidate gold-page hit: {combined['candidate_gold_page_hit_rate']:.2%}",
        f"- Context gold-page hit: {combined['context_gold_page_hit_rate']:.2%}",
        f"- Candidate → context losses: {combined['candidate_to_context_losses']}",
        f"- Oracle gap: {combined.get('oracle_gap', 'N/A')}",
        f"- Answer tokens: {combined['answer_total_tokens']} (avg {combined['average_answer_tokens']})",
        f"- Average latency: {combined['average_latency_ms']} ms",
        f"- Jina input chars: {combined['remote_rerank_input_chars']}",
        f"- Remote rerank attempts / cache hits / local fallbacks: {combined['remote_rerank_attempts']} / {combined['rerank_cache_hits']} / {combined['local_fallbacks']}",
        f"- EvidenceFrame questions / structured executions: {combined['evidence_frame_questions']} / {combined['structured_executions']}",
        f"- QuerySpec-related frames / structured answerable / execution ready: {combined['queryspec_related_frame_questions']} / {combined['structured_answerable_questions']} / {combined['structured_execution_ready_questions']}",
        f"- Consistency checked / repaired: {combined['answer_consistency_checked']} / {combined['answer_consistency_repaired']}",
        f"- Candidate / selected-page / compact coverage complete: {combined['candidate_coverage_complete']} / {combined['selected_page_coverage_complete']} / {combined['compact_context_coverage_complete']}",
        f"- Candidate-complete → selected-incomplete: {combined['candidate_complete_selected_incomplete']}",
        f"- Candidate-complete → selected-incomplete before protection: {combined['candidate_complete_selected_incomplete_before_protection']}",
        f"- Protected page questions / pages / budget violations: {combined['protected_page_questions']} / {combined['protected_page_count']} / {combined['protected_page_budget_violations']}",
        f"- Numeric display eligible / repaired: {combined['numeric_display_eligible']} / {combined['numeric_display_repaired']}",
        f"- Answer facet contracts / omissions: {combined['answer_facet_contract_questions']} / {combined['answer_facet_omission_questions']}",
        f"- Protected evidence questions / dropped protected units: {combined['protected_evidence_questions']} / {combined['dropped_protected_evidence']}",
        f"- Supplemental triggered / effective / recovered: {combined['supplemental_triggered']} / {combined['supplemental_effective']} / {combined['supplemental_recovered']}",
        "",
        "## Split results",
        "",
        "| Split | Questions | Correct | Accuracy | Candidate page hit | Context page hit |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, item in splits.items():
        lines.append(
            f"| {label} | {item['answers']} | {item['correct']} | {item['accuracy']:.2%} | "
            f"{item['candidate_gold_page_hit_rate']:.2%} | {item['context_gold_page_hit_rate']:.2%} |"
        )
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["combined"], ensure_ascii=False))
    print(f"Summary: {args.output}")
    print(f"Markdown: {markdown_output}")


if __name__ == "__main__":
    main()
