"""Run a controlled FinanceBench experiment locally or, when explicit, in LangSmith."""

import argparse
import csv
import faulthandler
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DATA_PATH = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
load_dotenv(ROOT / ".env", override=True)
sys.path.insert(0, str(BACKEND))

from runtime_profile import apply_runtime_profile, feature_state, print_feature_summary


def _development_ids(rows: list[dict[str, str]], size: int = 20) -> set[str]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get("question_type") or "unknown", []).append(row)
    selected: list[dict[str, str]] = []
    while len(selected) < size:
        added = False
        for _, group in sorted(groups.items()):
            group.sort(key=lambda item: item.get("financebench_id") or "")
            if group:
                selected.append(group.pop(0))
                added = True
                if len(selected) == size:
                    break
        if not added:
            break
    return {row.get("financebench_id") or "" for row in selected}


def _configure_static_baseline(
    rag_profile: str,
    max_completion_tokens: int,
    thinking_mode: str,
    diagnose: bool,
    field_aware: bool,
    enable_rerank: bool,
    local_rerank_fallback: bool,
    finance_policy: bool,
    evidence_frame: bool,
    structured_executor: bool,
    structured_coverage: bool,
    frame_alignment: bool,
    structured_task_executor: bool,
    answer_consistency_validator: bool,
    protected_evidence_slots: bool,
    stage_aware_coverage: bool,
    protected_page_slots: bool,
    numeric_display_validator: bool,
    answer_required_facets: bool,
    explicit_formula_advisory: bool,
    supplemental_find: bool,
) -> None:
    """Freeze the baseline before importing any backend module."""
    settings = {
            "RAG_QUERY_PLANNER_ENABLED": "false",
            "FINANCE_RAG_ENABLE_STEP_BACK": "false",
            "RAG_EXECUTION_MODE": "static",
            "RAG_PROFILE": rag_profile,
            "FINANCE_RAG_CANDIDATE_K": "40",
            "FINANCE_RAG_FINAL_TOP_K": "5",
            "RAG_PAGE_FIRST_ENABLED": "true",
            "RAG_FIELD_AWARE_ENABLED": "true" if field_aware else "false",
            "RAG_PAGE_NEIGHBOR_WINDOW": "2" if field_aware else "0",
            "RAG_CONTEXT_PAGE_WINDOW": "2" if field_aware else "0",
            "RAG_SUPPLEMENTAL_SEARCH_ENABLED": "true" if field_aware else "false",
            "RAG_SUPPLEMENTAL_CANDIDATE_K": "12",
            "FINANCE_RAG_W_STATEMENT": "0.18",
            "FINANCE_RAG_W_REQUIRED_FIELDS": "0.25",
            "FINANCE_RAG_W_REQUIRED_PERIODS": "0.20",
            "FINANCE_RAG_W_SELECTION_SCOPE": "0.30",
            "RAG_ANCHOR_GUARD_ENABLED": "false",
            "RAG_COVER_FILTER_ENABLED": "false",
            "FINANCE_RAG_ENABLE_PAGE_MERGE": "false",
            "FINANCE_RAG_ADJACENT_PAGE_WINDOW": "0",
            "FINANCE_RAG_ADJACENT_CHUNK_WINDOW": "0",
            "AUTO_MERGE_ENABLED": "false",
            "TABLE_AWARE_RETRIEVAL": "off",
            "RAG_EVIDENCE_GROUPING_ENABLED": "false",
            "ANSWER_MAX_COMPLETION_TOKENS": str(max_completion_tokens),
            "ANSWER_THINKING_MODE": thinking_mode,
            "ANSWER_TEMPERATURE": "0",
            "RAG_RETRIEVAL_DEBUG": "true" if diagnose else "false",
            "FINANCE_POLICY_ENABLED": "true" if finance_policy else "false",
            "EVIDENCE_FRAME_ENABLED": "true" if evidence_frame else "false",
            "STRUCTURED_EXECUTOR_ENABLED": "true" if structured_executor else "false",
            "STRUCTURED_COVERAGE_ENABLED": "true" if structured_coverage else "false",
            "FRAME_ALIGNMENT_ENABLED": "true" if frame_alignment else "false",
            "STRUCTURED_COVERAGE_ADVISORY_ENABLED": "true",
            "STRUCTURED_TASK_EXECUTOR_ENABLED": "true" if structured_task_executor else "false",
            "ANSWER_CONSISTENCY_VALIDATOR_ENABLED": "true" if answer_consistency_validator else "false",
            "RAG_PROTECTED_EVIDENCE_SLOTS_ENABLED": "true" if protected_evidence_slots else "false",
            "STAGE_AWARE_COVERAGE_ENABLED": "true" if stage_aware_coverage else "false",
            "PROTECTED_PAGE_SLOTS_ENABLED": "true" if protected_page_slots else "false",
            "NUMERIC_DISPLAY_VALIDATOR_ENABLED": "true" if numeric_display_validator else "false",
            "ANSWER_REQUIRED_FACETS_ENABLED": "true" if answer_required_facets else "false",
            "EXPLICIT_FORMULA_ADVISORY_ENABLED": "true" if explicit_formula_advisory else "false",
            "SUPPLEMENTAL_FIND_ENABLED": "true" if supplemental_find else "false",
    }
    if not enable_rerank:
        settings.update(
            {
                "RERANK_MODEL": "",
                "RERANK_BINDING_HOST": "",
                "RERANK_API_KEY": "",
                "LOCAL_RERANK_ENABLED": "false",
            }
        )
    elif local_rerank_fallback:
        settings["LOCAL_RERANK_ENABLED"] = "true"
    else:
        settings["LOCAL_RERANK_ENABLED"] = "false"
    os.environ.update(settings)
    apply_runtime_profile(rag_profile)


def _normalized_document_name(value: object) -> str:
    name = Path(str(value or "")).stem
    return name.strip().casefold()


def _example_financebench_id(example) -> str:
    if isinstance(example, dict):
        return str(example.get("financebench_id") or "")
    return str((getattr(example, "metadata", None) or {}).get("financebench_id") or "")


def citation_document_hit(run, example) -> dict:
    outputs = getattr(run, "outputs", None) or {}
    metadata = getattr(example, "metadata", None) or {}
    cited = {
        _normalized_document_name(item.get("filename"))
        for item in (outputs.get("citations") or [])
        if isinstance(item, dict)
    }
    gold = {
        _normalized_document_name(item.get("doc_name"))
        for item in (metadata.get("gold_evidence") or [])
        if isinstance(item, dict)
    }
    matched = sorted(cited & gold)
    return {
        "key": "citation_document_hit",
        "score": bool(matched),
        "comment": f"matched documents: {', '.join(matched) if matched else 'none'}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a formal EvidenceRAG FinanceBench static experiment")
    parser.add_argument(
        "--evaluation-backend",
        choices=("local", "langsmith"),
        default=os.getenv("FINANCEBENCH_EVALUATION_BACKEND", "local").strip().lower(),
        help="Evaluation storage backend. Local is the default and never accesses LangSmith.",
    )
    parser.add_argument("--dataset-name", default="evidencerag_financebench_all100_v1")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the whole selected split.")
    parser.add_argument(
        "--question-id",
        action="append",
        default=[],
        help="Evaluate only this FinanceBench ID; repeat the option for a targeted set.",
    )
    parser.add_argument("--experiment-prefix", default="evidencerag-finance-static")
    parser.add_argument(
        "--rag-profile",
        choices=("finance", "clean_baseline", "clean_baseline_formula_skill"),
        default="finance",
        help="Runtime profile. clean_baseline authoritatively disables all experimental answer paths.",
    )
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--thinking", choices=("disabled", "auto", "enabled"), default="disabled")
    parser.add_argument(
        "--finance-policy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add the local Financial Task Policy to answer prompts (default: disabled for v14 compatibility).",
    )
    parser.add_argument("--evidence-frame", action="store_true", help="Enable EvidenceFrame Lite table adaptation.")
    parser.add_argument("--structured-executor", action="store_true", help="Prefer the audited EvidenceFrame Decimal executor.")
    parser.add_argument("--structured-coverage", action="store_true", help="Use structured answerability coverage for numeric tasks.")
    parser.add_argument("--frame-alignment", action="store_true", help="Enable layered QuerySpec-to-EvidenceFrame matching.")
    parser.add_argument("--structured-task-executor", action="store_true", help="Execute high-confidence comparison and selection matrices.")
    parser.add_argument("--answer-consistency-validator", action="store_true", help="Repair answer conflicts with authoritative local results.")
    parser.add_argument("--protected-evidence-slots", action="store_true", help="Reserve existing context slots for operands, periods, and candidate matrices.")
    parser.add_argument("--stage-aware-coverage", action="store_true", help="Trace QuerySpec coverage across candidate, selected-page, and compact-context stages.")
    parser.add_argument("--protected-page-slots", action="store_true", help="Replace low-value selected pages with requirement-complete candidate pages without growing the page budget.")
    parser.add_argument("--numeric-display-validator", action="store_true", help="Repair only deterministic explicit rounding and unit-display errors.")
    parser.add_argument("--answer-required-facets", action="store_true", help="Add and validate a concise question-derived answer facet contract.")
    parser.add_argument(
        "--explicit-formula-advisory",
        action="store_true",
        help="Use question-defined formula operands for coverage and page protection only.",
    )
    parser.add_argument("--supplemental-find", action="store_true", help="Allow one document-scoped retrieval only for partial evidence.")
    parser.add_argument("--diagnose", action="store_true", help="Print retrieval and generation stage timing.")
    parser.add_argument(
        "--field-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic finance rewrite and required-field coverage tracing (default: enabled).",
    )
    parser.add_argument(
        "--enable-rerank",
        action="store_true",
        help="Use the configured remote reranker and preserve a local reranker fallback.",
    )
    parser.add_argument(
        "--disable-local-rerank-fallback",
        action="store_true",
        help="With --enable-rerank, do not fall back to the local cross-encoder after a remote failure.",
    )
    parser.add_argument("--slow-question-seconds", type=int, default=90, help="Dump a stack trace after this many seconds per question (0 disables it).")
    parser.add_argument("--output", type=Path, help="Write each completed answer, citations, and trace to JSONL.")
    parser.add_argument(
        "--skip-auto-judge",
        action="store_true",
        help="Do not run the configured FinanceBench judge after a successful experiment.",
    )
    parser.add_argument(
        "--judge-output",
        type=Path,
        help="Optional JSONL destination for automatic judge results.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip completed FinanceBench IDs in --output.")
    parser.add_argument(
        "--retry-empty-retrieval-from",
        type=Path,
        help="Run only IDs whose prior JSONL trace had zero fused retrieval candidates.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=2.0,
        help="One-time wait before retrying a failed zero-candidate retrieval.",
    )
    args = parser.parse_args()

    if args.evaluation_backend == "local":
        os.environ.update({
            "LANGSMITH_TRACING": "false",
            "LANGSMITH_TRACING_V2": "false",
            "LANGCHAIN_TRACING_V2": "false",
        })

    if args.structured_executor and not args.evidence_frame:
        parser.error("--structured-executor requires --evidence-frame.")
    if args.structured_coverage and not args.evidence_frame:
        parser.error("--structured-coverage requires --evidence-frame.")
    if args.frame_alignment and not args.evidence_frame:
        parser.error("--frame-alignment requires --evidence-frame.")
    if args.structured_task_executor and not (args.evidence_frame and args.frame_alignment and args.structured_coverage):
        parser.error("--structured-task-executor requires --evidence-frame, --frame-alignment, and --structured-coverage.")
    if args.answer_consistency_validator and not (args.structured_executor or args.structured_task_executor):
        parser.error("--answer-consistency-validator requires a structured executor.")
    if args.supplemental_find and not args.structured_coverage:
        parser.error("--supplemental-find requires --structured-coverage.")

    _configure_static_baseline(
        args.rag_profile,
        args.max_completion_tokens,
        args.thinking,
        args.diagnose,
        args.field_aware,
        args.enable_rerank,
        not args.disable_local_rerank_fallback,
        args.finance_policy,
        args.evidence_frame,
        args.structured_executor,
        args.structured_coverage,
        args.frame_alignment,
        args.structured_task_executor,
        args.answer_consistency_validator,
        args.protected_evidence_slots,
        args.stage_aware_coverage,
        args.protected_page_slots,
        args.numeric_display_validator,
        args.answer_required_facets,
        args.explicit_formula_advisory,
        args.supplemental_find,
    )
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    id_by_question = {row.get("question", ""): row.get("financebench_id", "") for row in source_rows}
    dev_ids = _development_ids(source_rows)
    client = None
    if args.evaluation_backend == "local":
        examples = list(source_rows)
    else:
        from langsmith import Client

        client = Client()
        try:
            examples = list(client.list_examples(dataset_name=args.dataset_name))
        except Exception as exc:
            raise SystemExit(
                "LangSmith is unavailable. Use --evaluation-backend local or check the LangSmith account."
            ) from exc
    if args.split != "all":
        use_dev = args.split == "dev"
        examples = [
            example
            for example in examples
            if (_example_financebench_id(example) in dev_ids) == use_dev
        ]
    if args.question_id:
        requested_ids = set(args.question_id)
        examples = [
            example
            for example in examples
            if _example_financebench_id(example) in requested_ids
        ]
        selected_ids = {_example_financebench_id(example) for example in examples}
        missing_ids = requested_ids - selected_ids
        if missing_ids:
            parser.error(f"question IDs not found in selected split: {', '.join(sorted(missing_ids))}")
    if args.limit > 0:
        examples = examples[: args.limit]
    if args.retry_empty_retrieval_from:
        if not args.retry_empty_retrieval_from.exists():
            raise SystemExit(f"Retry source does not exist: {args.retry_empty_retrieval_from}")
        retry_ids = set()
        for line in args.retry_empty_retrieval_from.read_text(encoding="utf-8").splitlines():
            try:
                prior = json.loads(line)
            except json.JSONDecodeError:
                continue
            trace = prior.get("rag_trace") or {}
            if trace.get("rrf_fused_candidate_count") == 0:
                retry_ids.add(prior.get("financebench_id"))
        examples = [
            example
            for example in examples
            if _example_financebench_id(example) in retry_ids
        ]
    if args.resume:
        if not args.output:
            raise SystemExit("--resume requires --output.")
        completed_ids = set()
        if args.output.exists():
            for line in args.output.read_text(encoding="utf-8").splitlines():
                try:
                    completed_ids.add(json.loads(line).get("financebench_id"))
                except json.JSONDecodeError:
                    continue
            examples = [
                example
                for example in examples
                if _example_financebench_id(example) not in completed_ids
            ]
    if not examples:
        raise SystemExit("No FinanceBench examples selected. Check the split and question IDs.")
    if args.evaluation_backend == "local" and args.output is None:
        args.output = ROOT / "reports" / f"{args.experiment_prefix}_answers.jsonl"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("a" if args.resume else "w", encoding="utf-8")
    else:
        output_handle = None

    # Do not load the local embedding model until the remote experiment service is reachable.
    from answer_generator import generate_answer
    from calculation_service import validate_numeric_display, validate_or_repair_structured_answer
    from query_parser import assess_answer_facets
    from rag_orchestrator import prepare_rag_response
    completed_run_ids: set[str] = set()

    if args.evaluation_backend == "langsmith":
        from langsmith.run_helpers import get_current_run_tree, traceable
    else:
        def traceable(*_args, **_kwargs):
            def decorator(function):
                return function
            return decorator

        def get_current_run_tree():
            return None

    @traceable(name="EvidenceRAG.retrieve", run_type="retriever")
    def retrieve(question: str) -> dict:
        return prepare_rag_response(question, profile=args.rag_profile, mode="static")

    def failed_empty_retrieval(prepared: dict) -> bool:
        trace = prepared.get("rag_trace") or {}
        route_counts = trace.get("per_query_retrieval_counts") or []
        return trace.get("rrf_fused_candidate_count") == 0 and any(
            entry.get("retrieval_mode") == "failed"
            for entry in route_counts
            if isinstance(entry, dict)
        )

    @traceable(name="EvidenceRAG.generate", run_type="llm")
    def generate(question: str, evidence: str, task_policy: str) -> tuple[str, dict]:
        return generate_answer(question, evidence, [], task_policy, args.rag_profile)

    @traceable(name="EvidenceRAG.financebench_static", run_type="chain")
    def target(inputs: dict) -> dict:
        question = str(inputs.get("question") or "")
        financebench_id = id_by_question.get(question, "")
        question_started = time.perf_counter()
        started = time.perf_counter()
        generation_seconds = 0.0
        if args.diagnose:
            print(f"[question] {financebench_id} retrieve starting", flush=True)
        if args.slow_question_seconds > 0:
            faulthandler.dump_traceback_later(args.slow_question_seconds, repeat=False)
        try:
            prepared = retrieve(question)
        finally:
            if args.slow_question_seconds > 0:
                faulthandler.cancel_dump_traceback_later()
        if failed_empty_retrieval(prepared):
            delay = max(0.0, args.retry_delay_seconds)
            print(
                f"[question] {financebench_id} retrieval failed with zero candidates; retrying after {delay:.1f}s",
                flush=True,
            )
            if delay:
                time.sleep(delay)
            prepared = retrieve(question)
        retrieval_seconds = time.perf_counter() - started
        if args.diagnose:
            print(f"[question] {financebench_id} retrieve finished in {retrieval_seconds:.2f}s", flush=True)
        if prepared.get("skill_applied"):
            answer, usage = str(prepared.get("skill_answer") or ""), {}
        elif prepared["evidence_status"] == "insufficient":
            answer, usage = "未检索到足够证据，无法基于当前知识库可靠回答。", {}
        else:
            if args.diagnose:
                print(f"[question] {financebench_id} generate starting", flush=True)
            started = time.perf_counter()
            if args.slow_question_seconds > 0:
                faulthandler.dump_traceback_later(args.slow_question_seconds, repeat=False)
            try:
                answer, usage = generate(
                    question,
                    prepared["evidence"],
                    prepared.get("task_policy", ""),
                )
                if args.rag_profile in {"clean_baseline", "clean_baseline_formula_skill"}:
                    prepared["rag_trace"].update({
                        "answer_consistency": {"enabled": False, "checked": False},
                        "numeric_display_validation": {"enabled": False, "checked": False},
                        "answer_facet_validation": {"enabled": False, "checked": False},
                    })
                else:
                    answer, consistency_trace = validate_or_repair_structured_answer(
                        answer,
                        prepared.get("query_spec") or {},
                        prepared.get("calculation"),
                    )
                    prepared["rag_trace"]["answer_consistency"] = consistency_trace
                    answer, numeric_trace = validate_numeric_display(
                        answer,
                        prepared.get("query_spec") or {},
                        prepared.get("calculation"),
                    )
                    prepared["rag_trace"]["numeric_display_validation"] = numeric_trace
                    facet_trace = assess_answer_facets(answer, prepared.get("query_spec") or {})
                    prepared["rag_trace"]["answer_facet_validation"] = facet_trace
                    if (
                        facet_trace.get("missing_facets")
                        and prepared["rag_trace"].get("evidence_flow_stage") == "evidence_ready_for_utilization"
                    ):
                        prepared["rag_trace"]["evidence_flow_stage"] = "evidence_utilization_failure"
                    elif (
                        facet_trace.get("complete")
                        and prepared["rag_trace"].get("evidence_flow_stage") == "evidence_ready_for_utilization"
                    ):
                        prepared["rag_trace"]["evidence_flow_stage"] = "evidence_utilization_complete"
            finally:
                if args.slow_question_seconds > 0:
                    faulthandler.cancel_dump_traceback_later()
            generation_seconds = time.perf_counter() - started
            if args.diagnose:
                print(f"[question] {financebench_id} generate finished in {generation_seconds:.2f}s", flush=True)
        run_tree = get_current_run_tree()
        total_seconds = time.perf_counter() - question_started
        prepared["rag_trace"].update({
            "answer_input_tokens": int(usage.get("input_tokens") or 0),
            "answer_output_tokens": int(usage.get("output_tokens") or 0),
            "answer_total_tokens": int(usage.get("total_tokens") or 0),
            "answer_latency_ms": round(generation_seconds * 1000, 2),
            "retrieval_latency_ms": round(retrieval_seconds * 1000, 2),
            "total_latency_ms": round(total_seconds * 1000, 2),
        })
        result = {
            "financebench_id": financebench_id,
            "question": question,
            "answer": answer,
            "citations": prepared["citations"],
            "evidence_status": prepared["evidence_status"],
            "calculation": prepared.get("calculation"),
            "execution_mode": prepared["execution_mode"],
            "route_reason": prepared["route_reason"],
            "rag_trace": prepared["rag_trace"],
            "usage": usage,
            "evaluation_latency": {
                "retrieval_ms": round(retrieval_seconds * 1000, 2),
                "generation_ms": round(generation_seconds * 1000, 2),
                "total_ms": round(total_seconds * 1000, 2),
            },
            "application_trace_id": prepared["trace_id"],
            "evaluation_run_id": prepared["trace_id"],
            "langsmith_trace_id": str(run_tree.id) if run_tree else "",
        }
        if output_handle:
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_handle.flush()
        completed_run_ids.add(financebench_id)
        return result

    active_features = feature_state(args.rag_profile)
    active_modules = active_features["modules"]
    metadata = {
        "profile": args.rag_profile,
        "execution_mode": "static",
        "query_planner": False,
        "step_back": False,
        "field_aware": feature_state(args.rag_profile)["field_aware"],
        "statement_aware": False if args.rag_profile in {"clean_baseline", "clean_baseline_formula_skill"} else args.field_aware,
        "required_field_page_scoring": False if args.rag_profile in {"clean_baseline", "clean_baseline_formula_skill"} else args.field_aware,
        "required_period_page_scoring": False if args.rag_profile in {"clean_baseline", "clean_baseline_formula_skill"} else args.field_aware,
        "legacy_supplemental_search": False if args.rag_profile in {"clean_baseline", "clean_baseline_formula_skill"} else args.field_aware and not args.supplemental_find,
        "supplemental_find": active_modules["Supplemental Retrieval"],
        "page_neighbor_window": 0 if args.rag_profile in {"clean_baseline", "clean_baseline_formula_skill"} else (2 if args.field_aware else 0),
        "context_page_window": 0 if args.rag_profile in {"clean_baseline", "clean_baseline_formula_skill"} else (2 if args.field_aware else 0),
        "rerank": args.enable_rerank,
        "rerank_remote_max_attempts": int(os.getenv("RERANK_REMOTE_MAX_ATTEMPTS", "2")),
        "rerank_remote_backoff_seconds": float(os.getenv("RERANK_REMOTE_BACKOFF_SECONDS", "0.8")),
        "rerank_cache_enabled": os.getenv("RERANK_CACHE_ENABLED", "true").lower() == "true",
        "rerank_cache_ttl_seconds": int(os.getenv("RERANK_CACHE_TTL_SECONDS", "604800")),
        "thinking": args.thinking,
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": 0,
        "finance_policy": active_modules["Finance Policy"],
        "evidence_frame": os.getenv("EVIDENCE_FRAME_ENABLED", "false").lower() == "true",
        "structured_executor": active_modules["Structured Executor"],
        "structured_coverage": active_modules["Structured Coverage"],
        "frame_alignment": args.frame_alignment,
        "structured_task_executor": args.structured_task_executor,
        "answer_consistency_validator": args.answer_consistency_validator,
        "protected_evidence_slots": args.protected_evidence_slots,
        "stage_aware_coverage": args.stage_aware_coverage,
        "protected_page_slots": args.protected_page_slots,
        "numeric_display_validator": args.numeric_display_validator,
        "answer_required_facets": args.answer_required_facets,
        "explicit_formula_advisory": args.explicit_formula_advisory,
        "benchmark_status": "fixed_seen_regression",
        "candidate_k": 40,
        "final_evidence_k": 5,
        "model": os.getenv("MODEL", ""),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_backend": args.evaluation_backend,
    }
    print_feature_summary(args.rag_profile)
    print(
        f"[setup] backend={args.evaluation_backend} profile={args.rag_profile} dataset={args.dataset_name} split={args.split} examples={len(examples)} "
        f"planner=false policy={str(args.finance_policy).lower()} frames={str(args.evidence_frame).lower()} "
        f"executor={str(args.structured_executor).lower()} coverage={str(args.structured_coverage).lower()} "
        f"alignment={str(args.frame_alignment).lower()} task_executor={str(args.structured_task_executor).lower()} "
        f"validator={str(args.answer_consistency_validator).lower()} protected={str(args.protected_evidence_slots).lower()} "
        f"stage_coverage={str(args.stage_aware_coverage).lower()} page_slots={str(args.protected_page_slots).lower()} "
        f"numeric_display={str(args.numeric_display_validator).lower()} facets={str(args.answer_required_facets).lower()} "
        f"formula_advisory={str(args.explicit_formula_advisory).lower()} "
        f"supplemental={str(args.supplemental_find).lower()} thinking={args.thinking} "
        "benchmark=fixed_seen_regression",
        flush=True,
    )
    try:
        if args.evaluation_backend == "local":
            for index, example in enumerate(examples, 1):
                target({"question": str(example.get("question") or "")})
                print(f"[{index:02d}/{len(examples)}] {_example_financebench_id(example)} completed", flush=True)
            experiment_name = args.experiment_prefix
            print(f"Local experiment: {experiment_name}", flush=True)
        else:
            from langsmith.evaluation import evaluate

            results = evaluate(
                target,
                data=examples,
                evaluators=[citation_document_hit],
                experiment_prefix=args.experiment_prefix,
                description="Controlled EvidenceRAG fixed, previously-seen FinanceBench regression.",
                metadata=metadata,
                max_concurrency=max(1, args.max_concurrency),
                client=client,
                blocking=True,
            )
            experiment_name = results.experiment_name
            print(f"Experiment: {experiment_name}", flush=True)
        if not args.skip_auto_judge:
            expected_ids = {_example_financebench_id(example) for example in examples}
            missing_completed_ids = set(filter(None, expected_ids)) - completed_run_ids
            if missing_completed_ids:
                print(
                    "[judge] skipped because target runs failed before producing answers: "
                    + ", ".join(sorted(missing_completed_ids)),
                    flush=True,
                )
                return
            judge_output = args.judge_output or (ROOT / "reports" / f"{experiment_name}_judge.jsonl")
            if args.evaluation_backend == "local":
                judge_command = [
                    sys.executable,
                    str(ROOT / "scripts" / "judge_financebench_local_answers.py"),
                    "--answers",
                    str(args.output),
                    "--output",
                    str(judge_output),
                ]
            else:
                judge_command = [
                    sys.executable,
                    str(ROOT / "scripts" / "judge_financebench_langsmith_experiment.py"),
                    "--experiment-name",
                    experiment_name,
                    "--output",
                    str(judge_output),
                ]
            print(f"[judge] starting output={judge_output}", flush=True)
            completed = subprocess.run(judge_command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                raise SystemExit(
                    f"Experiment completed, but automatic judge failed (exit code {completed.returncode}). "
                    f"Run the judge command manually for: {experiment_name}"
                )
            print(f"[judge] completed output={judge_output}", flush=True)
    finally:
        if output_handle:
            output_handle.close()


if __name__ == "__main__":
    main()
