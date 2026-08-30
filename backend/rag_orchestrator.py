"""Static and bounded-agentic retrieval orchestration for EvidenceRAG."""

import hashlib
import os
import re
import time
import uuid
from dataclasses import dataclass

from runtime_profile import (
    EXPLICIT_FORMULA_SKILL_PROFILE,
    FINANCE_SKILLS_V1_PROFILE,
    apply_runtime_profile,
    feature_state,
    uses_clean_baseline_path,
)

apply_runtime_profile()

from agent_tools import find_evidence, open_pages, select_pages
from calculation_service import build_calculation_result, format_calculation_evidence
from evidence_frame import build_evidence_frames
from evidence_coverage import (
    assess_stage_coverage,
    assess_structured_coverage,
    build_document_scoped_supplemental_query,
    coverage_transition_reason,
    protect_selected_page_slots,
    stage_aware_coverage_enabled,
    structured_coverage_enabled,
)
from finance_policy import load_finance_policy
from evidence_context import build_baseline_evidence, build_compact_evidence
from prompts import CLEAN_BASELINE_PROMPT_VERSION, PROMPT_VERSION
from query_parser import assess_required_field_coverage, build_answer_directives, build_finance_query_rewrite, parse_query
from rag_pipeline import run_rag_graph
from rag_utils import finalize_retrieved_documents, get_finance_rag_config, retrieve_document_scoped_candidates
from table_store import TableStore


VALID_PROFILES = {
    "general", "finance", "clean_baseline", EXPLICIT_FORMULA_SKILL_PROFILE, FINANCE_SKILLS_V1_PROFILE,
}
VALID_MODES = {"static", "agentic", "auto"}
_table_store = TableStore()


class RetrievalServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionConfig:
    profile: str
    requested_mode: str
    max_rounds: int = 3
    max_tool_calls: int = 5


def resolve_execution_config(profile: str | None = None, mode: str | None = None) -> ExecutionConfig:
    resolved_profile = apply_runtime_profile(profile)
    if resolved_profile not in VALID_PROFILES:
        resolved_profile = "finance"
    default_mode = "auto" if resolved_profile == "finance" else "static"
    resolved_mode = (mode or os.getenv("RAG_EXECUTION_MODE", default_mode)).strip().lower()
    if resolved_mode not in VALID_MODES:
        resolved_mode = default_mode
    return ExecutionConfig(
        profile=resolved_profile,
        requested_mode=resolved_mode,
        max_rounds=max(1, int(os.getenv("RAG_AGENT_MAX_ROUNDS", "3"))),
        max_tool_calls=max(1, int(os.getenv("RAG_AGENT_MAX_TOOL_CALLS", "5"))),
    )


def _is_complex_question(question: str) -> tuple[bool, str]:
    text = (question or "").lower()
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
    complex_markers = (
        "compare", "comparison", "versus", " vs ", "rank", "highest", "lowest",
        "increase", "decrease", "growth rate", "margin", "ratio", "percentage change",
        "比较", "对比", "排名", "最高", "最低", "增长率", "变化率", "占比", "计算",
    )
    if len(years) >= 2:
        return True, "multiple_reporting_periods"
    if any(marker in text for marker in complex_markers):
        return True, "comparison_or_calculation"
    if text.count("?") + text.count("？") > 1:
        return True, "multiple_questions"
    return False, "single_evidence_question"


def _deduplicate_docs(documents: list[dict]) -> list[dict]:
    seen, result = set(), []
    for document in documents:
        key = (
            document.get("chunk_id") or "",
            document.get("filename") or "",
            document.get("page_number"),
            document.get("text") or "",
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(document)
    return result


def _format_evidence(documents: list[dict]) -> str:
    if not documents:
        return "No supporting evidence was retrieved."
    blocks = []
    for index, document in enumerate(documents, 1):
        filename = document.get("filename") or "Unknown"
        page = document.get("page_number", "N/A")
        text = document.get("text") or document.get("page_text") or ""
        blocks.append(f"[{index}] Source: {filename} | Page: {page}\n{text}")
    return "\n\n---\n\n".join(blocks)


def build_citations(documents: list[dict]) -> list[dict]:
    citations, seen = [], set()
    for document in documents:
        filename = str(document.get("filename") or "").strip()
        page = document.get("page_number")
        if not filename:
            continue
        key = (filename, str(page))
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "id": f"evidence-{len(citations) + 1}",
                "filename": filename,
                "page_number": page,
                "text": str(document.get("text") or document.get("page_text") or "")[:1200],
                "score": document.get("rerank_score") if document.get("rerank_score") is not None else document.get("score"),
            }
        )
    return citations


def _resolve_evidence_status(citations: list[dict], coverage: dict) -> str:
    if not citations:
        return "insufficient"
    if coverage.get("status") in {"partial", "insufficient"}:
        return "limited"
    return "sufficient"


def _run_search(query: str) -> dict:
    try:
        return run_rag_graph(query)
    except Exception as exc:
        raise RetrievalServiceError(f"检索服务暂不可用：{exc}") from exc


def _open_retrieved_pages(documents: list[dict], limit: int = 15) -> tuple[list[dict], dict]:
    """Replace retrieved snippets with full text from the same already-retrieved pages."""
    requested_pages = select_pages(documents, limit=limit)
    if not requested_pages:
        return documents, {"answer_page_open_requested": 0, "answer_page_opened": 0}
    started = time.perf_counter()
    try:
        opened_pages = open_pages(requested_pages, limit=limit)
    except Exception as exc:
        return documents, {
            "answer_page_open_requested": len(requested_pages),
            "answer_page_opened": 0,
            "answer_page_open_error": f"{type(exc).__name__}: {exc}",
            "answer_page_open_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    opened_by_page = {
        (item.get("filename"), item.get("page_number")): item
        for item in opened_pages
    }
    enriched = [
        {
            **document,
            **opened_by_page.get((document.get("filename"), document.get("page_number")), {}),
        }
        for document in documents
    ]
    return enriched, {
        "answer_page_open_requested": len(requested_pages),
        "answer_page_opened": len(opened_pages),
        "answer_page_open_latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _agent_queries(question: str) -> list[str]:
    rewrite = build_finance_query_rewrite(question)
    return [rewrite] if rewrite and rewrite != question else []


def _build_evidence_frames_for_documents(documents: list[dict], company: str) -> tuple[list[dict], dict]:
    enabled = os.getenv("EVIDENCE_FRAME_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    base_trace = {
        "evidence_frame_enabled": enabled,
        "evidence_frame_count": 0,
        "table_frame_count": 0,
        "evidence_frame_tables_considered": 0,
        "evidence_frame_tables_accepted": 0,
        "frames_with_period": 0,
        "frames_with_unit_scale": 0,
        "frames_used_for_execution": 0,
        "evidence_frame_skipped": {},
        "evidence_frame_load_errors": [],
        "evidence_frame_page_window": 0,
        "evidence_frame_adjacent_page_tables": 0,
        "evidence_frame_load_ms": 0.0,
    }
    if not enabled or not documents:
        return [], base_trace
    started = time.perf_counter()
    max_tables = max(1, int(os.getenv("EVIDENCE_FRAME_MAX_TABLES", "8")))
    max_frames = max(1, int(os.getenv("EVIDENCE_FRAME_MAX_FRAMES", "500")))
    page_window = max(0, int(os.getenv("EVIDENCE_FRAME_PAGE_WINDOW", "1")))
    pages_by_filename: dict[str, set[int]] = {}
    for document in documents:
        filename = str(document.get("filename") or "").strip()
        try:
            page_number = int(document.get("page_number") or 0)
        except (TypeError, ValueError):
            continue
        if filename and page_number > 0:
            pages_by_filename.setdefault(filename, set()).add(page_number)
    tables: list[dict] = []
    seen: set[str] = set()
    errors: list[str] = []
    for filename, pages in pages_by_filename.items():
        try:
            candidates = _table_store.get_tables_by_filename(filename) or []
        except Exception as exc:
            errors.append(f"{filename}:{type(exc).__name__}:{exc}")
            continue
        for table in candidates:
            table_id = str(table.get("table_id") or "")
            table_page = int(table.get("page_number") or 0)
            if not any(abs(table_page - page) <= page_window for page in pages) or table_id in seen:
                continue
            seen.add(table_id)
            row_labels = [
                str(next(iter(row.values()), "")).strip()
                for row in (table.get("normalized_rows") or table.get("rows") or [])
                if isinstance(row, dict) and row
            ][:5]
            matched_document = next(
                (
                    document
                    for document in documents
                    if str(document.get("filename") or "") == filename
                    and sum(
                        bool(label) and label.casefold() in str(document.get("text") or document.get("page_text") or "").casefold()
                        for label in row_labels
                    ) >= min(2, len(row_labels))
                ),
                None,
            )
            tables.append({
                **table,
                "evidence_page_context": (
                    str(matched_document.get("text") or matched_document.get("page_text") or "")
                    if matched_document else ""
                ),
                "evidence_page_number": matched_document.get("page_number") if matched_document else None,
            })
            if len(tables) >= max_tables:
                break
        if len(tables) >= max_tables:
            break
    frames, frame_trace = build_evidence_frames(tables, company=company or None, max_frames=max_frames)
    return frames, {
        **base_trace,
        **frame_trace,
        "evidence_frame_load_errors": errors,
        "evidence_frame_page_window": page_window,
        "evidence_frame_adjacent_page_tables": sum(
            int(table.get("page_number") or 0) not in pages_by_filename.get(str(table.get("filename") or ""), set())
            for table in tables
        ),
        "evidence_frame_load_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _supplement_partial_evidence(
    question: str,
    query_parse: dict,
    documents: list[dict],
    coverage: dict,
    candidate_coverage: dict | None = None,
) -> tuple[list[dict], dict]:
    """Perform at most one deterministic retrieval inside selected documents."""
    enabled = os.getenv("SUPPLEMENTAL_FIND_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    candidate_coverage = candidate_coverage or {}
    candidate_diagnosis = str(candidate_coverage.get("candidate_miss_diagnosis") or "")
    candidate_missing_operands = [
        str(item) for item in candidate_coverage.get("missing_operands") or [] if str(item)
    ]
    explicit_formula_candidate_gap = bool(
        candidate_missing_operands
        and candidate_diagnosis == "target_document_hit_requirement_page_not_hit"
        and query_parse.get("explicit_formula_source") == "question_explicit_definition"
    )
    base_coverage_incomplete = coverage.get("status") in {"partial", "insufficient", "incomplete"}
    coverage_for_query = dict(coverage)
    if explicit_formula_candidate_gap:
        coverage_for_query["missing_operands"] = candidate_missing_operands
    trace = {
        "supplemental_find_enabled": enabled,
        "supplemental_triggered": False,
        "missing_evidence": list(dict.fromkeys([
            *list(coverage.get("missing_fields") or coverage.get("structured_missing") or []),
            *[f"operand:{item}" for item in candidate_missing_operands],
        ])),
        "supplemental_query": "",
        "searched_documents": [],
        "new_pages": [],
        "new_evidence_frames": 0,
        "new_evidence_hashes": [],
        "supplemental_effective": False,
        "coverage_before": coverage,
        "coverage_after": coverage,
        "supplemental_skip_reason": "",
        "explicit_formula_candidate_gap": explicit_formula_candidate_gap,
        "candidate_missing_operands": candidate_missing_operands,
    }
    if not enabled or not (base_coverage_incomplete or explicit_formula_candidate_gap):
        return documents, trace
    if candidate_coverage:
        if candidate_coverage.get("status") == "complete" and not base_coverage_incomplete:
            trace["supplemental_skip_reason"] = "candidate_coverage_complete"
            return documents, trace
        if candidate_diagnosis == "target_document_not_hit":
            trace["supplemental_skip_reason"] = "target_document_not_hit"
            return documents, trace
        if candidate_diagnosis != "target_document_hit_requirement_page_not_hit":
            trace["supplemental_skip_reason"] = "target_document_not_high_confidence"
            return documents, trace
    actionable_missing = (
        list(coverage.get("missing_fields") or [])
        + list(coverage.get("missing_periods") or [])
        + candidate_missing_operands
    )
    if not actionable_missing and coverage.get("page_supported") is not False:
        trace["supplemental_skip_reason"] = "structural_metadata_gap_not_retrieval_actionable"
        return documents, trace
    filenames = list(dict.fromkeys(str(doc.get("filename") or "") for doc in documents if doc.get("filename")))
    if not filenames:
        return documents, trace
    supplemental_query = build_document_scoped_supplemental_query(question, query_parse, coverage_for_query)
    if not supplemental_query:
        return documents, trace
    trace.update({
        "supplemental_triggered": True,
        "supplemental_query": supplemental_query,
        "searched_documents": filenames,
    })
    try:
        hits = retrieve_document_scoped_candidates(
            supplemental_query,
            filenames,
            top_k=max(1, int(os.getenv("SUPPLEMENTAL_FIND_CANDIDATE_K", "12"))),
        )
        page_requests = []
        seen_pages = set()
        for hit in hits[: max(1, int(os.getenv("SUPPLEMENTAL_FIND_TOP_PAGES", "3")))]:
            filename = str(hit.get("filename") or "")
            try:
                page = int(hit.get("page_number") or 0)
            except (TypeError, ValueError):
                continue
            for adjacent in (page - 1, page, page + 1):
                key = (filename, adjacent)
                if filename in filenames and adjacent > 0 and key not in seen_pages:
                    seen_pages.add(key)
                    page_requests.append({"filename": filename, "page_number": adjacent})
        opened = open_pages(page_requests, limit=len(page_requests)) if page_requests else []
        existing_pages = {(doc.get("filename"), doc.get("page_number")) for doc in documents}

        def evidence_hash(document: dict) -> str:
            text = re.sub(r"\s+", " ", str(document.get("text") or document.get("page_text") or "")).strip()
            return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""

        existing_hashes = {evidence_hash(doc) for doc in documents} - {""}
        new_pages = []
        new_hashes = []
        for doc in opened:
            key = (doc.get("filename"), doc.get("page_number"))
            item_hash = evidence_hash(doc)
            if key in existing_pages or not item_hash or item_hash in existing_hashes:
                continue
            new_pages.append(doc)
            new_hashes.append(item_hash)
            existing_hashes.add(item_hash)
        trace["new_evidence_hashes"] = new_hashes
        if not new_pages:
            trace["supplemental_skip_reason"] = "no_new_evidence_hash"
            return documents, trace
        trace["new_pages"] = [
            {"filename": doc.get("filename"), "page_number": doc.get("page_number")}
            for doc in new_pages
        ]
        supplemented_documents = _deduplicate_docs([*documents, *new_pages])
        if explicit_formula_candidate_gap:
            before_stage = assess_stage_coverage(
                query_parse, documents, stage="selected_page_before_supplemental",
            )
            after_stage = assess_stage_coverage(
                query_parse, supplemented_documents, stage="selected_page_after_supplemental",
            )
            before_missing = set(before_stage.get("missing_requirements") or [])
            after_missing = set(after_stage.get("missing_requirements") or [])
            resolved = sorted(before_missing - after_missing)
            trace["stage_coverage_before"] = before_stage
            trace["stage_coverage_after"] = after_stage
            trace["supplemental_requirement_improvements"] = [
                f"resolved:stage_requirement:{item}" for item in resolved
            ]
            trace["supplemental_effective"] = bool(resolved)
        else:
            # New content is not evidence of effectiveness. The caller checks
            # base/structured requirements again after rebuilding frames.
            trace["supplemental_effective"] = False
        return supplemented_documents, trace
    except Exception as exc:
        # Supplemental retrieval is optional. Preserve the primary answer path
        # and expose the exact failure in trace instead of hiding it as no hit.
        trace["supplemental_error"] = f"{type(exc).__name__}: {exc}"
        return documents, trace


def _prepare_clean_baseline_response(question: str, config: ExecutionConfig, started: float) -> dict:
    """Run the isolated retrieval, rerank, generic context, and answer path."""
    initial = _run_search(question)
    final_docs = list(initial.get("docs") or [])
    candidate_pool = list(initial.get("initial_candidate_docs") or final_docs)
    answer_docs, page_trace = _open_retrieved_pages(final_docs)
    citations = build_citations(answer_docs)
    evidence, context_meta = build_baseline_evidence(question, answer_docs)
    if not evidence:
        evidence = _format_evidence(answer_docs)
        context_meta = {
            **context_meta,
            "answer_context_chars": len(evidence),
            "answer_context_unit_count": len(answer_docs),
        }

    trace = dict(initial.get("rag_trace") or {})
    latency = dict(trace.get("latency_breakdown") or {})
    latency["orchestration_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    trace_id = str(uuid.uuid4())
    trace.update({
        "profile": config.profile,
        "execution_mode": "static",
        "route_reason": "clean_baseline_static",
        "original_question": question,
        "retrieval_queries": [question],
        "dense_candidate_count": trace.get("dense_candidate_count"),
        "bm25_candidate_count": trace.get("bm25_candidate_count"),
        "dense_candidate_requested": int(os.getenv("FINANCE_RAG_CANDIDATE_K", "40")) * 2,
        "bm25_candidate_requested": int(os.getenv("FINANCE_RAG_CANDIDATE_K", "40")) * 2,
        "candidate_count_observability": "milvus_hybrid_api_exposes_fused_results_only",
        "candidate_count": len(candidate_pool),
        "rrf_fused_candidate_count": trace.get("rrf_fused_candidate_count", len(candidate_pool)),
        "final_selected_pages": [
            {"filename": doc.get("filename"), "page_number": doc.get("page_number")}
            for doc in answer_docs
        ],
        "final_evidence_unit_count": context_meta.get("answer_context_unit_count", 0),
        "evidence_status": "sufficient" if citations else "insufficient",
        "trace_id": trace_id,
        "prompt_version": CLEAN_BASELINE_PROMPT_VERSION,
        "feature_state": feature_state(config.profile),
        "experiment_modules_in_answer_path": [],
        "shadow_query_spec": {},
        "shadow_evidence_frame_count": 0,
        "shadow_formula_detected": False,
        "queryspec_used_for_retrieval": False,
        "queryspec_used_for_context": False,
        "queryspec_used_for_answer": False,
        "finance_policy_enabled": False,
        "structured_authoritative": False,
        "agent_tool_calls": [{"tool": "search", "query": question, "new_evidence": len(final_docs)}],
        "agent_tool_call_count": 1,
        "latency_breakdown": latency,
        **page_trace,
        **context_meta,
    })
    skill_answer = ""
    skill_applied = False
    calculation = None
    if config.profile in {EXPLICIT_FORMULA_SKILL_PROFILE, FINANCE_SKILLS_V1_PROFILE}:
        from skills.registry import execute_matching_skill

        enabled_skills = ("explicit_formula",) if config.profile == EXPLICIT_FORMULA_SKILL_PROFILE else (
            "explicit_formula", "canonical_finance_metric",
        )
        skill_result = execute_matching_skill(question, answer_docs, candidate_pool, enabled_skills)
        skill_name = str(skill_result.trace.get("skill_name") or "none")
        trace["skill_router"] = {
            "enabled_skills": list(enabled_skills),
            "selected_skill": skill_name,
        }
        if skill_name != "none":
            trace[f"{skill_name}_skill"] = skill_result.trace
        skill_answer = skill_result.answer
        skill_applied = skill_result.applied
        verified_evidence = str(skill_result.trace.get("verified_evidence") or "")
        if skill_result.success and not skill_applied and verified_evidence:
            evidence = f"{evidence}\n\n---\n\n{verified_evidence}"
            existing_citations = {(item.get("filename"), item.get("page_number")) for item in citations}
            citations.extend(
                item for item in skill_result.citations
                if (item.get("filename"), item.get("page_number")) not in existing_citations
            )
        if skill_applied:
            citations = skill_result.citations
            calculation = {
                "source": f"{skill_name}_decimal",
                "authoritative": True,
                "formula": skill_result.trace.get("formula_text") or skill_result.trace.get("formula_variant", ""),
                "operands": skill_result.trace.get("resolved_operands", []),
                "result": skill_result.trace.get("full_precision_result") or skill_result.trace.get("metric_full_precision_result", ""),
                "display_result": skill_result.trace.get("display_result") or skill_result.trace.get("metric_display_result", ""),
            }
        if skill_result.success:
            trace["experiment_modules_in_answer_path"] = [f"{skill_name}_skill"]
        latency[f"{skill_name}_skill_latency_ms"] = skill_result.trace.get("skill_latency_ms", 0)
        latency["orchestration_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        trace["latency_breakdown"] = latency
    trace["answer_prompt_evidence_chars"] = len(evidence)
    trace["answer_prompt_policy_chars"] = 0
    trace["answer_prompt_total_chars"] = len(evidence)
    return {
        "evidence": evidence,
        "task_policy": "",
        "docs": final_docs,
        "rag_trace": trace,
        "profile": config.profile,
        "execution_mode": "static",
        "route_reason": "clean_baseline_static",
        "citations": citations,
        "evidence_status": "sufficient" if citations else "insufficient",
        "calculation": calculation,
        "query_spec": {},
        "evidence_frames": [],
        "trace_id": trace_id,
        "skill_applied": skill_applied,
        "skill_answer": skill_answer,
    }


def prepare_rag_response(question: str, profile: str | None = None, mode: str | None = None) -> dict:
    started = time.perf_counter()
    config = resolve_execution_config(profile, mode)
    if uses_clean_baseline_path(config.profile):
        return _prepare_clean_baseline_response(question, config, started)
    initial = _run_search(question)
    initial_docs = list(initial.get("docs") or [])
    initial_candidates = list(initial.get("initial_candidate_docs") or initial_docs)
    candidate_pool = list(initial_candidates)
    query_parse = parse_query(question)
    complex_question, route_reason = _is_complex_question(question)
    low_evidence = len(initial_docs) < 2

    execution_mode = config.requested_mode
    if execution_mode == "auto":
        execution_mode = "agentic" if complex_question or low_evidence else "static"
        if low_evidence:
            route_reason = "low_evidence_coverage"

    final_docs = initial_docs
    trace = dict(initial.get("rag_trace") or {})
    tool_calls = [{"tool": "search", "query": question, "new_evidence": len(initial_docs)}]

    if execution_mode == "agentic":
        all_candidates = list(initial_candidates)
        previous_count = len(_deduplicate_docs(all_candidates))
        no_progress_rounds = 0
        for query in _agent_queries(question)[: max(0, min(config.max_rounds - 1, config.max_tool_calls - 1))]:
            expanded = _run_search(query)
            new_candidates = list(expanded.get("initial_candidate_docs") or expanded.get("docs") or [])
            all_candidates.extend(new_candidates)
            current_count = len(_deduplicate_docs(all_candidates))
            added = max(0, current_count - previous_count)
            tool_calls.append({"tool": "search", "query": query, "new_evidence": added})
            no_progress_rounds = no_progress_rounds + 1 if added == 0 else 0
            previous_count = current_count
            if no_progress_rounds >= 2 or len(tool_calls) >= config.max_tool_calls:
                break

        if all_candidates:
            candidate_pool = _deduplicate_docs(all_candidates)
            finance_config = get_finance_rag_config()
            finalized = finalize_retrieved_documents(
                question,
                _deduplicate_docs(all_candidates),
                final_top_k=finance_config["final_top_k"],
                enable_page_merge=False,
                adjacent_page_window=0,
                adjacent_chunk_window=0,
            )
            final_docs = list(finalized.get("context_docs") or finalized.get("final_retrieved_docs") or final_docs)

        if len(tool_calls) < config.max_tool_calls:
            found_docs = find_evidence(question, all_candidates or initial_docs, limit=3)
            tool_calls.append({"tool": "find", "matches": len(found_docs)})
        else:
            found_docs = []
        if found_docs and len(tool_calls) < config.max_tool_calls:
            requested_pages = select_pages(found_docs, limit=3)
            opened_pages = open_pages(requested_pages, limit=3)
            tool_calls.append({"tool": "open_page", "pages": len(opened_pages)})
            opened_by_page = {
                (item.get("filename"), item.get("page_number")): item
                for item in opened_pages
            }
            final_docs = [
                {
                    **document,
                    **opened_by_page.get((document.get("filename"), document.get("page_number")), {}),
                }
                for document in final_docs
            ]

    candidate_coverage = (
        assess_stage_coverage(query_parse, candidate_pool, stage="candidate")
        if stage_aware_coverage_enabled() else {}
    )
    final_docs, protected_page_trace = protect_selected_page_slots(
        query_parse,
        candidate_pool,
        final_docs,
    )
    answer_docs, answer_page_open_trace = _open_retrieved_pages(final_docs)
    selected_page_coverage = (
        assess_stage_coverage(query_parse, answer_docs, stage="selected_page")
        if stage_aware_coverage_enabled() else {}
    )
    citations = build_citations(answer_docs)
    evidence_frames, evidence_frame_trace = _build_evidence_frames_for_documents(
        answer_docs,
        str(query_parse.get("company") or ""),
    )
    finance_policy = load_finance_policy(str(query_parse.get("task_type") or "lookup"))
    evidence_coverage = assess_required_field_coverage(query_parse, answer_docs)
    if structured_coverage_enabled():
        evidence_coverage = assess_structured_coverage(
            query_parse,
            answer_docs,
            evidence_frames,
            evidence_coverage,
        )
    evidence_coverage["supplemental_search_attempted"] = bool(
        trace.get("supplemental_search_attempted")
        or (trace.get("evidence_coverage") or {}).get("supplemental_search_attempted")
    )
    frames_before_supplement = {str(frame.get("evidence_id") or "") for frame in evidence_frames}
    answer_docs, supplemental_trace = _supplement_partial_evidence(
        question,
        query_parse,
        answer_docs,
        evidence_coverage,
        candidate_coverage if stage_aware_coverage_enabled() else None,
    )
    if supplemental_trace["supplemental_triggered"]:
        citations = build_citations(answer_docs)
        evidence_frames, evidence_frame_trace = _build_evidence_frames_for_documents(
            answer_docs,
            str(query_parse.get("company") or ""),
        )
        evidence_coverage = assess_required_field_coverage(query_parse, answer_docs)
        if structured_coverage_enabled():
            evidence_coverage = assess_structured_coverage(
                query_parse,
                answer_docs,
                evidence_frames,
                evidence_coverage,
            )
        evidence_coverage["supplemental_search_attempted"] = True
        supplemental_trace["new_evidence_frames"] = sum(
            str(frame.get("evidence_id") or "") not in frames_before_supplement
            for frame in evidence_frames
        )
        supplemental_trace["coverage_after"] = evidence_coverage
        coverage_before = supplemental_trace.get("coverage_before") or {}
        requirement_improvements = list(
            supplemental_trace.get("supplemental_requirement_improvements") or []
        )
        for name in ("missing_fields", "missing_periods", "structured_missing"):
            before_missing = {str(item) for item in coverage_before.get(name) or []}
            after_missing = {str(item) for item in evidence_coverage.get(name) or []}
            requirement_improvements.extend(f"resolved:{name}:{item}" for item in sorted(before_missing - after_missing))
        before_gates = coverage_before.get("structured_gate_trace") or {}
        after_gates = evidence_coverage.get("structured_gate_trace") or {}
        requirement_improvements.extend(
            f"gate:{gate}"
            for gate, supported in after_gates.items()
            if supported is True and before_gates.get(gate) is not True
        )
        requirement_improvements = list(dict.fromkeys(requirement_improvements))
        coverage_improved = bool(requirement_improvements)
        supplemental_trace["coverage_improved"] = coverage_improved
        supplemental_trace["supplemental_requirement_improvements"] = requirement_improvements
        supplemental_trace["supplemental_effective"] = coverage_improved
        if stage_aware_coverage_enabled():
            selected_page_coverage = assess_stage_coverage(
                query_parse,
                answer_docs,
                stage="selected_page_after_supplemental",
            )
    calculation = build_calculation_result(
        query_parse,
        evidence_coverage,
        answer_docs,
        evidence_frames=evidence_frames,
    )
    if calculation and calculation.get("executor") == "evidence_frame":
        evidence_frame_trace["frames_used_for_execution"] = len(
            calculation.get("operand_evidence_ids") or []
        )
    evidence_status = _resolve_evidence_status(citations, evidence_coverage)
    trace.update(
        {
            "profile": config.profile,
            "execution_mode": execution_mode,
            "route_reason": route_reason,
            "evidence_status": evidence_status,
            "evidence_coverage": evidence_coverage,
            "queryspec_concepts": evidence_coverage.get("queryspec_concepts") or query_parse.get("required_concepts") or [],
            "frame_match_candidates": evidence_coverage.get("frame_match_candidates") or [],
            "frame_match_method": evidence_coverage.get("frame_match_method") or "",
            "frame_match_score": evidence_coverage.get("frame_match_score") or 0.0,
            "relevant_frame_count": evidence_coverage.get("relevant_frame_count") or 0,
            "operand_resolution_failure_reason": evidence_coverage.get("operand_resolution_failure_reason") or "",
            "structured_gate_trace": evidence_coverage.get("structured_gate_trace") or {},
            "execution_contract": (calculation or {}).get("execution_contract") or {},
            "structured_authoritative": bool((calculation or {}).get("authoritative")),
            "calculation": calculation,
            "trace_id": str(uuid.uuid4()),
            "prompt_version": PROMPT_VERSION,
            "task_type": finance_policy["task_type"],
            "finance_policy_enabled": finance_policy["enabled"],
            "policy": finance_policy["policy_file"],
            "finance_policy_chars": finance_policy["chars"],
            "finance_policy_estimated_tokens": finance_policy["estimated_tokens"],
            "finance_policy_cache_hit": finance_policy["cache_hit"],
            "finance_policy_load_ms": finance_policy["load_ms"],
            "agent_tool_calls": tool_calls,
            "agent_tool_call_count": len(tool_calls),
            **supplemental_trace,
            **evidence_frame_trace,
            **answer_page_open_trace,
            "candidate_coverage": candidate_coverage,
            "selected_page_coverage": selected_page_coverage,
            "candidate_to_selected_transition": (
                coverage_transition_reason(candidate_coverage, selected_page_coverage)
                if candidate_coverage and selected_page_coverage else "disabled"
            ),
            "protected_page_slots_enabled": protected_page_trace.get("protected_page_slots_enabled", False),
            "protected_pages": protected_page_trace.get("protected_pages") or [],
            "protected_page_replacements": protected_page_trace.get("protected_page_replacements") or [],
            "protected_page_count": protected_page_trace.get("protected_page_count") or 0,
            "protected_page_coverage_before": protected_page_trace.get("coverage_before") or {},
            "protected_page_coverage_after": protected_page_trace.get("coverage_after") or {},
            "protected_page_coverage_transition": protected_page_trace.get("coverage_transition_reason") or "",
            "selected_page_count_before_protection": protected_page_trace.get("selected_page_count_before") or 0,
            "selected_page_count_after_protection": protected_page_trace.get("selected_page_count_after") or 0,
        }
    )
    latency = dict(trace.get("latency_breakdown") or {})
    latency["orchestration_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    trace["latency_breakdown"] = latency
    evidence, answer_context_meta = build_compact_evidence(
        question,
        answer_docs,
        query_parse,
        calculation,
    )
    if not evidence:
        evidence = _format_evidence(answer_docs)
        answer_context_meta = {
            **answer_context_meta,
            "answer_context_chars": len(evidence),
            "answer_context_unit_count": len(answer_docs),
        }
    compact_context_coverage = (
        assess_stage_coverage(
            query_parse,
            [{
                "filename": " ".join(dict.fromkeys(
                    str(document.get("filename") or "") for document in answer_docs if document.get("filename")
                )),
                "page_number": "compact",
                "text": evidence,
            }],
            stage="compact_context",
        )
        if stage_aware_coverage_enabled() else {}
    )
    if candidate_coverage and selected_page_coverage and compact_context_coverage:
        if candidate_coverage.get("status") != "complete":
            evidence_flow_stage = "retrieval_miss"
        elif selected_page_coverage.get("status") != "complete":
            evidence_flow_stage = "candidate_to_page_selection_loss"
        elif compact_context_coverage.get("status") != "complete":
            evidence_flow_stage = "compression_loss"
        else:
            evidence_flow_stage = "evidence_ready_for_utilization"
    else:
        evidence_flow_stage = "disabled"
    trace.update({
        "compact_context_coverage": compact_context_coverage,
        "selected_to_compact_transition": (
            coverage_transition_reason(selected_page_coverage, compact_context_coverage)
            if selected_page_coverage and compact_context_coverage else "disabled"
        ),
        "evidence_flow_stage": evidence_flow_stage,
    })
    trace.update(answer_context_meta)
    answer_directives = build_answer_directives(question, query_parse)
    if answer_directives:
        directive_text = "\n".join(f"- {directive}" for directive in answer_directives)
        evidence = f"Question-specific answer contract (instructions, not evidence):\n{directive_text}\n\n---\n\n{evidence}"
    calculation_evidence = format_calculation_evidence(calculation)
    if calculation_evidence:
        evidence = f"{evidence}\n\n---\n\n{calculation_evidence}"
    trace["answer_prompt_evidence_chars"] = len(evidence)
    trace["answer_prompt_policy_chars"] = finance_policy["chars"]
    trace["answer_prompt_total_chars"] = len(evidence) + finance_policy["chars"]
    return {
        "evidence": evidence,
        "task_policy": finance_policy["text"],
        "docs": final_docs,
        "rag_trace": trace,
        "profile": config.profile,
        "execution_mode": execution_mode,
        "route_reason": route_reason,
        "citations": citations,
        "evidence_status": evidence_status,
        "calculation": calculation,
        "query_spec": query_parse,
        "evidence_frames": evidence_frames,
        "trace_id": trace["trace_id"],
    }
