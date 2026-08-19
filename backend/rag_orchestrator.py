"""Static and bounded-agentic retrieval orchestration for EvidenceRAG."""

import os
import re
import time
import uuid
from dataclasses import dataclass

from agent_tools import find_evidence, open_pages, select_pages
from prompts import PROMPT_VERSION
from query_planner import plan_retrieval_queries
from rag_pipeline import run_rag_graph
from rag_utils import finalize_retrieved_documents, get_finance_rag_config


VALID_PROFILES = {"general", "finance"}
VALID_MODES = {"static", "agentic", "auto"}


class RetrievalServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionConfig:
    profile: str
    requested_mode: str
    max_rounds: int = 3
    max_tool_calls: int = 5


def resolve_execution_config(profile: str | None = None, mode: str | None = None) -> ExecutionConfig:
    resolved_profile = (profile or os.getenv("RAG_PROFILE", "finance")).strip().lower()
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


def _run_search(query: str) -> dict:
    try:
        return run_rag_graph(query)
    except Exception as exc:
        raise RetrievalServiceError(f"检索服务暂不可用：{exc}") from exc


def _agent_queries(question: str) -> list[str]:
    plan = plan_retrieval_queries(question)
    candidates = []
    for key in ("semantic_queries", "evidence_field_queries", "keyword_queries", "table_heading_queries"):
        candidates.extend(plan.get(key) or [])
    result = []
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text and text != question and text not in result:
            result.append(text)
        if len(result) >= 2:
            break
    return result


def prepare_rag_response(question: str, profile: str | None = None, mode: str | None = None) -> dict:
    started = time.perf_counter()
    config = resolve_execution_config(profile, mode)
    initial = _run_search(question)
    initial_docs = list(initial.get("docs") or [])
    initial_candidates = list(initial.get("initial_candidate_docs") or initial_docs)
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

    citations = build_citations(final_docs)
    evidence_status = "sufficient" if len(citations) >= 2 else ("limited" if citations else "insufficient")
    trace.update(
        {
            "profile": config.profile,
            "execution_mode": execution_mode,
            "route_reason": route_reason,
            "evidence_status": evidence_status,
            "trace_id": str(uuid.uuid4()),
            "prompt_version": PROMPT_VERSION,
            "agent_tool_calls": tool_calls,
            "agent_tool_call_count": len(tool_calls),
        }
    )
    latency = dict(trace.get("latency_breakdown") or {})
    latency["orchestration_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    trace["latency_breakdown"] = latency
    return {
        "evidence": _format_evidence(final_docs),
        "docs": final_docs,
        "rag_trace": trace,
        "profile": config.profile,
        "execution_mode": execution_mode,
        "route_reason": route_reason,
        "citations": citations,
        "evidence_status": evidence_status,
        "trace_id": trace["trace_id"],
    }
