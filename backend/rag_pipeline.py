from typing import List, Optional, TypedDict
import json
import logging
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from rag_utils import (
    finalize_retrieved_documents,
    get_doc_name,
    get_finance_rag_config,
    retrieve_candidate_documents,
    retrieve_documents,
    step_back_expand,
)
from tools import emit_rag_step

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("ARK_API_KEY")
GRADE_MODEL = os.getenv("GRADE_MODEL", "gpt-4.1")
BASE_URL = os.getenv("BASE_URL")

_grader_model = None


def _get_grader_model():
    global _grader_model
    if not API_KEY or not GRADE_MODEL:
        return None
    if _grader_model is None:
        _grader_model = init_chat_model(
            model=GRADE_MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
            stream_usage=True,
        )
    return _grader_model


GRADE_PROMPT = (
    "You are a grader assessing relevance of retrieved documents to a user question.\n"
    "Retrieved context:\n\n{context}\n\n"
    "User question:\n{question}\n\n"
    "If the retrieved documents are likely sufficient to answer the question, output yes.\n"
    "Otherwise output no."
)


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Relevance score: 'yes' if relevant, or 'no' if not relevant")


class RAGState(TypedDict):
    question: str
    query: str
    context: str
    docs: List[dict]
    route: Optional[str]
    rewritten_question: Optional[str]
    rewrite_used: Optional[bool]
    step_back_question: Optional[str]
    step_back_answer: Optional[str]
    initial_candidate_docs: List[dict]
    expanded_candidate_docs: List[dict]
    rag_trace: Optional[dict]


def _format_docs(docs: List[dict]) -> str:
    if not docs:
        return ""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = doc.get("filename", "Unknown")
        doc_name = doc.get("doc_name") or get_doc_name(source)
        page = doc.get("page_number", "N/A")
        evidence_type = doc.get("type", "chunk") or "chunk"
        text = doc.get("text", "")
        chunks.append(f"[{i}] {doc_name} | {source} | Page {page} | Type {evidence_type}:\n{text}")
    return "\n\n---\n\n".join(chunks)


def _deduplicate_docs(docs: List[dict]) -> List[dict]:
    out: List[dict] = []
    seen = set()
    for doc in docs:
        key = (
            doc.get("chunk_id") or "",
            doc.get("filename") or "",
            doc.get("page_number"),
            doc.get("chunk_idx"),
            doc.get("text") or "",
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def _as_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _normalize_trace_chunks(docs: List[dict]) -> List[dict]:
    normalized = []
    for idx, doc in enumerate(docs, 1):
        filename = doc.get("filename", "") or ""
        page_number = doc.get("page_number", "")
        source = filename
        if filename and page_number not in ("", None):
            source = f"{filename}#page={page_number}"
        normalized.append(
            {
                "rank": idx,
                "filename": filename,
                "doc_name": doc.get("doc_name") or get_doc_name(filename),
                "page_number": page_number if page_number is not None else "",
                "type": doc.get("type", "") or "",
                "text": doc.get("text", "") or "",
                "score": _as_float(doc.get("score")),
                "rerank_score": _as_float(doc.get("rerank_score")),
                "source": source,
                "chunk_id": doc.get("chunk_id", "") or "",
            }
        )
    return normalized


def _page_distribution(chunks: List[dict]) -> dict:
    distribution: dict = {}
    for chunk in chunks:
        page = chunk.get("page_number")
        key = "" if page is None else str(page)
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _update_trace_from_finalization(trace: dict, finalization: dict) -> dict:
    meta = finalization.get("meta", {})
    final_docs = finalization.get("final_retrieved_docs", [])
    context_docs = finalization.get("context_docs", [])
    trace.update(
        {
            "retrieved_chunks": _normalize_trace_chunks(context_docs or final_docs),
            "final_retrieved_chunks": _normalize_trace_chunks(final_docs),
            "query_planner_enabled": meta.get("query_planner_enabled", False),
            "planner_intent": meta.get("planner_intent", ""),
            "planner_must_keep_terms": meta.get("planner_must_keep_terms", []) or [],
            "planner_dense_queries": meta.get("planner_dense_queries", []) or [],
            "planner_semantic_queries": meta.get("planner_semantic_queries", []) or [],
            "planner_evidence_field_queries": meta.get("planner_evidence_field_queries", []) or [],
            "planner_table_heading_queries": meta.get("planner_table_heading_queries", []) or [],
            "planner_keyword_queries": meta.get("planner_keyword_queries", []) or [],
            "planner_table_queries": meta.get("planner_table_queries", []) or [],
            "planner_validation_dropped_queries": meta.get("planner_validation_dropped_queries", []) or [],
            "planner_parse_error": meta.get("planner_parse_error", ""),
            "per_query_retrieval_counts": meta.get("per_query_retrieval_counts", []) or [],
            "rrf_fused_candidate_count": meta.get("rrf_fused_candidate_count", 0),
            "page_level_fusion_enabled": meta.get("page_level_fusion_enabled", False),
            "fused_page_count": meta.get("fused_page_count", 0),
            "fused_top_pages": meta.get("fused_top_pages", []) or [],
            "fused_pages_after_anchor_guard": meta.get("fused_pages_after_anchor_guard", []) or [],
            "page_anchor_filtered_count": meta.get("page_anchor_filtered_count", 0),
            "page_contributing_routes": meta.get("page_contributing_routes", {}) or {},
            "page_fusion_used_for_final_context": meta.get("page_fusion_used_for_final_context", False),
            "page_first_anchor_window": meta.get("page_first_anchor_window", 0),
            "final_evidence_pack_source": meta.get("final_evidence_pack_source", "chunk_rerank_fallback"),
            "page_stage_candidates": _normalize_trace_chunks(meta.get("page_stage_candidates", []) or []),
            "doc_stage_selected_docs": meta.get("doc_stage_selected_docs", []) or [],
            "selected_docs": meta.get("selected_docs", []) or meta.get("doc_stage_selected_docs", []) or [],
            "selected_pages": meta.get("selected_pages", []) or [],
            "page_scores": meta.get("page_scores", []) or [],
            "final_evidence_pack": _normalize_trace_chunks(meta.get("final_evidence_pack", []) or context_docs or final_docs),
            "final_evidence_pack_debug": _normalize_trace_chunks(meta.get("final_evidence_pack_debug", []) or []),
            "final_evidence_pack_used": _normalize_trace_chunks(meta.get("final_evidence_pack_used", []) or context_docs or final_docs),
            "final_evidence_pack_debug_count": meta.get("final_evidence_pack_debug_count"),
            "final_evidence_pack_used_count": meta.get("final_evidence_pack_used_count"),
            "dropped_evidence_count": meta.get("dropped_evidence_count"),
            "dropped_reasons": meta.get("dropped_reasons"),
            "prompt_context_char_count_estimate": meta.get("prompt_context_char_count_estimate"),
            "query_parse": meta.get("query_parse"),
            "evidence_coverage": meta.get("evidence_coverage", {}),
            "latency_breakdown": meta.get("latency_breakdown"),
            "table_aware_retrieval_mode": meta.get("table_aware_retrieval_mode", "off"),
            "table_aware_auto_triggered": meta.get("table_aware_auto_triggered", False),
            "table_aware_trigger_reason": meta.get("table_aware_trigger_reason", []) or [],
            "query_anchors": meta.get("query_anchors", []) or [],
            "anchor_guard_applied": meta.get("anchor_guard_applied", False),
            "anchor_filtered_count": meta.get("anchor_filtered_count", 0),
            "table_context_source": meta.get("table_context_source", "none"),
            "table_evidence_hit_count": meta.get("table_evidence_hit_count", 0),
            "table_context_table_count": meta.get("table_context_table_count", 0),
            "table_context_char_count": meta.get("table_context_char_count", 0),
            "evidence_unit_count": meta.get("evidence_unit_count", 0),
            "evidence_units_with_tables": meta.get("evidence_units_with_tables", 0),
            "table_attached_count": meta.get("table_attached_count", 0),
            "table_attach_reasons": meta.get("table_attach_reasons", []) or [],
            "evidence_group_count": meta.get("evidence_group_count", 0),
            "selected_evidence_group_count": meta.get("selected_evidence_group_count", 0),
            "evidence_groups_debug": meta.get("evidence_groups_debug", []) or [],
            "selected_evidence_groups": meta.get("selected_evidence_groups", []) or [],
            "group_scores": meta.get("group_scores", []) or [],
            "expanded_snippet_count": meta.get("expanded_snippet_count", 0),
            "relevant_table_row_count": meta.get("relevant_table_row_count", 0),
            "dropped_group_reasons": meta.get("dropped_group_reasons", []) or [],
            "table_candidate_filenames": meta.get("table_candidate_filenames", []) or [],
            "table_candidate_pages": meta.get("table_candidate_pages", []) or [],
            "table_ids": meta.get("table_ids", []) or [],
            "table_context_skipped_reasons": meta.get("table_context_skipped_reasons", []) or [],
            "two_stage_retrieval": meta.get("two_stage_retrieval"),
            "doc_stage_top_n": meta.get("doc_stage_top_n"),
            "page_stage_top_n": meta.get("page_stage_top_n"),
            "rerank_enabled": meta.get("rerank_enabled"),
            "rerank_applied": meta.get("rerank_applied"),
            "rerank_model": meta.get("rerank_model"),
            "rerank_endpoint": meta.get("rerank_endpoint"),
            "rerank_error": meta.get("rerank_error"),
            "rerank_provider": meta.get("rerank_provider"),
            "remote_rerank_candidate_count": meta.get("remote_rerank_candidate_count"),
            "remote_rerank_input_chars": meta.get("remote_rerank_input_chars"),
            "remote_rerank_rate_limited": meta.get("remote_rerank_rate_limited"),
            "auto_merge_enabled": meta.get("auto_merge_enabled"),
            "auto_merge_applied": meta.get("auto_merge_applied"),
            "auto_merge_threshold": meta.get("auto_merge_threshold"),
            "auto_merge_replaced_chunks": meta.get("auto_merge_replaced_chunks"),
            "auto_merge_steps": meta.get("auto_merge_steps"),
            "page_merge_applied": meta.get("page_merge_applied"),
            "merged_chunk_count": meta.get("merged_chunk_count"),
            "final_context_chunk_count": meta.get("final_context_chunk_count"),
            "cover_page_filtered_count": meta.get("cover_page_filtered_count"),
            "fallback_used": meta.get("fallback_used"),
            "fallback_reason": meta.get("fallback_reason"),
        }
    )
    return trace


def _log_rag_diagnostics(rag_trace: dict | None) -> None:
    if not rag_trace:
        return
    final_chunks = rag_trace.get("final_retrieved_chunks", []) or []
    final_page_distribution = _page_distribution(final_chunks)
    payload = {
        "original_question": rag_trace.get("original_question"),
        "rewritten_question": rag_trace.get("rewritten_question"),
        "rewrite_used": rag_trace.get("rewrite_used"),
        "rewrite_strategy": rag_trace.get("rewrite_strategy"),
        "retrieval_mode": rag_trace.get("retrieval_mode"),
        "candidate_k": rag_trace.get("candidate_k"),
        "final_top_k": rag_trace.get("final_top_k"),
        "two_stage_retrieval": rag_trace.get("two_stage_retrieval"),
        "query_parse": rag_trace.get("query_parse"),
        "doc_stage_selected_docs": rag_trace.get("doc_stage_selected_docs"),
        "selected_pages": [
            {
                "filename": page.get("filename"),
                "page_number": page.get("page_number"),
                "page_score": page.get("page_score"),
                "type": page.get("type"),
            }
            for page in rag_trace.get("selected_pages", []) or []
        ],
        "final_evidence_pack_debug_count": rag_trace.get("final_evidence_pack_debug_count"),
        "final_evidence_pack_used_count": rag_trace.get("final_evidence_pack_used_count"),
        "dropped_evidence_count": rag_trace.get("dropped_evidence_count"),
        "dropped_reasons": rag_trace.get("dropped_reasons"),
        "prompt_context_char_count_estimate": rag_trace.get("prompt_context_char_count_estimate"),
        "latency_breakdown": rag_trace.get("latency_breakdown"),
        "initial_retrieved": [
            {
                "filename": chunk.get("filename"),
                "page_number": chunk.get("page_number"),
                "score": chunk.get("score"),
                "rerank_score": chunk.get("rerank_score"),
            }
            for chunk in rag_trace.get("initial_retrieved_chunks", []) or []
        ],
        "expanded_retrieved": [
            {
                "filename": chunk.get("filename"),
                "page_number": chunk.get("page_number"),
                "score": chunk.get("score"),
                "rerank_score": chunk.get("rerank_score"),
            }
            for chunk in rag_trace.get("expanded_retrieved_chunks", []) or []
        ],
        "final_retrieved": [
            {
                "filename": chunk.get("filename"),
                "page_number": chunk.get("page_number"),
                "score": chunk.get("score"),
                "rerank_score": chunk.get("rerank_score"),
            }
            for chunk in rag_trace.get("final_retrieved_chunks", []) or []
        ],
        "final_retrieved_chunk_count": len(final_chunks),
        "final_retrieved_page_distribution": final_page_distribution,
        "final_retrieved_page_zero_count": sum(
            count for page, count in final_page_distribution.items() if page in {"", "0"}
        ),
        "page_merge_applied": rag_trace.get("page_merge_applied"),
        "merged_chunk_count": rag_trace.get("merged_chunk_count"),
        "final_context_chunk_count": rag_trace.get("final_context_chunk_count"),
        "cover_page_filtered_count": rag_trace.get("cover_page_filtered_count"),
        "fallback_used": rag_trace.get("fallback_used"),
        "fallback_reason": rag_trace.get("fallback_reason"),
    }
    logger.info("finance_rag_trace %s", json.dumps(payload, ensure_ascii=False))


def retrieve_initial(state: RAGState) -> RAGState:
    config = get_finance_rag_config()
    query = state["question"]
    emit_rag_step("🔍", "正在检索知识库...", f"问题: {query[:120]}")
    retrieved = retrieve_documents(
        query,
        top_k=config["final_top_k"],
        candidate_k=config["candidate_k"],
        apply_page_merge=config["enable_page_merge"],
    )

    initial_candidates = retrieved.get("candidate_docs", [])
    final_docs = retrieved.get("final_retrieved_docs", [])
    context_docs = retrieved.get("context_docs", [])
    retrieve_meta = retrieved.get("meta", {})
    context = _format_docs(context_docs)

    emit_rag_step(
        "🧱",
        "候选召回",
        f"candidate_k={config['candidate_k']}，召回 {len(initial_candidates)} 个候选块",
    )
    emit_rag_step(
        "📚",
        "最终上下文",
        f"final_top_k={config['final_top_k']}，上下文块 {len(context_docs)} 个",
    )

    rag_trace = {
        "tool_used": True,
        "tool_name": "search_knowledge_base",
        "query": query,
        "original_question": query,
        "rewritten_question": query,
        "rewrite_used": False,
        "rewrite_strategy": "none",
        "candidate_k": config["candidate_k"],
        "final_top_k": config["final_top_k"],
        "two_stage_retrieval": retrieve_meta.get("two_stage_retrieval"),
        "doc_stage_top_n": retrieve_meta.get("doc_stage_top_n"),
        "page_stage_top_n": retrieve_meta.get("page_stage_top_n"),
        "initial_retrieved_chunks": _normalize_trace_chunks(initial_candidates),
        "expanded_retrieved_chunks": [],
        "retrieval_stage": "initial",
        "retrieval_mode": retrieve_meta.get("retrieval_mode"),
        "retrieve_error": retrieve_meta.get("retrieve_error"),
        "leaf_retrieve_level": retrieve_meta.get("leaf_retrieve_level"),
        "step_back_question": "",
        "step_back_answer": "",
    }
    rag_trace = _update_trace_from_finalization(
        rag_trace,
        {
            "final_retrieved_docs": final_docs,
            "context_docs": context_docs,
            "meta": retrieve_meta,
        },
    )

    return {
        "query": query,
        "docs": context_docs,
        "context": context,
        "initial_candidate_docs": initial_candidates,
        "expanded_candidate_docs": [],
        "rewritten_question": query,
        "rewrite_used": False,
        "step_back_question": "",
        "step_back_answer": "",
        "rag_trace": rag_trace,
    }


def grade_documents_node(state: RAGState) -> RAGState:
    config = get_finance_rag_config()
    if not config["enable_step_back"]:
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update(
            {
                "grade_score": "skipped",
                "grade_route": "generate_answer",
                "rewrite_needed": False,
                "rewrite_used": False,
                "rewrite_strategy": "none",
                "rewritten_question": state["question"],
            }
        )
        return {"route": "generate_answer", "rag_trace": rag_trace}

    grader = _get_grader_model()
    emit_rag_step("📊", "正在评估当前检索是否足够...")
    if not grader:
        rag_trace = state.get("rag_trace", {}) or {}
        rag_trace.update(
            {
                "grade_score": "unknown",
                "grade_route": "rewrite_question",
                "rewrite_needed": True,
            }
        )
        return {"route": "rewrite_question", "rag_trace": rag_trace}

    question = state["question"]
    context = state.get("context", "")
    response = grader.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": GRADE_PROMPT.format(question=question, context=context)}]
    )
    score = (response.binary_score or "").strip().lower()
    route = "generate_answer" if score == "yes" else "rewrite_question"
    emit_rag_step("✅" if route == "generate_answer" else "⚠️", "相关性判断完成", f"结果: {route}")
    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(
        {
            "grade_score": score,
            "grade_route": route,
            "rewrite_needed": route == "rewrite_question",
        }
    )
    return {"route": route, "rag_trace": rag_trace}


def rewrite_question_node(state: RAGState) -> RAGState:
    config = get_finance_rag_config()
    if not config["enable_step_back"]:
        return {
            "rewritten_question": state["question"],
            "rewrite_used": False,
            "step_back_question": "",
            "step_back_answer": "",
            "rag_trace": {
                **(state.get("rag_trace", {}) or {}),
                "rewrite_used": False,
                "rewrite_strategy": "none",
                "rewritten_question": state["question"],
            },
        }

    question = state["question"]
    emit_rag_step("✏️", "正在生成保守改写...", "保留公司名、年份、指标名、单位等关键词")
    step_back = step_back_expand(question)
    rewritten_question = (step_back.get("expanded_query") or question).strip() or question
    rewrite_used = rewritten_question != question

    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(
        {
            "rewrite_used": rewrite_used,
            "rewrite_strategy": "step_back" if rewrite_used else "none",
            "rewritten_question": rewritten_question,
            "step_back_question": step_back.get("step_back_question", "") or "",
            "step_back_answer": step_back.get("step_back_answer", "") or "",
        }
    )
    return {
        "rewritten_question": rewritten_question,
        "rewrite_used": rewrite_used,
        "step_back_question": step_back.get("step_back_question", "") or "",
        "step_back_answer": step_back.get("step_back_answer", "") or "",
        "rag_trace": rag_trace,
    }


def retrieve_expanded(state: RAGState) -> RAGState:
    config = get_finance_rag_config()
    rewritten_question = (state.get("rewritten_question") or state["question"]).strip()
    rewrite_used = bool(state.get("rewrite_used")) and rewritten_question != state["question"]

    if not rewrite_used:
        return {"rag_trace": state.get("rag_trace")}

    emit_rag_step("🔄", "使用改写问题补充召回...", rewritten_question[:120])
    expanded = retrieve_candidate_documents(rewritten_question, candidate_k=config["candidate_k"])
    expanded_candidates = expanded.get("docs", [])
    combined_candidates = _deduplicate_docs((state.get("initial_candidate_docs") or []) + expanded_candidates)

    finalization = finalize_retrieved_documents(
        state["question"],
        combined_candidates,
        final_top_k=config["final_top_k"],
        enable_page_merge=config["enable_page_merge"],
        adjacent_page_window=config["adjacent_page_window"],
        adjacent_chunk_window=config["adjacent_chunk_window"],
    )
    final_docs = finalization.get("final_retrieved_docs", [])
    context_docs = finalization.get("context_docs", [])
    context = _format_docs(context_docs)

    emit_rag_step(
        "📚",
        "统一重排完成",
        f"初始 {len(state.get('initial_candidate_docs') or [])} + 改写 {len(expanded_candidates)} -> 最终 {len(final_docs)}",
    )

    rag_trace = state.get("rag_trace", {}) or {}
    rag_trace.update(
        {
            "expanded_retrieved_chunks": _normalize_trace_chunks(expanded_candidates),
            "rewrite_used": True,
            "rewrite_strategy": "step_back",
            "rewritten_question": rewritten_question,
            "retrieval_stage": "expanded",
            "retrieval_mode": expanded.get("meta", {}).get("retrieval_mode"),
        }
    )
    rag_trace = _update_trace_from_finalization(rag_trace, finalization)

    return {
        "docs": context_docs,
        "context": context,
        "expanded_candidate_docs": expanded_candidates,
        "rag_trace": rag_trace,
    }


def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve_initial", retrieve_initial)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("rewrite_question", rewrite_question_node)
    graph.add_node("retrieve_expanded", retrieve_expanded)

    graph.set_entry_point("retrieve_initial")
    graph.add_edge("retrieve_initial", "grade_documents")
    graph.add_conditional_edges(
        "grade_documents",
        lambda state: state.get("route"),
        {
            "generate_answer": END,
            "rewrite_question": "rewrite_question",
        },
    )
    graph.add_edge("rewrite_question", "retrieve_expanded")
    graph.add_edge("retrieve_expanded", END)
    return graph.compile()


rag_graph = build_rag_graph()


def run_rag_graph(question: str) -> dict:
    result = rag_graph.invoke(
        {
            "question": question,
            "query": question,
            "context": "",
            "docs": [],
            "route": None,
            "rewritten_question": None,
            "rewrite_used": None,
            "step_back_question": None,
            "step_back_answer": None,
            "initial_candidate_docs": [],
            "expanded_candidate_docs": [],
            "rag_trace": None,
        }
    )
    _log_rag_diagnostics(result.get("rag_trace") if isinstance(result, dict) else None)
    return result
