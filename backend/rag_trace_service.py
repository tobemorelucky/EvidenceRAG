"""Persistent observability records for Conversation-aware RAG v6 shadow."""

from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

try:
    from database import SessionLocal
except ModuleNotFoundError:
    from backend.database import SessionLocal

try:
    from backend.memory.persistent_memory_store import MemoryBase, MemoryConversation, _utcnow
except ModuleNotFoundError:
    from memory.persistent_memory_store import MemoryBase, MemoryConversation, _utcnow


class RagTraceRecord(MemoryBase):
    __tablename__ = "rag_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    conversation_ref_id: Mapped[int] = mapped_column(
        ForeignKey("memory_conversations.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    conversation_id: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    conversation_understanding: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    policy_decision: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    standalone_query: Mapped[str] = mapped_column(Text, default="", nullable=False)
    retrieval_executed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    dense_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    bm25_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rrf_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rerank_parameters: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    evidence_items: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    answer_model: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    token_usage: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    latency_ms: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def build_rag_trace_payload(
    *,
    conversation_id: str,
    user_query: str,
    decision: dict,
    policy: dict,
    rag_result: dict,
    citations: list[dict],
    answer_model: str,
    understanding_usage: dict | None,
    answer_usage: dict | None,
    latency_ms: dict,
    profile: str | None = None,
    execution_mode: str | None = None,
    answer_prompt_mode: str | None = None,
) -> dict:
    rag_trace = dict(rag_result.get("rag_trace") or {})
    policy_payload = dict(policy)
    policy_payload.update({
        "profile": profile or rag_trace.get("profile") or os.getenv("RAG_PROFILE", ""),
        "execution_mode": execution_mode or rag_trace.get("execution_mode") or "static",
        "answer_prompt_mode": answer_prompt_mode or os.getenv("ANSWER_PROMPT_MODE", "baseline"),
        "profile_config": rag_trace.get("profile_config"),
        "query_rewrite_enabled": rag_trace.get("query_rewrite_enabled"),
        "query_rewrite_executed": bool(
            rag_trace.get("query_rewrite_executed")
            if "query_rewrite_executed" in rag_trace
            else policy.get("rewrite")
        ),
        "context_budget": rag_trace.get("context_budget"),
        "answer_prompt_name": rag_trace.get("answer_prompt_name"),
        "retrieval_config": dict(rag_trace.get("retrieval_config") or {}),
        "jina_config": dict(rag_trace.get("jina_config") or {}),
        "prompt_config": dict(rag_trace.get("prompt_config") or {}),
        "answer_config": dict(rag_trace.get("answer_config") or {}),
        "answer_temperature": rag_trace.get("answer_temperature"),
        "answer_thinking": rag_trace.get("answer_thinking"),
        "answer_max_tokens": rag_trace.get("answer_max_tokens"),
    })
    evidence_items = [
        {
            "id": item.get("id"), "filename": item.get("filename"),
            "page_number": item.get("page_number"), "text": str(item.get("text") or "")[:1200],
            "score": item.get("score"),
        }
        for item in citations
    ]
    if not evidence_items:
        evidence_items = [
            {
                "chunk_id": item.get("chunk_id"), "filename": item.get("filename"),
                "page_number": item.get("page_number"), "text": str(item.get("text") or "")[:1200],
                "score": item.get("rerank_score", item.get("score")),
            }
            for item in list(rag_result.get("docs") or [])[:20]
        ]
    understanding_usage = dict(understanding_usage or {})
    answer_usage = dict(answer_usage or {})
    query_rewrite_usage = dict(rag_trace.get("query_rewrite_usage") or {})
    token_usage = {
        "conversation_understanding": understanding_usage,
        "query_rewrite": query_rewrite_usage,
        "answer": answer_usage,
        "total_tokens": (
            _as_int(understanding_usage.get("total_tokens"))
            + _as_int(query_rewrite_usage.get("total_tokens"))
            + _as_int(answer_usage.get("total_tokens"))
        ),
    }
    return {
        "conversation_id": conversation_id,
        "user_query": user_query,
        "conversation_understanding": dict(decision),
        "policy_decision": policy_payload,
        "standalone_query": str(decision.get("standalone_query") or ""),
        "retrieval_executed": bool(policy.get("retrieval")),
        "dense_count": _as_int(rag_trace.get("dense_candidate_count", rag_trace.get("initial_dense_candidates"))),
        "bm25_count": _as_int(rag_trace.get("bm25_candidate_count", rag_trace.get("initial_bm25_candidates"))),
        "rrf_count": _as_int(rag_trace.get("rrf_fused_candidate_count", rag_trace.get("rrf_candidates"))),
        "rerank_parameters": {
            "enabled": bool(rag_trace.get("rerank_enabled")),
            "applied": bool(rag_trace.get("rerank_applied")),
            "model": rag_trace.get("rerank_model"),
            "provider": rag_trace.get("rerank_provider"),
            "candidate_k": rag_trace.get("candidate_k"),
            "final_top_k": rag_trace.get("final_top_k"),
            "dense_top_k": rag_trace.get("dense_top_k") or rag_trace.get("dense_candidate_requested") or os.getenv("DENSE_TOP_K"),
            "bm25_top_k": rag_trace.get("bm25_top_k") or rag_trace.get("bm25_candidate_requested") or os.getenv("BM25_TOP_K"),
            "rrf_top_k": rag_trace.get("rrf_top_k") or os.getenv("RRF_TOP_K") or rag_trace.get("candidate_k"),
            "input_k": rag_trace.get("jina_input_k") or rag_trace.get("remote_rerank_candidate_count") or os.getenv("JINA_INPUT_K") or rag_trace.get("candidate_k"),
            "output_k": rag_trace.get("jina_output_k") or os.getenv("JINA_OUTPUT_K") or rag_trace.get("final_top_k"),
            "context_budget": rag_trace.get("context_budget"),
            "profile_config": rag_trace.get("profile_config"),
            "status": rag_trace.get("rerank_status"),
            "input_max_chars": rag_trace.get("jina_input_max_chars"),
        },
        "evidence_items": evidence_items,
        "answer_model": answer_model,
        "token_usage": token_usage,
        "latency_ms": dict(latency_ms),
    }


class RagTraceService:
    def __init__(self, session_factory=SessionLocal):
        self.session_factory = session_factory

    @staticmethod
    def _conversation(db, user_id: str, conversation_id: str) -> MemoryConversation | None:
        return db.query(MemoryConversation).filter(
            MemoryConversation.user_id == user_id,
            MemoryConversation.conversation_id == conversation_id,
        ).first()

    def save(self, user_id: str, payload: dict) -> dict:
        db = self.session_factory()
        try:
            conversation = self._conversation(db, user_id, str(payload.get("conversation_id") or ""))
            if not conversation:
                raise KeyError("conversation not found")
            row = RagTraceRecord(
                trace_id=str(payload.get("trace_id") or uuid.uuid4()),
                conversation_ref_id=conversation.id,
                conversation_id=conversation.conversation_id,
                user_query=str(payload.get("user_query") or ""),
                conversation_understanding=dict(payload.get("conversation_understanding") or {}),
                policy_decision=dict(payload.get("policy_decision") or {}),
                standalone_query=str(payload.get("standalone_query") or ""),
                retrieval_executed=bool(payload.get("retrieval_executed")),
                dense_count=_as_int(payload.get("dense_count")),
                bm25_count=_as_int(payload.get("bm25_count")),
                rrf_count=_as_int(payload.get("rrf_count")),
                rerank_parameters=dict(payload.get("rerank_parameters") or {}),
                evidence_items=list(payload.get("evidence_items") or []),
                answer_model=str(payload.get("answer_model") or ""),
                token_usage=dict(payload.get("token_usage") or {}),
                latency_ms=dict(payload.get("latency_ms") or {}),
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            return self._serialize(row)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def list_for_conversation(
        self, user_id: str, conversation_id: str, *, limit: int = 100,
    ) -> list[dict]:
        db = self.session_factory()
        try:
            conversation = self._conversation(db, user_id, conversation_id)
            if not conversation:
                raise KeyError("conversation not found")
            rows = db.query(RagTraceRecord).filter(
                RagTraceRecord.conversation_ref_id == conversation.id,
            ).order_by(RagTraceRecord.id.desc()).limit(max(1, min(limit, 500))).all()
            rows.reverse()
            return [self._serialize(row) for row in rows]
        finally:
            db.close()

    @staticmethod
    def _serialize(row: RagTraceRecord) -> dict:
        return {
            "trace_id": row.trace_id, "conversation_id": row.conversation_id,
            "user_query": row.user_query,
            "conversation_understanding": dict(row.conversation_understanding or {}),
            "policy_decision": dict(row.policy_decision or {}),
            "standalone_query": row.standalone_query,
            "retrieval_executed": row.retrieval_executed,
            "retrieval_counts": {"dense": row.dense_count, "bm25": row.bm25_count, "rrf": row.rrf_count},
            "rerank_parameters": dict(row.rerank_parameters or {}),
            "evidence_items": list(row.evidence_items or []),
            "answer_model": row.answer_model,
            "token_usage": dict(row.token_usage or {}),
            "latency_ms": dict(row.latency_ms or {}),
            "created_at": row.created_at.isoformat(),
        }


rag_trace_service = RagTraceService()
