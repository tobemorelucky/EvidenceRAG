"""Feature-gated Conversation-aware RAG v5 orchestration."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.conversation_understanding_v3 import understand_conversation_v3
from backend.memory.conversation_policy import build_conversation_policy
from backend.memory.memory_update import MemoryUpdateService
from backend.memory.token_budget_manager import TokenBudgetManager
from backend.rag_trace_service import RagTraceService, build_rag_trace_payload


logger = logging.getLogger(__name__)


def conversation_memory_v5_enabled() -> bool:
    return os.getenv("ENABLE_CONVERSATION_MEMORY", "false").strip().lower() in {"1", "true", "yes", "on"}


class ConversationMemoryV5Service:
    def __init__(
        self,
        memory: MemoryUpdateService | None = None,
        budget_manager: TokenBudgetManager | None = None,
        *,
        understanding_fn=None,
        retrieval_fn=None,
        answer_fn=None,
        trace_service: RagTraceService | None = None,
    ):
        self.memory = memory or MemoryUpdateService()
        self.budget_manager = budget_manager or TokenBudgetManager()
        self.understanding_fn = understanding_fn or understand_conversation_v3
        self.retrieval_fn = retrieval_fn
        self.answer_fn = answer_fn
        self.trace_service = trace_service or RagTraceService()

    def create(self, user_id: str, metadata: dict | None = None) -> dict:
        conversation, trace = self.memory.create_conversation(user_id, metadata=metadata)
        return {**conversation, "memory_trace": trace}

    @staticmethod
    def _history(messages: list[dict], summary: str) -> list[Any]:
        result: list[Any] = []
        if summary:
            result.append(SystemMessage(content=f"Conversation memory summary:\n{summary}"))
        for item in messages:
            cls = HumanMessage if item.get("role") == "user" else AIMessage if item.get("role") == "assistant" else SystemMessage
            result.append(cls(content=str(item.get("content") or "")))
        return result

    def chat(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        *,
        profile: str | None = None,
        execution_mode: str | None = None,
    ) -> dict:
        if self.retrieval_fn is None:
            from backend.rag_orchestrator import prepare_rag_response
            retrieval_fn = prepare_rag_response
        else:
            retrieval_fn = self.retrieval_fn
        if self.answer_fn is None:
            from backend.answer_generator import generate_answer
            answer_fn = generate_answer
        else:
            answer_fn = self.answer_fn

        total_started = time.perf_counter()
        state, state_trace = self.memory.load_state(user_id, conversation_id)
        if state is None:
            raise KeyError("conversation not found")
        evidence_state, evidence_cache_trace = self.memory.load_evidence_state(user_id, conversation_id)
        previous_evidence = str(evidence_state.get("evidence") or "")

        understanding_started = time.perf_counter()
        decision, understanding_trace = self.understanding_fn(message, state)
        understanding_ms = round((time.perf_counter() - understanding_started) * 1000, 2)
        policy = build_conversation_policy(message, decision, has_previous_evidence=bool(previous_evidence))

        retrieval_started = time.perf_counter()
        rag_result: dict[str, Any] = {}
        fresh_evidence = ""
        if policy.retrieval:
            rag_result = retrieval_fn(
                policy.retrieval_query,
                profile=profile,
                mode=execution_mode or "static",
            )
            fresh_evidence = str(rag_result.get("evidence") or "")
        retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000, 2)

        if policy.reuse_previous_evidence and fresh_evidence:
            evidence = f"Previous evidence:\n{previous_evidence}\n\n---\n\nFresh evidence:\n{fresh_evidence}"
            evidence_source = "previous_and_fresh"
        elif policy.reuse_previous_evidence:
            evidence, evidence_source = previous_evidence, "previous"
        else:
            evidence, evidence_source = fresh_evidence, "fresh"

        messages = [
            {"role": turn.role, "content": turn.content, "evidence_refs": list(turn.evidence_refs)}
            for turn in state.turns
        ]
        latest_summary = self.memory.latest_summary(user_id, conversation_id)
        budgeted = self.budget_manager.build_context(
            messages,
            summary=str((latest_summary or {}).get("summary") or ""),
            evidence=evidence,
        )
        history = self._history(budgeted["messages"], budgeted["summary"])

        answer_started = time.perf_counter()
        answer, usage = answer_fn(
            message,
            budgeted["evidence"],
            history=history,
            profile=profile,
            prompt_mode="baseline",
        )
        answer_ms = round((time.perf_counter() - answer_started) * 1000, 2)
        citations = list(rag_result.get("citations") or evidence_state.get("citations") or [])
        trace = {
            "conversation_understanding": decision.to_dict(),
            "conversation_understanding_trace": understanding_trace,
            "policy_decision": policy.to_dict(),
            "retrieval_decision": {
                "called": policy.retrieval,
                "query": policy.retrieval_query if policy.retrieval else "",
                "route_reason": rag_result.get("route_reason"),
            },
            "evidence_reuse": {
                "source": evidence_source,
                "previous_available": bool(previous_evidence),
                "previous_reused": policy.reuse_previous_evidence and bool(previous_evidence),
                "fresh_chars": len(fresh_evidence),
                "final_chars": len(budgeted["evidence"]),
                "truncated": budgeted["evidence_truncated"],
            },
            "memory": {"state": state_trace, "evidence_cache": evidence_cache_trace},
            "latency_ms": {
                "understanding": understanding_ms,
                "retrieval": retrieval_ms,
                "answer": answer_ms,
                "total": round((time.perf_counter() - total_started) * 1000, 2),
            },
        }
        trace_source_result = rag_result or {
            "rag_trace": evidence_state.get("rag_trace") or {},
            "citations": evidence_state.get("citations") or [],
        }
        trace_payload = build_rag_trace_payload(
            conversation_id=conversation_id,
            user_query=message,
            decision=decision.to_dict(),
            policy=policy.to_dict(),
            rag_result=trace_source_result,
            citations=citations,
            answer_model=os.getenv("MODEL", ""),
            understanding_usage=dict(understanding_trace.get("usage") or {}),
            answer_usage=usage,
            latency_ms=trace["latency_ms"],
        )
        try:
            persisted_trace = self.trace_service.save(user_id, trace_payload)
            trace["observability"] = {
                "persisted": True, "trace_id": persisted_trace["trace_id"],
            }
        except Exception as exc:
            logger.exception("conversation v6 trace persistence failed conversation_id=%s", conversation_id)
            trace["observability"] = {
                "persisted": False, "error": f"{type(exc).__name__}: {exc}",
            }
        logger.info(
            "conversation_memory_v5 conversation_id=%s understanding=%s policy=%s retrieval=%s "
            "evidence_reuse=%s latency_ms=%s",
            conversation_id,
            decision.to_dict(),
            policy.to_dict(),
            trace["retrieval_decision"],
            trace["evidence_reuse"],
            trace["latency_ms"],
        )
        self.memory.append_message(user_id, conversation_id, "user", message)
        evidence_refs = [str(item.get("id") or "") for item in citations if item.get("id")]
        self.memory.append_message(
            user_id, conversation_id, "assistant", answer,
            evidence_refs=evidence_refs, trace=trace,
        )
        self.memory.save_evidence_state(
            user_id,
            conversation_id,
            {"evidence": budgeted["evidence"], "citations": citations, "rag_trace": rag_result.get("rag_trace")},
        )
        return {
            "conversation_id": conversation_id,
            "response": answer,
            "citations": citations,
            "usage": usage,
            "trace": trace,
        }

    def traces(self, user_id: str, conversation_id: str, *, limit: int = 100) -> list[dict]:
        return self.trace_service.list_for_conversation(user_id, conversation_id, limit=limit)

    def list_conversations(self, user_id: str) -> list[dict]:
        return self.memory.list_conversations(user_id)

    def messages(self, user_id: str, conversation_id: str) -> list[dict]:
        return self.memory.list_messages(user_id, conversation_id)

    def delete(self, user_id: str, conversation_id: str) -> tuple[bool, dict]:
        return self.memory.delete_conversation(user_id, conversation_id)


conversation_memory_v5_service = ConversationMemoryV5Service()
