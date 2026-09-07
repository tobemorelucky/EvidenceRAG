"""Conversation Understanding v3 schema for shadow evaluation only."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from backend.memory.context_builder import build_understanding_context
from backend.memory.conversation_state import ConversationState


SCHEMA_KEYS = {
    "need_rag", "need_retrieval", "depends_on_history", "query_resolution_needed",
    "required_memory_scope", "standalone_query", "response_mode",
}

SYSTEM_PROMPT = """Analyze the current user turn in its conversation. Return exactly one JSON object with no Markdown:
{
  "need_rag": true,
  "need_retrieval": true,
  "depends_on_history": false,
  "query_resolution_needed": false,
  "required_memory_scope": "none",
  "standalone_query": "self-contained representation of the current request",
  "response_mode": "free-form description of appropriate response behavior"
}

Field meanings are independent:
- need_rag: factual knowledge-base evidence is needed, including evidence already present in conversation state.
- need_retrieval: a new retrieval call is needed. It can be false while need_rag is true when existing cited evidence is enough.
- depends_on_history: the current response cannot be completed correctly without reading prior conversation turns.
- query_resolution_needed: the current utterance contains anaphora, omission, inherited conditions, corrections, or references to prior work that must be resolved into a self-contained standalone_query. This describes resolution already performed in your output; it does not ask whether another rewrite call is needed.
- required_memory_scope: use exactly "none" when depends_on_history is false; otherwise briefly name the minimum history needed, such as "last exchange", "relevant history", or "full conversation".
- standalone_query: always resolve required entities, periods, metrics, constraints, and requested action. Do not use unresolved phrases such as "it", "that", "the same", or "previously discussed" when query_resolution_needed is true.
- response_mode: is advisory and is not a fixed intent category. RAG supplies evidence but does not force the final reply style.

Do not answer the user, retrieve information, or expose hidden reasoning."""


@dataclass(frozen=True)
class ConversationDecisionV3:
    need_rag: bool
    need_retrieval: bool
    depends_on_history: bool
    query_resolution_needed: bool
    required_memory_scope: str
    standalone_query: str
    response_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_conversation_decision_v3(content: str) -> ConversationDecisionV3:
    payload = json.loads(str(content or "").strip())
    if not isinstance(payload, dict) or set(payload) != SCHEMA_KEYS:
        raise ValueError("Conversation Understanding v3 output must contain exactly the required keys")
    for key in ("need_rag", "need_retrieval", "depends_on_history", "query_resolution_needed"):
        if not isinstance(payload[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key in ("required_memory_scope", "standalone_query", "response_mode"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    scope_is_none = payload["required_memory_scope"].strip().casefold() == "none"
    if payload["depends_on_history"] == scope_is_none:
        raise ValueError("required_memory_scope must be none exactly when depends_on_history is false")
    if payload["query_resolution_needed"] and not payload["depends_on_history"]:
        raise ValueError("query_resolution_needed requires depends_on_history")
    if payload["need_retrieval"] and not payload["need_rag"]:
        raise ValueError("need_retrieval cannot be true when need_rag is false")
    return ConversationDecisionV3(**payload)


def first_turn_decision_v3(user_input: str) -> ConversationDecisionV3:
    query = str(user_input or "").strip()
    if not query:
        raise ValueError("user_input must not be empty")
    return ConversationDecisionV3(True, True, False, False, "none", query, "answer the current request using retrieved evidence")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text") or "") if isinstance(item, dict) else str(item) for item in content)
    return str(content or "")


def create_understanding_model_v3():
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model="deepseek-v4-flash-ga-260731", model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"), base_url=os.getenv("BASE_URL"), temperature=0,
        max_completion_tokens=int(os.getenv("CONVERSATION_UNDERSTANDING_MAX_TOKENS", "256")),
        timeout=float(os.getenv("CONVERSATION_UNDERSTANDING_TIMEOUT_SECONDS", "60")), max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )


def understand_conversation_v3(user_input: str, state: ConversationState, model=None) -> tuple[ConversationDecisionV3, dict]:
    if state.is_first_user_turn:
        return first_turn_decision_v3(user_input), {"model_called": False, "reason": "first_turn_original_retrieval"}
    from langchain_core.messages import HumanMessage, SystemMessage

    context = build_understanding_context(state, user_input)
    response = (model or create_understanding_model_v3()).invoke([
        SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=json.dumps(context, ensure_ascii=False)),
    ])
    decision = parse_conversation_decision_v3(_content_text(response.content))
    return decision, {
        "model_called": True, "usage": dict(getattr(response, "usage_metadata", {}) or {}),
        "history_turns": len(context["history"]),
    }
