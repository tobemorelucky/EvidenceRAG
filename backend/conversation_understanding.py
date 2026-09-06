"""LLM JSON contract for the Conversation-aware RAG v1 shadow layer."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from backend.memory.context_builder import build_understanding_context
from backend.memory.conversation_state import ConversationState


SCHEMA_KEYS = {
    "need_rag", "need_retrieval", "need_rewrite", "need_previous_context", "standalone_query", "response_mode",
}

SYSTEM_PROMPT = """Analyze the user's current conversational behavior and decide how a RAG application should handle this turn.
Return exactly one JSON object with these keys and no Markdown:
{
  "need_rag": true,
  "need_retrieval": true,
  "need_rewrite": false,
  "need_previous_context": false,
  "standalone_query": "self-contained retrieval query or the original input",
  "response_mode": "a concise free-form description of the appropriate response behavior"
}
Do not classify the request into a fixed intent taxonomy. Infer the behavior needed for this turn.
Set need_rewrite only when prior context is needed to make the retrieval query self-contained.
RAG supplies evidence; it does not force the final response mode. Do not answer the user or provide hidden reasoning."""


@dataclass(frozen=True)
class ConversationDecision:
    need_rag: bool
    need_retrieval: bool
    need_rewrite: bool
    need_previous_context: bool
    standalone_query: str
    response_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def conversation_memory_enabled(value: str | bool | None = None) -> bool:
    if isinstance(value, bool):
        return value
    resolved = value if value is not None else os.getenv("ENABLE_CONVERSATION_MEMORY", "false")
    return str(resolved).strip().lower() in {"1", "true", "yes", "on"}


def first_turn_decision(user_input: str) -> ConversationDecision:
    query = user_input.strip()
    if not query:
        raise ValueError("user_input must not be empty")
    return ConversationDecision(True, True, False, False, query, "answer the current request using retrieved evidence")


def parse_conversation_decision(content: str) -> ConversationDecision:
    payload = json.loads(str(content or "").strip())
    if not isinstance(payload, dict) or set(payload) != SCHEMA_KEYS:
        raise ValueError("Conversation understanding output must contain exactly the required keys")
    for key in ("need_rag", "need_retrieval", "need_rewrite", "need_previous_context"):
        if not isinstance(payload[key], bool):
            raise ValueError(f"{key} must be boolean")
    if not isinstance(payload["standalone_query"], str) or not payload["standalone_query"].strip():
        raise ValueError("standalone_query must be a non-empty string")
    if not isinstance(payload["response_mode"], str) or not payload["response_mode"].strip():
        raise ValueError("response_mode must be a non-empty free-form string")
    if payload["need_rewrite"] and not payload["need_previous_context"]:
        raise ValueError("need_rewrite requires need_previous_context")
    if payload["need_retrieval"] and not payload["need_rag"]:
        raise ValueError("need_retrieval cannot be true when need_rag is false")
    return ConversationDecision(**payload)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text") or "") if isinstance(item, dict) else str(item) for item in content)
    return str(content or "")


def create_understanding_model():
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=os.getenv("CONVERSATION_UNDERSTANDING_MODEL", os.getenv("MODEL", "deepseek-v4-flash-ga-260731")),
        model_provider="openai", api_key=os.getenv("ARK_API_KEY"), base_url=os.getenv("BASE_URL"),
        temperature=0, max_completion_tokens=int(os.getenv("CONVERSATION_UNDERSTANDING_MAX_TOKENS", "256")),
        timeout=float(os.getenv("CONVERSATION_UNDERSTANDING_TIMEOUT_SECONDS", "60")), max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )


def understand_conversation(user_input: str, state: ConversationState, model=None) -> tuple[ConversationDecision, dict]:
    """First turn preserves the original retrieval route; later turns use one LLM decision."""
    if state.is_first_user_turn:
        return first_turn_decision(user_input), {"model_called": False, "reason": "first_turn_original_retrieval"}
    from langchain_core.messages import HumanMessage, SystemMessage

    context = build_understanding_context(state, user_input)
    response = (model or create_understanding_model()).invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=json.dumps(context, ensure_ascii=False)),
    ])
    decision = parse_conversation_decision(_content_text(response.content))
    return decision, {
        "model_called": True,
        "usage": dict(getattr(response, "usage_metadata", {}) or {}),
        "history_turns": len(context["history"]),
    }
