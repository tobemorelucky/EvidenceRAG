"""LLM query rewriting shared by the validated shadow and finance online profiles.

The public helper always keeps the original question as the first retrieval
query and limits the complete query set to three.  It remains independent of
retrieval implementation details so the caller controls all retrieval depths.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


MAX_RETRIEVAL_QUERIES = 3

_SYSTEM_PROMPT = """You rewrite a user's financial question for document retrieval.
Return JSON only, using this exact shape: {"queries": ["query 1", "query 2"]}.
Produce at most two concise alternative search queries.
Preserve the requested entity, reporting period, metric, accounting scope, and
calculation operands. Expand useful accounting abbreviations when applicable.
Do not answer the question. Do not invent facts, values, companies, or years."""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "")
            if isinstance(block, dict) and block.get("type") == "text"
            else str(block)
            for block in content
        )
    return str(content or "")


def _parse_queries(content: str) -> list[str]:
    text = str(content or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates = [fenced.group(1)] if fenced else []
    candidates.extend(match.group(0) for match in re.finditer(r"\{.*?\}", text, re.DOTALL))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        values = payload.get("queries") if isinstance(payload, dict) else None
        if isinstance(values, list):
            return [str(value).strip() for value in values if str(value).strip()]
    return []


def normalize_retrieval_queries(question: str, alternatives: list[str]) -> list[str]:
    """Return original-first, distinct queries, capped at three total."""
    original = str(question or "").strip()
    if not original:
        raise ValueError("question must not be empty")
    queries: list[str] = []
    seen: set[str] = set()
    for value in [original, *(alternatives or [])]:
        query = re.sub(r"\s+", " ", str(value or "")).strip()[:500]
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            queries.append(query)
        if len(queries) == MAX_RETRIEVAL_QUERIES:
            break
    return queries


def build_rewrite_messages(question: str) -> list:
    return [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f"Original question:\n{str(question or '').strip()}"),
    ]


def create_query_rewrite_model():
    """Create the isolated DeepSeek Flash client used only by the shadow script."""
    return init_chat_model(
        model=os.getenv("MODEL", "deepseek-v4-flash-ga-260731"),
        model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0.0,
        max_completion_tokens=int(os.getenv("QUERY_REWRITE_MAX_COMPLETION_TOKENS", "256")),
        timeout=float(os.getenv("QUERY_REWRITE_TIMEOUT_SECONDS", "60")),
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )


def rewrite_queries(question: str, model) -> dict[str, Any]:
    """Generate and validate an original-first retrieval query set."""
    response = model.invoke(build_rewrite_messages(question))
    content = _content_text(getattr(response, "content", response))
    queries = normalize_retrieval_queries(question, _parse_queries(content))
    usage = (
        getattr(response, "usage_metadata", None)
        or getattr(response, "response_metadata", {}).get("token_usage")
        or {}
    )
    return {
        "queries": queries,
        "generated_alternatives": queries[1:],
        "raw_response": content,
        "usage": dict(usage or {}),
        "model": os.getenv("MODEL", "deepseek-v4-flash-ga-260731"),
    }
