"""One-shot calculation retry helpers for offline shadow evaluation only.

This module deliberately does not import or mutate ``answer_generator``.  It
reconstructs the frozen clean-baseline messages and adds exactly one short
system instruction for an eligible retry.
"""

from __future__ import annotations

import os
import re
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from prompts import (
    CLEAN_BASELINE_ANSWER_SYSTEM_PROMPT,
    CLEAN_BASELINE_ANSWER_USER_TEMPLATE,
    CLEAN_BASELINE_PROMPT_VERSION,
)


RETRY_SYSTEM_INSTRUCTION = """Recheck calculation using the provided evidence.
If required operands exist, compute explicitly.
Do not refuse when operands are available."""

_CALCULATION_QUESTION_RE = re.compile(
    r"\b(?:ratio|margin|turnover|growth|percentage|percent)\b",
    re.IGNORECASE,
)
_ELIGIBLE_WARNING_ALIASES = {
    "formula_failure": "formula_failure",
    "formula_inconsistent_or_not_verifiable": "formula_failure",
    "unnecessary_refusal": "unnecessary_refusal",
    "unnecessary_refusal_with_available_operands": "unnecessary_refusal",
}
_REFUSAL_RE = re.compile(
    r"\b(?:cannot|can't|unable to|insufficient|not enough|does not provide|"
    r"do not provide|no information|cannot be determined|cannot determine|"
    r"cannot calculate|not possible to calculate)\b",
    re.IGNORECASE,
)


def retry_eligibility(question: str, warnings: list[str] | tuple[str, ...]) -> dict[str, Any]:
    """Return a transparent, generic eligibility decision for the shadow retry."""
    question_match = _CALCULATION_QUESTION_RE.search(str(question or ""))
    normalized_warnings = sorted({
        _ELIGIBLE_WARNING_ALIASES[warning]
        for warning in (warnings or [])
        if warning in _ELIGIBLE_WARNING_ALIASES
    })
    return {
        "eligible": bool(question_match and normalized_warnings),
        "question_type_term": question_match.group(0).casefold() if question_match else None,
        "eligible_warnings": normalized_warnings,
    }


def refusal_detected(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(str(answer or "")))


def build_retry_messages(question: str, evidence: str) -> list:
    """Rebuild clean-baseline messages plus the sole targeted retry instruction."""
    return [
        SystemMessage(
            content=(
                f"Prompt-Version: {CLEAN_BASELINE_PROMPT_VERSION}\n\n"
                f"{CLEAN_BASELINE_ANSWER_SYSTEM_PROMPT}"
            )
        ),
        SystemMessage(content=RETRY_SYSTEM_INSTRUCTION),
        HumanMessage(
            content=CLEAN_BASELINE_ANSWER_USER_TEMPLATE.format(
                question=question,
                evidence=evidence,
            )
        ),
    ]


def create_shadow_model():
    """Create a lazy, isolated model with the frozen Jina baseline settings."""
    return init_chat_model(
        model=os.getenv("MODEL", "deepseek-v4-flash-ga-260731"),
        model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=float(os.getenv("ANSWER_TEMPERATURE", "0.1")),
        max_completion_tokens=int(os.getenv("ANSWER_MAX_COMPLETION_TOKENS", "1024")),
        timeout=float(os.getenv("ANSWER_TIMEOUT_SECONDS", "60")),
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "") if isinstance(block, dict) and block.get("type") == "text" else str(block)
            for block in content
        )
    return str(content or "")


def invoke_retry(model, question: str, evidence: str) -> tuple[str, dict]:
    response = model.invoke(build_retry_messages(question, evidence))
    usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {}).get("token_usage") or {}
    return content_text(getattr(response, "content", response)), dict(usage or {})

