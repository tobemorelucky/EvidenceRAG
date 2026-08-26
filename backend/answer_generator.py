"""Evidence-grounded answer generation."""

import os
from typing import AsyncIterator

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
    ANSWER_USER_WITH_POLICY_TEMPLATE,
    PROMPT_VERSION,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_TEMPLATE,
)


def _create_model():
    options = {
        "model": os.getenv("MODEL"),
        "model_provider": "openai",
        "api_key": os.getenv("ARK_API_KEY"),
        "base_url": os.getenv("BASE_URL"),
        "temperature": float(os.getenv("ANSWER_TEMPERATURE", "0.1")),
        "stream_usage": True,
        "timeout": float(os.getenv("ANSWER_TIMEOUT_SECONDS", "60")),
        "max_retries": int(os.getenv("ANSWER_MAX_RETRIES", "2")),
    }
    max_completion_tokens = os.getenv("ANSWER_MAX_COMPLETION_TOKENS")
    if max_completion_tokens:
        options["max_completion_tokens"] = int(max_completion_tokens)
    thinking_mode = os.getenv("ANSWER_THINKING_MODE", "").strip().lower()
    if thinking_mode in {"enabled", "disabled", "auto"}:
        options["extra_body"] = {"thinking": {"type": thinking_mode}}
    return init_chat_model(**options)


model = _create_model()


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(content or "")


def build_answer_messages(
    question: str,
    evidence: str,
    history: list | None = None,
    task_policy: str = "",
) -> list:
    messages = [SystemMessage(content=f"Prompt-Version: {PROMPT_VERSION}\n\n{ANSWER_SYSTEM_PROMPT}")]
    for message in (history or [])[-12:]:
        if getattr(message, "type", "") in {"human", "ai"}:
            messages.append(message)
    if task_policy:
        content = ANSWER_USER_WITH_POLICY_TEMPLATE.format(
            question=question,
            task_policy=task_policy,
            evidence=evidence,
        )
    else:
        content = ANSWER_USER_TEMPLATE.format(question=question, evidence=evidence)
    messages.append(HumanMessage(content=content))
    return messages


def generate_answer(
    question: str,
    evidence: str,
    history: list | None = None,
    task_policy: str = "",
) -> tuple[str, dict]:
    response = model.invoke(build_answer_messages(question, evidence, history, task_policy))
    usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {}).get("token_usage") or {}
    return _content_text(getattr(response, "content", response)), dict(usage or {})


async def stream_answer(
    question: str,
    evidence: str,
    history: list | None = None,
    task_policy: str = "",
) -> AsyncIterator[tuple[str, dict]]:
    usage = {}
    async for chunk in model.astream(build_answer_messages(question, evidence, history, task_policy)):
        text = _content_text(getattr(chunk, "content", chunk))
        chunk_usage = getattr(chunk, "usage_metadata", None)
        if chunk_usage:
            usage = dict(chunk_usage)
        if text:
            yield text, usage


def summarize_messages(messages: list) -> str:
    conversation = "\n".join(
        f"{'用户' if getattr(message, 'type', '') == 'human' else '系统回答'}: {getattr(message, 'content', '')}"
        for message in messages
        if getattr(message, "type", "") in {"human", "ai"}
    )
    response = model.invoke(
        [
            SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
            HumanMessage(content=SUMMARY_USER_TEMPLATE.format(conversation=conversation)),
        ]
    )
    return _content_text(getattr(response, "content", response))
