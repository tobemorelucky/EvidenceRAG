"""Evidence-grounded answer generation."""

import os
from typing import AsyncIterator

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from prompts import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
    ANSWER_USER_WITH_POLICY_TEMPLATE,
    CLEAN_BASELINE_ANSWER_SYSTEM_PROMPT,
    CLEAN_BASELINE_ANSWER_USER_TEMPLATE,
    CLEAN_BASELINE_PROMPT_VERSION,
    FINANCE_REASONING_ANSWER_SYSTEM_PROMPT,
    FINANCE_REASONING_PROMPT_VERSION,
    FINANCE_REASONING_V1_1_ANSWER_SYSTEM_PROMPT,
    FINANCE_REASONING_V1_1_PROMPT_VERSION,
    PROMPT_VERSION,
    RAG_CORE_V2_PROMPT_VERSION,
    RAG_CORE_V3_PROMPT_VERSION,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_TEMPLATE,
)
from runtime_profile import uses_clean_baseline_path, uses_rag_core_v2_path, uses_rag_core_v3_path


ANSWER_PROMPT_MODES = {"baseline", "finance_reasoning", "finance_reasoning_v1_1"}


def resolve_answer_prompt_mode(mode: str | None = None) -> str:
    resolved = (mode or os.getenv("ANSWER_PROMPT_MODE", "baseline")).strip().lower()
    if resolved not in ANSWER_PROMPT_MODES:
        raise ValueError(f"Unsupported ANSWER_PROMPT_MODE: {resolved}")
    return resolved


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
    profile: str | None = None,
    prompt_mode: str | None = None,
) -> list:
    clean_baseline = uses_clean_baseline_path(profile)
    mode = resolve_answer_prompt_mode(prompt_mode)
    if mode == "finance_reasoning_v1_1":
        prompt_version = FINANCE_REASONING_V1_1_PROMPT_VERSION
        system_prompt = FINANCE_REASONING_V1_1_ANSWER_SYSTEM_PROMPT
    elif mode == "finance_reasoning":
        prompt_version = FINANCE_REASONING_PROMPT_VERSION
        system_prompt = FINANCE_REASONING_ANSWER_SYSTEM_PROMPT
    else:
        prompt_version = (
            RAG_CORE_V3_PROMPT_VERSION if uses_rag_core_v3_path(profile)
            else RAG_CORE_V2_PROMPT_VERSION if uses_rag_core_v2_path(profile)
            else CLEAN_BASELINE_PROMPT_VERSION if clean_baseline
            else PROMPT_VERSION
        )
        system_prompt = CLEAN_BASELINE_ANSWER_SYSTEM_PROMPT if clean_baseline else ANSWER_SYSTEM_PROMPT
    messages = [SystemMessage(content=f"Prompt-Version: {prompt_version}\n\n{system_prompt}")]
    for message in (history or [])[-12:]:
        if getattr(message, "type", "") in {"human", "ai"}:
            messages.append(message)
    if clean_baseline:
        content = CLEAN_BASELINE_ANSWER_USER_TEMPLATE.format(question=question, evidence=evidence)
    elif task_policy:
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
    profile: str | None = None,
    prompt_mode: str | None = None,
) -> tuple[str, dict]:
    response = model.invoke(build_answer_messages(question, evidence, history, task_policy, profile, prompt_mode))
    usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {}).get("token_usage") or {}
    return _content_text(getattr(response, "content", response)), dict(usage or {})


async def stream_answer(
    question: str,
    evidence: str,
    history: list | None = None,
    task_policy: str = "",
    profile: str | None = None,
    prompt_mode: str | None = None,
) -> AsyncIterator[tuple[str, dict]]:
    usage = {}
    async for chunk in model.astream(build_answer_messages(question, evidence, history, task_policy, profile, prompt_mode)):
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
