"""Feature-flagged, question-only financial operation planner candidate."""

from __future__ import annotations

import json
import os
import re


PLANNER_MODEL = "deepseek-v4-flash-ga-260731"
TASK_TYPES = {"calculation", "comparison", "trend", "lookup", "judgment"}
REQUIRED_KEYS = {"task_type", "entity", "period", "metric", "required_facts", "operation"}
OPTIONAL_KEYS = {"formula", "direction"}

PLANNER_PROMPT = """You are a financial operation planner. Analyze only the question.
Do not answer it and do not use external knowledge, a reference answer, or gold evidence.
Return exactly one JSON object with no Markdown and no extra keys:
{{
  "task_type": "calculation|comparison|trend|lookup|judgment",
  "entity": "requested entity or scope, or unknown",
  "period": "requested fiscal period, or unknown",
  "metric": "requested metric or fact",
  "required_facts": ["evidence facts or operands required"],
  "operation": "concise operation",
  "formula": "formula if required, otherwise null",
  "direction": "comparison or selection direction if required, otherwise null"
}}
The plan is procedural metadata and never factual evidence.

Question:
{question}"""

GUIDANCE_TEMPLATE = """Financial operation plan (procedure only; not factual evidence):
{plan}

Use the plan only when every fact is supported by the unchanged Evidence below. Do not treat the plan as a source of values.

Original Evidence (unchanged):
{evidence}"""

_CALCULATION = re.compile(
    r"\b(calculate|computed?|formula|ratio|margin|turnover|growth|percentage|percent|average|dpo|eps|working capital|round)\b",
    re.IGNORECASE,
)
_COMPARISON = re.compile(
    r"\b(compare|comparison|versus|vs\.?|higher|lower|highest|lowest|most|least|difference|between)\b",
    re.IGNORECASE,
)
_TREND = re.compile(
    r"\b(increase(?:d)?|decrease(?:d)?|improv(?:e|ed|ing)|declin(?:e|ed|ing)|trend|accelerat(?:e|ed|ing)|decelerat(?:e|ed|ing)|fluctuat(?:e|ed|ing))\b",
    re.IGNORECASE,
)
_JUDGMENT = re.compile(
    r"\b(reasonably|healthy|meaningful|useful|relevant|capital[- ]intensive|adequate|sustainable|expected|risk(?:y)?|appropriate)\b",
    re.IGNORECASE,
)


def planner_enabled(value: str | bool | None = None) -> bool:
    if isinstance(value, bool):
        return value
    resolved = value if value is not None else os.getenv("FINANCIAL_OPERATION_PLANNER_ENABLED", "false")
    return str(resolved).strip().lower() in {"1", "true", "yes", "on"}


def planner_trigger(question: str) -> tuple[bool, str]:
    for reason, pattern in (
        ("calculation_or_metric", _CALCULATION),
        ("comparison", _COMPARISON),
        ("trend", _TREND),
        ("judgment", _JUDGMENT),
    ):
        if pattern.search(question or ""):
            return True, reason
    return False, "lookup"


def parse_plan(text: str) -> dict:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict) or set(payload) - REQUIRED_KEYS - OPTIONAL_KEYS:
        raise ValueError("Planner output must contain only schema keys")
    if not REQUIRED_KEYS.issubset(payload) or payload.get("task_type") not in TASK_TYPES:
        raise ValueError("Planner output has missing keys or invalid task_type")
    if not all(isinstance(payload.get(key), str) and payload[key].strip() for key in ("entity", "period", "metric", "operation")):
        raise ValueError("Planner scalar fields must be non-empty strings")
    facts = payload.get("required_facts")
    if not isinstance(facts, list) or not all(isinstance(item, str) and item.strip() for item in facts):
        raise ValueError("required_facts must be a list of non-empty strings")
    for key in OPTIONAL_KEYS:
        if key in payload and payload[key] is not None and not isinstance(payload[key], str):
            raise ValueError(f"{key} must be a string or null")
    return payload


def create_planner_model():
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model=PLANNER_MODEL,
        model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        temperature=0,
        max_completion_tokens=512,
        timeout=60,
        max_retries=0,
        extra_body={"thinking": {"type": "disabled"}},
    )


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text") or "") if isinstance(item, dict) else str(item) for item in content)
    return str(content or "")


def plan_question(question: str, model=None) -> tuple[dict, dict, str]:
    from langchain_core.messages import HumanMessage, SystemMessage

    response = (model or create_planner_model()).invoke([
        SystemMessage(content="Return strict JSON only."),
        HumanMessage(content=PLANNER_PROMPT.format(question=question)),
    ])
    raw = _content_text(response.content).strip()
    return parse_plan(raw), dict(getattr(response, "usage_metadata", {}) or {}), raw


def attach_plan(evidence: str, plan: dict) -> str:
    plan_json = json.dumps(plan, ensure_ascii=False, separators=(",", ":"))
    return GUIDANCE_TEMPLATE.format(plan=plan_json, evidence=evidence)
