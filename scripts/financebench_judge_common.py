"""FinanceBench judge primitives shared by local and LangSmith runners."""

import json
import os

from langchain.chat_models import init_chat_model


JUDGE_PROMPT = """You are a strict financial QA evaluator. Compare the candidate answer with the reference answer.
Score 1 only when the candidate directly answers the question and is materially consistent with the reference. Score 0 otherwise.
Do not reward unsupported detail. Output JSON only with keys score (0 or 1), verdict (correct or incorrect), and reason (one short sentence).

Question: {question}
Reference answer: {reference}
Candidate answer: {answer}
"""


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(item.get("text") or "") if isinstance(item, dict) else str(item) for item in content)
    return str(content or "")


def _judge_model():
    options = {
        "model": os.getenv("JUDGE_MODEL", "deepseek-v4-pro-ga-260813"),
        "model_provider": "openai",
        "api_key": os.getenv("JUDGE_API_KEY") or os.getenv("ARK_API_KEY"),
        "base_url": os.getenv("JUDGE_BASE_URL") or os.getenv("BASE_URL"),
        "temperature": 0,
        "max_completion_tokens": int(os.getenv("JUDGE_MAX_COMPLETION_TOKENS", "512")),
    }
    thinking_mode = os.getenv("JUDGE_THINKING_MODE", "disabled").strip().lower()
    if thinking_mode in {"enabled", "disabled", "auto"}:
        options["extra_body"] = {"thinking": {"type": thinking_mode}}
    return init_chat_model(**options)


def _parse_verdict(text: str) -> dict:
    try:
        payload = json.loads(text[text.find("{") : text.rfind("}") + 1])
        score = 1 if int(payload.get("score", 0)) == 1 else 0
        return {
            "score": score,
            "verdict": str(payload.get("verdict") or "incorrect"),
            "reason": str(payload.get("reason") or ""),
        }
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"score": 0, "verdict": "invalid_judge_output", "reason": text[:300]}


def _judge_once(model, prompt: str) -> dict:
    response = model.invoke(prompt)
    return _parse_verdict(_content_text(getattr(response, "content", response)))


def _judge_with_retry(model, prompt: str) -> dict:
    verdict = _judge_once(model, prompt)
    if verdict["verdict"] != "invalid_judge_output":
        return verdict
    retry_prompt = (
        "Return exactly one JSON object and no other text. "
        "The allowed keys are score, verdict, reason. score must be 0 or 1.\n\n"
        + prompt
    )
    retry = _judge_once(model, retry_prompt)
    retry["retried"] = True
    return retry
