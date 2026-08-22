"""Score an already-finished FinanceBench LangSmith experiment with a separate judge model."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langsmith import Client


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

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


def _requested_run_ids(values: list[str] | None) -> set[str]:
    return {
        item.strip()
        for value in (values or [])
        for item in value.split(",")
        if item.strip()
    }


def _parse_verdict(text: str) -> dict:
    try:
        payload = json.loads(text[text.find("{") : text.rfind("}") + 1])
        score = 1 if int(payload.get("score", 0)) == 1 else 0
        return {"score": score, "verdict": str(payload.get("verdict") or "incorrect"), "reason": str(payload.get("reason") or "")}
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"score": 0, "verdict": "invalid_judge_output", "reason": text[:300]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach DeepSeek-V4-Pro judge feedback to a LangSmith experiment")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "financebench_judge.jsonl")
    parser.add_argument("--run-id", action="append", help="Judge only this LangSmith root run ID; repeat or comma-separate values.")
    parser.add_argument("--append", action="store_true", help="Append selected re-judged runs to an existing JSONL output.")
    args = parser.parse_args()

    client = Client()
    runs = list(client.list_runs(project_name=args.experiment_name, is_root=True))
    requested_ids = _requested_run_ids(args.run_id)
    if requested_ids:
        runs = [run for run in runs if str(run.id) in requested_ids]
    if args.limit > 0:
        runs = runs[: args.limit]
    if not runs:
        raise SystemExit("No root runs found. Use the exact Experiment name printed by the baseline script.")
    model = _judge_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[setup] experiment={args.experiment_name} runs={len(runs)} judge={os.getenv('JUDGE_MODEL', 'deepseek-v4-pro-ga-260813')}", flush=True)
    with args.output.open("a" if args.append else "w", encoding="utf-8") as handle:
        for index, run in enumerate(runs, 1):
            example = client.read_example(run.reference_example_id)
            inputs = getattr(example, "inputs", None) or {}
            expected = getattr(example, "outputs", None) or {}
            outputs = getattr(run, "outputs", None) or {}
            prompt = JUDGE_PROMPT.format(
                question=inputs.get("question", ""),
                reference=expected.get("answer", ""),
                answer=outputs.get("answer", ""),
            )
            verdict = _judge_with_retry(model, prompt)
            client.create_feedback(
                run_id=run.id,
                key="answer_correctness_v4_pro",
                score=verdict["score"],
                value=verdict["verdict"],
                comment=verdict["reason"],
                source_info={"model": os.getenv("JUDGE_MODEL", "deepseek-v4-pro-ga-260813")},
            )
            record = {
                "run_id": str(run.id),
                "reference_example_id": str(run.reference_example_id),
                "judge_model": os.getenv("JUDGE_MODEL", "deepseek-v4-pro-ga-260813"),
                "thinking": os.getenv("JUDGE_THINKING_MODE", "disabled"),
                **verdict,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index:02d}/{len(runs)}] {verdict['verdict']}", flush=True)


if __name__ == "__main__":
    main()
