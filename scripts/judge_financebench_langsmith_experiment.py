"""Score an already-finished FinanceBench LangSmith experiment with a separate judge model."""

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

from financebench_judge_common import JUDGE_PROMPT, _judge_model, _judge_with_retry


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

def _requested_run_ids(values: list[str] | None) -> set[str]:
    return {
        item.strip()
        for value in (values or [])
        for item in value.split(",")
        if item.strip()
    }


def _wait_for_visible_answer(client: Client, run, timeout_seconds: float):
    """Refresh a just-finished root run until LangSmith exposes its answer."""
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    current = run
    while not str(((getattr(current, "outputs", None) or {}).get("answer") or "")).strip():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(2.0, remaining))
        current = client.read_run(run.id)
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description="Attach DeepSeek-V4-Pro judge feedback to a LangSmith experiment")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "financebench_judge.jsonl")
    parser.add_argument("--run-id", action="append", help="Judge only this LangSmith root run ID; repeat or comma-separate values.")
    parser.add_argument("--append", action="store_true", help="Append selected re-judged runs to an existing JSONL output.")
    parser.add_argument(
        "--visibility-wait-seconds",
        type=float,
        default=30.0,
        help="Wait for freshly uploaded LangSmith root outputs before judging.",
    )
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
            run = _wait_for_visible_answer(client, run, args.visibility_wait_seconds)
            example = client.read_example(run.reference_example_id)
            inputs = getattr(example, "inputs", None) or {}
            expected = getattr(example, "outputs", None) or {}
            example_metadata = getattr(example, "metadata", None) or {}
            outputs = getattr(run, "outputs", None) or {}
            if not str(outputs.get("answer") or "").strip():
                raise SystemExit(
                    f"Run {run.id} still has no visible answer after "
                    f"{args.visibility_wait_seconds:.1f}s; rerun the judge later instead of recording a false score."
                )
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
                "financebench_id": str(example_metadata.get("financebench_id") or inputs.get("financebench_id") or ""),
                "question": str(inputs.get("question") or ""),
                "judge_model": os.getenv("JUDGE_MODEL", "deepseek-v4-pro-ga-260813"),
                "thinking": os.getenv("JUDGE_THINKING_MODE", "disabled"),
                **verdict,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index:02d}/{len(runs)}] {verdict['verdict']}", flush=True)


if __name__ == "__main__":
    main()
