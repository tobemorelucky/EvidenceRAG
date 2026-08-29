"""Judge a local FinanceBench answer JSONL without accessing LangSmith."""

import argparse
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from financebench_judge_common import JUDGE_PROMPT, _judge_model, _judge_with_retry


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
load_dotenv(ROOT / ".env", override=True)
os.environ.update({
    "LANGSMITH_TRACING": "false",
    "LANGSMITH_TRACING_V2": "false",
    "LANGCHAIN_TRACING_V2": "false",
})


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    with args.dataset.open("r", encoding="utf-8-sig", newline="") as handle:
        references = {str(row.get("financebench_id") or ""): row for row in csv.DictReader(handle)}
    answers = _read_jsonl(args.answers)
    if args.limit > 0:
        answers = answers[: args.limit]
    if not answers:
        raise SystemExit("No local answers found.")

    model = _judge_model()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[setup] local_answers={args.answers} runs={len(answers)} "
        f"judge={os.getenv('JUDGE_MODEL', 'deepseek-v4-pro-ga-260813')}",
        flush=True,
    )
    with args.output.open("a" if args.append else "w", encoding="utf-8") as handle:
        for index, answer in enumerate(answers, 1):
            financebench_id = str(answer.get("financebench_id") or "")
            reference = references.get(financebench_id)
            if reference is None:
                raise SystemExit(f"Missing reference row for {financebench_id}.")
            verdict = _judge_with_retry(
                model,
                JUDGE_PROMPT.format(
                    question=reference.get("question", ""),
                    reference=reference.get("answer", ""),
                    answer=answer.get("answer", ""),
                ),
            )
            record = {
                "run_id": str(answer.get("evaluation_run_id") or answer.get("application_trace_id") or financebench_id),
                "reference_example_id": "",
                "financebench_id": financebench_id,
                "question": str(reference.get("question") or ""),
                "judge_model": os.getenv("JUDGE_MODEL", "deepseek-v4-pro-ga-260813"),
                "thinking": os.getenv("JUDGE_THINKING_MODE", "disabled"),
                **verdict,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index:02d}/{len(answers)}] {verdict['verdict']}", flush=True)


if __name__ == "__main__":
    main()
