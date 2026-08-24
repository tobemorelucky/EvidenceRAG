"""Replace infrastructure-failed FinanceBench runs with validated retry runs."""

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSONL file: {path}") from exc


def _answers_by_id(records: list[dict], label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        financebench_id = str(record.get("financebench_id") or "")
        if not financebench_id or financebench_id in indexed:
            raise SystemExit(f"{label}: missing or duplicate financebench_id: {financebench_id!r}")
        indexed[financebench_id] = record
    return indexed


def _judges_by_run(records: list[dict], label: str) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in records:
        run_id = str(record.get("run_id") or "")
        if not run_id or run_id in indexed:
            raise SystemExit(f"{label}: missing or duplicate run_id: {run_id!r}")
        indexed[run_id] = record
    return indexed


def merge_records(
    base_answers: list[dict],
    base_judges: list[dict],
    retry_answers: list[dict],
    retry_judges: list[dict],
) -> tuple[list[dict], list[dict]]:
    base_by_id = _answers_by_id(base_answers, "base answers")
    retry_by_id = _answers_by_id(retry_answers, "retry answers")
    unknown_ids = set(retry_by_id) - set(base_by_id)
    if unknown_ids:
        raise SystemExit(f"Retry IDs are not present in base answers: {sorted(unknown_ids)}")

    base_judges_by_run = _judges_by_run(base_judges, "base judges")
    retry_judges_by_run = _judges_by_run(retry_judges, "retry judges")
    resolved_answers: list[dict] = []
    resolved_judges: list[dict] = []
    for original in base_answers:
        financebench_id = str(original["financebench_id"])
        answer = retry_by_id.get(financebench_id, original)
        run_id = str(answer.get("langsmith_trace_id") or "")
        judge_source = retry_judges_by_run if financebench_id in retry_by_id else base_judges_by_run
        judge = judge_source.get(run_id)
        if judge is None:
            raise SystemExit(f"No matching Judge record for {financebench_id}, run_id={run_id!r}")
        resolved_answers.append(answer)
        resolved_judges.append(judge)

    if len(resolved_answers) != len(base_answers):
        raise SystemExit("Resolved answer count changed unexpectedly.")
    return resolved_answers, resolved_judges


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace selected FinanceBench base runs with retry runs.")
    parser.add_argument("--base-answers", type=Path, required=True)
    parser.add_argument("--base-judges", type=Path, required=True)
    parser.add_argument("--retry-answers", type=Path, required=True)
    parser.add_argument("--retry-judges", type=Path, required=True)
    parser.add_argument("--output-answers", type=Path, required=True)
    parser.add_argument("--output-judges", type=Path, required=True)
    args = parser.parse_args()

    retry_answers = _read_jsonl(args.retry_answers)
    resolved_answers, resolved_judges = merge_records(
        _read_jsonl(args.base_answers),
        _read_jsonl(args.base_judges),
        retry_answers,
        _read_jsonl(args.retry_judges),
    )
    _write_jsonl(args.output_answers, resolved_answers)
    _write_jsonl(args.output_judges, resolved_judges)
    print(
        json.dumps(
            {
                "answers": len(resolved_answers),
                "judges": len(resolved_judges),
                "replaced_ids": [record["financebench_id"] for record in retry_answers],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
