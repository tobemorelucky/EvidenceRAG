"""Create a compact, reproducible FinanceBench summary from answer and judge JSONL files."""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSONL file: {path}") from exc


def _summarize_split(label: str, answers_path: Path, judge_path: Path) -> tuple[dict, set[str]]:
    answers = _read_jsonl(answers_path)
    judges = _read_jsonl(judge_path)
    judges_by_run = {str(item.get("run_id") or ""): item for item in judges}
    answer_ids = [str(item.get("financebench_id") or "") for item in answers]
    if not all(answer_ids) or len(answer_ids) != len(set(answer_ids)):
        raise SystemExit(f"{label}: missing or duplicate financebench_id in answer file.")
    matched = [judges_by_run.get(str(item.get("langsmith_trace_id") or "")) for item in answers]
    if any(item is None for item in matched):
        raise SystemExit(f"{label}: not every answer has a matching Judge run_id.")

    traces = [item.get("rag_trace") or {} for item in answers]
    correct = sum(int(item.get("score") or 0) for item in matched if item)
    return (
        {
            "answers": len(answers),
            "judge_records": len(judges),
            "correct": correct,
            "accuracy": round(correct / len(answers), 4) if answers else 0.0,
            "invalid_judge_outputs": sum(item.get("verdict") == "invalid_judge_output" for item in matched if item),
            "empty_retrievals": sum(trace.get("rrf_fused_candidate_count") == 0 for trace in traces),
            "rerank_providers": dict(sorted(Counter(trace.get("rerank_provider") or "unknown" for trace in traces).items())),
            "remote_rerank_input_chars": sum(int(trace.get("remote_rerank_input_chars") or 0) for trace in traces),
            "answer_file": str(answers_path.relative_to(ROOT)),
            "judge_file": str(judge_path.relative_to(ROOT)),
        },
        set(answer_ids),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed FinanceBench answer and Judge JSONL files.")
    parser.add_argument(
        "--split",
        action="append",
        nargs=3,
        metavar=("LABEL", "ANSWERS_JSONL", "JUDGE_JSONL"),
        required=True,
        help="Repeat for each split to combine.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    splits: dict[str, dict] = {}
    all_ids: set[str] = set()
    for label, answers_name, judge_name in args.split:
        if label in splits:
            raise SystemExit(f"Duplicate split label: {label}")
        summary, ids = _summarize_split(label, ROOT / answers_name, ROOT / judge_name)
        overlap = all_ids & ids
        if overlap:
            raise SystemExit(f"Duplicate FinanceBench IDs across splits: {sorted(overlap)[:3]}")
        splits[label] = summary
        all_ids.update(ids)

    total = sum(item["answers"] for item in splits.values())
    correct = sum(item["correct"] for item in splits.values())
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "splits": splits,
        "combined": {
            "questions": total,
            "unique_financebench_ids": len(all_ids),
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "empty_retrievals": sum(item["empty_retrievals"] for item in splits.values()),
            "remote_rerank_input_chars": sum(item["remote_rerank_input_chars"] for item in splits.values()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["combined"], ensure_ascii=False))
    print(f"Summary: {args.output}")


if __name__ == "__main__":
    main()
