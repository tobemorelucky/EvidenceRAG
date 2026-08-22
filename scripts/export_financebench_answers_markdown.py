"""Export completed FinanceBench answers and Judge feedback to a compact Markdown review file."""

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing input file: {path}")
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSONL file: {path}") from exc


def _citations(record: dict) -> str:
    values = []
    for citation in record.get("citations") or []:
        if not isinstance(citation, dict):
            continue
        filename = str(citation.get("filename") or "unknown file")
        page = citation.get("page_number")
        values.append(f"{filename}, p. {page}" if page is not None else filename)
    return "; ".join(dict.fromkeys(values)) or "None"


def _split_records(label: str, answers_path: Path, judge_path: Path) -> tuple[list[dict], int]:
    answers = _read_jsonl(answers_path)
    judges = {str(item.get("run_id") or ""): item for item in _read_jsonl(judge_path)}
    rows = []
    for answer in answers:
        financebench_id = str(answer.get("financebench_id") or "")
        judge = judges.get(str(answer.get("langsmith_trace_id") or ""))
        if not financebench_id or judge is None:
            raise SystemExit(f"{label}: answer/Judge records cannot be matched.")
        trace = answer.get("rag_trace") or {}
        rows.append(
            {
                "financebench_id": financebench_id,
                "question": str(answer.get("question") or ""),
                "answer": str(answer.get("answer") or ""),
                "citations": _citations(answer),
                "verdict": str(judge.get("verdict") or "incorrect"),
                "score": int(judge.get("score") or 0),
                "judge_reason": str(judge.get("reason") or ""),
                "evidence_status": str(answer.get("evidence_status") or ""),
                "rerank_provider": str(trace.get("rerank_provider") or "unknown"),
            }
        )
    return rows, sum(row["score"] for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export FinanceBench answers and Judge feedback to Markdown.")
    parser.add_argument(
        "--split",
        action="append",
        nargs=3,
        metavar=("LABEL", "ANSWERS_JSONL", "JUDGE_JSONL"),
        required=True,
        help="Repeat for each split to include.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped = []
    all_ids: set[str] = set()
    total_correct = 0
    for label, answers_name, judge_name in args.split:
        rows, correct = _split_records(label, ROOT / answers_name, ROOT / judge_name)
        ids = {row["financebench_id"] for row in rows}
        if len(ids) != len(rows) or all_ids & ids:
            raise SystemExit(f"{label}: duplicate FinanceBench IDs detected.")
        all_ids.update(ids)
        grouped.append((label, rows, correct))
        total_correct += correct

    total = sum(len(rows) for _, rows, _ in grouped)
    lines = ["# EvidenceRAG FinanceBench 100-Question Results", ""]
    lines.extend(
        [
            f"- Questions: {total}",
            f"- Correct: {total_correct}",
            f"- Accuracy: {total_correct / total:.1%}" if total else "- Accuracy: n/a",
            "- Judge: DeepSeek-V4-Pro",
            "",
            "Each entry retains only the question, generated answer, citations, Judge result, and exceptional retrieval state.",
        ]
    )
    for label, rows, correct in grouped:
        lines.extend(["", f"## {label} ({correct}/{len(rows)}, {correct / len(rows):.1%})"])
        for index, row in enumerate(rows, 1):
            verdict = "Correct" if row["score"] else "Incorrect"
            lines.extend(
                [
                    "",
                    f"### {index}. {row['financebench_id']} — {verdict}",
                    "",
                    f"**Question:** {row['question']}",
                    "",
                    f"**Answer:** {row['answer']}",
                    "",
                    f"**Citations:** {row['citations']}",
                    "",
                    f"**Judge:** {row['judge_reason']}",
                ]
            )
            if row["evidence_status"] != "sufficient" or row["rerank_provider"] != "remote":
                lines.extend(
                    [
                        "",
                        f"**Run note:** evidence={row['evidence_status'] or 'unknown'}; rerank={row['rerank_provider']}",
                    ]
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Markdown: {args.output}")


if __name__ == "__main__":
    main()
