"""Run a small FinanceBench end-to-end baseline with LangSmith tracing enabled by .env."""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DATA_PATH = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
load_dotenv(ROOT / ".env", override=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a traced FinanceBench answer baseline")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--mode", choices=("static", "agentic"), default="static")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "financebench_answer_baseline.jsonl")
    parser.add_argument("--resume", action="store_true", help="Skip IDs already written to --output after an interrupted run.")
    args = parser.parse_args()

    sys.path.insert(0, str(BACKEND))
    from answer_generator import generate_answer
    from rag_orchestrator import prepare_rag_response
    from evaluate_financebench_retrieval import select_development_rows

    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    dev_ids = {row.get("financebench_id") or "" for row in select_development_rows(rows)}
    if args.split == "dev":
        rows = [row for row in rows if row.get("financebench_id") in dev_ids]
    elif args.split == "holdout":
        rows = [row for row in rows if row.get("financebench_id") not in dev_ids]
    rows = rows[: max(1, args.limit)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                completed_ids.add(json.loads(line).get("financebench_id"))
            except json.JSONDecodeError:
                continue
        rows = [row for row in rows if row.get("financebench_id") not in completed_ids]
    print(f"[setup] mode={args.mode} split={args.split} questions={len(rows)}", flush=True)
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            prepared = prepare_rag_response(row["question"], profile="finance", mode=args.mode)
            if prepared["evidence_status"] == "insufficient":
                answer, usage = "未检索到足够证据，无法基于当前知识库可靠回答。", {}
            else:
                answer, usage = generate_answer(
                    row["question"],
                    prepared["evidence"],
                    [],
                    prepared.get("task_policy", ""),
                )
            record = {
                "financebench_id": row.get("financebench_id"),
                "question": row["question"],
                "reference_answer": row.get("answer", ""),
                "answer": answer,
                "citations": prepared["citations"],
                "execution_mode": prepared["execution_mode"],
                "route_reason": prepared["route_reason"],
                "evidence_status": prepared["evidence_status"],
                "trace_id": prepared["trace_id"],
                "usage": usage,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{index:02d}/{len(rows)}] {record['financebench_id']} {record['evidence_status']}", flush=True)
    print(f"完成：{args.output} ({datetime.now(timezone.utc).isoformat()})", flush=True)


if __name__ == "__main__":
    main()
