"""Replay frozen Core v2 candidates through Page Selector v3 without retrieval or LLM calls."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from rag_core_v3 import select_core_v3_pages  # noqa: E402


DEFAULT_ANSWERS = ROOT / "reports" / "evidencerag-rag-core-v2-skills-all100-final_answers.jsonl"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json"
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_OUTPUT = ROOT / "reports" / "rag_core_v3_page_selector_replay.json"


def _read_jsonl(path: Path) -> dict[str, dict]:
    return {
        str(item.get("financebench_id") or ""): item
        for item in (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    }


def _document(value: object) -> str:
    return Path(str(value or "")).stem.casefold()


def _pages(items: list[dict]) -> set[tuple[str, int]]:
    result = set()
    for item in items or []:
        try:
            page = int(item.get("page_number") or 0)
        except (TypeError, ValueError):
            continue
        filename = _document(item.get("filename") or item.get("doc_name"))
        if filename:
            result.add((filename, page))
    return result


def _gold(row: dict) -> set[tuple[str, int]]:
    return {
        (_document(item.get("doc_name")), int(item.get("evidence_page_num") or 0))
        for item in json.loads(row.get("evidence") or "[]")
    }


def _hit(gold: set[tuple[str, int]], actual: set[tuple[str, int]]) -> bool:
    return bool(gold & actual)


def _fixture_ids(fixture: dict, groups: list[str]) -> list[str]:
    result = []
    for group in groups:
        for item in fixture.get(group) or []:
            financebench_id = item if isinstance(item, str) else item.get("financebench_id")
            if financebench_id and financebench_id not in result:
                result.append(financebench_id)
    return result


def evaluate(args: argparse.Namespace) -> dict:
    answers = _read_jsonl(args.answers)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    with args.dataset.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["financebench_id"]: row for row in csv.DictReader(handle)}
    ids = _fixture_ids(fixture, args.group)
    experiments = {}
    for final_page_k in args.final_page_k:
        records = []
        for financebench_id in ids:
            answer = answers[financebench_id]
            trace = answer.get("rag_trace") or {}
            candidates = list(trace.get("initial_retrieved_chunks") or [])
            reranked = list(trace.get("reranked_chunks") or [])
            selected, selection_trace = select_core_v3_pages(
                rows[financebench_id]["question"],
                candidates,
                reranked,
                final_page_k=final_page_k,
            )
            gold = _gold(rows[financebench_id])
            candidate_pages = _pages(candidates)
            selected_pages = _pages(selected)
            records.append({
                "financebench_id": financebench_id,
                "candidate_page_hit": _hit(gold, candidate_pages),
                "selected_page_hit": _hit(gold, selected_pages),
                "selected_pages": selection_trace["selected_pages"],
                "global_escape_pages": selection_trace["global_escape_pages"],
                "selected_document_count": len({_document(item.get("filename")) for item in selected}),
                "average_redundancy": statistics.fmean(selection_trace["selected_page_redundancy"]),
            })
        experiments[str(final_page_k)] = {
            "questions": len(records),
            "candidate_page_hits": sum(item["candidate_page_hit"] for item in records),
            "selected_page_hits": sum(item["selected_page_hit"] for item in records),
            "average_selected_document_count": statistics.fmean(
                item["selected_document_count"] for item in records
            ) if records else 0,
            "average_selected_page_redundancy": statistics.fmean(
                item["average_redundancy"] for item in records
            ) if records else 0,
            "records": records,
        }
    return {
        "source_answers": str(args.answers),
        "fixture": str(args.fixture),
        "groups": args.group,
        "retrieval_calls": 0,
        "llm_calls": 0,
        "experiments": experiments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--group", action="append", choices=("selection_loss10", "candidate_miss10", "correct_regression10"),
        default=[],
    )
    parser.add_argument("--final-page-k", action="append", type=int, default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.group:
        args.group = ["selection_loss10", "correct_regression10"]
    if not args.final_page_k:
        args.final_page_k = [6, 8]
    report = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        key: {name: value for name, value in result.items() if name != "records"}
        for key, result in report["experiments"].items()
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
