"""Offline metrics for the fixed RAG Core v2 diagnostic set."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "rag_core_v2_diagnostic_ids.json"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _document(value: object) -> str:
    return Path(str(value or "")).stem.casefold()


def _page_set(items: list[dict]) -> set[tuple[str, int]]:
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


def _gold(row: dict) -> tuple[set[tuple[str, int]], str]:
    evidence = json.loads(row.get("evidence") or "[]")
    pages = {
        (_document(item.get("doc_name")), int(item.get("evidence_page_num") or 0))
        for item in evidence
    }
    text = "\n".join(str(item.get("evidence_text") or "") for item in evidence)
    return pages, text


def _hit(gold: set[tuple[str, int]], actual: set[tuple[str, int]], offset: int = 0) -> bool:
    return any((filename, page + offset) in actual for filename, page in gold)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9.%'-]*", text or "") if len(token) > 1}


def _text_recall(gold_text: str, evidence_text: str) -> float | None:
    gold_tokens = _tokens(gold_text)
    if not gold_tokens or not evidence_text:
        return None
    return len(gold_tokens & _tokens(evidence_text)) / len(gold_tokens)


def _bigram_recall(gold_text: str, evidence_text: str) -> float | None:
    gold = re.findall(r"[A-Za-z0-9][A-Za-z0-9.%'-]*", gold_text.lower())
    evidence = re.findall(r"[A-Za-z0-9][A-Za-z0-9.%'-]*", evidence_text.lower())
    gold_pairs = set(zip(gold, gold[1:]))
    if not gold_pairs or not evidence_text:
        return None
    evidence_pairs = set(zip(evidence, evidence[1:]))
    return len(gold_pairs & evidence_pairs) / len(gold_pairs)


def _gold_rank(gold: set[tuple[str, int]], items: list[dict], *, offset: int = 0) -> int | None:
    for rank, item in enumerate(items or [], 1):
        if _page_set([item]) & {(filename, page + offset) for filename, page in gold}:
            return rank
    return None


def suggest_fixture(clean_answers: Path, rows: dict[str, dict]) -> dict[str, list[str]]:
    answers = _read_jsonl(clean_answers)
    categories = {key: [] for key in (
        "candidate_miss", "candidate_to_context_loss", "gold_context_refusal", "table_or_calculation",
    )}
    refusal = re.compile(r"insufficient|cannot determine|does not contain|not enough|missing", re.I)
    calculation = re.compile(
        r"calculate|ratio|percent|percentage|change|difference|margin|average|working capital|turnover|how much",
        re.I,
    )
    for answer in answers:
        item_id = str(answer.get("financebench_id") or "")
        row = rows.get(item_id)
        if not row:
            continue
        gold_pages, _ = _gold(row)
        trace = answer.get("rag_trace") or {}
        candidate = _page_set(trace.get("initial_retrieved_chunks") or [])
        context = _page_set(trace.get("answer_context_pages") or [])
        candidate_hit = _hit(gold_pages, candidate) or _hit(gold_pages, candidate, -1)
        context_hit = _hit(gold_pages, context) or _hit(gold_pages, context, -1)
        if not candidate_hit:
            categories["candidate_miss"].append(item_id)
        elif not context_hit:
            categories["candidate_to_context_loss"].append(item_id)
        elif refusal.search(str(answer.get("answer") or "")):
            categories["gold_context_refusal"].append(item_id)
        if calculation.search(row.get("question") or ""):
            categories["table_or_calculation"].append(item_id)

    selected: dict[str, list[str]] = {}
    used = set()
    for category, candidates in categories.items():
        chosen = [item for item in candidates if item not in used][:5]
        selected[category] = chosen
        used.update(chosen)
    return selected


def evaluate(answers_path: Path, fixture_path: Path | None, rows: dict[str, dict]) -> dict:
    if fixture_path is None:
        expected_ids = list(rows)
    else:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        expected_ids = [item for values in fixture["categories"].values() for item in values]
    answers = {str(item.get("financebench_id") or ""): item for item in _read_jsonl(answers_path)}
    records = []
    for item_id in expected_ids:
        answer = answers.get(item_id)
        row = rows.get(item_id)
        if not answer or not row:
            records.append({"financebench_id": item_id, "missing": True})
            continue
        gold_pages, gold_text = _gold(row)
        trace = answer.get("rag_trace") or {}
        candidate = _page_set(trace.get("initial_retrieved_chunks") or [])
        selected = _page_set(trace.get("selected_pages") or trace.get("final_selected_pages") or [])
        context = _page_set(trace.get("answer_context_pages") or [])
        citations = _page_set(answer.get("citations") or [])
        gold_documents = {item[0] for item in gold_pages}
        cited_documents = {item[0] for item in citations}
        records.append({
            "financebench_id": item_id,
            "candidate_page_hit": _hit(gold_pages, candidate),
            "candidate_page_hit_offset_minus_one": _hit(gold_pages, candidate, -1),
            "selected_page_hit": _hit(gold_pages, selected),
            "selected_page_hit_offset_minus_one": _hit(gold_pages, selected, -1),
            "context_page_hit": _hit(gold_pages, context),
            "context_page_hit_offset_minus_one": _hit(gold_pages, context, -1),
            "citation_document_hit": bool(gold_documents & cited_documents),
            "gold_evidence_token_recall": _text_recall(gold_text, str(answer.get("evidence_context") or "")),
            "gold_evidence_bigram_recall": _bigram_recall(gold_text, str(answer.get("evidence_context") or "")),
            "gold_candidate_rank": _gold_rank(gold_pages, trace.get("initial_retrieved_chunks") or []),
            "gold_rerank_rank": _gold_rank(gold_pages, trace.get("reranked_chunks") or []),
            "answer_input_tokens": int((answer.get("usage") or {}).get("input_tokens") or 0),
            "answer_context_chars": int(trace.get("answer_context_chars") or 0),
            "tables_available": int(trace.get("tables_available_on_selected_pages") or 0),
            "tables_attached": int(trace.get("tables_attached") or 0),
            "empty_retrieval": int(trace.get("rrf_candidates") or trace.get("rrf_fused_candidate_count") or 0) == 0,
            "latency_ms": float((answer.get("evaluation_latency") or {}).get("total_ms") or 0),
        })
    present = [item for item in records if not item.get("missing")]
    recalls = [item["gold_evidence_token_recall"] for item in present if item["gold_evidence_token_recall"] is not None]
    bigram_recalls = [item["gold_evidence_bigram_recall"] for item in present if item["gold_evidence_bigram_recall"] is not None]
    return {
        "answers": str(answers_path),
        "fixture": str(fixture_path) if fixture_path else "all_dataset_rows",
        "expected_questions": len(expected_ids),
        "evaluated_questions": len(present),
        "missing_questions": [item["financebench_id"] for item in records if item.get("missing")],
        "metrics": {
            "candidate_page_hit_rate": sum(item["candidate_page_hit"] for item in present) / len(present) if present else 0,
            "selected_page_hit_rate": sum(item["selected_page_hit"] for item in present) / len(present) if present else 0,
            "context_page_hit_rate": sum(item["context_page_hit"] for item in present) / len(present) if present else 0,
            "citation_document_hit_rate": sum(item["citation_document_hit"] for item in present) / len(present) if present else 0,
            "average_gold_evidence_token_recall": statistics.fmean(recalls) if recalls else None,
            "average_gold_evidence_bigram_recall": statistics.fmean(bigram_recalls) if bigram_recalls else None,
            "average_answer_input_tokens": statistics.fmean(item["answer_input_tokens"] for item in present) if present else 0,
            "average_answer_context_chars": statistics.fmean(item["answer_context_chars"] for item in present) if present else 0,
            "questions_with_tables_available": sum(item["tables_available"] > 0 for item in present),
            "questions_with_tables_attached": sum(item["tables_attached"] > 0 for item in present),
            "empty_retrievals": sum(item["empty_retrieval"] for item in present),
            "average_latency_ms": statistics.fmean(item["latency_ms"] for item in present) if present else 0,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--all", action="store_true", help="Evaluate every dataset row instead of the fixed fixture.")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "rag_core_v2_fixed20_diagnostic.json")
    parser.add_argument("--suggest-from-clean-baseline", type=Path)
    args = parser.parse_args()
    with DATA.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["financebench_id"]: row for row in csv.DictReader(handle)}
    if args.suggest_from_clean_baseline:
        print(json.dumps(suggest_fixture(args.suggest_from_clean_baseline, rows), ensure_ascii=False, indent=2))
        return
    if not args.answers:
        parser.error("--answers is required unless --suggest-from-clean-baseline is used")
    report = evaluate(args.answers, None if args.all else args.fixture, rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
