"""Evaluate broad recall plus one page-level Jina rerank on diagnostic30."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from retrieval_ablation import (  # noqa: E402
    build_page_candidates,
    page_key,
    page_level_jina_rerank,
    retrieve_independent_routes,
)


DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json"


def _normalize_filename(value: object) -> str:
    name = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return name.casefold() if name.casefold().endswith(".pdf") else f"{name}.pdf".casefold()


def _gold(row: dict) -> set[tuple[str, int]]:
    return {
        (_normalize_filename(item.get("doc_name")), int(item.get("evidence_page_num") or 0))
        for item in json.loads(row.get("evidence") or "[]")
    }


def _rank(items: list[dict], gold: set[tuple[str, int]]) -> int | None:
    for rank, item in enumerate(items, 1):
        filename, page = page_key(item)
        if (_normalize_filename(filename), page) in gold:
            return rank
    return None


def _rows(dataset: Path, fixture: Path) -> list[dict]:
    groups = json.loads(fixture.read_text(encoding="utf-8"))
    ids = []
    for group in ("candidate_miss10", "selection_loss10", "correct_regression10"):
        ids.extend(item if isinstance(item, str) else item["financebench_id"] for item in groups[group])
    with dataset.open(encoding="utf-8-sig", newline="") as handle:
        by_id = {row["financebench_id"]: row for row in csv.DictReader(handle)}
    return [by_id[item] for item in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--max-k", type=int, default=120)
    parser.add_argument("--rrf-chunk-k", type=int, default=100)
    parser.add_argument("--page-candidate-k", type=int, default=30)
    parser.add_argument("--final-page-k", type=int, default=6)
    parser.add_argument("--representation-chars", type=int, default=1500)
    parser.add_argument("--skip-jina", action="store_true", help="Measure the page-candidate upper bound without reranking.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "page_level_jina_diagnostic30.json")
    args = parser.parse_args()
    rows = _rows(args.dataset, args.fixture)
    if args.limit:
        rows = rows[: args.limit]
    records = []
    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        routes = retrieve_independent_routes(row["question"], max_k=args.max_k)
        if args.skip_jina:
            candidates = build_page_candidates(
                row["question"], routes["rrf"], rrf_chunk_k=args.rrf_chunk_k,
                page_candidate_k=args.page_candidate_k, representation_chars=args.representation_chars,
            )
            prototype = {
                "page_candidates": candidates,
                "reranked_pages": [],
                "selected_pages": [],
                "rerank_meta": {"rerank_provider": "skipped", "remote_rerank_input_chars": 0},
            }
        else:
            prototype = page_level_jina_rerank(
                row["question"],
                routes["rrf"],
                rrf_chunk_k=args.rrf_chunk_k,
                page_candidate_k=args.page_candidate_k,
                final_page_k=args.final_page_k,
                representation_chars=args.representation_chars,
            )
        gold = _gold(row)
        meta = prototype["rerank_meta"]
        record = {
            "financebench_id": row["financebench_id"],
            "rrf_rank": _rank(routes["rrf"], gold),
            "page_candidate_rank": _rank(prototype["page_candidates"], gold),
            "page_jina_rank": _rank(prototype["reranked_pages"], gold),
            "selected_hit": _rank(prototype["selected_pages"], gold) is not None if not args.skip_jina else False,
            "selected_pages": [list(page_key(item)) for item in prototype["selected_pages"]],
            "page_candidate_count": len(prototype["page_candidates"]),
            "representation_chars": sum(item["representation_chars"] for item in prototype["page_candidates"]),
            "table_representation_chars": sum(item["table_representation_chars"] for item in prototype["page_candidates"]),
            "jina_chars": int(meta.get("remote_rerank_input_chars") or 0),
            "jina_calls": int(bool(meta.get("remote_success") or meta.get("rerank_cache_hit"))),
            "rerank_provider": meta.get("rerank_provider") or "none",
            "rerank_error": meta.get("rerank_error"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        records.append(record)
        print(
            f"[{index:02d}/{len(rows)}] {row['financebench_id']} rrf={record['rrf_rank']} "
            f"page_jina={record['page_jina_rank']} selected={record['selected_hit']} "
            f"provider={record['rerank_provider']}",
            flush=True,
        )
    questions = len(records)
    metrics = {
        "questions": questions,
        "rrf_candidate_hit": round(sum(item["rrf_rank"] is not None and item["rrf_rank"] <= args.max_k for item in records) / max(1, questions), 4),
        "page_candidate_hit": round(sum(item["page_candidate_rank"] is not None for item in records) / max(1, questions), 4),
        "page_jina_hit": round(sum(item["page_jina_rank"] is not None for item in records) / max(1, questions), 4),
        "selected_hit": round(sum(item["selected_hit"] for item in records) / max(1, questions), 4),
        "jina_calls": sum(item["jina_calls"] for item in records),
        "jina_chars": sum(item["jina_chars"] for item in records),
        "average_jina_chars": round(sum(item["jina_chars"] for item in records) / max(1, questions), 2),
        "average_latency_ms": round(sum(item["latency_ms"] for item in records) / max(1, questions), 2),
        "rerank_providers": {
            provider: sum(item["rerank_provider"] == provider for item in records)
            for provider in sorted({item["rerank_provider"] for item in records})
        },
    }
    payload = {
        "config": {
            "max_k": args.max_k,
            "rrf_chunk_k": args.rrf_chunk_k,
            "page_candidate_k": args.page_candidate_k,
            "final_page_k": args.final_page_k,
            "representation_chars": args.representation_chars,
            "jina_calls_per_question": 0 if args.skip_jina else 1,
            "skip_jina": args.skip_jina,
        },
        "metrics": metrics,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(
        "\n".join([
            "# Page-level Jina Retrieval Ablation", "",
            "> diagnostic30 only; one page-level Jina call per question; no answer model or Judge.", "",
            f"- RRF candidate hit: {metrics['rrf_candidate_hit']:.2%}",
            f"- Page candidate hit: {metrics['page_candidate_hit']:.2%}",
            f"- Selected Top-{args.final_page_k} hit: {metrics['selected_hit']:.2%}",
            f"- Jina calls/chars: {metrics['jina_calls']} / {metrics['jina_chars']}",
            f"- Average latency: {metrics['average_latency_ms']:.2f} ms", "",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"JSON: {args.output}\nMarkdown: {markdown}")


if __name__ == "__main__":
    main()
