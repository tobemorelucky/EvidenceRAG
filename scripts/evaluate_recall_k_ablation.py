"""Evaluate retrieval-only candidate K on the frozen 30-question diagnostic set."""

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

from embedding import embedding_service  # noqa: E402
from retrieval_ablation import page_key, retrieve_independent_routes  # noqa: E402


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


def _ids(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for group in ("candidate_miss10", "selection_loss10", "correct_regression10"):
        result.extend(item if isinstance(item, str) else item["financebench_id"] for item in data[group])
    return list(dict.fromkeys(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--candidate-k", type=int, action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "retrieval_recall_k_ablation.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "reports" / "retrieval_recall_k_ablation.md")
    args = parser.parse_args()
    ks = sorted(set(args.candidate_k or [40, 60, 80, 100, 120]))
    wanted = set(_ids(args.fixture))
    with args.dataset.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["financebench_id"] in wanted]
    if args.limit:
        rows = rows[: args.limit]

    records = []
    for index, row in enumerate(rows, 1):
        vector_started = time.perf_counter()
        vector = embedding_service.get_embeddings([row["question"]])[0]
        embedding_ms = round((time.perf_counter() - vector_started) * 1000, 2)
        gold = _gold(row)
        for k in ks:
            result = retrieve_independent_routes(row["question"], max_k=k, dense_vector=vector)
            rrf_rank = _rank(result["rrf"], gold)
            records.append({
                "financebench_id": row["financebench_id"],
                "candidate_k": k,
                "dense_rank": _rank(result["dense"], gold),
                "bm25_rank": _rank(result["bm25"], gold),
                "rrf_rank": rrf_rank,
                "candidate_hit": rrf_rank is not None and rrf_rank <= k,
                "unique_pages": len({page_key(item) for item in result["rrf"][:k]}),
                "embedding_ms": embedding_ms,
                "retrieval_ms": result["latency_ms"]["total"],
            })
        print(f"[{index:02d}/{len(rows)}] {row['financebench_id']} complete", flush=True)

    metrics = {}
    for k in ks:
        subset = [item for item in records if item["candidate_k"] == k]
        metrics[str(k)] = {
            "questions": len(subset),
            "candidate_gold_page_hit": round(sum(item["candidate_hit"] for item in subset) / max(1, len(subset)), 4),
            "dense_gold_page_hit": round(sum(item["dense_rank"] is not None and item["dense_rank"] <= k for item in subset) / max(1, len(subset)), 4),
            "bm25_gold_page_hit": round(sum(item["bm25_rank"] is not None and item["bm25_rank"] <= k for item in subset) / max(1, len(subset)), 4),
            "average_unique_pages": round(sum(item["unique_pages"] for item in subset) / max(1, len(subset)), 2),
            "average_retrieval_ms": round(sum(item["retrieval_ms"] for item in subset) / max(1, len(subset)), 2),
        }
    payload = {"questions": len(rows), "candidate_k": ks, "metrics": metrics, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Recall@K Retrieval Ablation", "",
        "> Frozen 30-question retrieval-only diagnostic; no answer model or Judge.", "",
        "| K | Candidate hit | Dense hit | BM25 hit | Avg unique pages | Avg retrieval ms |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ks:
        item = metrics[str(k)]
        lines.append(
            f"| {k} | {item['candidate_gold_page_hit']:.2%} | {item['dense_gold_page_hit']:.2%} | "
            f"{item['bm25_gold_page_hit']:.2%} | {item['average_unique_pages']:.2f} | {item['average_retrieval_ms']:.2f} |"
        )
    args.markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"JSON: {args.output}\nMarkdown: {args.markdown_output}")


if __name__ == "__main__":
    main()
