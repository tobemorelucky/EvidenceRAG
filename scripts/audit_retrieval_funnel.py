"""Audit Dense/BM25/RRF and replay frozen Core v3 Jina/page ranks.

This is retrieval-only. Gold evidence is loaded only by this offline evaluator
and is never passed to retrieval code.
"""

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

from retrieval_ablation import page_key, retrieve_independent_routes  # noqa: E402


KS = (10, 20, 40, 60, 80, 100)
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_ANSWERS = ROOT / "reports" / "evidencerag-rag-core-v3-skills-all100-final_answers.jsonl"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json"


def _normalize_filename(value: object) -> str:
    name = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return name.casefold() if name.casefold().endswith(".pdf") else f"{name}.pdf".casefold()


def _gold_pages(row: dict) -> set[tuple[str, int]]:
    result = set()
    for evidence in json.loads(row.get("evidence") or "[]"):
        result.add((_normalize_filename(evidence.get("doc_name")), int(evidence.get("evidence_page_num") or 0)))
    return result


def _rank(items: list[dict], gold: set[tuple[str, int]]) -> int | None:
    for rank, item in enumerate(items, 1):
        filename, page = page_key(item)
        if (_normalize_filename(filename), page) in gold:
            return rank
    return None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _diagnostic_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for group in ("candidate_miss10", "selection_loss10", "correct_regression10"):
        for item in payload[group]:
            financebench_id = item if isinstance(item, str) else item["financebench_id"]
            if financebench_id not in result:
                result.append(financebench_id)
    return result


def _first_loss(record: dict) -> str:
    dense = record["dense_rank"]
    bm25 = record["bm25_rank"]
    rrf = record["rrf_rank"]
    if dense is None and bm25 is None:
        return "dense_and_bm25_miss"
    if rrf is None:
        return "rrf_fusion_loss"
    if record["current_rrf_rank"] is None:
        if rrf > int(record.get("frozen_candidate_k") or 60):
            return "frozen_candidate_k_cutoff"
        return "hybrid_rrf_difference"
    if not record["entered_jina"]:
        return "jina_candidate_cutoff"
    if record["jina_rank"] is None:
        return "jina_downranked_below_output"
    if record["page_aggregate_rank"] is None:
        return "page_aggregation_loss"
    if record["selected_rank"] is None:
        return "page_selection_loss"
    return "retained"


def _recall(records: list[dict], field: str, k: int) -> float:
    return round(sum(item.get(field) is not None and item[field] <= k for item in records) / max(1, len(records)), 4)


def _markdown(payload: dict) -> str:
    lines = [
        "# Retrieval Funnel Audit",
        "",
        "> Offline evaluation only. Gold pages are never passed to retrieval or ranking.",
        "",
        f"- Questions: {payload['questions']}",
        f"- Independent max K: {payload['max_k']}",
        f"- Mean retrieval latency: {payload['mean_latency_ms']:.2f} ms",
        "",
        "## Gold-page recall",
        "",
        "| Stage | @10 | @20 | @40 | @60 | @80 | @100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, metrics in payload["recall"].items():
        lines.append("| " + stage + " | " + " | ".join(f"{metrics[str(k)]:.2%}" for k in KS) + " |")
    lines.extend(["", "## First loss stage", ""])
    for name, count in payload["first_loss_counts"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Route overlap", ""])
    for name, count in payload["route_overlap"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "## Stage transitions", ""])
    for name, count in payload["transitions"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend([
        "",
        "## Per-question losses",
        "",
        "| ID | Gold document/page | Dense | BM25 | RRF | Current RRF | Jina input | Jina rank | Page rank | Selected | First loss |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ])
    for item in payload["records"]:
        if item["first_loss_stage"] == "retained":
            continue
        gold = ", ".join(f"{doc} p.{page}" for doc, page in item["gold_pages"])
        value = lambda name: item.get(name) if item.get(name) is not None else "—"
        lines.append(
            f"| `{item['financebench_id']}` | {gold} | {value('dense_rank')} | {value('bm25_rank')} | "
            f"{value('rrf_rank')} | {value('current_rrf_rank')} | {'yes' if item['entered_jina'] else 'no'} | "
            f"{value('jina_rank')} | {value('page_aggregate_rank')} | {value('selected_rank')} | "
            f"`{item['first_loss_stage']}` |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--historical-answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--diagnostic30", action="store_true")
    parser.add_argument("--max-k", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "reports" / "retrieval_funnel_audit_rows_v2.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "retrieval_funnel_audit.json")
    parser.add_argument("--markdown-output", type=Path, default=ROOT / "reports" / "retrieval_funnel_audit.md")
    args = parser.parse_args()

    with args.dataset.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.diagnostic30:
        ids = set(_diagnostic_ids(args.fixture))
        rows = [row for row in rows if row["financebench_id"] in ids]
    if args.limit:
        rows = rows[: args.limit]
    historical = {
        item["financebench_id"]: item
        for item in _load_jsonl(args.historical_answers)
    }
    existing = {item["financebench_id"]: item for item in _load_jsonl(args.checkpoint)}
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with args.checkpoint.open("a", encoding="utf-8") as checkpoint:
        for index, row in enumerate(rows, 1):
            financebench_id = row["financebench_id"]
            if financebench_id in existing:
                print(f"[{index:02d}/{len(rows)}] {financebench_id}: checkpoint", flush=True)
                continue
            gold = _gold_pages(row)
            started = time.perf_counter()
            routes = retrieve_independent_routes(row["question"], max_k=args.max_k)
            prior = historical.get(financebench_id, {})
            trace = prior.get("rag_trace") or {}
            current_chunks = trace.get("initial_retrieved_chunks") or []
            reranked = trace.get("reranked_chunks") or []
            page_scores = trace.get("page_scores") or []
            selected = trace.get("selected_pages") or []
            current_rrf_rank = _rank(current_chunks, gold)
            jina_candidate_k = int(trace.get("remote_rerank_candidate_count") or 0)
            record = {
                "financebench_id": financebench_id,
                "question": row["question"],
                "gold_pages": sorted([list(item) for item in gold]),
                "dense_rank": _rank(routes["dense"], gold),
                "bm25_rank": _rank(routes["bm25"], gold),
                "rrf_rank": _rank(routes["rrf"], gold),
                "current_rrf_rank": current_rrf_rank,
                "frozen_candidate_k": len(current_chunks),
                "entered_jina": current_rrf_rank is not None and current_rrf_rank <= jina_candidate_k,
                "jina_rank": _rank(reranked, gold),
                "page_aggregate_rank": _rank(page_scores, gold),
                "selected_rank": _rank(selected, gold),
                "independent_unique_pages": {
                    str(k): len({page_key(item) for item in routes["rrf"][:k]}) for k in KS
                },
                "latency_ms": routes["latency_ms"],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            record["first_loss_stage"] = _first_loss(record)
            checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
            checkpoint.flush()
            existing[financebench_id] = record
            print(f"[{index:02d}/{len(rows)}] {financebench_id}: {record['first_loss_stage']}", flush=True)

    records = [existing[row["financebench_id"]] for row in rows if row["financebench_id"] in existing]
    for record in records:
        record["first_loss_stage"] = _first_loss(record)
    recall_fields = {
        "Dense": "dense_rank",
        "BM25": "bm25_rank",
        "Dense+BM25 RRF": "rrf_rank",
        "Frozen Core v3 RRF": "current_rrf_rank",
        "Frozen chunk Jina": "jina_rank",
        "Frozen page aggregation": "page_aggregate_rank",
    }
    recall = {
        stage: {str(k): _recall(records, field, k) for k in KS}
        for stage, field in recall_fields.items()
    }
    loss_counts: dict[str, int] = {}
    for record in records:
        name = record["first_loss_stage"]
        loss_counts[name] = loss_counts.get(name, 0) + 1
    route_overlap = {
        "dense_only": sum(item["dense_rank"] is not None and item["bm25_rank"] is None for item in records),
        "bm25_only": sum(item["dense_rank"] is None and item["bm25_rank"] is not None for item in records),
        "both": sum(item["dense_rank"] is not None and item["bm25_rank"] is not None for item in records),
        "neither": sum(item["dense_rank"] is None and item["bm25_rank"] is None for item in records),
    }
    transitions = {
        "dense_hit_at_120_but_rrf_miss_at_120": sum(
            item["dense_rank"] is not None and item["dense_rank"] <= 120
            and (item["rrf_rank"] is None or item["rrf_rank"] > 120)
            for item in records
        ),
        "frozen_rrf_hit_but_outside_jina_input": sum(
            item["current_rrf_rank"] is not None and not item["entered_jina"] for item in records
        ),
        "entered_jina_but_absent_from_jina_output": sum(
            item["entered_jina"] and item["jina_rank"] is None for item in records
        ),
        "page_ranked_but_not_selected": sum(
            item["page_aggregate_rank"] is not None and item["selected_rank"] is None for item in records
        ),
    }
    payload = {
        "questions": len(records),
        "max_k": args.max_k,
        "recall": recall,
        "first_loss_counts": dict(sorted(loss_counts.items())),
        "route_overlap": route_overlap,
        "transitions": transitions,
        "mean_latency_ms": round(sum(item["elapsed_ms"] for item in records) / max(1, len(records)), 2),
        "records": records,
        "limitations": [
            "Frozen Jina ranks are replayed from the existing Core v3 top-output trace; ranks below its output cutoff are unobservable.",
            "Independent Dense/BM25 queries are diagnostic only and do not alter production hybrid retrieval.",
        ],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_output.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"questions": len(records), "recall": recall, "first_loss": loss_counts}, ensure_ascii=False, indent=2))
    print(f"JSON: {args.output}\nMarkdown: {args.markdown_output}")


if __name__ == "__main__":
    main()
