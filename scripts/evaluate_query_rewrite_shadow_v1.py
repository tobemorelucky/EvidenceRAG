"""Compare original-query RRF with DeepSeek query-rewrite RRF on diagnostic30."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_reranker_shadow_v1 import digest, fixture_rows, metrics  # noqa: E402
from query_rewriter import create_query_rewrite_model, rewrite_queries  # noqa: E402


DEFAULT_OUTPUT = ROOT / "reports/query_rewrite_shadow_v1.json"
LEAF_FILTER = '(evidence_type == "text_chunk" or evidence_type == "") and chunk_level == 3'


def _chunk_key(item: dict) -> tuple:
    return (
        str(item.get("chunk_id") or item.get("id") or ""),
        str(item.get("filename") or ""),
        int(item.get("page_number") or 0),
        int(item.get("chunk_idx") or 0),
    )


def rrf_fuse(routes: list[tuple[str, list[dict]]], *, rrf_k: int = 60) -> list[dict]:
    """Fuse any number of frozen route rankings without production imports."""
    fused: dict[tuple, dict] = {}
    for route_name, candidates in routes:
        for rank, candidate in enumerate(candidates, 1):
            key = _chunk_key(candidate)
            item = fused.setdefault(
                key,
                {**candidate, "route_ranks": {}, "rrf_score": 0.0},
            )
            item["route_ranks"][route_name] = rank
            item["rrf_score"] += 1.0 / (rrf_k + rank)
    ordered = sorted(
        fused.values(),
        key=lambda item: (
            -item["rrf_score"],
            min(item["route_ranks"].values()),
            _chunk_key(item),
        ),
    )
    return [
        {**item, "rrf_rank": rank, "score": item["rrf_score"]}
        for rank, item in enumerate(ordered, 1)
    ]


def _route_summary(records: list[dict], route: str) -> dict:
    values = [record["routes"][route]["metrics"] for record in records]
    chunk_ranks = [item["gold_chunk_rank"] for item in values if item["gold_chunk_rank"] is not None]
    page_ranks = [item["gold_page_rank"] for item in values if item["gold_page_rank"] is not None]
    return {
        "questions": len(values),
        "candidate_hit": fmean(item["candidate_gold_page_hit"] for item in values),
        "gold_chunk_rank": {
            "hits": len(chunk_ranks),
            "mean_on_hits": fmean(chunk_ranks) if chunk_ranks else None,
        },
        "gold_page_rank": {
            "hits": len(page_ranks),
            "mean_on_hits": fmean(page_ranks) if page_ranks else None,
        },
    }


def summarize(records: list[dict]) -> dict:
    baseline = _route_summary(records, "baseline")
    rewritten = _route_summary(records, "query_rewrite")
    delta = rewritten["candidate_hit"] - baseline["candidate_hit"]
    gate = delta >= 0.03
    return {
        "baseline": baseline,
        "query_rewrite": rewritten,
        "candidate_hit_delta": round(delta, 6),
        "candidate_gains": [
            record["question_id"] for record in records
            if record["routes"]["query_rewrite"]["metrics"]["candidate_gold_page_hit"]
            and not record["routes"]["baseline"]["metrics"]["candidate_gold_page_hit"]
        ],
        "candidate_regressions": [
            record["question_id"] for record in records
            if not record["routes"]["query_rewrite"]["metrics"]["candidate_gold_page_hit"]
            and record["routes"]["baseline"]["metrics"]["candidate_gold_page_hit"]
        ],
        "rewrite_statuses": dict(Counter(record["rewrite"]["status"] for record in records)),
        "average_query_count": fmean(len(record["rewrite"]["queries"]) for record in records),
        "candidate_gate": ">= +3 percentage points",
        "candidate_gate_passed": gate,
        "decision": "continue" if gate else "stop",
        "external_calls": {
            "query_rewrite_llm": sum(record["rewrite"]["status"] == "ok" for record in records),
            "jina": 0,
            "answer_llm": 0,
            "judge": 0,
            "langsmith": 0,
        },
    }


def markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Query Rewrite Shadow v1", "",
        "固定 diagnostic30。每组最多3条总检索query，原query始终为第一条；每条query分别执行Dense与BM25，再以相同RRF(k=60)融合为Top120。Gold仅用于排序完成后的离线评估。", "",
        "| Route | Candidate hit | Gold chunk rank* | Gold page rank |",
        "|---|---:|---:|---:|",
    ]
    for route in ("baseline", "query_rewrite"):
        item = summary[route]
        lines.append(
            f"| `{route}` | {item['candidate_hit']:.2%} | "
            f"{item['gold_chunk_rank']['mean_on_hits']} | {item['gold_page_rank']['mean_on_hits']} |"
        )
    lines.extend([
        "", f"- Candidate delta：`{summary['candidate_hit_delta']:+.2%}`",
        f"- Gains：`{summary['candidate_gains']}`",
        f"- Regressions：`{summary['candidate_regressions']}`",
        f"- Rewrite状态：`{summary['rewrite_statuses']}`；平均query数：`{summary['average_query_count']:.2f}`",
        f"- +3pp门槛：`{summary['candidate_gate_passed']}`；决定：`{summary['decision']}`",
        "- *Gold chunk rank是gold页内参考原文行匹配代理，并非官方chunk标注。", "",
        "## 逐题", "",
        "| ID | Group | Queries | A hit/rank | B hit/rank | Rewrite status |",
        "|---|---|---:|---|---|---|",
    ])
    for record in payload["records"]:
        a = record["routes"]["baseline"]["metrics"]
        b = record["routes"]["query_rewrite"]["metrics"]
        lines.append(
            f"| {record['question_id']} | {record['group']} | {len(record['rewrite']['queries'])} | "
            f"{a['candidate_gold_page_hit']}/{a['gold_page_rank']} | "
            f"{b['candidate_gold_page_hit']}/{b['gold_page_rank']} | {record['rewrite']['status']} |"
        )
    return "\n".join(lines) + "\n"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    path.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")


def _load_completed(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if payload.get("schema") != "query_rewrite_shadow_v1":
        return {}
    return {record["question_id"]: record for record in payload.get("records", [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows, groups = fixture_rows()
    completed = _load_completed(args.output) if args.resume else {}
    records = [completed[question_id] for question_id in groups if question_id in completed]
    pending = [(question_id, group) for question_id, group in groups.items() if question_id not in completed]
    if pending:
        os.environ.update(
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
            LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false",
        )
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        from embedding import embedding_service
        from milvus_client import MilvusManager

        manager = MilvusManager()
        if not manager.uses_builtin_bm25:
            raise RuntimeError("Query Rewrite Shadow v1 requires Milvus built-in BM25")
        model = create_query_rewrite_model()

        for index, (question_id, group) in enumerate(pending, len(records) + 1):
            question = rows[question_id]["question"]
            started = time.perf_counter()
            try:
                rewrite = {"status": "ok", **rewrite_queries(question, model)}
            except Exception as exc:
                rewrite = {
                    "status": "error_fallback_original",
                    "queries": [question], "generated_alternatives": [],
                    "raw_response": "", "usage": {}, "error": f"{type(exc).__name__}: {exc}",
                }

            retrieval_routes: dict[str, tuple[list[dict], list[dict]]] = {}
            for query_index, query in enumerate(rewrite["queries"]):
                vector = embedding_service.get_embeddings([query])[0]
                retrieval_routes[f"q{query_index}"] = (
                    manager.dense_retrieve(vector, top_k=240, filter_expr=LEAF_FILTER),
                    manager.bm25_retrieve(query, top_k=240, filter_expr=LEAF_FILTER),
                )

            original_dense, original_bm25 = retrieval_routes["q0"]
            baseline_chunks = rrf_fuse([
                ("q0_dense", original_dense), ("q0_bm25", original_bm25),
            ])[:120]
            rewrite_routes = [
                (f"{name}_dense", dense) if kind == "dense" else (f"{name}_bm25", sparse)
                for name, (dense, sparse) in retrieval_routes.items()
                for kind in ("dense", "bm25")
            ]
            rewrite_chunks = rrf_fuse(rewrite_routes)[:120]
            record = {
                "question_id": question_id, "group": group, "question": question,
                "rewrite": rewrite,
                "routes": {
                    "baseline": {
                        "chunks": baseline_chunks, "candidate_sha256": digest(baseline_chunks),
                        "metrics": metrics(rows[question_id], baseline_chunks),
                    },
                    "query_rewrite": {
                        "chunks": rewrite_chunks, "candidate_sha256": digest(rewrite_chunks),
                        "metrics": metrics(rows[question_id], rewrite_chunks),
                    },
                },
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            records.append(record)
            records.sort(key=lambda item: list(groups).index(item["question_id"]))
            partial = {"schema": "query_rewrite_shadow_v1", "records": records}
            partial["summary"] = summarize(records)
            _write(args.output, partial)
            print(
                f"[{index:02d}/30] {question_id} queries={len(rewrite['queries'])} "
                f"A={record['routes']['baseline']['metrics']['candidate_gold_page_hit']} "
                f"B={record['routes']['query_rewrite']['metrics']['candidate_gold_page_hit']}",
                flush=True,
            )

    if len(records) != 30:
        raise RuntimeError(f"Expected 30 completed records, got {len(records)}")
    records.sort(key=lambda item: list(groups).index(item["question_id"]))
    payload = {
        "schema": "query_rewrite_shadow_v1",
        "manifest": {
            "questions": 30, "max_total_queries": 3,
            "dense_top_k_per_query": 240, "bm25_top_k_per_query": 240,
            "rrf_top_k": 120, "rrf_k": 60,
            "original_query_retained": True,
            "dense_and_bm25_use_same_query_set": True,
            "gold_used_after_ranking_only": True,
        },
        "records": records,
    }
    payload["summary"] = summarize(records)
    _write(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
