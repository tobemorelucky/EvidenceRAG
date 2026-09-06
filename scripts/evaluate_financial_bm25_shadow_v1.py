"""Evaluate original vs financially expanded BM25 on frozen diagnostic30."""

from __future__ import annotations

import argparse
import hashlib
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
from financial_query_expander import explain_financial_query_expansion  # noqa: E402


DEFAULT_OUTPUT = ROOT / "reports/financial_bm25_shadow_v1.json"
FROZEN_RRF = ROOT / "reports/reranker_shadow_v1_rrf_top120.json"
FROZEN_JINA = ROOT / "reports/reranker_shadow_v1.json"
LEAF_FILTER = '(evidence_type == "text_chunk" or evidence_type == "") and chunk_level == 3'


def _chunk_key(item: dict) -> tuple:
    return (
        str(item.get("chunk_id") or item.get("id") or ""), str(item.get("filename") or ""),
        int(item.get("page_number") or 0), int(item.get("chunk_idx") or 0),
    )


def rrf_fuse(dense: list[dict], bm25: list[dict], *, rrf_k: int = 60) -> list[dict]:
    """Local shadow RRF; it does not import or alter the production pipeline."""
    fused = {}
    for route, candidates in (("dense", dense), ("bm25", bm25)):
        for rank, candidate in enumerate(candidates, 1):
            key = _chunk_key(candidate)
            item = fused.setdefault(key, {**candidate, "dense_rank": None, "bm25_rank": None, "rrf_score": 0.0})
            item[f"{route}_rank"] = rank
            item["rrf_score"] += 1.0 / (rrf_k + rank)
    result = sorted(
        fused.values(),
        key=lambda item: (-item["rrf_score"], min(rank for rank in (item["dense_rank"], item["bm25_rank"]) if rank is not None), _chunk_key(item)),
    )
    for rank, item in enumerate(result, 1):
        item["rrf_rank"], item["score"] = rank, item["rrf_score"]
    return result


def _route_summary(records: list[dict], route: str) -> dict:
    values = [record["routes"][route]["metrics"] for record in records]
    chunk_ranks = [item["gold_chunk_rank"] for item in values if item["gold_chunk_rank"] is not None]
    page_ranks = [item["gold_page_rank"] for item in values if item["gold_page_rank"] is not None]
    return {
        "questions": len(values),
        "candidate_hit": fmean(item["candidate_gold_page_hit"] for item in values),
        "gold_chunk_rank": {"hits": len(chunk_ranks), "mean_on_hits": fmean(chunk_ranks) if chunk_ranks else None},
        "gold_page_rank": {"hits": len(page_ranks), "mean_on_hits": fmean(page_ranks) if page_ranks else None},
        "jina_context_hit": fmean(
            record["routes"][route]["jina"]["metrics"]["context_hit"]
            for record in records if record["routes"][route].get("jina", {}).get("status") == "ok"
        ) if any(record["routes"][route].get("jina", {}).get("status") == "ok" for record in records) else None,
        "jina_completed": sum(record["routes"][route].get("jina", {}).get("status") == "ok" for record in records),
    }


def summarize(records: list[dict]) -> dict:
    baseline = _route_summary(records, "baseline")
    financial = _route_summary(records, "financial_bm25")
    delta = financial["candidate_hit"] - baseline["candidate_hit"]
    gate = delta >= 0.05
    return {
        "baseline": baseline, "financial_bm25": financial,
        "candidate_hit_delta": round(delta, 6),
        "candidate_gains": [r["question_id"] for r in records if r["routes"]["financial_bm25"]["metrics"]["candidate_gold_page_hit"] and not r["routes"]["baseline"]["metrics"]["candidate_gold_page_hit"]],
        "candidate_regressions": [r["question_id"] for r in records if not r["routes"]["financial_bm25"]["metrics"]["candidate_gold_page_hit"] and r["routes"]["baseline"]["metrics"]["candidate_gold_page_hit"]],
        "candidate_gate": ">= +5 percentage points",
        "candidate_gate_passed": gate,
        "decision": "continue_to_jina" if gate else "stop",
        "expansion_changed_questions": sum(record["expansion"]["changed"] for record in records),
        "external_calls": {"llm": 0, "judge": 0, "langsmith": 0},
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    path.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")


def markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Financial-aware BM25 Shadow v1", "",
        "固定diagnostic30。Dense向量和Dense候选完全共享；仅BM25 query不同，再使用相同RRF(k=60)产生Top120。Gold只在排序完成后用于离线指标。", "",
        "| Route | Candidate hit | Gold chunk rank* | Gold page rank | Jina context hit | Jina完成 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route in ("baseline", "financial_bm25"):
        item = summary[route]
        chunk_rank = item["gold_chunk_rank"]["mean_on_hits"]
        page_rank = item["gold_page_rank"]["mean_on_hits"]
        jina = item["jina_context_hit"]
        lines.append(
            f"| `{route}` | {item['candidate_hit']:.2%} | {chunk_rank if chunk_rank is not None else 'N/A'} | "
            f"{page_rank if page_rank is not None else 'N/A'} | {f'{jina:.2%}' if jina is not None else '未运行'} | {item['jina_completed']}/30 |"
        )
    lines.extend([
        "", f"- Candidate delta：`{summary['candidate_hit_delta']:+.2%}`",
        f"- Gains：`{summary['candidate_gains']}`",
        f"- Regressions：`{summary['candidate_regressions']}`",
        f"- +5pp门槛：`{summary['candidate_gate_passed']}`；决定：`{summary['decision']}`",
        "- *Gold chunk rank为gold页内参考原文行匹配代理，不是官方chunk真值。", "",
        "## 逐题", "",
        "| ID | Group | Expanded | A hit/rank | B hit/rank | A Jina context | B Jina context |",
        "|---|---|---|---|---|---|---|",
    ])
    for record in payload["records"]:
        a, b = record["routes"]["baseline"], record["routes"]["financial_bm25"]
        lines.append(
            f"| {record['question_id']} | {record['group']} | {record['expansion']['changed']} | "
            f"{a['metrics']['candidate_gold_page_hit']}/{a['metrics']['gold_page_rank']} | "
            f"{b['metrics']['candidate_gold_page_hit']}/{b['metrics']['gold_page_rank']} | "
            f"{a.get('jina', {}).get('metrics', {}).get('context_hit')} | {b.get('jina', {}).get('metrics', {}).get('context_hit')} |"
        )
    return "\n".join(lines) + "\n"


def _reuse_baseline_jina(records: list[dict]) -> None:
    if not FROZEN_RRF.exists() or not FROZEN_JINA.exists():
        return
    frozen = {row["question_id"]: row for row in json.loads(FROZEN_RRF.read_text(encoding="utf-8"))["records"]}
    jina = {row["question_id"]: row for row in json.loads(FROZEN_JINA.read_text(encoding="utf-8"))["records"]}
    for record in records:
        source, result = frozen.get(record["question_id"]), jina.get(record["question_id"])
        current_chunks = record["routes"]["baseline"]["chunks"]
        same_candidates = bool(source) and [
            (chunk["chunk_id"], chunk["filename"], int(chunk["page_number"]), chunk["text"])
            for chunk in current_chunks
        ] == [
            (chunk["chunk_id"], chunk["filename"], int(chunk["page_number"]), chunk["text"])
            for chunk in source["chunks"]
        ]
        if not source or not result or not same_candidates:
            record["routes"]["baseline"]["jina"] = {"status": "candidate_hash_mismatch_no_reuse"}
            continue
        cached = result.get("routes", {}).get("jina", {})
        record["routes"]["baseline"]["jina"] = cached if cached.get("status") == "ok" else {"status": "cached_jina_unavailable"}


def _run_financial_jina(records: list[dict], output: Path) -> None:
    from dotenv import dotenv_values
    from shadow_rerankers_v1 import JinaReranker

    env = {**dotenv_values(ROOT / ".env"), **os.environ}
    key = env.get("RERANK_API_KEY") or env.get("JINA_API_KEY")
    if not key:
        for record in records:
            record["routes"]["financial_bm25"]["jina"] = {"status": "credential_not_configured"}
        return
    host = (env.get("RERANK_BINDING_HOST") or "https://api.jina.ai").rstrip("/")
    endpoint = host if host.endswith("/v1/rerank") else host + "/v1/rerank"
    reranker = JinaReranker(key, env.get("RERANK_MODEL") or "jina-reranker-v3", endpoint, interval=8)
    rows, _ = fixture_rows()
    for index, record in enumerate(records, 1):
        route = record["routes"]["financial_bm25"]
        if route.get("jina", {}).get("status") == "ok":
            continue
        started = time.perf_counter()
        ranked, trace = reranker.rank(record["question"], [chunk["text"] for chunk in route["chunks"]])
        ordered = [route["chunks"][item["index"]] for item in ranked]
        route["jina"] = {
            "status": "ok", "ranked": ranked, "trace": trace,
            "metrics": metrics(rows[record["question_id"]], ordered),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        payload = {"schema": "financial_bm25_shadow_v1", "records": records, "summary": summarize(records)}
        _write(output, payload)
        print(f"[jina {index:02d}/30] {record['question_id']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-jina-if-gate-passes", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        _reuse_baseline_jina(payload["records"])
        payload["summary"] = summarize(payload["records"])
        _write(args.output, payload)
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        return

    rows, groups = fixture_rows()
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    from embedding import embedding_service
    from milvus_client import MilvusManager

    manager = MilvusManager()
    if not manager.uses_builtin_bm25:
        raise RuntimeError("Financial BM25 shadow requires Milvus built-in BM25")
    records = []
    for index, (question_id, group) in enumerate(groups.items(), 1):
        question = rows[question_id]["question"]
        expansion = explain_financial_query_expansion(question)
        started = time.perf_counter()
        vector = embedding_service.get_embeddings([question])[0]
        dense = manager.dense_retrieve(vector, top_k=240, filter_expr=LEAF_FILTER)
        baseline_bm25 = manager.bm25_retrieve(question, top_k=240, filter_expr=LEAF_FILTER)
        financial_bm25 = manager.bm25_retrieve(expansion["expanded_query"], top_k=240, filter_expr=LEAF_FILTER)
        routes = {}
        for name, sparse in (("baseline", baseline_bm25), ("financial_bm25", financial_bm25)):
            chunks = rrf_fuse(dense, sparse, rrf_k=60)[:120]
            routes[name] = {
                "chunks": chunks, "candidate_sha256": digest(chunks),
                "metrics": metrics(rows[question_id], chunks),
            }
        records.append({
            "question_id": question_id, "group": group, "question": question,
            "dense_query": question, "expansion": expansion, "routes": routes,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        })
        payload = {"schema": "financial_bm25_shadow_v1", "records": records, "summary": summarize(records)}
        _write(args.output, payload)
        print(f"[{index:02d}/30] {question_id} A={routes['baseline']['metrics']['candidate_gold_page_hit']} B={routes['financial_bm25']['metrics']['candidate_gold_page_hit']}", flush=True)
    _reuse_baseline_jina(records)
    summary = summarize(records)
    if summary["candidate_gate_passed"] and args.run_jina_if_gate_passes:
        _run_financial_jina(records, args.output)
    else:
        status = "not_run_candidate_gate_failed" if not summary["candidate_gate_passed"] else "not_run_flag_required"
        for record in records:
            record["routes"]["financial_bm25"].setdefault("jina", {"status": status})
    payload = {
        "schema": "financial_bm25_shadow_v1",
        "manifest": {
            "questions": 30, "dense_top_k": 240, "bm25_top_k": 240, "rrf_top_k": 120, "rrf_k": 60,
            "dense_query_shared": True, "gold_used_after_ranking_only": True,
            "implementation_sha256": hashlib.sha256((ROOT / "backend/financial_query_expander.py").read_bytes()).hexdigest(),
        },
        "records": records,
    }
    payload["summary"] = summarize(records)
    _write(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
