"""Rerank frozen original/query-rewrite Top120 candidates with Jina.

This shadow evaluator never runs retrieval, answer generation, Judge, or
LangSmith.  Successful Jina routes are checkpointed after every request.
"""

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
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_reranker_shadow_v1 import digest, fixture_rows, metrics  # noqa: E402
from jina_full_baseline_v1 import build_context  # noqa: E402
from shadow_rerankers_v1 import JinaReranker, validate_order  # noqa: E402


DEFAULT_INPUT = ROOT / "reports/query_rewrite_shadow_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/query_rewrite_jina_shadow_v1.json"
OLD_SNAPSHOT = ROOT / "reports/reranker_shadow_v1_rrf_top120.json"
OLD_JINA_CACHE = ROOT / "reports/reranker_shadow_v1.json"
ROUTES = ("baseline", "query_rewrite")
CONTEXT_CONFIG = {"top_k": 12, "max_chars": 28000}


def validate_frozen_input(payload: dict) -> list[dict]:
    if payload.get("schema") != "query_rewrite_shadow_v1":
        raise ValueError("Expected query_rewrite_shadow_v1 input")
    records = payload.get("records") or []
    rows, groups = fixture_rows()
    if len(records) != 30 or {record.get("question_id") for record in records} != set(rows):
        raise ValueError("Input must contain the fixed diagnostic30")
    for record in records:
        question_id = record["question_id"]
        rewrite = record.get("rewrite") or {}
        queries = rewrite.get("queries") or []
        if record.get("question") != rows[question_id]["question"] or record.get("group") != groups[question_id]:
            raise ValueError(f"Question/group drift: {question_id}")
        if not 1 <= len(queries) <= 3 or queries[0] != record["question"]:
            raise ValueError(f"Invalid original-first rewrite set: {question_id}")
        for route_name in ROUTES:
            route = record.get("routes", {}).get(route_name, {})
            chunks = route.get("chunks") or []
            if len(chunks) != 120 or len({chunk.get("chunk_id") for chunk in chunks}) != 120:
                raise ValueError(f"Incomplete {route_name} Top120: {question_id}")
            if [chunk.get("rrf_rank") for chunk in chunks] != list(range(1, 121)):
                raise ValueError(f"Invalid {route_name} RRF ranks: {question_id}")
            if digest(chunks) != route.get("candidate_sha256"):
                raise ValueError(f"Candidate hash drift: {question_id}/{route_name}")
    return records


def candidate_signature(chunks: list[dict]) -> list[tuple]:
    """Fingerprint exactly what Jina sees plus stable source identity."""
    return [
        (
            chunk.get("chunk_id"), chunk.get("filename"),
            int(chunk.get("page_number") or 0), chunk.get("text"),
        )
        for chunk in chunks
    ]


def route_metrics(row: dict, chunks: list[dict], ranked: list[dict]) -> dict:
    ordered = [chunks[item["index"]] for item in validate_order(ranked, len(chunks))]
    full = metrics(row, ordered)
    top12 = metrics(row, ordered[:12], context_chunks=12, budget=10**9)
    _, citations, context_documents = build_context(ordered, CONTEXT_CONFIG)
    context = metrics(
        row, context_documents,
        context_chunks=max(1, len(context_documents)), budget=10**9,
    )
    return {
        "candidate_hit": metrics(row, chunks)["candidate_gold_page_hit"],
        "jina_top12_gold_chunk_hit": top12["gold_chunk_rank"] is not None,
        "jina_top12_gold_page_hit": top12["gold_page_rank"] is not None,
        "context_hit": context["candidate_gold_page_hit"],
        "context_evidence_span_hit": context["gold_chunk_rank"] is not None,
        "gold_chunk_rank": full["gold_chunk_rank"],
        "gold_page_rank": full["gold_page_rank"],
        "context_chars": sum(citation["included_chars"] for citation in citations)
            + sum(len(f"Source: {citation['filename']} | Page: {citation['page_number']}\n") for citation in citations)
            + max(0, len(citations) - 1) * 2,
        "context_documents": len(context_documents),
        "context_citations": citations,
    }


def _reuse_old_baseline(frozen: list[dict]) -> dict[str, dict]:
    if not OLD_SNAPSHOT.exists() or not OLD_JINA_CACHE.exists():
        return {}
    old_source = json.loads(OLD_SNAPSHOT.read_text(encoding="utf-8"))
    old_cache = json.loads(OLD_JINA_CACHE.read_text(encoding="utf-8"))
    source_by_id = {record["question_id"]: record for record in old_source.get("records", [])}
    cache_by_id = {record["question_id"]: record for record in old_cache.get("records", [])}
    reused = {}
    for record in frozen:
        question_id = record["question_id"]
        source = source_by_id.get(question_id)
        cached = cache_by_id.get(question_id, {}).get("routes", {}).get("jina", {})
        current = record["routes"]["baseline"]["chunks"]
        if (
            source and cached.get("status") == "ok"
            and source.get("question") == record["question"]
            and candidate_signature(source.get("chunks") or []) == candidate_signature(current)
        ):
            reused[question_id] = cached
    return reused


def summarize(records: list[dict]) -> dict:
    result = {}
    for route_name in ROUTES:
        completed = [record for record in records if record.get("routes", {}).get(route_name, {}).get("status") == "ok"]
        values = [record["routes"][route_name]["metrics"] for record in completed]
        page_ranks = [item["gold_page_rank"] for item in values if item["gold_page_rank"] is not None]
        chunk_ranks = [item["gold_chunk_rank"] for item in values if item["gold_chunk_rank"] is not None]
        result[route_name] = {
            "completed": len(completed),
            "statuses": dict(Counter(record.get("routes", {}).get(route_name, {}).get("status", "pending") for record in records)),
            "candidate_hit": fmean(item["candidate_hit"] for item in values) if values else None,
            "jina_top12_gold_chunk_hit": fmean(item["jina_top12_gold_chunk_hit"] for item in values) if values else None,
            "jina_top12_gold_page_hit": fmean(item["jina_top12_gold_page_hit"] for item in values) if values else None,
            "context_hit": fmean(item["context_hit"] for item in values) if values else None,
            "gold_chunk_rank": {"hits": len(chunk_ranks), "mean_on_hits": fmean(chunk_ranks) if chunk_ranks else None},
            "gold_page_rank": {"hits": len(page_ranks), "mean_on_hits": fmean(page_ranks) if page_ranks else None},
            "jina_cache_hits": sum(record["routes"][route_name].get("cache_hit", False) for record in completed),
            "jina_new_requests": sum(not record["routes"][route_name].get("cache_hit", False) for record in completed),
        }
    if all(result[route]["completed"] == 30 for route in ROUTES):
        delta = result["query_rewrite"]["context_hit"] - result["baseline"]["context_hit"]
        result["context_hit_delta"] = round(delta, 6)
        result["context_gate"] = ">= +5 percentage points"
        result["context_gate_passed"] = delta >= 0.05
        result["decision"] = "continue_to_all100" if delta >= 0.05 else "stop"
        result["context_gains"] = [
            record["question_id"] for record in records
            if record["routes"]["query_rewrite"]["metrics"]["context_hit"]
            and not record["routes"]["baseline"]["metrics"]["context_hit"]
        ]
        result["context_regressions"] = [
            record["question_id"] for record in records
            if not record["routes"]["query_rewrite"]["metrics"]["context_hit"]
            and record["routes"]["baseline"]["metrics"]["context_hit"]
        ]
    return result


def markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Query Rewrite + Jina Shadow v1", "",
        "输入为冻结的 Query Rewrite Shadow v1 Top120。未重新执行rewrite或retrieval；Jina input=120、output=12，context使用原有build_context(top_k=12, max_chars=28000)。", "",
        "| Route | Candidate hit | Jina Top12 chunk hit* | Jina Top12 page hit | Context hit | Gold page rank |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for route_name in ROUTES:
        item = summary[route_name]
        value = lambda key: "N/A" if item[key] is None else f"{item[key]:.2%}"
        lines.append(
            f"| `{route_name}` | {value('candidate_hit')} | {value('jina_top12_gold_chunk_hit')} | "
            f"{value('jina_top12_gold_page_hit')} | {value('context_hit')} | {item['gold_page_rank']['mean_on_hits']} |"
        )
    if "context_hit_delta" in summary:
        lines.extend([
            "", f"- Context delta：`{summary['context_hit_delta']:+.2%}`",
            f"- Gains：`{summary['context_gains']}`",
            f"- Regressions：`{summary['context_regressions']}`",
            f"- +5pp门槛：`{summary['context_gate_passed']}`；决定：`{summary['decision']}`",
        ])
    lines.extend([
        "", "- *Gold chunk是gold页内参考原文行匹配代理，不是官方chunk ID。", "",
        "## 逐题", "",
        "| ID | Group | A candidate/top12/context/rank | B candidate/top12/context/rank |",
        "|---|---|---|---|",
    ])
    for record in payload["records"]:
        cells = []
        for route_name in ROUTES:
            route = record.get("routes", {}).get(route_name, {})
            if route.get("status") != "ok":
                cells.append(route.get("status", "pending"))
                continue
            item = route["metrics"]
            cells.append(
                f"{item['candidate_hit']}/{item['jina_top12_gold_page_hit']}/{item['context_hit']}/{item['gold_page_rank']}"
            )
        lines.append(f"| {record['question_id']} | {record['group']} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines) + "\n"


def write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    path.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    # A Top120 request is about 80k rerank tokens for this snapshot.  The
    # current Jina key allows 100k tokens/minute, so two requests in the same
    # rolling minute are rejected even when spaced by the older 8s setting.
    parser.add_argument("--jina-interval-seconds", type=float, default=60.0)
    args = parser.parse_args()
    if not 0 <= args.jina_interval_seconds <= 60:
        parser.error("Jina interval must be between 0 and 60 seconds")

    input_payload = json.loads(args.input.read_text(encoding="utf-8"))
    frozen = validate_frozen_input(input_payload)
    input_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    rows, _ = fixture_rows()
    old_reuse = _reuse_old_baseline(frozen)

    saved = {}
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous.get("manifest", {}).get("input_sha256") != input_sha256:
            raise ValueError("Existing checkpoint belongs to a different frozen input")
        saved = {record["question_id"]: record for record in previous.get("records", [])}

    os.environ.update(LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false")
    from dotenv import dotenv_values
    env = {**dotenv_values(ROOT / ".env"), **os.environ}
    key = env.get("RERANK_API_KEY") or env.get("JINA_API_KEY")
    if not key:
        raise RuntimeError("Jina credential is not configured")
    host = env.get("RERANK_BINDING_HOST") or "https://api.jina.ai"
    endpoint = host.rstrip("/") if host.rstrip("/").endswith("/v1/rerank") else host.rstrip("/") + "/v1/rerank"
    reranker = JinaReranker(
        key, env.get("RERANK_MODEL") or "jina-reranker-v3", endpoint,
        interval=args.jina_interval_seconds,
    )
    manifest = {
        "input_sha256": input_sha256,
        "jina": reranker.config,
        "jina_input_k": 120, "jina_output_k": 12,
        "context": CONTEXT_CONFIG,
        "retrieval_executed": False,
        "answer_llm_calls": 0, "judge_calls": 0, "langsmith_calls": 0,
    }
    records = []
    for index, source in enumerate(frozen, 1):
        question_id = source["question_id"]
        record = saved.get(question_id, {
            "question_id": question_id, "group": source["group"],
            "question": source["question"], "routes": {},
        })
        for route_name in ROUTES:
            chunks = source["routes"][route_name]["chunks"]
            existing = record["routes"].get(route_name, {})
            if existing.get("status") == "ok":
                if existing.get("candidate_sha256") != source["routes"][route_name]["candidate_sha256"]:
                    raise ValueError(f"Checkpoint candidate drift: {question_id}/{route_name}")
                continue

            cached = old_reuse.get(question_id) if route_name == "baseline" else None
            if cached:
                ranked = validate_order(cached["ranked"], 120)
                trace = cached.get("trace") or {}
                cache_hit = True
            else:
                print(f"[{index:02d}/30] {route_name} Jina starting", flush=True)
                started = time.perf_counter()
                try:
                    ranked, trace = reranker.rank(source["question"], [chunk["text"] for chunk in chunks])
                    ranked = validate_order(ranked, 120)
                except Exception as exc:
                    record["routes"][route_name] = {
                        "status": "error", "error_type": type(exc).__name__,
                        "reason": str(exc)[:500],
                        "candidate_sha256": source["routes"][route_name]["candidate_sha256"],
                    }
                    raise
                cache_hit = False
                print(f"[{index:02d}/30] {route_name} Jina finished in {time.perf_counter() - started:.2f}s", flush=True)

            record["routes"][route_name] = {
                "status": "ok", "candidate_sha256": source["routes"][route_name]["candidate_sha256"],
                "cache_hit": cache_hit, "ranked": ranked, "trace": trace,
                "metrics": route_metrics(rows[question_id], chunks, ranked),
            }
            saved[question_id] = record
            partial = [saved[item["question_id"]] for item in frozen if item["question_id"] in saved]
            payload = {"schema": "query_rewrite_jina_shadow_v1", "manifest": manifest, "records": partial}
            payload["summary"] = summarize(partial)
            write_output(args.output, payload)
        records.append(record)
        saved[question_id] = record
        print(f"[{index:02d}/30] {question_id} completed", flush=True)

    payload = {"schema": "query_rewrite_jina_shadow_v1", "manifest": manifest, "records": records}
    payload["summary"] = summarize(records)
    write_output(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
