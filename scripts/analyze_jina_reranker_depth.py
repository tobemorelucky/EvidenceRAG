"""Offline Jina input-depth replay over the frozen diagnostic30 cache.

No network/model client is imported or called. The cached Top120 Jina scores are
filtered to each RRF prefix, which is a counterfactual depth simulation rather
than a new API measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import fmean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_reranker_shadow_v1 import fixture_rows, metrics, validate_snapshot
from scripts.jina_full_baseline_v1 import digest, read_profile
from scripts.shadow_rerankers_v1 import validate_order

DEFAULT_SNAPSHOT = ROOT / "reports/reranker_shadow_v1_rrf_top120.json"
DEFAULT_CACHE = ROOT / "reports/reranker_shadow_v1.json"
DEFAULT_JSON = ROOT / "reports/jina_reranker_depth_analysis_30.json"
DEFAULT_MD = ROOT / "reports/jina_reranker_depth_analysis_30.md"
DEPTHS = (40, 60, 80, 100, 120)


def _cache_routes(cache, snapshot_records):
    config = cache.get("manifest", {}).get("backends", {}).get("jina", {})
    if config.get("model") != "jina-reranker-v3" or config.get("endpoint") != "https://api.jina.ai/v1/rerank":
        raise ValueError("Expected the verified official Jina v3 cache")
    cached = {record["question_id"]: record for record in cache.get("records", [])}
    routes = {}
    for source in snapshot_records:
        item = cached.get(source["question_id"])
        route = (item or {}).get("routes", {}).get("jina", {})
        if not item or item.get("candidate_sha256") != source["candidate_sha256"] or route.get("status") != "ok":
            raise ValueError(f"Missing/drifted Jina cache: {source['question_id']}")
        routes[source["question_id"]] = {
            **route,
            "ranked": validate_order(route["ranked"], 120),
        }
    return routes


def analyze(snapshot, cache, output_k=8):
    rows, groups = fixture_rows()
    sources = validate_snapshot(snapshot, rows, groups)
    routes = _cache_routes(cache, sources)
    details, summary = [], {}
    for depth in DEPTHS:
        depth_rows = []
        for source in sources:
            route = routes[source["question_id"]]
            # Cached Jina scores are point-wise replayed only for documents that
            # existed in the RRF prefix. No request and no re-scoring occurs.
            replay = [item for item in route["ranked"] if item["index"] < depth]
            if len(replay) != depth:
                raise ValueError("Cached ranking is not a complete Top120 permutation")
            ordered = [source["chunks"][item["index"]] for item in replay]
            score = metrics(rows[source["question_id"]], ordered, context_chunks=output_k, budget=28000)
            full_chars = len(source["question"]) + sum(len(chunk["text"]) for chunk in source["chunks"])
            selected_chars = len(source["question"]) + sum(len(chunk["text"]) for chunk in source["chunks"][:depth])
            reported = int((route.get("trace", {}).get("usage") or {}).get("total_tokens", 0))
            estimated_tokens = round(reported * selected_chars / full_chars) if reported and full_chars else None
            depth_rows.append({
                "question_id": source["question_id"],
                "group": source["group"],
                "candidate_recall": score["candidate_gold_page_hit"],
                "gold_page_rank": score["gold_page_rank"],
                "context_hit": score["context_hit"],
                "estimated_tokens": estimated_tokens,
                "cached_top120_reported_tokens": reported,
            })
        ranks = [row["gold_page_rank"] for row in depth_rows if row["gold_page_rank"] is not None]
        tokens = sum(row["estimated_tokens"] or 0 for row in depth_rows)
        full_tokens = sum(row["cached_top120_reported_tokens"] for row in depth_rows)
        summary[str(depth)] = {
            "questions": len(depth_rows),
            "candidate_recall": fmean(row["candidate_recall"] for row in depth_rows),
            "gold_page_rank": {"hit_questions": len(ranks), "mean_on_hits": fmean(ranks) if ranks else None,
                               "median_on_hits": median(ranks) if ranks else None},
            "context_hit": fmean(row["context_hit"] for row in depth_rows),
            "estimated_token_cost": {"tokens": tokens, "relative_to_cached_top120": tokens / full_tokens if full_tokens else None,
                                     "reduction_vs_top120": 1 - tokens / full_tokens if full_tokens else None},
        }
        details.append({"jina_input_k": depth, "records": depth_rows})
    return {
        "schema": "jina_reranker_depth_analysis_v1",
        "questions": len(sources),
        "depths": list(DEPTHS),
        "jina_output_k": output_k,
        "snapshot_sha256": digest(snapshot),
        "cache_input_sha256": cache.get("manifest", {}).get("input_sha256"),
        "method": {
            "ranking": "filter cached full-Top120 Jina permutation to each original RRF prefix",
            "token_estimate": "per-question cached Top120 reported tokens multiplied by (query+prefix document chars)/(query+Top120 document chars)",
            "limitations": "Counterfactual replay and character-calibrated token estimate; no Jina request, tokenizer, price, LLM, Judge, or LangSmith",
        },
        "summary": summary,
        "details": details,
    }


def markdown(report):
    lines = ["# Jina Reranker Depth Analysis（固定30题，离线缓存重放）", "",
        "本报告读取统一 RRF Top120 snapshot 和既有 Jina Top120 cache；未调用 Jina、LLM、Judge 或 LangSmith。", "",
        f"- `JINA_OUTPUT_K={report['jina_output_k']}`，context budget 固定 28000 chars。",
        "- Candidate recall：RRF 前 K 中是否含 gold page。",
        "- Gold page rank：将既有 Jina Top120 排名过滤到 RRF 前 K 后的唯一页面排名，仅对命中题统计。",
        "- Estimated token cost：按每题缓存的 Top120 reported tokens，以 query+documents 字符比例插值；它不是实际 tokenizer 计数或货币账单。", "",
        "| JINA_INPUT_K | Candidate recall | Gold page rank mean / median (hits) | Context hit | Estimated tokens | vs Top120 |", "|---:|---:|---:|---:|---:|---:|"]
    for depth in report["depths"]:
        row = report["summary"][str(depth)]
        rank = row["gold_page_rank"]
        cost = row["estimated_token_cost"]
        lines.append(f"| {depth} | {row['candidate_recall']:.2%} | {rank['mean_on_hits']:.2f} / {rank['median_on_hits']:.2f} ({rank['hit_questions']}) | {row['context_hit']:.2%} | {cost['tokens']:,} | {cost['relative_to_cached_top120']:.2%} |")
    lines += ["", "## 解释边界", "",
        "该实验能回答‘若只把 RRF 前 K 个候选交给同一套既有 Jina relevance scores，召回、排序与上下文命中如何变化’。它不能证明重新请求 Jina 后分数完全相同，因此正式选择深度前仍应把结果视为低成本筛选依据。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    profile = read_profile()
    if hashlib.sha256(args.snapshot.read_bytes()).hexdigest() != json.loads(args.cache.read_text(encoding="utf-8")).get("manifest", {}).get("input_sha256"):
        raise ValueError("Jina cache was not produced from this exact snapshot file")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    report = analyze(snapshot, cache, output_k=profile["reranker"]["output_k"])
    for path, content in ((args.output_json, json.dumps(report, ensure_ascii=False, indent=2)),
                          (args.output_md, markdown(report))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output_md}")


if __name__ == "__main__":
    main()
