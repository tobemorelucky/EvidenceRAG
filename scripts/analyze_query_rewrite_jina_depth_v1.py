"""Offline input-depth replay for the frozen Query Rewrite + Jina diagnostic30.

No model or network client is imported. Cached Top120 Jina scores are filtered
to each original RRF prefix, so this is a counterfactual replay rather than a
new Jina measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_evidence_assembly_ab import _contains_all, _numbers, _periods, _required_numbers  # noqa: E402
from evaluate_reranker_shadow_v1 import fixture_rows, metrics  # noqa: E402
from jina_full_baseline_v1 import build_context  # noqa: E402
from shadow_rerankers_v1 import validate_order  # noqa: E402


DEPTHS = (40, 60, 80, 100, 120)
OUTPUT_K = 12
CONTEXT_CONFIG = {"top_k": OUTPUT_K, "max_chars": 28000}
DEFAULT_CANDIDATES = ROOT / "reports/query_rewrite_shadow_v1.json"
DEFAULT_JINA = ROOT / "reports/query_rewrite_jina_shadow_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/query_rewrite_jina_depth_analysis_v1"


def _rate(values: list[bool | None]) -> float | None:
    available = [value for value in values if value is not None]
    return fmean(available) if available else None


def _load_frozen(candidates: dict, jina: dict) -> tuple[list[dict], dict[str, dict]]:
    rows, groups = fixture_rows()
    if candidates.get("schema") != "query_rewrite_shadow_v1" or jina.get("schema") != "query_rewrite_jina_shadow_v1":
        raise ValueError("Expected Query Rewrite and Query Rewrite + Jina v1 caches")
    sources = candidates.get("records") or []
    cached = {record["question_id"]: record for record in jina.get("records") or []}
    if len(sources) != 30 or len(cached) != 30 or {record["question_id"] for record in sources} != set(rows):
        raise ValueError("Both caches must contain the same fixed diagnostic30")
    for source in sources:
        question_id = source["question_id"]
        chunks = source.get("routes", {}).get("query_rewrite", {}).get("chunks") or []
        route = cached[question_id].get("routes", {}).get("query_rewrite", {})
        if source.get("question") != rows[question_id]["question"] or source.get("group") != groups[question_id]:
            raise ValueError(f"Question/group drift: {question_id}")
        if len(chunks) != 120 or [chunk.get("rrf_rank") for chunk in chunks] != list(range(1, 121)):
            raise ValueError(f"Incomplete RRF Top120: {question_id}")
        if route.get("status") != "ok" or route.get("candidate_sha256") != source["routes"]["query_rewrite"].get("candidate_sha256"):
            raise ValueError(f"Missing/drifted Jina ranking: {question_id}")
        validate_order(route.get("ranked") or [], 120)
    return sources, cached


def analyze(candidates: dict, jina: dict) -> dict:
    dataset, _ = fixture_rows()
    sources, cached = _load_frozen(candidates, jina)
    details: dict[str, list[dict]] = {}
    summary: dict[str, dict] = {}
    for depth in DEPTHS:
        records = []
        for source in sources:
            question_id = source["question_id"]
            row = dataset[question_id]
            chunks = source["routes"]["query_rewrite"]["chunks"]
            route = cached[question_id]["routes"]["query_rewrite"]
            ranked = [item for item in validate_order(route["ranked"], 120) if item["index"] < depth]
            if len(ranked) != depth:
                raise ValueError("Cached ranking is not a complete Top120 permutation")
            ordered = [chunks[item["index"]] for item in ranked]
            top12 = ordered[:OUTPUT_K]
            evidence, _, context_documents = build_context(ordered, CONTEXT_CONFIG)
            candidate_score = metrics(row, chunks[:depth])
            top12_score = metrics(row, top12, context_chunks=OUTPUT_K, budget=10**9)
            context_score = metrics(row, context_documents, context_chunks=max(1, len(context_documents)), budget=10**9)
            required_numbers = _required_numbers(row)
            required_periods = _periods(row["question"])

            full_chars = len(row["question"]) + sum(len(chunk["text"]) for chunk in chunks)
            depth_chars = len(row["question"]) + sum(len(chunk["text"]) for chunk in chunks[:depth])
            reported = int((route.get("trace", {}).get("usage") or {}).get("total_tokens") or 0)
            estimated_tokens = round(reported * depth_chars / full_chars) if reported and full_chars else math.ceil(depth_chars / 4)
            records.append({
                "question_id": question_id, "group": source["group"],
                "candidate_hit": candidate_score["candidate_gold_page_hit"],
                "gold_chunk_hit": top12_score["gold_chunk_rank"] is not None,
                "gold_page_hit": top12_score["gold_page_rank"] is not None,
                "context_hit": context_score["candidate_gold_page_hit"],
                "required_number_hit": _contains_all(required_numbers, evidence, _numbers),
                "required_period_hit": _contains_all(required_periods, evidence, _periods),
                "gold_chunk_rank_after_jina": metrics(row, ordered)["gold_chunk_rank"],
                "gold_page_rank_after_jina": metrics(row, ordered)["gold_page_rank"],
                "context_chars": len(evidence),
                "estimated_jina_tokens": estimated_tokens,
                "cached_top120_reported_tokens": reported,
            })
        details[str(depth)] = records
        summary[str(depth)] = {
            "input_k": depth,
            "candidate_hit": _rate([record["candidate_hit"] for record in records]),
            "gold_chunk_hit": _rate([record["gold_chunk_hit"] for record in records]),
            "gold_page_hit": _rate([record["gold_page_hit"] for record in records]),
            "context_hit": _rate([record["context_hit"] for record in records]),
            "required_number_hit": _rate([record["required_number_hit"] for record in records]),
            "required_period_hit": _rate([record["required_period_hit"] for record in records]),
            "estimated_jina_tokens": {
                "total": sum(record["estimated_jina_tokens"] for record in records),
                "mean_per_question": fmean(record["estimated_jina_tokens"] for record in records),
            },
            "mean_context_chars": fmean(record["context_chars"] for record in records),
        }

    top = summary["120"]
    for depth in DEPTHS:
        item = summary[str(depth)]
        item["relative_to_120"] = {
            key: item[key] - top[key]
            for key in ("candidate_hit", "gold_chunk_hit", "gold_page_hit", "context_hit", "required_number_hit", "required_period_hit")
            if item[key] is not None and top[key] is not None
        }
        item["token_saving_vs_120"] = 1 - item["estimated_jina_tokens"]["total"] / top["estimated_jina_tokens"]["total"]

    context_delta_80 = summary["80"]["context_hit"] - top["context_hit"]
    context_delta_100 = summary["100"]["context_hit"] - top["context_hit"]
    context_drop_80 = max(0.0, -context_delta_80)
    context_drop_100 = max(0.0, -context_delta_100)
    if context_drop_80 <= 0.03:
        recommendation, reason = 80, "K=80 context hit相对K=120没有超过3pp的质量下降"
    elif context_drop_100 <= 0.03:
        recommendation, reason = 100, "K=80未达标，但K=100接近K=120"
    else:
        recommendation, reason = 120, "K=80和K=100的context hit均未保持在3pp以内"
    return {
        "schema": "query_rewrite_jina_depth_analysis_v1",
        "questions": 30, "input_depths": list(DEPTHS), "output_k": OUTPUT_K,
        "context_budget_chars": 28000,
        "method": {
            "ranking": "filter cached Jina Top120 permutation to each original Query Rewrite RRF prefix",
            "token_estimate": "cached per-question Top120 reported tokens scaled by query+prefix document character ratio; chars/4 fallback only when usage is absent",
            "limitation": "counterfactual replay; no Jina request or rescoring",
        },
        "network_calls": {"jina": 0, "llm": 0, "judge": 0},
        "summary": summary, "details": details,
        "recommendation": {
            "input_k": recommendation, "reason": reason,
            "context_delta_80_vs_120": context_delta_80,
            "context_delta_100_vs_120": context_delta_100,
            "context_drop_80_vs_120": context_drop_80,
            "context_drop_100_vs_120": context_drop_100,
        },
    }


def markdown(report: dict) -> str:
    lines = [
        "# Query Rewrite + Jina Depth Optimization Shadow v1", "",
        "固定 Query Rewrite RRF Top120 和既有 Jina Top120 排名，离线过滤到不同input K后取Top12，并使用原build_context(max_chars=28000)。未调用Jina、LLM或Judge。", "",
        "| Input K | Candidate hit | Top12 chunk hit* | Top12 page hit | Context hit | Required number | Required period | Est. Jina tokens | Token saving |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in DEPTHS:
        item = report["summary"][str(depth)]
        lines.append(
            f"| {depth} | {item['candidate_hit']:.2%} | {item['gold_chunk_hit']:.2%} | {item['gold_page_hit']:.2%} | "
            f"{item['context_hit']:.2%} | {item['required_number_hit']:.2%} | {item['required_period_hit']:.2%} | "
            f"{item['estimated_jina_tokens']['total']:,} | {item['token_saving_vs_120']:.2%} |"
        )
    lines.extend(["", "## 相对K=120", "",
        "| Input K | Candidate Δ | Top12 page Δ | Context Δ | Number Δ | Period Δ |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for depth in DEPTHS:
        delta = report["summary"][str(depth)]["relative_to_120"]
        lines.append(
            f"| {depth} | {delta['candidate_hit']:+.2%} | {delta['gold_page_hit']:+.2%} | "
            f"{delta['context_hit']:+.2%} | {delta['required_number_hit']:+.2%} | {delta['required_period_hit']:+.2%} |"
        )
    recommendation = report["recommendation"]
    lines.extend(["", "## 建议", "",
        f"推荐 `JINA_INPUT_K={recommendation['input_k']}`：{recommendation['reason']}。",
        f"K=80/100相对120的context变化分别为 `{recommendation['context_delta_80_vs_120']:+.2%}` / `{recommendation['context_delta_100_vs_120']:+.2%}`。", "",
        "## 解释边界", "",
        "这是既有Top120分数的反事实过滤，不是对较小候选集重新请求Jina。它适合做零成本深度筛选，但正式100题配置仍应将该限制写入实验说明。",
        "*Gold chunk hit使用gold页内参考原文行匹配代理，并非官方chunk ID。", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--jina-cache", type=Path, default=DEFAULT_JINA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = analyze(
        json.loads(args.candidates.read_text(encoding="utf-8")),
        json.loads(args.jina_cache.read_text(encoding="utf-8")),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["recommendation"], ensure_ascii=False, indent=2))
    print(f"Report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
