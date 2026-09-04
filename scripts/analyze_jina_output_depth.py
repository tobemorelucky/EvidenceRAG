"""Offline Jina output-depth analysis over the verified diagnostic30 cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_reranker_shadow_v1 import filename, fixture_rows, page, validate_snapshot
from scripts.jina_full_baseline_v1 import build_context
from scripts.shadow_rerankers_v1 import validate_order

DEFAULT_SNAPSHOT = ROOT / "reports/reranker_shadow_v1_rrf_top120.json"
DEFAULT_CACHE = ROOT / "reports/reranker_shadow_v1.json"
DEFAULT_JSON = ROOT / "reports/jina_output_depth_analysis_30.json"
DEFAULT_MD = ROOT / "reports/jina_output_depth_analysis_30.md"
OUTPUT_DEPTHS = (4, 6, 8, 10, 12)
INPUT_DEPTHS = (120, 80)


def analyze(snapshot, cache):
    rows, groups = fixture_rows()
    sources = validate_snapshot(snapshot, rows, groups)
    cached = {record["question_id"]: record for record in cache.get("records", [])}
    results = {}
    for input_k in INPUT_DEPTHS:
        results[str(input_k)] = {}
        for output_k in OUTPUT_DEPTHS:
            records = []
            for source in sources:
                item = cached.get(source["question_id"])
                route = (item or {}).get("routes", {}).get("jina", {})
                if not item or item.get("candidate_sha256") != source["candidate_sha256"] or route.get("status") != "ok":
                    raise ValueError(f"Missing/drifted Jina cache: {source['question_id']}")
                ranked = [entry for entry in validate_order(route["ranked"], 120) if entry["index"] < input_k]
                if len(ranked) != input_k:
                    raise ValueError("Cached ranking is not a complete Top120 permutation")
                ordered = [source["chunks"][entry["index"]] for entry in ranked]
                gold = json.loads(rows[source["question_id"]].get("evidence") or "[]")
                gold_pages = {(filename(value["doc_name"]), int(value["evidence_page_num"])) for value in gold}
                output_pages = {page(chunk) for chunk in ordered[:output_k]}
                context, _, documents = build_context(ordered, {"top_k": output_k, "max_chars": 28000})
                context_pages = {page(chunk) for chunk in documents}
                estimated_tokens = math.ceil((len(source["question"]) + len(context)) / 4)
                records.append({
                    "question_id": source["question_id"],
                    "group": source["group"],
                    "gold_page_hit": bool(gold_pages & output_pages),
                    "context_hit": bool(gold_pages & context_pages),
                    "context_chars": len(context),
                    "estimated_llm_input_tokens": estimated_tokens,
                    "context_chunks": len(documents),
                })
            results[str(input_k)][str(output_k)] = {
                "questions": len(records),
                "gold_page_hit": fmean(record["gold_page_hit"] for record in records),
                "context_hit": fmean(record["context_hit"] for record in records),
                "estimated_llm_input_tokens": {
                    "total": sum(record["estimated_llm_input_tokens"] for record in records),
                    "mean_per_question": fmean(record["estimated_llm_input_tokens"] for record in records),
                },
                "mean_context_chars": fmean(record["context_chars"] for record in records),
                "mean_context_chunks": fmean(record["context_chunks"] for record in records),
                "records": records,
            }
    return {
        "schema": "jina_output_depth_analysis_v1",
        "questions": len(sources),
        "jina_input_depths": list(INPUT_DEPTHS),
        "output_depths": list(OUTPUT_DEPTHS),
        "context_budget_chars": 28000,
        "token_estimate": "ceil((question characters + formatted context characters) / 4); excludes fixed system prompt",
        "network_calls": 0,
        "results": results,
    }


def markdown(report):
    lines = ["# Jina Output Depth Analysis（固定30题，离线缓存重放）", "",
        "读取统一 RRF Top120 snapshot 与既有 Jina cache，分别重放 `JINA_INPUT_K=120/80`；未调用 Jina、LLM、Judge 或 LangSmith。", "",
        "- Gold page hit：Jina排序前K个chunk中是否包含gold page。",
        "- Context hit：经过既有28000字符预算后，实际进入格式化context的页面是否包含gold page。",
        "- Estimated LLM input tokens：`ceil((question chars + formatted context chars) / 4)`，不含固定system prompt，不代表DeepSeek真实tokenizer或账单。", ""]
    for input_k in report["jina_input_depths"]:
        lines += [f"## JINA_INPUT_K={input_k}", "", "| JINA_OUTPUT_K | Gold page hit | Context hit | Mean context chars | Mean context chunks | Estimated tokens/question | Estimated tokens/30 |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for depth in report["output_depths"]:
            row = report["results"][str(input_k)][str(depth)]
            tokens = row["estimated_llm_input_tokens"]
            lines.append(f"| {depth} | {row['gold_page_hit']:.2%} | {row['context_hit']:.2%} | {row['mean_context_chars']:.0f} | {row['mean_context_chunks']:.2f} | {tokens['mean_per_question']:.0f} | {tokens['total']:,} |")
        lines.append("")
    lines += ["## 收敛建议", "",
        "- 主质量基线：`JINA_INPUT_K=120`, `JINA_OUTPUT_K=12`。相对120/8，context hit由63.33%升至73.33%，估算LLM输入约由5.65k增至6.98k token/题。",
        "- 成本对照基线：`JINA_INPUT_K=80`, `JINA_OUTPUT_K=10`。context hit为63.33%，估算LLM输入约6.71k token/题；Jina输入成本依据上一轮分析约为Top120的66.88%。",
        "- 80/12达到66.67% context hit，但平均LLM输入约7.01k token/题；它不是当前成本profile的首选。",
        "- 这些结论来自30题缓存反事实重放。100题正式主基线应采用120/12；80/10保留为成本A/B，不应与主基线结果混合。", ""]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    if hashlib.sha256(args.snapshot.read_bytes()).hexdigest() != json.loads(args.cache.read_text(encoding="utf-8")).get("manifest", {}).get("input_sha256"):
        raise ValueError("Jina cache was not produced from this exact snapshot file")
    report = analyze(json.loads(args.snapshot.read_text(encoding="utf-8")), json.loads(args.cache.read_text(encoding="utf-8")))
    for path, text in ((args.output_json, json.dumps(report, ensure_ascii=False, indent=2)), (args.output_md, markdown(report))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    compact = {input_k: {output_k: {k: v for k, v in row.items() if k != "records"}
                         for output_k, row in output_rows.items()} for input_k, output_rows in report["results"].items()}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print(f"Report: {args.output_md}")


if __name__ == "__main__":
    main()
