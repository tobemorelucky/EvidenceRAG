"""One fixed30 metadata-aware rerank of cached BGE scores; zero inference/API."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.bge_metadata_reranker_v1 import rerank, VERSION, BGE_WEIGHT, METADATA_WEIGHT
from scripts.evaluate_reranker_shadow_v1 import DEFAULT_INPUT, GROUPS, fixture_rows, validate_snapshot, finalize_report, metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_view(chunk):
    return {k: chunk.get(k) for k in ("text", "company", "report_year", "section", "table_title")}


def summarize_comparison(records):
    def compare(items):
        old = lambda r: r["routes"]["bge"]["metrics"]["context_hit"]
        new = lambda r: r["routes"]["bge_metadata_v1"]["metrics"]["context_hit"]
        return {"questions": len(items), "before_hits": sum(old(r) for r in items),
                "after_hits": sum(new(r) for r in items),
                "gains": [r["question_id"] for r in items if new(r) and not old(r)],
                "regressions": [r["question_id"] for r in items if old(r) and not new(r)]}
    return {"all": compare(records), **{g: compare([r for r in records if r["group"] == g]) for g in GROUPS}}


def markdown(payload):
    lines = ["# BGE Metadata-aware Reranker Shadow v1", "",
        "固定30题/3600个RRF候选，复用原BGE logits与排序，仅执行一次预先固定的metadata融合。无模型推理、外部API或GPU加载。", "",
        "## 冻结策略", "",
        "`score = 0.75 × BGE排名百分位 + 0.25 × mean(有效metadata信号)`；本轮无权重搜索。",
        "实体：从候选company中匹配问题显式名称，匹配+1、已解析但不同公司−1、未知0。不加公司别名。",
        "期间：匹配局部句行及相邻行=强信号，chunk其他位置=半权，report_year=四分之一权。报告年份不同不作为事实年份冲突。",
        "指标：去通用任务词、实体和年份后，按候选内IDF计算局部词重叠；没有金融指标/公式映射。",
        "全部120候选保留，只变顺序；context仍是原始完整chunk Top8/≤28000字符，不改生产Packing或Assembly。",
        "实体名称、年份共现与指标词面匹配不是精确事实绑定；多实体角色、会计期间、表格列归属可能仍然不明确。", "",
        f"- Snapshot SHA256: `{payload['manifest']['input_sha256']}`",
        f"- 本轮模型/API调用：0；配置：`{payload['experiment_config']}`", "",
        "## 结果", "", "| Backend | 完成 | Candidate hit | Page@5 | Page@10 | Page@20 | Context hit | Context span hit | Gold page rank* | Gold chunk rank* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in payload["summary"].items():
        page_rank = s["gold_page_rank"]["mean_on_hits"]
        chunk_rank = s["gold_chunk_rank"]["mean_on_hits"]
        lines.append(f"| {name} | {s['completed']}/30 | {s['candidate_gold_page_hit']:.2%} | {s['page_hit_at_5']:.2%} | {s['page_hit_at_10']:.2%} | {s['page_hit_at_20']:.2%} | {s['context_hit']:.2%} | {s['context_evidence_span_hit']:.2%} | {page_rank:.2f} | {chunk_rank:.2f} |")
    lines += ["", "identity/BGE/Jina均为相同快照的历史缓存，本轮没有重新运行。排名均值只统计命中题；gold chunk为同页≥40字符完整原文行代理，不是官方chunk标注。所有内部页码保持0-based。",
        "Context hit是固定shadow投影的gold页出现率，不是答案正确率，也不能代表事实约束全部满足。", "",
        "## 分组", "", "| Group | 原BGE | Metadata BGE | 新增 | 回退 |", "|---|---:|---:|---:|---:|"]
    for group, d in payload["comparison_vs_bge"].items():
        lines.append(f"| {group} | {d['before_hits']}/{d['questions']} | {d['after_hits']}/{d['questions']} | {len(d['gains'])} | {len(d['regressions'])} |")
    lines += ["", f"逐题新增/回退：`{json.dumps(payload['comparison_vs_bge'], ensure_ascii=False)}`", "",
        "## Metadata可用性与成本", "", f"`{json.dumps(payload['metadata_diagnostics'], ensure_ascii=False)}`", "",
        "无额外token消耗；耗时只包含本地特征提取和排序，不含加载JSON、gold评估、报告写入。",
        "未匹配到名称不等于公司错误，report_year不等于值所属年份，metric仅为词面代理，诊断不得当作真实entity/period/metric正确率。", "",
        "## 逐题", "", "| ID | Group | 原BGE context | 新context | 原page rank | 新page rank | 解析实体 | 年份 |", "|---|---|---|---|---:|---:|---|---|"]
    for r in payload["records"]:
        old = r["routes"]["bge"]["metrics"]
        new = r["routes"]["bge_metadata_v1"]["metrics"]
        t = r["routes"]["bge_metadata_v1"]["trace"]
        lines.append(f"| {r['question_id']} | {r['group']} | {old['context_hit']} | {new['context_hit']} | {old['gold_page_rank']} | {new['gold_page_rank']} | {t['resolved_query_entities']} | {t['required_years']} |")
    old = payload["summary"]["bge"]["context_hit"]
    new = payload["summary"]["bge_metadata_v1"]["context_hit"]
    jina = payload["summary"]["jina"]["context_hit"]
    lines += ["", "## 判读与停止条件", "",
        f"Context hit变化：{old:.2%} → {new:.2%}；距历史Jina {jina:.2%}尚有{(jina-new)*100:.2f}个百分点。",
        "本轮只验证这一个固定策略；无论结果如何，都不在此30题上追加参数搜索，也不接入生产。",
        "如有提升，仅说明该策略对开发诊断集排序有帮助，不能证明实体/期间已精确绑定，或推断未见题答案准确率。若无提升，只否定当前轻量方案，不证明全部metadata方法无效。",
        "全部120×4组排名与指标已重算；逐候选原始BGE分数、metadata信号、最终分数及局部片段见JSON。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--baseline", type=Path, default=ROOT / "reports/reranker_shadow_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/bge_metadata_shadow_v1.json")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in {args.input.resolve(), args.baseline.resolve()}:
        parser.error("Output already exists or targets a frozen artifact; refusing overwrite")
    rows, groups = fixture_rows()
    frozen = validate_snapshot(json.loads(args.input.read_text(encoding="utf-8")), rows, groups)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    dataset = ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"
    if baseline["manifest"]["input_sha256"] != sha(args.input) or baseline["manifest"]["dataset_sha256"] != sha(dataset):
        raise ValueError("Cached baseline does not match frozen input/dataset")
    baseline = finalize_report(baseline, frozen, rows)
    if any(baseline["summary"][b]["completed"] != 30 for b in ("identity", "bge", "jina")):
        raise ValueError("Completed cached baseline required; never fetch missing scores")
    manifest = {"input_sha256": sha(args.input), "baseline_sha256": sha(args.baseline),
                "dataset_sha256": sha(dataset), "ranker_sha256": sha(ROOT / "scripts/bge_metadata_reranker_v1.py"),
                "evaluator_sha256": sha(Path(__file__)), "context_chunks": 8, "context_budget": 28000,
                "backends": {**baseline["manifest"]["backends"], "bge_metadata_v1": {"version": VERSION, "source": "cached_bge_plus_metadata"}}}
    records, saved = [], {r["question_id"]: r for r in baseline["records"]}
    print("[setup] frozen30 RRF Top120; cached BGE only; model/API calls=0", flush=True)
    for i, source in enumerate(frozen, 1):
        record = copy.deepcopy(saved[source["question_id"]])
        start = time.perf_counter()
        ranked, trace = rerank(source["question"], [input_view(c) for c in source["chunks"]], record["routes"]["bge"]["ranked"])
        elapsed = (time.perf_counter() - start) * 1000
        # Only here are gold annotations used, after the final order exists.
        record["routes"]["bge_metadata_v1"] = {"status": "ok", "ranked": ranked, "trace": trace,
            "latency_ms": elapsed, "input_chars": 0,
            "metrics": metrics(rows[source["question_id"]], [source["chunks"][v["index"]] for v in ranked])}
        records.append(record)
        if i % 10 == 0:
            print(f"[offline] completed {i}/30", flush=True)
    payload = {"manifest": manifest, "records": records,
               "experiment_config": {"version": VERSION, "bge_weight": BGE_WEIGHT, "metadata_weight": METADATA_WEIGHT,
                                     "parameter_search": False, "new_model_calls": 0, "new_network_calls": 0},
               "comparison_vs_bge": summarize_comparison(records)}
    traces = [r["routes"]["bge_metadata_v1"]["trace"] for r in records]
    units = [u for t in traces for u in t["units"]]
    payload["metadata_diagnostics"] = {
        "resolved_entity_questions": sum(bool(t["resolved_query_entities"]) for t in traces),
        "explicit_year_questions": sum(bool(t["required_years"]) for t in traces),
        "entity_status_counts": dict(Counter(u["entity_status"] for u in units)),
        "period_status_counts": dict(Counter(u["period_status"] for u in units)),
        "nonzero_metric_lexical_pairs": sum(u["metric_relevance"] > 0 for u in units),
        "mean_rerank_ms": fmean(r["routes"]["bge_metadata_v1"]["latency_ms"] for r in records),
        "company_populated_pairs": sum(bool(c.get("company")) for r in frozen for c in r["chunks"]),
        "table_title_populated_pairs": sum(bool(c.get("table_title")) for r in frozen for c in r["chunks"])}
    payload = finalize_report(payload, frozen, rows)
    if sha(args.input) != manifest["input_sha256"] or sha(args.baseline) != manifest["baseline_sha256"]:
        raise ValueError("Frozen inputs changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    args.output.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload["comparison_vs_bge"], ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
