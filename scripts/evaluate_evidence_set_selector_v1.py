"""Offline fixed diagnostic30: Packing v1 vs dynamic Evidence Set Selector v1."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import socket
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.evidence_assembly_v5 import EvidenceUnit
from backend.evidence_packing_v1 import _rank, _render_selection
from backend.evidence_set_selector_v1 import select_evidence_set_v1
from scripts.evaluate_evidence_metadata_counterfactual_v1 import _context_metrics, _gold, _gold_retention, _load_dataset
from scripts.evaluate_evidence_packing_optimization_v1 import _route_summary

GROUPS = ("selection_loss10", "correct_regression10", "candidate_miss10")
ROUTES = ("packing_v1", "set_selector_v1")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def evaluate_record(source: dict, prior: dict, row: dict) -> dict:
    units = [{name: copy.deepcopy(unit[name]) for name in (*EvidenceUnit.__dataclass_fields__, "current_ranking")}
             for unit in source["candidate_units"]]
    fingerprint = digest(units)
    baseline = prior["routes"]["packing_optimization_v1"]
    old_selected = [unit for unit in units if _rank(unit) in baseline["selected_unit_ranks"]]
    if _context_metrics(row, _render_selection(old_selected)[0], old_selected) != baseline["metrics"]:
        raise ValueError("Frozen baseline metrics do not reproduce")
    start = time.perf_counter()
    context, selected, trace = select_evidence_set_v1(units, max_context_chars=28000)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if fingerprint != digest(units):
        raise AssertionError("Selector mutated candidate units/ranking")
    return {
        "financebench_id": source["financebench_id"], "group": source["group"], "question": source["question"],
        "candidate_count": len(units), "candidate_sha256": fingerprint,
        "candidate_and_ranking_unchanged": True, "baseline_reproduced": True,
        "selection_ms": round(elapsed_ms, 2),
        "sequential_coverage": prior["routes"]["sequential_first_fit"]["metrics"]["answer_evidence_coverage"]["ratio"],
        "candidate_manifest": [{"rank": _rank(unit), "score": unit["current_ranking"]["score"],
                                "document_id": unit["document_id"], "page_id": unit["page_id"], "source_type": unit["source_type"],
                                "chunk_id": unit["metadata"].get("chunk_id"), "table_id": unit["metadata"].get("table_id"),
                                "row_index": unit["metadata"].get("row_index")} for unit in units],
        "routes": {
            "packing_v1": {
                "metrics": baseline["metrics"], "gold_evidence_retention": baseline["gold_evidence_retention"],
                "selected_ranks": baseline["selected_unit_ranks"],
                "replacement_count": baseline["packing_trace"]["replacement_count"],
            },
            "set_selector_v1": {
                "metrics": _context_metrics(row, context, selected),
                "gold_evidence_retention": _gold_retention(_gold(row), "\n\n".join(u["source_text"] for u in units), context),
                "selected_ranks": [_rank(unit) for unit in selected], "replacement_count": 0,
                "trace": trace,
            },
        },
    }


def summarize(records: list[dict]) -> dict:
    def section(items):
        result = {route: _route_summary(items, route) for route in ROUTES}
        for route in ROUTES:
            result[route]["replacement_count"] = sum(r["routes"][route]["replacement_count"] for r in items)
            documents, pages, scores = [], [], []
            for record in items:
                chosen = set(record["routes"][route]["selected_ranks"])
                selected = [unit for unit in record["candidate_manifest"] if unit["rank"] in chosen]
                documents.append(len({unit["document_id"] for unit in selected}))
                pages.append(len({unit["page_id"] for unit in selected}))
                scores.append(statistics.fmean(unit["score"] for unit in selected) if selected else 0.0)
            result[route]["average_selected_documents"] = round(statistics.fmean(documents), 4)
            result[route]["average_selected_pages"] = round(statistics.fmean(pages), 4)
            result[route]["mean_selected_original_score"] = round(statistics.fmean(scores), 4)
            result[route]["regressions_vs_sequential"] = [r["financebench_id"] for r in items if
                r["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"] < r["sequential_coverage"]]
        result["gains_vs_packing_v1"], result["regressions_vs_packing_v1"] = [], []
        for item in items:
            before, after = [item["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"] for route in ROUTES]
            if before != after:
                result["gains_vs_packing_v1" if after > before else "regressions_vs_packing_v1"].append(item["financebench_id"])
        return result
    summary = section(records)
    summary["questions"] = len(records)
    summary["candidate_count"] = sum(r["candidate_count"] for r in records)
    summary["groups"] = {group: section([r for r in records if r["group"] == group]) for group in GROUPS}
    summary["mean_selection_ms"] = round(sum(r["selection_ms"] for r in records) / len(records), 2)
    steps = [step for record in records for step in record["routes"]["set_selector_v1"]["trace"]["steps"]]
    summary["novelty_dominance"] = {
        "selected_steps": len(steps),
        "gain_exceeds_relevance_steps": sum(step["marginal_information_gain"] > step["relevance"] for step in steps),
        "mean_gain": round(statistics.fmean(step["marginal_information_gain"] for step in steps), 4) if steps else 0.0,
        "mean_relevance": round(statistics.fmean(step["relevance"] for step in steps), 4) if steps else 0.0,
    }
    correct = summary["groups"]["correct_regression10"]
    summary["decision"] = {
        "aggregate_coverage_improved": summary["set_selector_v1"]["evidence_coverage"] > summary["packing_v1"]["evidence_coverage"],
        "correct_regression_not_worse": len(correct["set_selector_v1"]["regressions_vs_sequential"]) <= len(correct["packing_v1"]["regressions_vs_sequential"]),
        "production_integration": False,
        "causal_limit": "Dynamic recomputation AND utility change together. A negative result does not rule out set selection; a positive result alone does not prove fixed order was the only bottleneck.",
    }
    summary["invariants"] = {
        "candidate_and_ranking_unchanged": all(r["candidate_and_ranking_unchanged"] for r in records),
        "baseline_reproduced": all(r["baseline_reproduced"] for r in records),
        "max_context_chars": max(r["routes"][route]["metrics"]["context_chars"] for r in records for route in ROUTES),
        "gold_used_for_selection": False,
    }
    return summary


def render_markdown(payload: dict) -> str:
    s = payload["summary"]
    def pct(value):
        return "N/A" if value is None else f"{value:.2%}"
    lines = ["# Evidence Set Selector v1 shadow — 固定30题", "",
             "## 实验边界与固定定义", "",
             "- 9205个冻结Evidence Unit、原Ranking v1分数/原始文本/metadata不变；不接入生产，不改Retrieval、Unit生成、Prompt、Skills或现有Packing。",
             "- 每步重算所有可装入Unit的 utility，取最大值；并列使用原rank。无replacement、无Bundle、无新ranker。",
             "- utility = (原ranking score + metadata marginal gain) / (1 + same_page + text_similarity + numeric_similarity)。",
             "- marginal gain：entity/period/metric/value/unit五类中，各自新值数量÷该Unit该类值数量，然后取五类均值；缺失值为0。只读已有字段，无新metadata推断；限制各类贡献，避免数值数量本身无限加分。",
             "- redundancy：同文档/page_id已选数量n对应n/(1+n)；文本3-token shingle最大Jaccard；已有value数值集合最大Jaccard。空集合为0，不按文件名判断页面。均为软惩罚，无硬页数限制。",
             "- 数值仅作逗号/Decimal规范化，保留负号/币种/百分号，不转换口径；value中重复出现的已知period不视为新金额。",
             "- 长度只作28K准入限制，不加入utility；完整Unit不截断，最终仍按原rank及原Evidence渲染格式输出。每步增量字符与最终实际字符核验。",
             "- 原Packing v1也有marginal coverage，但遍历次序预先固定；本实验不仅改变每步选择次序，还改变utility定义，因此不是纯粹单因素排序实验。",
             "- novelty不等于问题所需信息，多一个entity/period/数值也可能是干扰项；metadata错误会直接影响增益。",
             "- Coverage沿用历史gold-line词/数值代理；numeric/period hit不验证值与实体/期间绑定关系。Gold只用于离线评测，不是答案正确率。", "",
             "## A/B结果", "",
             "| Group | Route | Evidence coverage | Gold retention | Number hit | Period hit | Selected units | Context chars |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for group in ("overall", *GROUPS):
        item = s if group == "overall" else s["groups"][group]
        for route in ROUTES:
            m = item[route]
            lines.append(f"| {group} | {route} | {pct(m['evidence_coverage'])} | {pct(m['gold_evidence_retention'])} | {pct(m['required_number_hit'])} | {pct(m['required_period_hit'])} | {m['average_selected_units']} | {m['context_characters']} |")
    lines += ["", "## 稳定性与结论边界", "",
              f"- 相对Packing v1：提升 `{len(s['gains_vs_packing_v1'])}` 题，下降 `{len(s['regressions_vs_packing_v1'])}` 题。",
              f"- vs sequential回退数量：`{len(s['packing_v1']['regressions_vs_sequential'])} → {len(s['set_selector_v1']['regressions_vs_sequential'])}`。",
              f"- 整体coverage提升：`{s['decision']['aggregate_coverage_improved']}`；correct-regression未恶化：`{s['decision']['correct_regression_not_worse']}`。",
              "- 只验证这一种贪心集合目标；失败不等于集合选择无效，成功也不能证明旧排序是唯一原因。没有独立objective/order消融，不能强行二选一归因。",
              "- 不继续调权重、不接入生产；该30题重复用于诊断，不代表未见问题泛化表现。", "",
              "## 选择行为诊断（不使用gold）", "",
              f"- 每题平均选中文档：`{s['packing_v1']['average_selected_documents']} → {s['set_selector_v1']['average_selected_documents']}`；页面：`{s['packing_v1']['average_selected_pages']} → {s['set_selector_v1']['average_selected_pages']}`。",
              f"- 选中Unit的原分数均值：`{s['packing_v1']['mean_selected_original_score']} → {s['set_selector_v1']['mean_selected_original_score']}`。",
              f"- 动态选择增益/相关性：`{s['novelty_dominance']}`。",
              "- 新增metadata与任务所需的证据不是同一个目标；覆盖更多文档/字段值不能直接视为答案证据更完整。这些统计描述行为，不能单独证明错误来自哪一项。", "",
              "## 验证", "",
              f"- 约束：`{s['invariants']}`；网络防护：`{payload['network']}`。",
              f"- 平均selector时间：`{s['mean_selection_ms']}` ms。",
              "- JSON保存每题candidate manifest、动态选择次序、每步utility/增益/冗余/字符预算及未选原因。", "",
              "## 每题（Packing v1 → Set Selector）", ""]
    for r in payload["records"]:
        a, b = [r["routes"][route] for route in ROUTES]
        lines += [f"### {r['financebench_id']} — {r['group']}", "", r["question"], "",
                  f"- Coverage：`{a['metrics']['answer_evidence_coverage']['ratio']} → {b['metrics']['answer_evidence_coverage']['ratio']}`；retention：`{a['gold_evidence_retention']['ratio']} → {b['gold_evidence_retention']['ratio']}`。",
                  f"- Number/period：`{a['metrics']['required_number_hit']}/{a['metrics']['required_period_hit']} → {b['metrics']['required_number_hit']}/{b['metrics']['required_period_hit']}`。",
                  f"- Units/chars：`{a['metrics']['selected_unit_count']}/{a['metrics']['context_chars']} → {b['metrics']['selected_unit_count']}/{b['metrics']['context_chars']}`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", type=Path, default=ROOT / "reports/evidence_metadata_counterfactual_v1.json")
    parser.add_argument("--baseline-json", type=Path, default=ROOT / "reports/evidence_packing_optimization_v1.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    records = json.loads(args.candidate_json.read_text(encoding="utf-8"))["records"]
    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))["records"]
    ids = {r["financebench_id"] for r in records}
    if len(records) != 30 or len(ids) != 30 or len(baseline) != 30 or ids != {r["financebench_id"] for r in baseline}:
        raise ValueError("Only the fixed matching diagnostic30 is allowed")
    if Counter(r["group"] for r in records) != Counter({group: 10 for group in GROUPS}):
        raise ValueError("Expected three fixed groups of ten")
    if sum(len(r["candidate_units"]) for r in records) != 9205:
        raise ValueError("Expected 9205 unchanged candidate units")
    rows = _load_dataset(args.dataset)
    by_id = {r["financebench_id"]: r for r in baseline}
    network = {"blocked_attempts": 0, "outbound_calls": 0, "socket_guard_enabled": True}
    def denied(*args, **kwargs):
        network["blocked_attempts"] += 1
        raise RuntimeError("Network calls forbidden in offline shadow evaluation")
    output = []
    with patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied), patch.object(socket, "create_connection", denied):
        for index, record in enumerate(records, 1):
            result = evaluate_record(record, by_id[record["financebench_id"]], rows[record["financebench_id"]])
            output.append(result)
            ratios = [result["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"] for route in ROUTES]
            print(f"[{index:02d}/30] {record['financebench_id']} coverage={ratios[0]}->{ratios[1]}", flush=True)
    paths = {"candidates": args.candidate_json, "baseline": args.baseline_json,
             "packing_v1": ROOT / "backend/evidence_packing_v1.py", "selector": ROOT / "backend/evidence_set_selector_v1.py"}
    payload = {"evaluation": "evidence_set_selector_v1_shadow", "network": network,
               "file_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()},
               "summary": summarize(output), "records": output}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for suffix, content in (("json", json.dumps(payload, ensure_ascii=False, indent=2)), ("md", render_markdown(payload))):
        path = args.report_dir / f"evidence_set_selector_v1.{suffix}"
        path.write_text(content + "\n", encoding="utf-8")
        print(f"Report: {path}", flush=True)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
