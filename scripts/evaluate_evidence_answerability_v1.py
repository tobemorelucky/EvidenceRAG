"""Offline diagnostic30: frozen Ranking v1 vs Answerability Ranking, same packer."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.evidence_answerability_ranker import WEIGHTS, rank_answerability
from backend.evidence_assembly_v5 import EvidenceUnit
from backend.evidence_packing_v1 import _rank, _render_selection, select_evidence_packing_v1
from scripts.evaluate_evidence_metadata_counterfactual_v1 import _context_metrics, _gold, _gold_retention, _load_dataset
from scripts.evaluate_evidence_packing_optimization_v1 import _route_summary

GROUPS = ("selection_loss10", "correct_regression10", "candidate_miss10")
ROUTES = ("packing_v1", "answerability_v1")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def shadow_inputs(units: list[dict], sidecar: list[dict]) -> list[dict]:
    by_rank = {item["original_rank"]: item for item in sidecar}
    if set(by_rank) != {_rank(unit) for unit in units} or len(sidecar) != len(units):
        raise ValueError("Ranking sidecar must cover the same units exactly once")
    result = []
    for unit in units:
        value = copy.deepcopy(unit)
        rank = by_rank[_rank(unit)]
        # Adapt a copy to the existing packer interface; original units stay frozen.
        value["current_ranking"] = {"rank": rank["rank"], "score": rank["score"]}
        result.append(value)
    return result


def evaluate_record(source: dict, prior: dict, row: dict) -> dict:
    units = [{name: copy.deepcopy(unit[name]) for name in (*EvidenceUnit.__dataclass_fields__, "current_ranking")}
             for unit in source["candidate_units"]]
    before_hash = digest(units)
    baseline = prior["routes"]["packing_optimization_v1"]
    old_selected = [unit for unit in units if _rank(unit) in baseline["selected_unit_ranks"]]
    old_context = _render_selection(old_selected)[0]
    if _context_metrics(row, old_context, old_selected) != baseline["metrics"]:
        raise ValueError("Frozen baseline context does not reproduce archived metrics")
    start = time.perf_counter()
    sidecar = rank_answerability(source["question"], units)
    ranking_ms = (time.perf_counter() - start) * 1000
    experimental = shadow_inputs(units, sidecar)
    for old, new in zip(units, experimental):
        if any(old[field] != new[field] for field in EvidenceUnit.__dataclass_fields__):
            raise AssertionError("Ranking changed Evidence Unit fields")
    start = time.perf_counter()
    context, selected, trace = select_evidence_packing_v1(source["question"], experimental, max_context_chars=28000)
    packing_ms = (time.perf_counter() - start) * 1000
    if digest(units) != before_hash:
        raise AssertionError("Input candidates or frozen Ranking v1 mutated")
    selected_ranks = {_rank(unit) for unit in selected}
    original_lookup = {_rank(unit): unit for unit in units}
    for item in sidecar:
        original = original_lookup[item["original_rank"]]
        metadata = original.get("metadata") or {}
        item.update({
            "document_id": original["document_id"], "page_id": original["page_id"],
            "source_type": original["source_type"], "chunk_id": metadata.get("chunk_id"),
            "table_id": metadata.get("table_id"), "row_index": metadata.get("row_index"),
            "original_score": original["current_ranking"]["score"],
            "selected": item["rank"] in selected_ranks,
        })
    return {
        "financebench_id": source["financebench_id"], "question": source["question"], "group": source["group"],
        "candidate_count": len(units), "frozen_candidate_sha256": before_hash,
        "candidate_and_unit_fields_unchanged": True, "baseline_reproduced": True,
        "ranking_ms": round(ranking_ms, 2), "packing_ms": round(packing_ms, 2),
        "sequential_coverage": prior["routes"]["sequential_first_fit"]["metrics"]["answer_evidence_coverage"]["ratio"],
        "routes": {
            "packing_v1": {key: baseline[key] for key in ("metrics", "gold_evidence_retention", "selected_unit_ranks", "packing_trace")},
            "answerability_v1": {
                "metrics": _context_metrics(row, context, selected),
                "gold_evidence_retention": _gold_retention(_gold(row), "\n\n".join(u["source_text"] for u in units), context),
                "selected_original_ranks": [item["original_rank"] for item in sidecar if item["selected"]],
                "packing_trace": trace,
            },
        },
        "answerability_ranking": sidecar,
    }


def summarize(records: list[dict]) -> dict:
    def group_summary(items):
        result = {route: _route_summary(items, route) for route in ROUTES}
        for route in ROUTES:
            counts = [r["routes"][route]["packing_trace"]["replacement_count"] for r in items]
            result[route].update({
                "replacement_count": sum(counts), "max_replacements": max(counts, default=0),
                "replacement_distribution": dict(sorted(Counter(counts).items())),
                "regressions_vs_sequential": [r["financebench_id"] for r in items if
                    r["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"] < r["sequential_coverage"]],
            })
        result["gains_vs_packing_v1"], result["regressions_vs_packing_v1"] = [], []
        for r in items:
            a, b = [r["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"] for route in ROUTES]
            if a != b:
                result["gains_vs_packing_v1" if b > a else "regressions_vs_packing_v1"].append(r["financebench_id"])
        return result
    result = group_summary(records)
    result["groups"] = {group: group_summary([r for r in records if r["group"] == group]) for group in GROUPS}
    result["questions"] = len(records)
    result["candidate_count"] = sum(r["candidate_count"] for r in records)
    result["period_status_counts"] = dict(Counter(item["period_status"] for r in records for item in r["answerability_ranking"]))
    result["entity_status_counts"] = dict(Counter(item["entity_status"] for r in records for item in r["answerability_ranking"]))
    result["ranking_ms_average"] = round(sum(r["ranking_ms"] for r in records) / len(records), 2)
    correct = result["groups"]["correct_regression10"]
    result["decision"] = {
        "coverage_improved_with_same_packing": result["answerability_v1"]["evidence_coverage"] > result["packing_v1"]["evidence_coverage"],
        "correct_regression_count_not_worse": len(correct["answerability_v1"]["regressions_vs_sequential"]) <= len(correct["packing_v1"]["regressions_vs_sequential"]),
        "production_integration": False,
        "causal_limit": "Only ranking inputs change. Improvement supports sensitivity to valuation; failure of this heuristic cannot prove packing is the primary cause. No validated oracle valuation or independent packing factorial experiment is included.",
    }
    result["invariants"] = {
        "candidate_fields_unchanged": all(r["candidate_and_unit_fields_unchanged"] for r in records),
        "baseline_reproduced": all(r["baseline_reproduced"] for r in records),
        "max_context_chars": max(r["routes"][route]["metrics"]["context_chars"] for r in records for route in ROUTES),
        "gold_used_in_ranking": False,
    }
    return result


def render_markdown(payload: dict) -> str:
    s = payload["summary"]
    lines = [
        "# Answerability Ranking v1 shadow — 固定30题", "",
        "## 实验约束", "",
        "- 只新增独立ranker与离线脚本，不接入生产、不改候选/Unit/Packing v1/28K预算，不继续开发Bundle或检索。",
        "- 9205个冻结Unit；原始Ranking v1不变。新分数及rank仅写入传给原packer的副本，原rank只作新分数并列时的排序依据。",
        "- answerability是通用可观察特征代理，不代表操作数唯一、事实关系正确或模型必定能回答。",
        f"- 固定权重（实验前确定，无网格调参）：`{payload['weights']}`。",
        "- 词汇相关性兼顾整段及单句/表格行；numeric有无采用饱和值，不按数字数量累加。",
        "- period优先原文年份，metadata-only/unknown不给确定性匹配分；entity不能确定时为中性，不把未识别实体误标成冲突。",
        "- 实现核验修复了FY2024/FY 2024连写年份解析；未改权重或packing。修复前报告另存为同名前缀的pre_period_fix文件，不用作正式结论。",
        "- answer type只用通用任务用词；没有公司名单、金融指标映射、参考答案或FinanceBench ID规则。",
        "- completeness结合source文档页标识、句段/表格结构和局部词/数字共现；不更改任何原始证据文本。",
        "- gold仅用于离线评测。Coverage沿用历史词/数值gold-line代理，number/period hit不等于同一事实关系成立；本轮不报告LLM正确率。", "",
        "## A/B结果", "",
        "| Group | Route | Evidence coverage | Gold retention | Number hit | Period hit | Context chars | 替换数/最大每题 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    def pct(value):
        return "N/A" if value is None else f"{value:.2%}"
    for group in ("overall", *GROUPS):
        section = s if group == "overall" else s["groups"][group]
        for route in ROUTES:
            m = section[route]
            lines.append(f"| {group} | {route} | {pct(m['evidence_coverage'])} | {pct(m['gold_evidence_retention'])} | {pct(m['required_number_hit'])} | {pct(m['required_period_hit'])} | {m['context_characters']} | {m['replacement_count']}/{m['max_replacements']} |")
    lines += ["", "## 稳定性与归因边界", "",
              f"- 本轮同packer整体coverage：`{pct(s['packing_v1']['evidence_coverage'])} → {pct(s['answerability_v1']['evidence_coverage'])}`；correct-regression coverage：`{pct(s['groups']['correct_regression10']['packing_v1']['evidence_coverage'])} → {pct(s['groups']['correct_regression10']['answerability_v1']['evidence_coverage'])}`。",
              f"- 相对Packing v1：提升 `{len(s['gains_vs_packing_v1'])}` 题、下降 `{len(s['regressions_vs_packing_v1'])}` 题。",
              f"- vs sequential回退题：`{len(s['packing_v1']['regressions_vs_sequential'])} → {len(s['answerability_v1']['regressions_vs_sequential'])}`。",
              f"- 同packer下coverage提升：`{s['decision']['coverage_improved_with_same_packing']}`。",
              f"- correct-regression回退数量未恶化：`{s['decision']['correct_regression_count_not_worse']}`。",
              "- 本轮只控制packing不变，比较一种新的估值代理；即使下降，也不能推出packing必然是主因。两种估值都可能存在代理误差，且packing会对score/rank变化作非线性响应。",
              "- 若提升且回归组稳定，说明估值值得进一步验证；若未通过，停止本轮，不以单题调权重、不改packing补救。",
              "- 此30题已被反复诊断，结果不能视为未见样本泛化验证；不接入生产。", "",
              "## 可复现性", "",
              f"- 约束：`{s['invariants']}`。",
              f"- 网络防护：`{payload['network']}`。",
              f"- 平均纯ranker耗时：`{s['ranking_ms_average']}` ms。",
              f"- period状态：`{s['period_status_counts']}`。",
              f"- entity状态：`{s['entity_status_counts']}`。", "",
              "## 每题结果（Packing v1 → answerability）", ""]
    for r in payload["records"]:
        a, b = [r["routes"][route] for route in ROUTES]
        lines += [f"### {r['financebench_id']} — {r['group']}", "", r["question"], "",
                  f"- Coverage：`{a['metrics']['answer_evidence_coverage']['ratio']} → {b['metrics']['answer_evidence_coverage']['ratio']}`；retention：`{a['gold_evidence_retention']['ratio']} → {b['gold_evidence_retention']['ratio']}`。",
                  f"- Number/period：`{a['metrics']['required_number_hit']}/{a['metrics']['required_period_hit']} → {b['metrics']['required_number_hit']}/{b['metrics']['required_period_hit']}`。",
                  f"- 替换数：`{a['packing_trace']['replacement_count']} → {b['packing_trace']['replacement_count']}`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-json", type=Path, default=ROOT / "reports/evidence_metadata_counterfactual_v1.json")
    parser.add_argument("--baseline-json", type=Path, default=ROOT / "reports/evidence_packing_optimization_v1.json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    source = json.loads(args.candidate_json.read_text(encoding="utf-8"))["records"]
    prior = json.loads(args.baseline_json.read_text(encoding="utf-8"))["records"]
    ids = {r["financebench_id"] for r in source}
    if len(source) != 30 or len(ids) != 30 or len(prior) != 30 or ids != {r["financebench_id"] for r in prior}:
        raise ValueError("Only the same fixed diagnostic30 is allowed")
    if Counter(r["group"] for r in source) != Counter({group: 10 for group in GROUPS}):
        raise ValueError("Expected three frozen groups of 10")
    if sum(len(r["candidate_units"]) for r in source) != 9205:
        raise ValueError("Expected 9205 frozen candidates")
    rows = _load_dataset(args.dataset)
    by_id = {r["financebench_id"]: r for r in prior}
    network = {"blocked_attempts": 0, "outbound_calls": 0, "socket_guard_enabled": True}
    def denied(*args, **kwargs):
        network["blocked_attempts"] += 1
        raise RuntimeError("Network calls prohibited in shadow ranking evaluation")
    records = []
    with patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied), patch.object(socket, "create_connection", denied):
        for index, record in enumerate(source, 1):
            result = evaluate_record(record, by_id[record["financebench_id"]], rows[record["financebench_id"]])
            records.append(result)
            scores = [result["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"] for route in ROUTES]
            print(f"[{index:02d}/30] {record['financebench_id']} coverage={scores[0]}->{scores[1]}", flush=True)
    payload = {"evaluation": "evidence_answerability_ranking_v1_shadow", "weights": WEIGHTS, "network": network,
               "candidate_file_sha256": hashlib.sha256(args.candidate_json.read_bytes()).hexdigest(),
               "baseline_file_sha256": hashlib.sha256(args.baseline_json.read_bytes()).hexdigest(),
               "packer_file_sha256": hashlib.sha256((ROOT / 'backend/evidence_packing_v1.py').read_bytes()).hexdigest(),
               "ranker_file_sha256": hashlib.sha256((ROOT / 'backend/evidence_answerability_ranker.py').read_bytes()).hexdigest(),
               "summary": summarize(records), "records": records}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for suffix, content in (("json", json.dumps(payload, ensure_ascii=False, indent=2)), ("md", render_markdown(payload))):
        path = args.report_dir / f"evidence_answerability_ranking_v1.{suffix}"
        path.write_text(content + "\n", encoding="utf-8")
        print(f"Report: {path}", flush=True)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
