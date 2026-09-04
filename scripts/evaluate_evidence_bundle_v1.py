"""Frozen diagnostic30: Packing v1 units versus atomic Evidence Bundles."""

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

from backend.evidence_assembly_v5 import EvidenceUnit
from backend.evidence_bundle_v1 import build_evidence_bundles, pack_evidence_bundles, unit_id
from backend.evidence_packing_v1 import _key, _rank, _render_selection, _unit_utility
from scripts.evaluate_evidence_metadata_counterfactual_v1 import (
    _context_metrics, _gold, _gold_retention, _load_dataset, evidence_coverage,
)
from scripts.evaluate_evidence_packing_optimization_v1 import _route_summary

ROUTES = ("packing_v1", "bundle_v1")
GROUPS = ("selection_loss10", "correct_regression10", "candidate_miss10")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def packing_inputs(units: list[dict]) -> list[dict]:
    """Whitelist existing Unit fields and frozen score. No benchmark annotations."""
    fields = [*EvidenceUnit.__dataclass_fields__, "current_ranking"]
    return [{key: copy.deepcopy(unit[key]) for key in fields} for unit in units]


def replay_baseline_events(question: str, units: list[dict], prior: dict) -> list[dict]:
    """Recover accepted replacements from v1's frozen trace; verify final state.

    An evicted unit retains its incoming replaced_unit_rank. The overwritten
    reason still identifies it as a unit that was selected during the replay.
    """
    traces = {item["rank"]: item for item in prior["packing_trace"]["trace"]}
    initial = {_rank(unit): _unit_utility(question, unit, [], set())["utility"] for unit in units}
    selected, events = {}, []
    for unit in sorted(units, key=lambda item: (-initial[_rank(item)], _rank(item), _key(item))):
        rank = _rank(unit)
        entry = traces[rank]
        if entry["selection_reason"] not in {
            "selected_direct", "selected_by_replacement", "replaced_by_higher_total_utility",
        }:
            continue
        removed = entry.get("replaced_unit_rank")
        before = [unit_id(selected[key]) for key in sorted(selected)]
        if removed is not None:
            if removed not in selected:
                raise ValueError("Frozen replacement trace cannot be replayed exactly")
            del selected[removed]
        selected[rank] = unit
        if removed is not None:
            events.append({
                "removed_rank": removed, "added_rank": rank,
                "before_unit_ids": before,
                "after_unit_ids": [unit_id(selected[key]) for key in sorted(selected)],
            })
    if sorted(selected) != sorted(prior["selected_unit_ranks"]):
        raise ValueError("Frozen baseline replay differs from archived selected units")
    if len(events) != prior["packing_trace"]["replacement_count"]:
        raise ValueError("Frozen baseline replacement count differs")
    return events


def audit_events(gold: list[dict], units: list[dict], events: list[dict]) -> dict:
    """Post-selection annotation only; no gold is passed to the builder/packer."""
    lookup = {unit_id(unit): unit for unit in units}
    annotated = []
    for event in events:
        counts = []
        for name in ("before_unit_ids", "after_unit_ids"):
            # No rendered unit ordinals: evaluate the actual evidence text.
            text = "\n\n".join(lookup[uid]["source_text"] for uid in event[name])
            counts.append(evidence_coverage(gold, text)["matched_lines"])
        annotated.append(dict(event, before_matched_lines=counts[0], after_matched_lines=counts[1],
                              source_coverage_decreased=counts[1] < counts[0]))
    return {
        "count": len(events),
        "coverage_decreasing_count": sum(event["source_coverage_decreased"] for event in annotated),
        "events": annotated,
        "definition": "Offline proxy: fewer matched gold lines in source_text after an accepted replacement; not answer correctness.",
    }


def evaluate_record(source: dict, archived: dict, row: dict) -> dict:
    units = packing_inputs(source["candidate_units"])
    fingerprint = digest(units)
    prior = archived["routes"]["packing_optimization_v1"]
    ranks = prior["selected_unit_ranks"]
    if len({_rank(unit) for unit in units}) != len(units):
        raise ValueError("Frozen ranks must be unique")
    selected_before = [unit for unit in units if _rank(unit) in ranks]
    baseline_context = _render_selection(selected_before)[0]
    baseline_metrics = _context_metrics(row, baseline_context, selected_before)
    if baseline_metrics != prior["metrics"]:
        raise ValueError("Archived baseline cannot be reproduced from frozen units")
    candidate_context = "\n\n".join(unit["source_text"] for unit in units)
    baseline_events = replay_baseline_events(source["question"], units, prior)
    start = time.perf_counter()
    bundles = build_evidence_bundles(source["question"], units)
    build_ms = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    context, selected, trace = pack_evidence_bundles(units, bundles)
    pack_ms = (time.perf_counter() - start) * 1000
    if digest(units) != fingerprint:
        raise AssertionError("Frozen candidate unit content/ranking changed")
    gold = _gold(row)
    return {
        "financebench_id": source["financebench_id"], "group": source["group"], "question": source["question"],
        "candidate_count": len(units), "candidate_sha256": fingerprint,
        "baseline_reproduced": True, "candidate_unchanged": True,
        "build_ms": round(build_ms, 2), "packing_ms": round(pack_ms, 2),
        "sequential_coverage": archived["routes"]["sequential_first_fit"]["metrics"]["answer_evidence_coverage"]["ratio"],
        "bundle_counts": dict(Counter(bundle["bundle_type"] for bundle in bundles)),
        "oversize_bundle_count": sum(bundle["length"] > 28000 for bundle in bundles),
        "bundles": bundles,
        "routes": {
            "packing_v1": {
                "metrics": baseline_metrics, "gold_evidence_retention": prior["gold_evidence_retention"],
                "selected_unit_ids": [unit_id(unit) for unit in selected_before],
                "replacement_audit": audit_events(gold, units, baseline_events),
            },
            "bundle_v1": {
                "metrics": _context_metrics(row, context, selected),
                "gold_evidence_retention": _gold_retention(gold, candidate_context, context),
                "selected_unit_ids": [unit_id(unit) for unit in selected],
                "packing_trace": {key: value for key, value in trace.items() if key != "replacement_events"},
                "replacement_audit": audit_events(gold, units, trace["replacement_events"]),
            },
        },
    }


def summarize(records: list[dict]) -> dict:
    def section(items):
        summary = {route: _route_summary(items, route) for route in ROUTES}
        for route in ROUTES:
            counts = [item["routes"][route]["replacement_audit"]["count"] for item in items]
            harmful = sum(item["routes"][route]["replacement_audit"]["coverage_decreasing_count"] for item in items)
            summary[route].update({
                "replacement_count": sum(counts), "coverage_decreasing_replacements": harmful,
                "coverage_decreasing_replacement_rate": round(harmful / sum(counts), 4) if sum(counts) else None,
                "replacement_distribution": dict(sorted(Counter(counts).items())),
                "regressions_vs_sequential": [item["financebench_id"] for item in items if (
                    item["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"] < item["sequential_coverage"]
                )],
            })
        summary["gains_vs_packing_v1"], summary["regressions_vs_packing_v1"] = [], []
        for item in items:
            before, after = [item["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"] for route in ROUTES]
            if before != after:
                summary["gains_vs_packing_v1" if after > before else "regressions_vs_packing_v1"].append(item["financebench_id"])
        return summary

    overall = section(records)
    groups = {group: section([r for r in records if r["group"] == group]) for group in GROUPS}
    overall["groups"] = groups
    overall["questions"] = len(records)
    overall["candidate_count"] = sum(r["candidate_count"] for r in records)
    overall["bundle_counts"] = dict(sum((Counter(r["bundle_counts"]) for r in records), Counter()))
    overall["oversize_bundle_count"] = sum(r["oversize_bundle_count"] for r in records)
    old, new = overall["packing_v1"], overall["bundle_v1"]
    old_regressions, new_regressions = set(old["regressions_vs_sequential"]), set(new["regressions_vs_sequential"])
    overall["recovered_old_regressions"] = sorted(old_regressions - new_regressions)
    overall["introduced_regressions"] = sorted(new_regressions - old_regressions)
    lost_count, lost_oversize_count = 0, 0
    for record in records:
        by_unit = {uid: bundle for bundle in record["bundles"] for uid in bundle["unit_ids"]}
        lost = set(record["routes"]["packing_v1"]["selected_unit_ids"]) - set(record["routes"]["bundle_v1"]["selected_unit_ids"])
        lost_count += len(lost)
        lost_oversize_count += sum(by_unit[uid]["length"] > 28000 for uid in lost)
    overall["packing_v1_units_lost"] = lost_count
    overall["packing_v1_units_lost_in_oversize_bundles"] = lost_oversize_count
    correct = groups["correct_regression10"]
    overall["decision"] = {
        "fewer_regressions_vs_sequential": len(new["regressions_vs_sequential"]) < len(old["regressions_vs_sequential"]),
        "correct_regression_not_worse": len(correct["bundle_v1"]["regressions_vs_sequential"]) <= len(correct["packing_v1"]["regressions_vs_sequential"]),
        "aggregate_coverage_not_lower": new["evidence_coverage"] >= old["evidence_coverage"],
        "production_integration": False,
        "scope": "One frozen diagnostic30 run. Evidence proxies only, no answer-generation or Judge score.",
    }
    overall["invariants"] = {
        "candidate_and_ranking_unchanged": all(r["candidate_unchanged"] for r in records),
        "archived_baseline_reproduced": all(r["baseline_reproduced"] for r in records),
        "max_context_chars": max(r["routes"][route]["metrics"]["context_chars"] for r in records for route in ROUTES),
        "gold_used_for_selection": False,
    }
    return overall


def render_markdown(payload: dict) -> str:
    s = payload["summary"]
    lines = [
        "# Evidence Bundle Shadow v1 — 离线30题", "",
        "## 实验边界与固定构造规则", "",
        "- 无生产改动、无重新检索、无 LLM/Jina/Judge/LangSmith。9205 Units 按题分别分组，不跨问题混合候选。",
        "- same-page：同 document_id/page_id；已知 entity 必须一致，已知 periods 必须有共同交集；unknown 只允许无冲突加入，不能桥接冲突。",
        "- adjacent-page：同文档、页距1–2、所有成员 period 非空且共同交集非空、词集合 Jaccard ≥0.20；按相似度配对，禁止传递连锁合并。",
        "- 每个 Unit 恰好归属一个 Bundle；单例保留。Bundle score 是成员冻结 Ranking v1 score 的字符长度加权平均，不重新计算 Unit rank/score。",
        "- utility、重复惩罚、单次替换收益条件沿用 Packing v1；替换单位改为整个 Bundle，无 guard/anchor/替换次数限制。",
        "- 最终原始 Unit 渲染、原 rank 展示，28K硬预算；无新增 bundle 标题挤占回答预算。超大 Bundle 拒绝装入，不拆分、不扩预算。",
        "- 本实验同时改变组合粒度与派生 bundle score/coverage，因此是组合策略验证，不是仅替换动作的单因素因果实验。",
        "- Coverage 沿用历史 gold-line 词/数值覆盖代理，可跨片段匹配；number/period hit 也不验证 operand 关系。不能解释为真实答案正确率。",
        "- 错误替换仅用离线代理：替换后 source_text 的 matched gold lines 减少；不把减少替换次数本身视为成功。", "",
        "## A/B 汇总", "",
        "| Group | Route | Evidence coverage | Gold retention | Number hit | Period hit | Context chars | 替换数 | 覆盖下降替换数/比例 | vs sequential回退题数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def pct(value):
        return "N/A" if value is None else f"{value:.2%}"
    for group in ("overall", *GROUPS):
        summary = s if group == "overall" else s["groups"][group]
        for route in ROUTES:
            m = summary[route]
            lines.append(f"| {group} | {route} | {pct(m['evidence_coverage'])} | {pct(m['gold_evidence_retention'])} | {pct(m['required_number_hit'])} | {pct(m['required_period_hit'])} | {m['context_characters']} | {m['replacement_count']} | {m['coverage_decreasing_replacements']} / {pct(m['coverage_decreasing_replacement_rate'])} | {len(m['regressions_vs_sequential'])} |")
    lines += ["", "## 构造、稳定性与约束", "",
              f"- Bundle类型统计：`{s['bundle_counts']}`；超28K Bundle：`{s['oversize_bundle_count']}`。",
              f"- 相对Packing v1：提升 `{len(s['gains_vs_packing_v1'])}` 题；下降 `{len(s['regressions_vs_packing_v1'])}` 题。",
              f"- 原回退修复 `{len(s['recovered_old_regressions'])}` 题：`{s['recovered_old_regressions']}`；新增回退 `{len(s['introduced_regressions'])}` 题：`{s['introduced_regressions']}`。",
              f"- Packing v1选中、Bundle未保留的Unit共 `{s['packing_v1_units_lost']}` 个，其中 `{s['packing_v1_units_lost_in_oversize_bundles']}` 个属于超28K的原子Bundle。此统计不是gold判定。",
              f"- 降低原始回退：`{s['decision']['fewer_regressions_vs_sequential']}`；正确回归组未恶化：`{s['decision']['correct_regression_not_worse']}`；平均coverage未下降：`{s['decision']['aggregate_coverage_not_lower']}`。",
              f"- Invariants：`{s['invariants']}`。",
              f"- 网络调用防护：`{payload['network']}`。",
              "- 不接入生产；只依据本次一次性结果判断，不继续调Bundle参数。", "",
              "## 结果解读与局限", "",
              "- 应联合看回退数量、回退幅度与替换损伤率；原子Bundle的替换总数较低，不能单独证明替换更安全。",
              "- 相同entity/period和邻页词汇相似只说明位置/元数据兼容，不证明这些Unit是问题所需的互补事实。",
              "- 同页全量聚合与跨页配对会绑定有用/无用Unit；拒绝或替换Bundle会同时丢失所有成员。超预算Bundle没有被拆分，这也是本次设计的已知限制。",
              "- Bundle score是派生加权平均，特征仍使用Packing v1的词/数字代理；本次结果不能独立归因于单一因素，也不能否定所有答案级证据组合方法。",
              "- 若本次三项稳定性判断未通过，应保留Packing v1 shadow作为比较基准，不将本原型投入生产，不继续调整本轮分组/替换参数。", "",
              "## 每题结果（baseline → bundle）", ""]
    for record in payload["records"]:
        a, b = [record["routes"][route] for route in ROUTES]
        lines += [f"### {record['financebench_id']} — {record['group']}", "",
                  record["question"], "",
                  f"- Coverage：`{a['metrics']['answer_evidence_coverage']['ratio']} → {b['metrics']['answer_evidence_coverage']['ratio']}`；retention：`{a['gold_evidence_retention']['ratio']} → {b['gold_evidence_retention']['ratio']}`。",
                  f"- Number/period：`{a['metrics']['required_number_hit']}/{a['metrics']['required_period_hit']} → {b['metrics']['required_number_hit']}/{b['metrics']['required_period_hit']}`。",
                  f"- 替换数/覆盖下降替换：`{a['replacement_audit']['count']}/{a['replacement_audit']['coverage_decreasing_count']} → {b['replacement_audit']['count']}/{b['replacement_audit']['coverage_decreasing_count']}`。",
                  f"- 构造：`{record['bundle_counts']}`；build/pack ms：`{record['build_ms']}/{record['packing_ms']}`。", ""]
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
    if len(source) != 30 or Counter(r["group"] for r in source) != Counter({g: 10 for g in GROUPS}):
        raise ValueError("Only the fixed 30-question diagnostic set is permitted")
    if {r["financebench_id"] for r in source} != {r["financebench_id"] for r in prior} or len(prior) != 30:
        raise ValueError("Candidate and baseline question sets differ")
    if sum(len(r["candidate_units"]) for r in source) != 9205:
        raise ValueError("Expected exactly 9205 frozen candidate units")
    by_id = {r["financebench_id"]: r for r in prior}
    rows = _load_dataset(args.dataset)
    network = {"blocked_attempts": 0, "outbound_calls": 0, "socket_guard_enabled": True}
    def denied(*args, **kwargs):
        network["blocked_attempts"] += 1
        raise RuntimeError("Offline evidence bundle audit prohibits network access")
    records = []
    with patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied), patch.object(socket, "create_connection", denied):
        for index, record in enumerate(source, 1):
            result = evaluate_record(record, by_id[record["financebench_id"]], rows[record["financebench_id"]])
            records.append(result)
            values = [result["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"] for route in ROUTES]
            print(f"[{index:02d}/30] {record['financebench_id']} coverage={values[0]}->{values[1]}", flush=True)
    payload = {"evaluation": "evidence_bundle_shadow_v1", "network": network,
               "source_sha256": hashlib.sha256(args.candidate_json.read_bytes()).hexdigest(),
               "baseline_sha256": hashlib.sha256(args.baseline_json.read_bytes()).hexdigest(),
               "summary": summarize(records), "records": records}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for suffix, content in (("json", json.dumps(payload, ensure_ascii=False, indent=2)), ("md", render_markdown(payload))):
        path = args.report_dir / f"evidence_bundle_shadow_v1.{suffix}"
        path.write_text(content + "\n", encoding="utf-8")
        print(f"Report: {path}", flush=True)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
