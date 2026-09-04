"""Fixed30 shadow: A Packing v1, B local requirement matching, C frozen Query Requirement v1."""

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
from backend.evidence_packing_v1 import _rank, _render_selection, select_evidence_packing_v1
from backend.query_requirement_v1 import parse_query_requirement
from backend.requirement_evidence_match_v1 import matching_inputs
from scripts.audit_evidence_selection_failure_v1 import evidence_coverage, _normalized
from scripts.evaluate_evidence_metadata_counterfactual_v1 import _context_metrics, _gold, _gold_retention, _load_dataset
from scripts.evaluate_evidence_packing_optimization_v1 import _route_summary

GROUPS = ("selection_loss10", "correct_regression10", "candidate_miss10")
ROUTES = ("A_packing_v1", "B_requirement_match", "C_requirement_aware_v1")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def frozen_route(row, units, route):
    ranks = set(route["selected_ranks"])
    selected = [u for u in units if _rank(u) in ranks]
    context = _render_selection(selected)[0]
    if len(selected) != len(route["selected_ranks"]) or _context_metrics(row, context, selected) != route["metrics"]:
        raise ValueError("Frozen selected Units/metrics do not reproduce")
    retention = _gold_retention(_gold(row), "\n\n".join(u["source_text"] for u in units), context)
    if retention != route["gold_evidence_retention"] or len(context) > 28000:
        raise ValueError("Frozen retention or context budget drift")
    return context, selected


def covered_lines(gold, context):
    lines = set()
    for item in gold:
        for line in str(item.get("evidence_text") or "").splitlines():
            line = _normalized(line)
            if len(line) >= 3 and evidence_coverage([{"evidence_text": line}], context)["matched_lines"]:
                lines.add(line)
    return lines


def regression_audit(row, units, contexts, prior, result, matches):
    """Gold enters only this post-selection audit, never the matcher or packer."""
    a, b, c = [result["routes"][route] for route in ROUTES]
    lost = sorted(covered_lines(_gold(row), contexts[0]) - covered_lines(_gold(row), contexts[2]))
    recovered = covered_lines(_gold(row), contexts[1]) & set(lost)
    old_support = {v["rank"]: v for v in prior["requirement_trace"]["unit_support"]}
    new_matches = {v["rank"]: v for v in matches["unit_matches"]}
    old_trace = {v["rank"]: v for v in prior["routes"]["requirement_guided"]["packing_trace"]["trace"]}
    new_trace = {v["rank"]: v for v in b["packing_trace"]["trace"]}
    old_ranks, current_ranks = set(a["selected_ranks"]), set(c["selected_ranks"])
    def unit_detail(u):
        rank = _rank(u)
        supported = [i for i, line in enumerate(lost) if evidence_coverage([{"evidence_text": line}], u["source_text"])["matched_lines"]]
        match = new_matches[rank]
        return {
            "rank": rank, "document_id": u["document_id"], "page_id": u["page_id"], "source_type": u["source_type"],
            "entity": u["entity"], "period": u["period"], "metric": u["metric"], "source_chars": len(u["source_text"]),
            "source_text": u["source_text"], "original_score": u["current_ranking"]["score"],
            "C_support": old_support[rank], "C_packing_decision": old_trace[rank],
            "B_match": match, "B_packing_decision": new_trace[rank],
            "lost_line_indices_supported_by_unit": supported, "selected_in_B": rank in b["selected_ranks"],
        }
    removed = [unit_detail(u) for u in units if _rank(u) in old_ranks - current_ranks]
    added = [unit_detail(u) for u in units if _rank(u) in current_ranks - old_ranks]
    supported_indices = {i for u in removed for i in u["lost_line_indices_supported_by_unit"]}
    return {
        "requirement_annotation_mismatches": prior.get("requirement_annotation", {}).get("mismatches", {}),
        "lost_gold_lines_A_to_C": lost, "lost_line_count": len(lost),
        "lost_lines_recovered_by_B": sorted(recovered), "recovered_count": len(recovered),
        "removed_unit_count": len(removed), "added_unit_count": len(added),
        "removed_units": removed, "added_units": added,
        "removed_rejection_reasons": dict(Counter(u["C_packing_decision"]["selection_reason"] for u in removed)),
        "lost_lines_with_single_removed_unit_support": len(supported_indices),
        "distributed_or_rendered_proxy_loss_count": len(lost) - len(supported_indices),
        "interpretation": "Recorded packing decisions are mechanics, not independently proven semantic causes. Missing single-unit support may be distributed token overlap or rendered metadata, not a missing complete fact.",
    }


def evaluate_record(source, prior, row, baseline):
    units = [{key: copy.deepcopy(u[key]) for key in (*EvidenceUnit.__dataclass_fields__, "current_ranking")}
             for u in source["candidate_units"]]
    fingerprint = digest(units)
    if fingerprint != prior["candidate_sha256"] or source["question"] != prior["question"] or source["question"] != row["question"]:
        raise ValueError("Frozen question/candidate mismatch")
    req = parse_query_requirement(source["question"])
    if digest(req.to_dict()) != digest(prior["requirement_trace"]["requirement"]):
        raise ValueError("Frozen Query Requirement v1 output changed")
    old_a = prior["routes"]["packing_v1"]
    archived_a = baseline["routes"]["packing_optimization_v1"]
    if old_a["metrics"] != archived_a["metrics"] or old_a["selected_ranks"] != archived_a["selected_unit_ranks"]:
        raise ValueError("Packing v1 archives disagree")
    context_a, _ = frozen_route(row, units, old_a)
    old_c = prior["routes"]["requirement_guided"]
    context_c, _ = frozen_route(row, units, old_c)
    start = time.perf_counter()
    adapted, matches = matching_inputs(source["question"], req, units)
    match_ms = (time.perf_counter() - start) * 1000
    context_b, selected_shadow, trace = select_evidence_packing_v1(source["question"], adapted, max_context_chars=28000)
    ranks = {_rank(u) for u in selected_shadow}
    selected = [u for u in units if _rank(u) in ranks]
    if context_b != _render_selection(selected)[0] or len(context_b) > 28000 or digest(units) != fingerprint:
        raise AssertionError("Candidate/text/ranking/budget drift")
    result = {
        "financebench_id": source["financebench_id"], "group": source["group"], "question": source["question"],
        "candidate_count": len(units), "candidate_sha256": fingerprint, "frozen_invariants_verified": True,
        "requirement": req.to_dict(), "matching_trace": matches, "match_ms": match_ms,
        "match_and_packing_ms": (time.perf_counter() - start) * 1000,
        "routes": {
            ROUTES[0]: {k: v for k, v in old_a.items() if k != "packing_trace"},
            ROUTES[1]: {"metrics": _context_metrics(row, context_b, selected),
                        "gold_evidence_retention": _gold_retention(_gold(row), "\n\n".join(u["source_text"] for u in units), context_b),
                        "selected_ranks": sorted(ranks), "replacement_count": trace["replacement_count"], "packing_trace": trace},
            ROUTES[2]: {k: v for k, v in old_c.items() if k != "packing_trace"},
        },
    }
    if old_c["metrics"]["answer_evidence_coverage"]["ratio"] < old_a["metrics"]["answer_evidence_coverage"]["ratio"]:
        result["prior_regression_audit"] = regression_audit(row, units, (context_a, context_b, context_c), prior, result, matches)
    return result


def summarize(records):
    def section(items):
        value = {route: _route_summary(items, route) for route in ROUTES}
        for route in ROUTES:
            value[route]["replacement_count"] = sum(r["routes"][route]["replacement_count"] for r in items)
        value["B_vs_A"], value["B_vs_C"] = {}, {}
        for comparison, baseline in (("B_vs_A", ROUTES[0]), ("B_vs_C", ROUTES[2])):
            for sign, label in ((1, "gains"), (-1, "regressions")):
                value[comparison][label] = [r["financebench_id"] for r in items if sign * (
                    r["routes"][ROUTES[1]]["metrics"]["answer_evidence_coverage"]["ratio"] -
                    r["routes"][baseline]["metrics"]["answer_evidence_coverage"]["ratio"]) > 0]
        return value
    result = section(records)
    result["groups"] = {group: section([r for r in records if r["group"] == group]) for group in GROUPS}
    audits = [r for r in records if "prior_regression_audit" in r]
    result["prior_regression_cases"] = [{"id": r["financebench_id"],
        "coverage": {route: r["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"] for route in ROUTES},
        "lost_lines_A_to_C": r["prior_regression_audit"]["lost_line_count"],
        "recovered_by_B": r["prior_regression_audit"]["recovered_count"],
        "C_removed_reasons": r["prior_regression_audit"]["removed_rejection_reasons"],
        "annotation_mismatches": r["prior_regression_audit"]["requirement_annotation_mismatches"]} for r in audits]
    result["invariants"] = {"questions": len(records), "candidate_units": sum(r["candidate_count"] for r in records),
        "A_C_reproduced_candidates_requirement_unchanged": all(r["frozen_invariants_verified"] for r in records),
        "max_chars": max(r["routes"][route]["metrics"]["context_chars"] for r in records for route in ROUTES),
        "prior_regression_cases": len(audits), "gold_used_for_matching": False, "production_integration": False}
    result["mean_match_ms"] = sum(r["match_ms"] for r in records) / len(records)
    result["mean_match_and_packing_ms"] = sum(r["match_and_packing_ms"] for r in records) / len(records)
    result["decision"] = {"selection_loss_improved_vs_C": result["groups"]["selection_loss10"][ROUTES[1]]["evidence_coverage"] > result["groups"]["selection_loss10"][ROUTES[2]]["evidence_coverage"],
        "correct_regression_no_loss_vs_C": not result["groups"]["correct_regression10"]["B_vs_C"]["regressions"],
        "production_integration": False, "no_parameter_search": True}
    return result


def render_markdown(payload):
    summary = payload["summary"]
    def pct(value):
        return "N/A" if value is None else f"{value:.2%}"
    lines = ["# Requirement–Evidence Matching v1 shadow", "", "## 三组定义与边界", "",
        "- A：冻结Packing v1；B：原Ranking分数 × (1 + compatibility)，调用原Packing v1默认参数；C：冻结Query Requirement v1的Requirement-aware packing（上一轮结果）。C不是另一个新打包算法。",
        "- 使用同一30题、9205个Unit、原rank/原文/metadata及28,000字符预算。A/C从存档重建并校验metrics、retention、候选hash和需求输出；仅B重新执行本地匹配和原Packing。",
        "- 未改生产、Retrieval、Unit生成、Query Requirement分类、Packing、Prompt或Skills；不调用LLM/Jina/Judge/LangSmith，不重新检索，不跑100题。",
        "- compatibility = metric relevance × active support均值；仍为[0,1]，沿用上一轮[1,2]有效分数倍率，不做权重/参数搜索。",
        "- entity：通用连续query token与已有entity匹配，忽略空格及法律后缀差异；无简称/公司字典。未知或未提及不冒充冲突，仅不给加分。",
        "- metric：问题词剔除通用指令词、period及已匹配entity后，与局部原文匹配；既有metric metadata仅给半额支持。不补充金融指标映射。",
        "- text按原有换行/句子切为检查片段，不改输出；table Unit保持已有title/header/row整体。数字和计算支持只在最高词重叠的片段中检查，不把其他段落数字当成支持。",
        "- period：局部匹配优先；非局部原文/metadata只给半额支持。两个数字、同一片段的年份仍不构成操作数绑定或计算验证；comparison也只用这一局部支持代理。",
        "- Gold和参考答案只由评测函数读取；匹配模块不接收这些字段。所有指标为冻结离线证据代理，不是strict answer accuracy。", "",
        "## A/B/C结果", "", "| Group | Route | Coverage | Gold retention | Number hit | Period hit | Units | Chars | Replacements |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in ("overall", *GROUPS):
        s = summary if group == "overall" else summary["groups"][group]
        for route in ROUTES:
            m = s[route]
            lines.append(f"| {group} | {route} | {pct(m['evidence_coverage'])} | {pct(m['gold_evidence_retention'])} | {pct(m['required_number_hit'])} | {pct(m['required_period_hit'])} | {m['average_selected_units']} | {m['context_characters']} | {m['replacement_count']} |")
    lines += ["", "## 总结", "",
        f"- B vs A：{len(summary['B_vs_A']['gains'])}题提升、{len(summary['B_vs_A']['regressions'])}题下降；B vs C：{len(summary['B_vs_C']['gains'])}题提升、{len(summary['B_vs_C']['regressions'])}题下降。",
        f"- Selection-loss相对C均值改善：{summary['decision']['selection_loss_improved_vs_C']}；correct-regression相对C无逐题回退：{summary['decision']['correct_regression_no_loss_vs_C']}。",
        "- 需求分类冻结，只改变匹配函数，因此可观察匹配代理的效果。但同时改了实体表面匹配、局部指标/数字/期间支持，不能将变化归因于其中唯一一项。",
        "- 不接入生产、不继续调参数。下一步判断需结合五题回退明细，不能只看整体均值。", "",
        "## 上轮5个回退样本：A → C丢失及B恢复", "",
        "lost lines采用历史gold-line词/数字覆盖代理；单Unit无法覆盖某行不代表其毫无贡献，可能是多个Unit共同覆盖或渲染metadata影响。以下定位是离线诊断，未反馈到匹配逻辑。", ""]
    for r in payload["records"]:
        if "prior_regression_audit" not in r:
            continue
        audit = r["prior_regression_audit"]
        scores = [r["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"] for route in ROUTES]
        lines += [f"### {r['financebench_id']} — {r['group']}", "", r["question"], "",
            f"- Coverage A/B/C：{scores}；旧C丢失{audit['lost_line_count']}行，B恢复其中{audit['recovered_count']}行。",
            f"- 原需求标签差异：`{audit['requirement_annotation_mismatches']}`（继承临时silver标签，非人工金标准）。",
            f"- C相对A移除{audit['removed_unit_count']}个Unit，加入{audit['added_unit_count']}个；移除原因：`{audit['removed_rejection_reasons']}`。",
            f"- 丢失行中有{audit['lost_lines_with_single_removed_unit_support']}行可在单个被移除Unit中匹配，其余{audit['distributed_or_rendered_proxy_loss_count']}行可能由分散内容或渲染字段共同覆盖。", "",
            "| Removed rank | Page | Type | Chars | C effective score | B effective score | B selected | Lost-line support | C rejection |",
            "|---:|---|---|---:|---:|---:|---|---|---|"]
        for u in audit["removed_units"]:
            lines.append(f"| {u['rank']} | {u['page_id']} | {u['source_type']} | {u['source_chars']} | {u['C_support']['effective_score']:.5f} | {u['B_match']['effective_score']:.5f} | {u['selected_in_B']} | {u['lost_line_indices_supported_by_unit']} | {u['C_packing_decision']['selection_reason']} |")
        lines += ["", "丢失gold行（完整Unit原文、局部匹配、增删和replacement trace见JSON）：", ""]
        for i, line in enumerate(audit["lost_gold_lines_A_to_C"]):
            lines.append(f"- [{i}] B恢复={line in audit['lost_lines_recovered_by_B']}：{line}")
        lines.append("")
    lines += ["## 全部30题逐题变化", "", "| ID | Group | Coverage A/B/C | Retention A/B/C | Number A/B/C | Period A/B/C |", "|---|---|---|---|---|---|"]
    for r in payload["records"]:
        values = [r["routes"][route] for route in ROUTES]
        cols = [[pct(v['metrics']['answer_evidence_coverage']['ratio']) for v in values],
                [pct(v['gold_evidence_retention']['ratio']) for v in values],
                [str(v['metrics']['required_number_hit']) for v in values],
                [str(v['metrics']['required_period_hit']) for v in values]]
        lines.append(f"| {r['financebench_id']} | {r['group']} | " + " | ".join(" / ".join(col) for col in cols) + " |")
    lines += ["", "## 验证与限制", "", f"- 冻结约束：`{summary['invariants']}`。",
        f"- 网络防护：`{payload['network']}`；平均匹配{summary['mean_match_ms']:.2f}ms，匹配+打包{summary['mean_match_and_packing_ms']:.2f}ms。",
        "- 30题已重复使用，不能证明未见样本泛化；未调用Judge，不能把coverage当答案正确率。",
        "- JSON记录源码/候选/历史文件hash、每个Unit兼容度及其组成、Packing决策、每题selected ranks、五题回退的完整Unit差异。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    paths = {"candidates": ROOT / "reports/evidence_metadata_counterfactual_v1.json",
        "baseline": ROOT / "reports/evidence_packing_optimization_v1.json", "previous": ROOT / "reports/query_requirement_shadow_v1.json",
        "dataset": ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv", "requirements": ROOT / "backend/query_requirement_v1.py",
        "packing": ROOT / "backend/evidence_packing_v1.py", "assembly": ROOT / "backend/evidence_assembly_v5.py",
        "matcher": ROOT / "backend/requirement_evidence_match_v1.py", "evaluator": Path(__file__)}
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    sources = json.loads(paths["candidates"].read_text(encoding="utf-8"))["records"]
    old = json.loads(paths["previous"].read_text(encoding="utf-8"))
    baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))["records"]
    ids = {r["financebench_id"] for r in sources}
    if len(sources) != 30 or len(ids) != 30 or len(old["records"]) != 30 or len(baseline) != 30 or ids != {r["financebench_id"] for r in old["records"]} or ids != {r["financebench_id"] for r in baseline}:
        raise ValueError("Exactly the matching frozen diagnostic30 is required")
    if Counter(r["group"] for r in sources) != Counter({g: 10 for g in GROUPS}) or sum(len(r["candidate_units"]) for r in sources) != 9205:
        raise ValueError("Frozen candidate manifest drift")
    for key in ("candidates", "baseline", "dataset", "requirements", "packing", "assembly"):
        if hashes[key] != old["file_sha256"][key]:
            raise ValueError(f"Frozen source changed: {key}")
    rows = _load_dataset(paths["dataset"])
    prior = {r["financebench_id"]: r for r in old["records"]}
    by_id = {r["financebench_id"]: r for r in baseline}
    network = {"blocked_attempts": 0, "outbound_calls": 0, "socket_guard_enabled": True}
    def denied(*args, **kwargs):
        network["blocked_attempts"] += 1
        raise RuntimeError("Network forbidden in matching shadow")
    output = []
    with patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied), patch.object(socket, "create_connection", denied):
        for i, source in enumerate(sources, 1):
            key = source["financebench_id"]
            item = evaluate_record(source, prior[key], rows[key], by_id[key])
            output.append(item)
            print(f"[{i:02d}/30] {key} A/B/C=" + "/".join(str(item['routes'][route]['metrics']['answer_evidence_coverage']['ratio']) for route in ROUTES), flush=True)
    if hashes != {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}:
        raise AssertionError("Sources changed during replay")
    payload = {"evaluation": "requirement_evidence_match_v1_shadow", "file_sha256": hashes, "network": network,
               "summary": summarize(output), "records": output}
    if payload["summary"]["invariants"]["prior_regression_cases"] != 5:
        raise ValueError("Expected the five frozen Query Requirement regressions")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for suffix, content in (("json", json.dumps(payload, ensure_ascii=False, indent=2)), ("md", render_markdown(payload))):
        path = args.report_dir / f"requirement_evidence_match_v1.{suffix}"
        path.write_text(content + "\n", encoding="utf-8")
        print(f"Report: {path}", flush=True)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
