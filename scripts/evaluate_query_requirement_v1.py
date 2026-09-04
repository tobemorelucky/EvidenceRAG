"""Fixed diagnostic30 offline replay: Packing v1 vs requirement-guided Packing v1."""

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
from backend.query_requirement_v1 import requirement_guided_inputs
from scripts.evaluate_evidence_metadata_counterfactual_v1 import _context_metrics, _gold, _gold_retention, _load_dataset
from scripts.evaluate_evidence_packing_optimization_v1 import _route_summary

GROUPS = ("selection_loss10", "correct_regression10", "candidate_miss10")
ROUTES = ("packing_v1", "requirement_guided")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def evaluate_record(source: dict, prior: dict, row: dict) -> dict:
    # No evaluation annotations cross the adapter boundary.
    units = [{key: copy.deepcopy(unit[key]) for key in (*EvidenceUnit.__dataclass_fields__, "current_ranking")}
             for unit in source["candidate_units"]]
    frozen_hash = digest(units)
    baseline = prior["routes"]["packing_optimization_v1"]
    selected_before = [u for u in units if _rank(u) in baseline["selected_unit_ranks"]]
    before_context = _render_selection(selected_before)[0]
    if len(selected_before) != len(baseline["selected_unit_ranks"]) or _context_metrics(row, before_context, selected_before) != baseline["metrics"]:
        raise ValueError("Frozen Packing v1 baseline does not reproduce")
    start = time.perf_counter()
    adapted, requirement_trace = requirement_guided_inputs(source["question"], units)
    adapter_ms = (time.perf_counter() - start) * 1000
    # No new replacement threshold, anchors or replacement limit.
    context, selected_shadow, packing_trace = select_evidence_packing_v1(source["question"], adapted, max_context_chars=28000)
    ranks = {_rank(u) for u in selected_shadow}
    selected = [u for u in units if _rank(u) in ranks]
    if context != _render_selection(selected)[0] or len(context) > 28000:
        raise AssertionError("Evidence rendering/budget drift")
    if digest(units) != frozen_hash:
        raise AssertionError("Frozen candidates/ranking changed")
    return {
        "financebench_id": source["financebench_id"], "group": source["group"], "question": source["question"],
        "candidate_count": len(units), "candidate_sha256": frozen_hash,
        "baseline_reproduced": True, "candidate_and_original_ranking_unchanged": True,
        "adapter_ms": round(adapter_ms, 2), "selection_ms": round((time.perf_counter() - start) * 1000, 2),
        "requirement_trace": requirement_trace,
        "routes": {
            "packing_v1": {"metrics": baseline["metrics"], "gold_evidence_retention": baseline["gold_evidence_retention"],
                           "selected_ranks": baseline["selected_unit_ranks"], "replacement_count": baseline["packing_trace"]["replacement_count"]},
            "requirement_guided": {"metrics": _context_metrics(row, context, selected),
                                   "gold_evidence_retention": _gold_retention(_gold(row), "\n\n".join(u["source_text"] for u in units), context),
                                   "selected_ranks": sorted(ranks), "replacement_count": packing_trace["replacement_count"],
                                   "packing_trace": packing_trace},
        },
    }


def requirement_accuracy(records: list[dict], annotations: dict) -> dict:
    """Agreement with prewritten question-only silver labels, not a self-consistency test."""
    fields = annotations["fields"]
    result = {"provenance": annotations["provenance"], "fields": {}}
    exact_known, full_exact, full_count, correct, total = 0, 0, 0, 0, 0
    for record in records:
        expected = dict(zip(fields, annotations["labels"][record["financebench_id"]], strict=True))
        actual = record["requirement_trace"]["requirement"]
        mismatches = {key: {"expected": val, "predicted": actual[key]} for key, val in expected.items()
                      if val is not None and val != actual[key]}
        record["requirement_annotation"] = {"expected": expected, "mismatches": mismatches,
                                             "unscored_fields": [k for k, v in expected.items() if v is None]}
        exact_known += not mismatches
        full_count += all(v is not None for v in expected.values())
        full_exact += not mismatches and all(v is not None for v in expected.values())
    for field in fields:
        pairs = [(r["requirement_annotation"]["expected"][field], r["requirement_trace"]["requirement"][field]) for r in records]
        eligible = [(a, b) for a, b in pairs if a is not None]
        matches = sum(a == b for a, b in eligible)
        correct += matches
        total += len(eligible)
        value = {"correct": matches, "evaluated": len(eligible), "unscored": len(pairs) - len(eligible),
                 "accuracy": matches / len(eligible) if eligible else None,
                 "confusion": dict(Counter(f"{a}->{b}" for a, b in eligible))}
        if field != "answer_type":
            tp = sum(a is True and b is True for a, b in eligible)
            fp = sum(a is False and b is True for a, b in eligible)
            fn = sum(a is True and b is False for a, b in eligible)
            value.update(precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None)
        result["fields"][field] = value
    result.update(micro_accuracy=correct / total if total else None, scored_fields=total,
                  exact_on_known_fields={"correct": exact_known, "total": len(records)},
                  full_spec_exact={"correct": full_exact, "total": full_count})
    return result


def summarize(records: list[dict], annotations: dict) -> dict:
    def group_summary(items):
        result = {route: _route_summary(items, route) for route in ROUTES}
        for route in ROUTES:
            result[route]["replacement_count"] = sum(r["routes"][route]["replacement_count"] for r in items)
        result["coverage_gains"], result["coverage_regressions"] = [], []
        for r in items:
            a, b = [r["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"] for route in ROUTES]
            if a != b:
                result["coverage_gains" if b > a else "coverage_regressions"].append(r["financebench_id"])
        return result
    result = group_summary(records)
    result["requirement_accuracy"] = requirement_accuracy(records, annotations)
    result["groups"] = {group: group_summary([r for r in records if r["group"] == group]) for group in GROUPS}
    for group in GROUPS:
        result["groups"][group]["requirement_accuracy"] = requirement_accuracy([r for r in records if r["group"] == group], annotations)
    result["mean_adapter_ms"] = sum(r["adapter_ms"] for r in records) / len(records)
    result["mean_selection_ms"] = sum(r["selection_ms"] for r in records) / len(records)
    result["invariants"] = {
        "questions": len(records), "candidate_units": sum(r["candidate_count"] for r in records),
        "baseline_reproduced": all(r["baseline_reproduced"] for r in records),
        "candidate_and_original_ranking_unchanged": all(r["candidate_and_original_ranking_unchanged"] for r in records),
        "maximum_chars": max(r["routes"][route]["metrics"]["context_chars"] for r in records for route in ROUTES),
        "gold_used_for_selection": False, "production_integration": False,
    }
    result["interpretation"] = {
        "selection_loss_improved": result["groups"]["selection_loss10"]["requirement_guided"]["evidence_coverage"] > result["groups"]["selection_loss10"]["packing_v1"]["evidence_coverage"],
        "correct_regression_no_per_question_loss": not result["groups"]["correct_regression10"]["coverage_regressions"],
        "causal_limit": "Parser quality and requirement-to-evidence proxy both affect this result. Does not isolate question understanding as the sole bottleneck.",
    }
    return result


def render_markdown(payload: dict) -> str:
    s = payload["summary"]
    def pct(value):
        return "N/A" if value is None else f"{value:.2%}"
    lines = ["# Query Requirement Shadow v1 — 固定30题", "", "## 实验边界", "",
             "- 冻结9205个候选Unit、原始ranking、Retrieval、Evidence Unit、Packing v1代码和28,000字符预算；未接入生产。",
             "- 只改变Packing输入副本的effective score：original score × (1 + query lexical gate × active requirement support均值)，倍率[1,2]。没有重写问题、过滤候选、修改文本、增设anchor或调replacement参数。",
             "- 原rank不变；调用原Packing v1默认参数；选中后映射回原Unit并核验渲染完全相同。不是新通用ranker，但有效分数确实变化，不能声称分数完全不变。",
             "- requirement仅从question提取：主回答类型和numeric/period/entity/comparison/calculation需求。条件性的解释指令不覆盖主任务。无公司或指标映射；未明确表达的领域需求可能漏判。",
             "- 数字支持按非年份数字存在计；comparison/calculation最多两个数即饱和；period按原文或既有metadata中的token匹配；entity按既有entity与query词交集。只给软加分，不把unknown判为冲突、不进行运算或补造metadata。",
             "- FY两位年份采用1950/2050窗口；quarter/year分别匹配，不证明同一行或同一报告口径。多个数字也不证明它们是正确操作数。",
             "- 未运行LLM/Jina/Judge/LangSmith，未重新检索、未跑100题、未修改Prompt/Skills。Baseline复用冻结Packing结果并逐题重建核验，不重复API。", "",
             "## Evidence A/B", "", "| Group | Route | Coverage | Gold retention | Number hit | Period hit | Selected page hit | Units | Chars | Replacements |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in ("overall", *GROUPS):
        section = s if group == "overall" else s["groups"][group]
        for route in ROUTES:
            m = section[route]
            lines.append(f"| {group} | {route} | {pct(m['evidence_coverage'])} | {pct(m['gold_evidence_retention'])} | {pct(m['required_number_hit'])} | {pct(m['required_period_hit'])} | {pct(m['selected_gold_page_hit'])} | {m['average_selected_units']} | {m['context_characters']} | {m['replacement_count']} |")
    accuracy = s["requirement_accuracy"]
    lines += ["", "## Requirement accuracy：临时标签一致率，不是人工金标准", "",
              "标签在实现解析器前按30题question-only语义人工式编写（由本次开发助手标注），没有读取答案或gold来推导需求；未根据解析器输出修改标签。标注与实现并非独立标注团队，仍可能有同源偏差，需用户复核。",
              "计算需求无法从题目确定时标为null并排除分母；accuracy不是规则与自身输出比较，也不是FinanceBench答案正确率。",
              f"- 总字段一致率：{pct(accuracy['micro_accuracy'])}，分母{accuracy['scored_fields']}；known-fields exact：{accuracy['exact_on_known_fields']}；完整六字段exact：{accuracy['full_spec_exact']}。", "",
              "| Field | Correct / scored | Unscored | Accuracy | Precision | Recall |",
              "|---|---:|---:|---:|---:|---:|"]
    for field, value in accuracy["fields"].items():
        lines.append(f"| {field} | {value['correct']}/{value['evaluated']} | {value['unscored']} | {pct(value['accuracy'])} | {pct(value.get('precision'))} | {pct(value.get('recall'))} |")
    lines += ["", "## 结论与限制", "",
              f"- 相对Packing v1 coverage：提升{len(s['coverage_gains'])}题，下降{len(s['coverage_regressions'])}题。",
              f"- Selection-loss：提升{s['groups']['selection_loss10']['coverage_gains']}；下降{s['groups']['selection_loss10']['coverage_regressions']}。",
              f"- Correct-regression：提升{s['groups']['correct_regression10']['coverage_gains']}；下降{s['groups']['correct_regression10']['coverage_regressions']}。",
              f"- Selection-loss均值提升：{s['interpretation']['selection_loss_improved']}；correct-regression无逐题回退：{s['interpretation']['correct_regression_no_per_question_loss']}。",
              "- 本实验同时依赖需求解析正确性和需求到Unit支持的弱映射。粗粒度numeric/comparison需求仍不能指出需要哪个事实/哪一组操作数；不能从失败直接得出需求建模无效，也不能从提升断言问题理解是唯一瓶颈。",
              "- Evidence coverage沿用历史词重叠+数字匹配代理；gold retention分母是candidate已覆盖gold行。Number/period hit检查字符串存在而非归属/口径正确，不是答案accuracy。",
              "- 这30题被反复诊断，结果不构成泛化证明；不根据本轮题目补金融规则，不继续搜索倍率，不接入生产。",
              f"- 平均adapter {s['mean_adapter_ms']:.2f}ms；adapter+packing {s['mean_selection_ms']:.2f}ms（本地CPU，不含任何生成）。",
              f"- 约束核验：`{s['invariants']}`；socket防护：`{payload['network']}`。", "",
              "## 每题：需求与证据变化", ""]
    for r in payload["records"]:
        a, b = [r["routes"][route] for route in ROUTES]
        lines += [f"### {r['financebench_id']} — {r['group']}", "", r["question"], "",
                  f"- 预测需求：`{r['requirement_trace']['requirement']}`。",
                  f"- 标签差异：`{r['requirement_annotation']['mismatches']}`；未评分字段：`{r['requirement_annotation']['unscored_fields']}`。",
                  f"- Coverage：{a['metrics']['answer_evidence_coverage']['ratio']} → {b['metrics']['answer_evidence_coverage']['ratio']}；retention：{a['gold_evidence_retention']['ratio']} → {b['gold_evidence_retention']['ratio']}。",
                  f"- Number/period：{a['metrics']['required_number_hit']}/{a['metrics']['required_period_hit']} → {b['metrics']['required_number_hit']}/{b['metrics']['required_period_hit']}。",
                  f"- Units/chars：{a['metrics']['selected_unit_count']}/{a['metrics']['context_chars']} → {b['metrics']['selected_unit_count']}/{b['metrics']['context_chars']}。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    paths = {"candidates": ROOT / "reports/evidence_metadata_counterfactual_v1.json",
             "baseline": ROOT / "reports/evidence_packing_optimization_v1.json",
             "annotations": ROOT / "tests/fixtures/query_requirement_v1_labels.json",
             "dataset": ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv",
             "packing": ROOT / "backend/evidence_packing_v1.py", "assembly": ROOT / "backend/evidence_assembly_v5.py",
             "requirements": ROOT / "backend/query_requirement_v1.py"}
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    records = json.loads(paths["candidates"].read_text(encoding="utf-8"))["records"]
    baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))["records"]
    annotations = json.loads(paths["annotations"].read_text(encoding="utf-8"))
    ids = {r["financebench_id"] for r in records}
    if len(records) != 30 or len(ids) != 30 or len(baseline) != 30 or ids != {r["financebench_id"] for r in baseline} or ids != set(annotations["labels"]):
        raise ValueError("Exactly the frozen matching diagnostic30 and its labels required")
    if Counter(r["group"] for r in records) != Counter({g: 10 for g in GROUPS}) or sum(len(r["candidate_units"]) for r in records) != 9205:
        raise ValueError("Frozen diagnostic groups or Unit count changed")
    rows = _load_dataset(paths["dataset"])
    by_id = {r["financebench_id"]: r for r in baseline}
    network = {"blocked_attempts": 0, "outbound_calls": 0, "socket_guard_enabled": True}
    def denied(*args, **kwargs):
        network["blocked_attempts"] += 1
        raise RuntimeError("Network forbidden in offline requirement shadow")
    output = []
    with patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied), patch.object(socket, "create_connection", denied):
        for i, record in enumerate(records, 1):
            item = evaluate_record(record, by_id[record["financebench_id"]], rows[record["financebench_id"]])
            output.append(item)
            values = [item["routes"][route]["metrics"]["answer_evidence_coverage"]["ratio"] for route in ROUTES]
            print(f"[{i:02d}/30] {record['financebench_id']} coverage={values[0]}->{values[1]}", flush=True)
    if hashes != {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}:
        raise AssertionError("Frozen input or implementation changed during evaluation")
    payload = {"evaluation": "query_requirement_shadow_v1", "file_sha256": hashes, "network": network,
               "annotation_rubric": annotations["rubric"], "summary": summarize(output, annotations), "records": output}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for suffix, content in (("json", json.dumps(payload, ensure_ascii=False, indent=2)), ("md", render_markdown(payload))):
        path = args.report_dir / f"query_requirement_shadow_v1.{suffix}"
        path.write_text(content + "\n", encoding="utf-8")
        print(f"Report: {path}", flush=True)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
