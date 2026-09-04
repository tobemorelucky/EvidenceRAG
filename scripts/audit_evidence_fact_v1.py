"""Offline fact compatibility audit of frozen Packing v1; never run selection.

This is a conservative partial verifier, not a replacement semantic Judge.
Unknown bindings and absent evidence are not contradictory facts.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import socket
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.evidence_assembly_v5 import EvidenceUnit
from backend.evidence_packing_v1 import _rank
from scripts.audit_evidence_identity_v1 import (
    GROUPS, audit_record, build_registry, canonical_entity, literal_hit,
    _normalized, periods, scope_terms,
)
from scripts.evaluate_evidence_metadata_counterfactual_v1 import _load_dataset

CATEGORIES = {
    "A": "exact fact match", "B": "compatible but different source",
    "C": "entity mismatch", "D": "period mismatch", "E": "metric mismatch",
    "F": "number mismatch", "G": "unknown",
}
# Formatting normalization only. No metric, company, or accounting formula map.
NUMBER = re.compile(r"(?<![\w.])(?:[$€£¥]\s*)?\(?[+−-]?\d[\d,]*(?:\.\d+)?\)?(?:\s*%)?(?!\w|\.\d)")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def numeric_signature(text):
    """Preserve sign/currency/percent and order; never infer scale or column.

    Years remain in the signature; callers compare periods separately. A scale
    word remains in the surrounding skeleton, so million != billion.
    """
    result = []
    for match in NUMBER.finditer(text):
        raw = match.group().strip()
        value = re.sub(r"[$€£¥%,()\s]", "", raw).replace("−", "-")
        number = Decimal(value)
        if "(" in raw and ")" in raw:
            number = -abs(number)
        result.append({"raw": raw, "value": str(number.normalize()),
                       "currency": next((c for c in "$€£¥" if c in raw), None),
                       "percent": "%" in raw})
    return result


def signature_values(text):
    return [(n["value"], n["currency"], n["percent"]) for n in numeric_signature(text)]


def skeleton(text):
    return _normalized(NUMBER.sub(" <number> ", text)).strip(" .;:")


def fragments(text):
    # Keep original lines. No guessed joining of bare cells/headers or periods.
    return list(dict.fromkeys(line.strip() for line in text.splitlines() if line.strip()))


def period_binding(reference_text, evidence_text, requested):
    expected, reference, visible = set(requested), set(periods(reference_text)), set(periods(evidence_text))
    if not expected:
        if reference and visible and reference != visible:
            return "reference_period_conflict"
        return "not_requested"
    if not reference or not visible:
        return "unknown_local_binding"
    if not expected & reference:
        return "question_reference_period_disagreement"
    if reference != visible:
        return "reference_period_conflict"
    # A multi-column claim cannot establish value -> period ownership.
    if len([p for p in reference if p.isdigit()]) > 1:
        return "multiple_period_value_binding_unknown"
    for kind in (lambda p: p.isdigit(), lambda p: p.startswith("q")):
        wanted, actual = {p for p in expected if kind(p)}, {p for p in visible if kind(p)}
        if wanted and (not actual or not wanted & actual):
            return "unknown_local_binding"
    return "explicit_local_match"


def compare_claim(reference, text, unit, requirement):
    """Compare one labelled reference span and one local candidate claim.

    Exact wording/skeleton is needed, not a bag of matching tokens or numbers.
    A/B certify reference-compatible *partial support*, never full answerability.
    """
    exact = literal_hit(reference["text"], text)
    same_shape = skeleton(reference["text"]) == skeleton(text)
    if not exact and not same_shape:
        return None
    ref_text = reference["text"]
    # For substring support, only the actual matched span supplies bindings.
    local = ref_text if exact else text
    result = {"reference_fact_id": reference["fact_id"], "reference_text": ref_text,
              "evidence_text": local, "category": "G", "reason": None,
              "period_binding": period_binding(ref_text, local, requirement["period"]),
              "numbers": numeric_signature(local), "reference_numbers": numeric_signature(ref_text)}
    entity = canonical_entity(unit.get("entity"))
    if not requirement["entity_reliable"] or not entity or entity in {"unknown", "none", "null"}:
        result["reason"] = "entity_unresolved"
        return result
    if entity not in requirement["entity"]:
        result.update(category="C", reason="frozen_entity_conflicts_with_reference_entity")
        return result
    metric = str(unit.get("metric") or "").strip()
    if not metric or not literal_hit(metric, requirement["question"]) or not literal_hit(metric, local):
        result["reason"] = "metric_identity_unverified"
        return result
    if result["period_binding"] == "reference_period_conflict":
        result.update(category="D", reason="same_claim_shape_explicit_period_differs_from_reference")
        return result
    if result["period_binding"] not in {"explicit_local_match", "not_requested"}:
        result["reason"] = result["period_binding"]
        return result
    needed_scope = set(requirement["scope"])
    if not needed_scope <= scope_terms(local):
        result["reason"] = "scope_binding_unknown_or_conflicting"
        return result
    values = signature_values(local)
    expected_values = signature_values(ref_text)
    if values != expected_values:
        # Only a single non-year scalar, unchanged wording, periods and units
        # can justify a numeric contradiction. Missing data != contradiction.
        ref_scalars = numeric_signature(YEAR.sub(" ", ref_text))
        local_scalars = numeric_signature(YEAR.sub(" ", local))
        if len(ref_scalars) == len(local_scalars) == 1:
            if (ref_scalars[0]["currency"], ref_scalars[0]["percent"]) == (local_scalars[0]["currency"], local_scalars[0]["percent"]):
                result.update(category="F", reason="same_local_claim_and_period_different_scalar")
            else:
                result["reason"] = "currency_or_percent_conversion_unverified"
        else:
            result["reason"] = "multi_value_binding_unknown"
        return result
    if not exact and not same_shape:
        raise AssertionError("Unmatched claims cannot be certified")
    # Broad gold paragraphs and lexical anchors can be irrelevant to the
    # actual requested metric. Require an explicit stored label, present in
    # both the question and the claim; absence is UNKNOWN, not metric mismatch.
    same_source = bool(reference.get("page_id")) and unit["document_id"] == reference["document_id"] and unit["page_id"] == reference["page_id"]
    result.update(category="A" if same_source else "B",
                  reason="same_source_verified_partial_fact" if same_source else "alternate_source_verified_partial_fact")
    return result


def inspect_fact_unit(unit, requirement, reference_facts):
    matches = []
    lines = fragments(unit["source_text"])
    for ref in reference_facts:
        # A wrapped exact span may be present even if PDF line breaks differ.
        if literal_hit(ref["text"], unit["source_text"]):
            candidates = [ref["text"]]
        else:
            candidates = [line for line in lines if skeleton(line) == skeleton(ref["text"])]
        for line in candidates:
            pair = compare_claim(ref, line, unit, requirement)
            if pair:
                matches.append(pair)
    entity = canonical_entity(unit.get("entity"))
    known = bool(entity) and entity not in {"unknown", "none", "null"} and requirement["entity_reliable"]
    conflict = known and entity not in requirement["entity"]
    categories = sorted({p["category"] for p in matches if p["category"] != "G"})
    if conflict and "C" not in categories:
        categories.append("C")
    if not categories:
        categories = ["G"]
    compatible = sorted({p["reference_fact_id"] for p in matches if p["category"] in {"A", "B"}})
    return {"rank": _rank(unit), "document_id": unit["document_id"], "page_id": unit["page_id"],
            "source_type": unit["source_type"], "entity": unit.get("entity"),
            "entity_basis": "frozen metadata, not independently verified corporate identity",
            "metric": {"stored": unit.get("metric"), "inferred": None},
            "period": {"stored": unit.get("period"), "text_mentions": sorted(periods(unit["source_text"]))},
            "scope": {"stored": unit["metadata"].get("scope"), "text_mentions": sorted(scope_terms(unit["source_text"]))},
            "numbers": numeric_signature(unit["source_text"]), "source_text": unit["source_text"],
            "categories": sorted(categories), "compatible_reference_fact_ids": compatible,
            "claim_comparisons": matches,
            "unknown_reasons": sorted({p["reason"] for p in matches if p["category"] == "G"}) or
                               ([] if compatible else ["no_locally_bound_reference_claim"]),
            "metric_lexical_absence_is_not_E": True}


def audit(source, previous, row, registry):
    # Reproduce/check selection and legacy metric without running a selector.
    old = audit_record(source, previous, row, registry)
    requirement = copy.deepcopy(old["question_requirement"])
    # Question-explicit identities from existing metadata take precedence over
    # gold provenance, including multi-company questions. No alias dictionary.
    from backend.requirement_evidence_match_v1 import entity_match
    named = sorted({e for values in registry[1].values() for e in values if entity_match(row["question"], e)[0]})
    if named:
        requirement.update(entity=named, entity_reliable=True,
                           entity_source="question surface matched against frozen metadata identities; no gold selection")
    candidates = [{k: copy.deepcopy(u[k]) for k in (*EvidenceUnit.__dataclass_fields__, "current_ranking")} for u in source["candidate_units"]]
    inspected = [inspect_fact_unit(u, requirement, old["reference_facts"]) for u in candidates]
    selected = [u for u in inspected if u["rank"] in old["selected_ranks"]]
    candidate_ids = set().union(*(set(u["compatible_reference_fact_ids"]) for u in inspected))
    selected_ids = set().union(*(set(u["compatible_reference_fact_ids"]) for u in selected))
    denominator = len(old["reference_facts"])
    return {"financebench_id": source["financebench_id"], "group": source["group"], "question": row["question"],
            "question_fact": requirement, "reference_span_candidates": old["reference_facts"],
            "excluded_reference_lines": old["excluded_reference_lines"],
            "candidate_sha256": old["candidate_sha256"], "candidate_count": len(candidates),
            "selected_ranks": old["selected_ranks"], "frozen_selection_verified": old["frozen_selection_verified"],
            "selected_evidence_facts": selected,
            "candidate_compatible_providers": {str(f): [u["rank"] for u in inspected if f in u["compatible_reference_fact_ids"]] for f in sorted(candidate_ids)},
            "selected_compatible_fact_ids": sorted(selected_ids),
            "verified_fact_ids_dropped_by_selection": sorted(candidate_ids - selected_ids),
            "coverage": {"original": old["coverage"]["legacy"],
                         "entity_bound_legacy_proxy": old["coverage"]["entity_bound"],
                         "reference_span_denominator": denominator,
                         "fact_compatible_partial_support_lower_bound": len(selected_ids) / denominator if denominator else None,
                         "true_fact_coverage": None},
            "verifiability": {
                "selected_with_stored_metric": sum(bool(u["metric"]["stored"]) for u in selected),
                "selected_with_locally_comparable_reference_span": sum(bool(u["claim_comparisons"]) for u in selected),
                "strict_verifier_limitation": "requires explicit frozen metric label in question and local claim; paraphrases/unlabelled text remain unknown, even if useful"},
            "selected_category_counts": dict(Counter(c for u in selected for c in u["categories"])),
            "identity_conflicting_legacy_lines": old["identity_conflicting_line_hits"],
            "source_flow": old["source_flow"],
            "legacy_qualified_fact_loss_for_manual_review": old["candidate_fact_loss"],
            "actual_answer_effectiveness": None}


def summarize(records):
    def part(items):
        units = [u for r in items for u in r["selected_evidence_facts"]]
        eligible = [r for r in items if r["coverage"]["reference_span_denominator"]]
        supported = sum(bool(u["compatible_reference_fact_ids"]) for u in units)
        return {"questions": len(items), "selected_units": len(units),
                "original_coverage": fmean(r["coverage"]["original"] for r in items),
                "entity_bound_legacy_proxy": fmean(r["coverage"]["entity_bound_legacy_proxy"] for r in items),
                "reference_span_eligible_questions": len(eligible),
                "reference_span_count": sum(r["coverage"]["reference_span_denominator"] for r in items),
                "original_coverage_on_same_eligible_subset": fmean(r["coverage"]["original"] for r in eligible) if eligible else None,
                "fact_compatible_partial_support_lower_bound": fmean(r["coverage"]["fact_compatible_partial_support_lower_bound"] for r in eligible) if eligible else None,
                "compatible_selected_units": supported,
                "verified_unit_support_lower_bound": supported / len(units) if units else None,
                "unknown_units": sum(u["categories"] == ["G"] for u in units),
                "category_counts": dict(Counter(c for u in units for c in u["categories"])),
                "selected_with_stored_metric": sum(bool(u["metric"]["stored"]) for u in units),
                "selected_with_locally_comparable_reference_span": sum(bool(u["claim_comparisons"]) for u in units),
                "unknown_claim_reason_counts": dict(Counter(p["reason"] for u in units for p in u["claim_comparisons"] if p["category"] == "G")),
                "identity_conflicting_legacy_lines": sum(r["identity_conflicting_legacy_lines"] for r in items),
                "source_flow": dict(Counter(r["source_flow"]["stage"] for r in items)),
                "strict_verified_candidate_fact_loss_questions": [r["financebench_id"] for r in items if r["verified_fact_ids_dropped_by_selection"]],
                "true_fact_coverage": None, "packing_actual_effectiveness": None}
    summary = part(records)
    summary["groups"] = {g: part([r for r in records if r["group"] == g]) for g in GROUPS}
    summary["candidate_units"] = sum(r["candidate_count"] for r in records)
    summary["frozen_selection_verified"] = all(r["frozen_selection_verified"] for r in records)
    return summary


def markdown(payload):
    s = payload["summary"]
    def pct(value):
        return "unknown" if value is None else f"{value:.2%}"
    lines = ["# Evidence Fact Audit v1", "", "## 结论", "",
        "当前原coverage不能解释为事实正确率。该审计只读冻结30题、9205个候选Unit及Packing v1既有选择，不重新检索、打分或packing。",
        "优先补足可验证的Evidence Identity/局部事实绑定及评测真值，再评价Requirement Matching；当前审计不足以证明Packing是主要因果瓶颈。未实施任何生产改动。", "",
        "## 判定契约与局限", "",
        "- A：同稳定document_id/page_id、完整参考片段、实体兼容、局部期间/scope可验证，且已有metric标签同时出现在问题和片段。仅表示该片段的部分事实支持。",
        "- B：不同来源且满足同样的局部事实约束；不要求必须来自gold页。A/B均允许只规范化千分位及小数尾零。相同公司/年份/数字的散落共现不算B。",
        "- C：冻结entity元数据与问题显式命名的目标不匹配；无法解析问题实体时以gold文档元数据为离线参照。此为身份冲突，不代表该Unit每句话均错，合法第三方信息仍需人工判断。",
        "- D：相同声明形态的局部期间与参考不同；参考期间与问题期间本身不一致时归G，避免强判历史操作数错误。",
        "- E保留为metric mismatch类，但现有数据没有独立规范metric真值，故不以词面缺失强行输出E。",
        "- F：声明措辞、局部期间一致、单个非年份数值不同。保留负号、币种、百分号；不推断million/billion换算或多列对应。不包含数字/缺操作数归G而非F。",
        "- G：期间/指标/scope/实体未绑定、裸单元格、多个年份列无法绑定或没有可核查参考声明。unknown不等于错误。",
        "- gold只在审计选择完成后的判定中使用。required numbers保留参考justification/answer的来源说明，可能是计算结果，不能作为每个Unit必须包含的操作数清单。",
        "- 参考片段候选沿用既有审计的逐行抽取，它可能含无关表头/跨行残片。新增严格绑定门槛会把这些候选降为G，不把旧proxy当成已确认事实。",
        "- A/B是部分支持且依赖冻结元数据，不是完整问题可回答性。真实fact coverage与Packing真实答案有效率保留null；下界低不证明实际效果低。", "",
        "## 汇总", "", "| 组别 | 原coverage | 同实体coverage代理 | 参考片段可审计题 | 严格兼容片段下界 | A/B Units | G Units |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name, v in [("overall", s), *s["groups"].items()]:
        lines.append(f"| {name} | {pct(v['original_coverage'])} | {pct(v['entity_bound_legacy_proxy'])} | {v['reference_span_eligible_questions']}/{v['questions']} | {pct(v['fact_compatible_partial_support_lower_bound'])} | {v['compatible_selected_units']}/{v['selected_units']} | {v['unknown_units']} |")
    lines += ["", f"分类计数（可重叠）：`{s['category_counts']}`。",
        f"严格兼容率分母仅{s['reference_span_count']}条候选片段、{s['reference_span_eligible_questions']}题；同一题子集的原coverage={pct(s['original_coverage_on_same_eligible_subset'])}。两者仍是不同的行/事实口径，不能当准确率差值。",
        f"原coverage中的身份冲突行：{s['identity_conflicting_legacy_lines']}。已确认部分支持Unit比例下界={pct(s['verified_unit_support_lower_bound'])}；真实有效率无法从无独立事实标注的数据算出。",
        f"G原因：`{s['unknown_claim_reason_counts']}`。", "", "## Selection-loss定位与下一步", "",
        "| 问题 | 参考页候选/已选 | 旧coverage | 分类 | 严格合格候选事实丢失 |",
        "|---|---:|---:|---|---|"]
    for r in payload["records"]:
        if r["group"] == "selection_loss10":
            flow = r["source_flow"]
            lines.append(f"| {r['financebench_id']} | {len(flow['candidate_reference_page_ranks'])}/{len(flow['selected_reference_page_ranks'])} | {pct(r['coverage']['original'])} | {r['selected_category_counts']} | {r['verified_fact_ids_dropped_by_selection']} |")
    lines += ["", f"判定能力限制：{s['selected_with_stored_metric']}/{s['selected_units']}个已选Unit具有metric字段，只有{s['selected_with_locally_comparable_reference_span']}个能匹配到本审计提取的局部参考片段。严格A/B={s['compatible_selected_units']}不是系统事实正确率为零，而是本方法未能确认；不能用这个结果证明生产metadata一定错误，也不能把G全部归因于Identity层。", "",
        "1. Identity layer/评测绑定优先：为实体、行标签、期间、scope、数值保存可验证的局部来源；先人工标注少量目标事实，核验C和G，不据本轮规则直接启用硬过滤。",
        "2. Requirement matching其次：只在身份与局部事实可验证的样本上测匹配；不要用当前宽松coverage继续调权重。",
        "3. Packing暂不调：参考页未选只能定位到选择阶段，不能区分ranking与packing因果。旧宽松审计丢失的候选保留在JSON供人工复查，不能冒充本轮严格合格事实。", "",
        "## 逐题结果", ""]
    for r in payload["records"]:
        q = r["question_fact"]
        lines += [f"### {r['financebench_id']} / {r['group']}", "", r["question"], "",
                  f"- Question Fact：entity={q['entity']}；metric词={q['metric']['question_terms']}；period={q['period']}；scope={q['scope']}；参考数字={q['required_number']}。",
                  f"- Coverage：`{r['coverage']}`；分类：`{r['selected_category_counts']}`。", "",
                  "| Rank | Entity | Metric | Period metadata | 分类 | 未验证原因 |",
                  "|---:|---|---|---|---|---|"]
        for u in r["selected_evidence_facts"]:
            metric = str(u["metric"]["stored"] or "unknown").replace("|", "/").replace("\n", " ")[:80]
            lines.append(f"| {u['rank']} | {u['entity']} | {metric} | {u['period']['stored']} | {','.join(u['categories'])} | {', '.join(u['unknown_reasons'])} |")
        lines.append("")
    lines += ["## 可复现性", "", "运行：`conda run --no-capture-output -n rag python -u scripts/audit_evidence_fact_v1.py`", "",
              f"- 冻结输入/选择校验={s['frozen_selection_verified']}；候选={s['candidate_units']}；网络guard={payload['network']}。",
              "- JSON包含输入哈希、完整selected source_text、局部声明比较、数字签名与来源ID；不修改生产代码，不调用LLM/Jina/Judge/LangSmith，不运行100题。", ""]
    return "\n".join(lines)


def run():
    paths = {"candidates": ROOT / "reports/evidence_metadata_counterfactual_v1.json",
             "baseline": ROOT / "reports/evidence_packing_optimization_v1.json",
             "previous": ROOT / "reports/query_requirement_shadow_v1.json",
             "dataset": ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv",
             "packing": ROOT / "backend/evidence_packing_v1.py",
             "requirements": ROOT / "backend/query_requirement_v1.py",
             "assembly": ROOT / "backend/evidence_assembly_v5.py"}
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    sources = json.loads(paths["candidates"].read_text(encoding="utf-8"))["records"]
    previous = json.loads(paths["previous"].read_text(encoding="utf-8"))
    baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))["records"]
    for key in ("candidates", "baseline", "dataset", "packing", "requirements", "assembly"):
        if hashes[key] != previous["file_sha256"][key]:
            raise ValueError(f"Frozen input changed: {key}")
    ids = {r["financebench_id"] for r in sources}
    if any(len(rs) != 30 or {r["financebench_id"] for r in rs} != ids for rs in (sources, previous["records"], baseline)) or len(ids) != 30:
        raise ValueError("Only frozen diagnostic30 is allowed")
    if Counter(r["group"] for r in sources) != Counter({g: 10 for g in GROUPS}) or sum(len(r["candidate_units"]) for r in sources) != 9205:
        raise ValueError("Frozen candidate/group manifest drift")
    by_prior = {r["financebench_id"]: r for r in previous["records"]}
    by_base = {r["financebench_id"]: r for r in baseline}
    rows = _load_dataset(paths["dataset"])
    registry = build_registry(sources)
    records = []
    for i, source in enumerate(sources, 1):
        key = source["financebench_id"]
        base = by_base[key]["routes"]["packing_optimization_v1"]
        if base["selected_unit_ranks"] != by_prior[key]["routes"]["packing_v1"]["selected_ranks"]:
            raise ValueError("Frozen selections disagree")
        record = audit({**source, "_packing_trace": base["packing_trace"]["trace"]}, by_prior[key], rows[key], registry)
        records.append(record)
        print(f"[{i:02d}/30] {key} {record['selected_category_counts']}", flush=True)
    if hashes != {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}:
        raise AssertionError("Frozen files changed during audit")
    return {"audit": "evidence_fact_v1", "file_sha256": hashes, "categories": CATEGORIES,
            "summary": summarize(records), "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    network = {"blocked_attempts": 0, "outbound_calls": 0, "socket_guard_enabled": True}
    def denied(*args, **kwargs):
        network["blocked_attempts"] += 1
        raise RuntimeError("Network forbidden in offline fact audit")
    with patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied), patch.object(socket, "create_connection", denied):
        payload = run()
    payload["network"] = network
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for suffix, content in (("json", json.dumps(payload, ensure_ascii=False, indent=2)), ("md", markdown(payload))):
        path = args.report_dir / f"evidence_fact_audit_v1.{suffix}"
        path.write_text(content + "\n", encoding="utf-8")
        print(f"Report: {path}", flush=True)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
