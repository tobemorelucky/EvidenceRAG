"""Offline Evidence Identity Audit v1. Read frozen selection; never select again."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import socket
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.evidence_assembly_v5 import EvidenceUnit
from backend.evidence_packing_v1 import _rank, _render_selection
from backend.query_requirement_v1 import parse_query_requirement, periods, _YEAR, _QUARTER
from backend.requirement_evidence_match_v1 import _lexical_terms, _TASK_WORDS, entity_match
from scripts.audit_evidence_selection_failure_v1 import _filename, _normalized, _words, _numbers, _required_numbers, evidence_coverage
from scripts.evaluate_evidence_metadata_counterfactual_v1 import _load_dataset, _gold, _context_metrics

GROUPS = ("selection_loss10", "correct_regression10", "candidate_miss10")
CATEGORIES = {"A": "entity mismatch", "B": "period mismatch", "C": "metric mismatch",
              "D": "scope mismatch", "E": "numeric missing", "F": "reference-aligned evidence"}
# General statement/scope names, not financial metric or company mappings.
_SCOPES = {
    "balance_sheet": r"\bbalance sheets?\b",
    "income_statement": r"\bincome statements?\b|\bstatements? of (?:income|operations)\b",
    "cash_flow_statement": r"\bstatements? of cash flows?\b|\bcash flow statements?\b",
    "consolidated": r"\bconsolidated\b",
    "segment": r"\b(?:business|operating|reportable) segments?\b",
}
_STATEMENTS = {"balance_sheet", "income_statement", "cash_flow_statement"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def canonical_entity(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def profile(text):
    return {"normalized": _normalized(text), "words": _words(text), "numbers": set(_numbers(text))}


def legacy_hit(line, value):
    words = _words(line)
    return line in value["normalized"] or (len(words & value["words"]) / max(1, len(words)) >= .7 and set(_numbers(line)) <= value["numbers"])


def literal_hit(line, text):
    # Unlike legacy coverage, no token aggregation and no numeric-prefix match.
    return re.search(r"(?<!\w)" + re.escape(_normalized(line)) + r"(?!\w)", _normalized(text)) is not None


def scope_terms(text):
    return {name for name, pattern in _SCOPES.items() if re.search(pattern, str(text or ""), re.I)}


def build_registry(records):
    """Resolve reference filenames once to stable IDs using frozen Unit metadata."""
    files, entities, pages = defaultdict(set), defaultdict(set), defaultdict(set)
    for record in records:
        for u in record["candidate_units"]:
            doc, page, meta = u["document_id"], u["page_id"], u["metadata"]
            files[_filename(meta.get("filename"))].add(doc)
            if u.get("entity") and canonical_entity(u["entity"]) not in {"unknown", "none", "null"}:
                entities[doc].add(canonical_entity(u["entity"]))
            if meta.get("page_number") is not None:
                pages[(doc, int(meta["page_number"]))].add(page)
    return files, entities, pages


def question_requirement(row, registry):
    files, entities, pages = registry
    gold = _gold(row)
    refs, unresolved = [], []
    for item in gold:
        docs = files.get(item["filename"], set())
        if len(docs) != 1:
            unresolved.append({"filename": item["filename"], "reason": "missing_or_ambiguous_document_mapping"})
            continue
        doc = next(iter(docs))
        page_ids = pages.get((doc, item["page_number"]), set())
        refs.append({**item, "document_id": doc, "page_id": next(iter(page_ids)) if len(page_ids) == 1 else None})
    expected_entities = set().union(*(entities.get(r["document_id"], set()) for r in refs))
    # An internally inconsistent document cannot establish a target entity.
    entity_reliable = bool(refs) and all(len(entities.get(r["document_id"], set())) == 1 for r in refs)
    if not entity_reliable:
        expected_entities = set()
    question = row["question"]
    named = set()
    remove_terms = set()
    for entity in expected_entities:
        score, _, tokens = entity_match(question, entity)
        if score:
            named.add(entity)
            remove_terms.update(_lexical_terms(" ".join(tokens)))
    lexical = _lexical_terms(_QUARTER.sub(" ", _YEAR.sub(" ", question))) - _TASK_WORDS - remove_terms
    parsed = parse_query_requirement(question).to_dict()
    return {
        "question": question, "entity": sorted(expected_entities), "entity_named_in_question": sorted(named),
        "entity_source": "offline gold document -> frozen document_id/entity metadata; not runtime inference",
        "entity_reliable": entity_reliable,
        "metric": {"question_terms": sorted(lexical), "canonical_metric": None, "status": "lexical_descriptor_only"},
        "period": list(parsed["explicit_periods"]), "period_source": "frozen Query Requirement v1, question only",
        "scope": sorted(scope_terms(question)), "scope_source": "explicit generic statement/scope words in question",
        "required_number": _required_numbers(row),
        "required_number_source": "offline legacy justification operands / reference answer fallback; may include result, not a verified operand inventory",
        "question_explicit_numbers": _numbers(_QUARTER.sub(" ", _YEAR.sub(" ", question))),
        "parsed_requirement": parsed, "reference_documents": refs, "unresolved_reference_mappings": unresolved,
    }


def facts_from_reference(requirement):
    """Conservative eligible labelled spans. Never attach bare cells to guessed rows."""
    facts, excluded, seen = [], Counter(), set()
    terms = set(requirement["metric"]["question_terms"])
    for ref in requirement["reference_documents"]:
        for raw_line in ref["evidence_text"].splitlines():
            line = _normalized(raw_line)
            key = (ref["document_id"], ref["page_number"], line)
            if len(line) < 3 or key in seen:
                continue
            seen.add(key)
            content_words = _lexical_terms(line) - _TASK_WORDS
            numeric = _numbers(_QUARTER.sub(" ", _YEAR.sub(" ", line)))
            if len(content_words) < (1 if numeric else 2):
                excluded["bare_value_or_short_header"] += 1
                continue
            if not terms & content_words:
                excluded["no_explicit_question_term_anchor"] += 1
                continue
            if requirement["parsed_requirement"]["requires_numeric_evidence"] and not numeric:
                excluded["no_value_bound_to_label_in_same_line"] += 1
                continue
            facts.append({"fact_id": len(facts), "text": raw_line.strip(), "document_id": ref["document_id"],
                          "page_id": ref["page_id"], "page_number": ref["page_number"],
                          "anchoring_terms": sorted(terms & content_words), "numeric_tokens": numeric})
    return facts, dict(excluded)


def check_period(requested, raw, metadata):
    expected = set(requested)
    if not expected:
        return "not_requested", []
    visible = set(raw)
    if not visible:
        return "metadata_only" if set(metadata) & expected else "unknown", sorted(expected)
    for kind in (lambda x: x.isdigit(), lambda x: x.startswith("q")):
        needed = {p for p in expected if kind(p)}
        present = {p for p in visible if kind(p)}
        if needed and present and not needed & present:
            return "visible_period_conflict", sorted(expected - visible)
    if expected <= visible:
        return "visible_match", []
    return "partial" if expected & visible else "unknown", sorted(expected - visible)


def inspect_unit(unit, req, facts):
    text, meta = unit["source_text"], unit["metadata"]
    entity = canonical_entity(unit.get("entity"))
    entity_status = "unknown" if not entity or entity in {"unknown", "none", "null"} or not req["entity_reliable"] else "match" if entity in req["entity"] else "mismatch"
    raw_periods = periods(text)
    meta_periods = periods(" ".join(map(str, unit.get("period") or [])))
    period_status, missing_periods = check_period(req["period"], raw_periods, meta_periods)
    metric = unit.get("metric")
    table_row = re.search(r"(?im)^Row:\s*(.*)", text)
    descriptor = str(metric or (table_row.group(1).split("|")[0].strip() if table_row else ""))
    query_terms = set(req["metric"]["question_terms"])
    row_terms = _lexical_terms(descriptor)
    lexical_overlap = sorted(query_terms & row_terms)
    required_scope = set(req["scope"])
    raw_scope = scope_terms(text)
    stored_scope = scope_terms(" ".join(str(meta.get(k) or "") for k in ("scope", "section", "statement_type")))
    scope_status = "not_requested" if not required_scope else "visible_match" if required_scope <= raw_scope | stored_scope else "unknown"
    if required_scope & _STATEMENTS and stored_scope & _STATEMENTS and not required_scope & stored_scope & _STATEMENTS:
        scope_status = "explicit_metadata_scope_conflict"
    raw_values = set(_numbers(text))
    required = set(req["required_number"])
    anchored = [f["fact_id"] for f in facts if f["page_id"] and unit["document_id"] == f["document_id"] and unit["page_id"] == f["page_id"] and literal_hit(f["text"], text)]
    # Matching one operand's period in a comparison is partial support, not wrong.
    period_compatible = period_status in {"visible_match", "not_requested", "partial"}
    qualified = anchored if entity_status == "match" and period_compatible and scope_status in {"visible_match", "not_requested"} else []
    confirmed, suspected = [], []
    if entity_status == "mismatch":
        confirmed.append("A")
    if period_status == "visible_period_conflict":
        suspected.append("B")  # historical/implicit operands can be legitimate
    if descriptor and query_terms and not lexical_overlap:
        suspected.append("C")  # synonyms and metadata quality are unresolved
    if scope_status == "explicit_metadata_scope_conflict":
        suspected.append("D")
    if required and not required & raw_values:
        suspected.append("E")  # a narrative Unit need not contain an operand
    if qualified:
        confirmed.append("F")
        suspected = [c for c in suspected if c != "C"]
    return {
        "rank": _rank(unit), "document_id": unit["document_id"], "page_id": unit["page_id"],
        "source_type": unit["source_type"], "source_chars": len(text), "source_text": text,
        "entity": unit.get("entity"), "entity_status": entity_status,
        "metric": {"stored": metric, "row_descriptor": descriptor, "query_overlap": lexical_overlap,
                   "status": "reference_anchor" if anchored else "lexical_only" if lexical_overlap else "unverified"},
        "period": {"stored": unit.get("period"), "raw": sorted(raw_periods), "status": period_status, "missing": missing_periods},
        "scope": {"stored": sorted(stored_scope), "raw": sorted(raw_scope), "status": scope_status},
        "values": {"stored": unit.get("value"), "raw": sorted(raw_values), "required_present": sorted(required & raw_values)},
        "confirmed_categories": confirmed, "suspected_categories": suspected,
        "unverified": "F" not in confirmed and "A" not in confirmed,
        "reference_anchor_ids": anchored, "qualified_fact_ids": qualified,
    }


def audit_record(source, prior, row, registry):
    units = [{k: copy.deepcopy(u[k]) for k in (*EvidenceUnit.__dataclass_fields__, "current_ranking")} for u in source["candidate_units"]]
    fingerprint = digest(units)
    ranks = prior["routes"]["packing_v1"]["selected_ranks"]
    selected = [u for u in units if _rank(u) in set(ranks)]
    context = _render_selection(selected)[0]
    baseline = prior["routes"]["packing_v1"]["metrics"]
    if source["question"] != row["question"] or fingerprint != prior["candidate_sha256"] or len(selected) != len(ranks) or _context_metrics(row, context, selected) != baseline:
        raise ValueError("Frozen Packing v1 selection/question/candidates did not reproduce")
    req = question_requirement(row, registry)
    facts, excluded = facts_from_reference(req)
    all_audits = [inspect_unit(u, req, facts) for u in units]
    chosen = [u for u in all_audits if u["rank"] in ranks]
    selected_fact_ids = set().union(*(set(u["qualified_fact_ids"]) for u in chosen))
    candidate_fact_ids = set().union(*(set(u["qualified_fact_ids"]) for u in all_audits))
    gold = _gold(row)
    source_profile = profile("\n\n".join(u["source_text"] for u in selected))
    legacy_profile = profile(context)
    profiles = {_rank(u): profile(u["source_text"]) for u in selected}
    audit_by_rank = {u["rank"]: u for u in chosen}
    entity_context = "\n\n".join(u["source_text"] for u in chosen if u["entity_status"] == "match")
    entity_profile = profile(entity_context)
    gold_lines = sorted({ _normalized(line) for g in gold for line in g["evidence_text"].splitlines() if len(_normalized(line)) >= 3})
    collisions = []
    for line in gold_lines:
        if not legacy_hit(line, legacy_profile):
            continue
        providers = [rank for rank, p in profiles.items() if legacy_hit(line, p)]
        foreign_only = bool(providers) and all(audit_by_rank[r]["entity_status"] == "mismatch" for r in providers)
        collisions.append({"line": line, "provider_ranks": providers, "all_single_unit_providers_foreign_entity": foreign_only,
                           "identity_conflicting_coverage": foreign_only and not legacy_hit(line, entity_profile),
                           "distributed_or_rendered_only": not providers, "raw_package_hit": legacy_hit(line, source_profile)})
    expected_docs = {ref["document_id"] for ref in req["reference_documents"]}
    expected_pages = {ref["page_id"] for ref in req["reference_documents"] if ref["page_id"]}
    document_context = "\n\n".join(u["source_text"] for u in chosen if u["document_id"] in expected_docs)
    providers = {f["fact_id"]: [u["rank"] for u in all_audits if f["fact_id"] in u["qualified_fact_ids"]] for f in facts}
    missing_qualified = sorted(candidate_fact_ids - selected_fact_ids)
    baseline_trace = {v["rank"]: v for v in source.get("_packing_trace", [])}
    reference_page_candidates = [u for u in units if u["document_id"] in expected_docs and u["page_id"] in expected_pages]
    reference_page_selected = [u for u in reference_page_candidates if _rank(u) in ranks]
    if not expected_docs:
        page_stage = "reference_identity_unresolved"
    elif not reference_page_candidates:
        page_stage = "no_reference_page_unit_in_frozen_candidates"
    elif not reference_page_selected:
        page_stage = "reference_page_candidates_not_selected"
    else:
        page_stage = "reference_page_selected_fact_still_needs_validation"
    lost_details = [{"fact_id": f, "provider_ranks": providers[f],
                     "packing_decisions": [baseline_trace.get(r, {}) for r in providers[f]]} for f in missing_qualified]
    if not facts:
        stage = "unverifiable_reference_fact_structure"
    elif missing_qualified:
        stage = "qualified_candidate_fact_lost_in_frozen_selection"
    elif not candidate_fact_ids:
        stage = "no_identity_qualified_candidate_fact"
    elif len(selected_fact_ids) < len(facts):
        stage = "reference_fact_partially_verifiable_in_candidates"
    else:
        stage = "all_auditable_reference_facts_retained"
    if fingerprint != digest(units):
        raise AssertionError("Audit mutated candidates")
    return {
        "financebench_id": source["financebench_id"], "group": source["group"], "question": row["question"],
        "question_requirement": req, "candidate_sha256": fingerprint, "selected_ranks": ranks,
        "selected_units": chosen, "frozen_selection_verified": True,
        "classification": {"confirmed_unit_counts": dict(Counter(c for u in chosen for c in u["confirmed_categories"])),
                           "suspected_unit_counts": dict(Counter(c for u in chosen for c in u["suspected_categories"])),
                           "unverified_units": sum(u["unverified"] for u in chosen)},
        "coverage": {"legacy": baseline["answer_evidence_coverage"]["ratio"],
                     "raw_source": evidence_coverage(gold, "\n\n".join(u["source_text"] for u in selected))["ratio"],
                     "entity_bound": evidence_coverage(gold, entity_context)["ratio"] if req["entity_reliable"] else None,
                     "reference_document_bound": evidence_coverage(gold, document_context)["ratio"] if expected_docs else None,
                     "fact_coverage_proxy": len(selected_fact_ids) / len(facts) if facts else None,
                     "fact_denominator": len(facts), "selected_qualified_facts": len(selected_fact_ids),
                     "candidate_qualified_facts": len(candidate_fact_ids)},
        "legacy_covered_lines_audit": collisions,
        "foreign_entity_only_line_hits": sum(c["all_single_unit_providers_foreign_entity"] for c in collisions),
        "identity_conflicting_line_hits": sum(c["identity_conflicting_coverage"] for c in collisions),
        "coverage_but_fact_wrong_count": None,
        "fact_wrong_count_note": "Cannot determine semantic truth without independently verified target facts; report identity-conflicting coverage separately, never call all unverified facts false.",
        "reference_facts": facts, "excluded_reference_lines": excluded,
        "source_flow": {"stage": page_stage,
                        "candidate_reference_document_units": sum(u["document_id"] in expected_docs for u in units),
                        "selected_reference_document_units": sum(u["document_id"] in expected_docs for u in selected),
                        "candidate_reference_page_ranks": [_rank(u) for u in reference_page_candidates],
                        "selected_reference_page_ranks": [_rank(u) for u in reference_page_selected],
                        "unselected_reference_page_decisions": [baseline_trace.get(_rank(u), {}) for u in reference_page_candidates if _rank(u) not in ranks],
                        "limitation": "Source identity/selection ledger, not proof that every Unit on a reference page contains a needed fact."},
        "fact_providers": providers, "candidate_fact_loss": lost_details, "selection_loss_diagnosis": stage,
        "candidate_audit_summary": {"units": len(all_audits), "reference_anchored_units": sum(bool(u["reference_anchor_ids"]) for u in all_audits),
                                    "identity_qualified_units": sum(bool(u["qualified_fact_ids"]) for u in all_audits)},
    }


def summarize(records):
    def part(items):
        coverage = {field: (fmean(values) if values else None) for field in ("legacy", "raw_source", "entity_bound", "reference_document_bound", "fact_coverage_proxy")
                    for values in [[r["coverage"][field] for r in items if r["coverage"][field] is not None]]}
        eligible = [r for r in items if r["coverage"]["fact_denominator"]]
        return {
            "questions": len(items), "selected_units": sum(len(r["selected_units"]) for r in items), "coverage_means": coverage,
            "fact_eligible_questions": len(eligible), "fact_denominator": sum(r["coverage"]["fact_denominator"] for r in items),
            "legacy_mean_on_fact_eligible_subset": fmean(r["coverage"]["legacy"] for r in eligible) if eligible else None,
            "confirmed_unit_counts": dict(Counter(c for r in items for u in r["selected_units"] for c in u["confirmed_categories"])),
            "suspected_unit_counts": dict(Counter(c for r in items for u in r["selected_units"] for c in u["suspected_categories"])),
            "unverified_units": sum(r["classification"]["unverified_units"] for r in items),
            "foreign_entity_only_line_hits": sum(r["foreign_entity_only_line_hits"] for r in items),
            "questions_with_foreign_entity_only_hits": [r["financebench_id"] for r in items if r["foreign_entity_only_line_hits"]],
            "identity_conflicting_line_hits": sum(r["identity_conflicting_line_hits"] for r in items),
            "questions_with_identity_conflicting_coverage": [r["financebench_id"] for r in items if r["identity_conflicting_line_hits"]],
            "selection_loss_stages": dict(Counter(r["selection_loss_diagnosis"] for r in items)),
            "source_flow_stages": dict(Counter(r["source_flow"]["stage"] for r in items)),
            "coverage_but_semantically_wrong_questions": None,
        }
    result = part(records)
    result["groups"] = {g: part([r for r in records if r["group"] == g]) for g in GROUPS}
    result["candidate_units"] = sum(r["candidate_audit_summary"]["units"] for r in records)
    result["frozen_selection_verified"] = all(r["frozen_selection_verified"] for r in records)
    result["recommendation"] = "Evidence Identity and reference-fact validation first; only investigate packing for qualified candidate facts demonstrably dropped. No changes in this audit."
    return result


def markdown(payload):
    s = payload["summary"]
    def pct(v):
        return "N/A" if v is None else f"{v:.2%}"
    lines = ["# Evidence Identity Audit v1", "", "## 结论边界与分类契约", "",
        "仅回放冻结Packing v1已选Unit，不重新选择/检索/排序。全量9205个candidate只用于核查已有合格证据是否被丢弃。", "",
        "- A：相对gold文档所关联的冻结entity元数据不匹配（身份冲突，不等于模型答案错误）。",
        "- B：可见期间冲突；C：已有row/metric与问题词不匹配；D：显式statement scope metadata冲突；E：参考数字没有出现在该Unit。B–E仅为疑点，历史操作数、同义词、scope元数据错误和叙述性证据可能合法。",
        "- F：同参考document_id/page_id的原文完整包含带问题词锚点的参考事实行，且entity/可见period/scope约束兼容。它是保守reference-aligned support，不是答案正确/完整可回答性；也不是独立人工语义标注。",
        "- 未满足F又未确认A时标unverified，不能强塞进F或认作错。多类别不互斥，不能相加作为错误Unit总数。",
        "- Question的entity以gold文档和冻结metadata为离线身份参照，另记问题中是否显式提及。别名不强行转换；映射歧义标unknown。",
        "- metric只保留问题词描述，不新增指标映射；scope仅显式通用报表/聚合范围；required number沿用参考justification/answer来源，可能包含最终结果，并非验证过的操作数清单。",
        "- 参考事实仅取同一gold行内有问题词锚点的片段：数值行至少有一个内容词标签，纯文本行至少有两个内容词；numeric任务要求同一行有数值。裸数字、无法绑定表头的单元格不猜测组合，列入未可审计。",
        "- 期间partial可支持多期题的一个操作数；metadata-only不能升级为F；即使F也不验证数值与列/年份的完整会计绑定。", "",
        "## 汇总", "", "| Group | Units | Legacy coverage | Entity-bound | Reference-document-bound | Fact proxy | Fact eligible questions |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("overall", *GROUPS):
        v = s if name == "overall" else s["groups"][name]
        m = v["coverage_means"]
        lines.append(f"| {name} | {v['selected_units']} | {pct(m['legacy'])} | {pct(m['entity_bound'])} | {pct(m['reference_document_bound'])} | {pct(m['fact_coverage_proxy'])} | {v['fact_eligible_questions']}/{v['questions']} |")
    lines += ["", f"- 确认分类：`{s['confirmed_unit_counts']}`；疑似分类：`{s['suspected_unit_counts']}`；unverified={s['unverified_units']}。",
        f"- 原coverage命中且所有可定位单Unit来源均为其他entity的行数：{s['foreign_entity_only_line_hits']}；涉及{len(s['questions_with_foreign_entity_only_hits'])}/30题。它证明身份约束漏洞，不等于这些问题答案错误。",
        f"- 进一步排除同entity多个Unit共同支持的情况，身份冲突覆盖为{s['identity_conflicting_line_hits']}行、{len(s['questions_with_identity_conflicting_coverage'])}题（同entity完整package也无法满足原行覆盖规则）。这是更严格的身份冲突统计，仍不是语义错误答案数。",
        "- ‘覆盖但语义事实错误’精确数量保持null：没有独立target-fact真值，不能把unknown算错误。上面的跨entity行数是可观察的身份冲突统计。",
        f"- Fact proxy只在{s['fact_eligible_questions']}/30题可审计，共{s['fact_denominator']}条片段；同一可审计子集legacy均值为{pct(s['legacy_mean_on_fact_eligible_subset'])}。Fact与全gold-line口径/分母不同，不能直接作同口径准确率下降。",
        "- Entity/document-bound仍沿用原词/数值覆盖代理；fact proxy则要求单Unit同源完整原文和可见约束。两者都可能漏掉合法替代证据，不能用作新的唯一优化目标。", "",
        "## Selection-loss 10题定位", "", "| ID | Source flow | Reference page units candidate/selected | Fact audit stage | Qualified facts candidate/selected | A mismatch units |",
        "|---|---|---|---|---:|---:|"]
    for r in payload["records"]:
        if r["group"] == "selection_loss10":
            flow = r["source_flow"]
            lines.append(f"| {r['financebench_id']} | {flow['stage']} | {len(flow['candidate_reference_page_ranks'])}/{len(flow['selected_reference_page_ranks'])} | {r['selection_loss_diagnosis']} | {r['coverage']['candidate_qualified_facts']}/{r['coverage']['selected_qualified_facts']} | {r['classification']['confirmed_unit_counts'].get('A', 0)} |")
    lines += ["", "只有qualified candidate facts存在且未被选择，才能定位到冻结选择阶段；JSON保留其provider ranks及原Packing拒绝/替换原因。该证据仍不能单独区分ranking和packing的因果贡献。其余情况是身份/事实结构未验证，不直接叫retrieval miss。", "",
        "## 下一阶段优先级", "", "先建立可核查的Evidence Identity及目标事实评测口径；随后在身份可验证的样本上评价Requirement Matching；Packing只针对有明确合格provider但丢失的案例分析。不据本轮coverage继续调参数。", "",
        "## 每题与已选Unit", ""]
    for r in payload["records"]:
        q = r["question_requirement"]
        lines += [f"### {r['financebench_id']} — {r['group']}", "", r["question"], "",
            f"- Requirement entity={q['entity']}；metric词={q['metric']['question_terms']}；period={q['period']}；scope={q['scope']}；参考numeric tokens={q['required_number']}。",
            f"- Coverage：`{r['coverage']}`；定位：`{r['selection_loss_diagnosis']}`。",
            f"- 确认/疑似分类：`{r['classification']}`；跨entity单Unit碰撞行={r['foreign_entity_only_line_hits']}。", "",
            "| Rank | Entity | Period status | Metric descriptor | Scope status | A/F | B–E suspected |",
            "|---:|---|---|---|---|---|---|"]
        for u in r["selected_units"]:
            desc = u["metric"]["row_descriptor"].replace("|", " / ").replace("\n", " ")[:90]
            lines.append(f"| {u['rank']} | {u['entity']} | {u['period']['status']} | {desc or 'unknown'} | {u['scope']['status']} | {u['confirmed_categories']} | {u['suspected_categories']} |")
        examples = [x for x in r["legacy_covered_lines_audit"] if x["all_single_unit_providers_foreign_entity"]][:3]
        lines += ["", f"跨entity覆盖示例：`{examples}`。完整source text、values、ID、fact providers及决策见JSON。", ""]
    lines += ["## 执行验证", "", f"- 冻结选择核验：{s['frozen_selection_verified']}；候选{s['candidate_units']}；网络：`{payload['network']}`。",
              "- 不修改生产、Packing、Ranking、Requirement score、Selector；无模型/Jina/Judge/LangSmith调用；未跑100题。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    paths = {"candidates": ROOT / "reports/evidence_metadata_counterfactual_v1.json",
             "baseline": ROOT / "reports/evidence_packing_optimization_v1.json", "previous": ROOT / "reports/query_requirement_shadow_v1.json",
             "dataset": ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv",
             "packing": ROOT / "backend/evidence_packing_v1.py", "requirements": ROOT / "backend/query_requirement_v1.py",
             "assembly": ROOT / "backend/evidence_assembly_v5.py", "matcher": ROOT / "backend/requirement_evidence_match_v1.py"}
    hashes = {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}
    sources = json.loads(paths["candidates"].read_text(encoding="utf-8"))["records"]
    previous = json.loads(paths["previous"].read_text(encoding="utf-8"))
    baseline = json.loads(paths["baseline"].read_text(encoding="utf-8"))["records"]
    ids = {r["financebench_id"] for r in sources}
    if len(sources) != 30 or len(ids) != 30 or len(previous["records"]) != 30 or len(baseline) != 30 or ids != {r["financebench_id"] for r in previous["records"]} or ids != {r["financebench_id"] for r in baseline}:
        raise ValueError("Exactly the frozen matching diagnostic30 required")
    if Counter(r["group"] for r in sources) != Counter({g: 10 for g in GROUPS}) or sum(len(r["candidate_units"]) for r in sources) != 9205:
        raise ValueError("Frozen group/candidate manifest drift")
    for key in ("candidates", "baseline", "dataset", "packing", "requirements", "assembly"):
        if hashes[key] != previous["file_sha256"][key]:
            raise ValueError(f"Frozen input changed: {key}")
    by_prior = {r["financebench_id"]: r for r in previous["records"]}
    by_base = {r["financebench_id"]: r for r in baseline}
    rows = _load_dataset(paths["dataset"])
    registry = build_registry(sources)
    network = {"blocked_attempts": 0, "outbound_calls": 0, "socket_guard_enabled": True}
    def denied(*args, **kwargs):
        network["blocked_attempts"] += 1
        raise RuntimeError("Network forbidden in offline identity audit")
    output = []
    with patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied), patch.object(socket, "create_connection", denied):
        for i, source in enumerate(sources, 1):
            key = source["financebench_id"]
            base = by_base[key]["routes"]["packing_optimization_v1"]
            if base["selected_unit_ranks"] != by_prior[key]["routes"]["packing_v1"]["selected_ranks"]:
                raise ValueError("Packing archives disagree")
            source = {**source, "_packing_trace": base["packing_trace"]["trace"]}
            record = audit_record(source, by_prior[key], rows[key], registry)
            output.append(record)
            print(f"[{i:02d}/30] {key} categories={record['classification']['confirmed_unit_counts']} coverage={record['coverage']}", flush=True)
    if hashes != {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in paths.items()}:
        raise AssertionError("Frozen source changed during audit")
    payload = {"audit": "evidence_identity_v1", "file_sha256": hashes, "categories": CATEGORIES,
               "network": network, "summary": summarize(output), "records": output}
    args.report_dir.mkdir(parents=True, exist_ok=True)
    for suffix, content in (("json", json.dumps(payload, ensure_ascii=False, indent=2)), ("md", markdown(payload))):
        path = args.report_dir / f"evidence_identity_audit_v1.{suffix}"
        path.write_text(content + "\n", encoding="utf-8")
        print(f"Report: {path}", flush=True)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
