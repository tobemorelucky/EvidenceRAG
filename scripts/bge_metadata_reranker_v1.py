"""One fixed offline metadata rerank; no model calls or production imports.

Metadata compatibility is a surface/co-occurrence proxy, NOT verified facts.
Only question, source text, company, report_year and optional headings are used.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from scripts.shadow_rerankers_v1 import validate_order

VERSION = "bge_metadata_v1"
BGE_WEIGHT = 0.75
METADATA_WEIGHT = 0.25
LEGAL = frozenset("inc incorporated corp corporation ltd limited plc co company".split())
STOP = frozenset("a an the and or of for to in on at by with from is are was were be been what which how did does do as it its that this than during would could should much many calculate compute compare explain why please state question answer based data if then not useful relevant meaningful metric company like roughly only use details shown fiscal year years period fy end between have has had".split())
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
SHORT_FY = re.compile(r"\bfy\s*['’]?\s*(\d{2})(?!\d)", re.I)


def words(text):
    text = re.sub(r"(?<=\w)['’]s\b", "", str(text or ""), flags=re.I)
    return re.findall(r"[^\W_]+", text.casefold())


def years(text):
    value = str(text or "")
    result = {int(x) for x in YEAR.findall(value)}
    for short in SHORT_FY.findall(value):
        n = int(short)
        result.add(2000 + n if n < 70 else 1900 + n)
    return result


def company_key(value):
    return "".join(w for w in words(value) if w not in LEGAL and w != "and")


def resolve_entities(question, chunks):
    # Candidate vocabulary only; no external company list/acronym expansion.
    query_words = [w for w in words(question) if w not in LEGAL and w != "and"]
    targets, matched_words = set(), set()
    names = {str(c.get("company") or "") for c in chunks}
    for name in sorted(names):
        key = company_key(name)
        if key in {"", "unknown", "none", "null"}:
            continue
        for start in range(len(query_words)):
            joined = ""
            for end in range(start, len(query_words)):
                joined += query_words[end]
                if joined == key:
                    targets.add(key)
                    matched_words.update(query_words[start:end + 1])
                if len(joined) >= len(key):
                    break
    return targets, matched_words


def metric_terms(text):
    return set(words(SHORT_FY.sub(" ", YEAR.sub(" ", str(text or ""))))) - STOP - {"s"}


def rerank(question, chunks, bge_ranked):
    """Return full permutation + per-candidate audit. Gold/IDs are never read."""
    if not question.strip() or not chunks:
        raise ValueError("Question and candidates required")
    base = validate_order(bge_ranked, len(chunks))
    targets, entity_words = resolve_entities(question, chunks)
    required_years = years(question)
    target_terms = metric_terms(question) - entity_words
    candidate_terms = [metric_terms(c.get("text", "")) for c in chunks]
    df = Counter(t for ts in candidate_terms for t in ts)
    weights = {t: 1 + math.log((len(chunks) + 1) / (df[t] + 1)) for t in target_terms}
    denominator = sum(weights.values())
    bge_by_index = {v["index"]: (rank, float(v["score"])) for rank, v in enumerate(base, 1)}
    trace, scored = [], []
    for index, chunk in enumerate(chunks):
        text = str(chunk.get("text") or "")
        fragments = [f for f in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", text) if f.strip()]
        heading = " ".join(str(chunk.get(k) or "") for k in ("table_title", "section"))
        if heading.strip():
            fragments.append(heading)
        local_scores = [sum(weights[t] for t in metric_terms(f) & target_terms) / denominator if denominator else 0.0 for f in fragments]
        best = max(range(len(fragments)), key=lambda i: (local_scores[i], -i), default=None)
        lexical = local_scores[best] if best is not None else 0.0
        # Periods from a matched local sentence/row plus its adjacent rows are
        # stronger than document metadata. No row/column binding is claimed.
        relevant_years = set()
        if lexical > 0:
            for i, score in enumerate(local_scores):
                if score == lexical:
                    for fragment in fragments[max(0, i - 1):i + 2]:
                        relevant_years.update(years(fragment))
        text_years = years(text)
        report_years = years(chunk.get("report_year"))
        def fraction(values):
            return len(required_years & values) / len(required_years) if required_years else 0.0
        local_fraction = fraction(relevant_years)
        period_score = max(local_fraction, 0.5 * fraction(text_years), 0.25 * fraction(report_years))
        if local_fraction:
            period_status = "local_cooccurrence"
        elif fraction(text_years):
            period_status = "elsewhere_in_chunk"
        elif fraction(report_years):
            period_status = "report_year_only_weak"
        else:
            period_status = "unknown_or_not_visible"
        key = company_key(chunk.get("company"))
        if not targets:
            entity_score, entity_status = 0.0, "query_entity_unresolved"
        elif key in {"", "none", "unknown", "null"}:
            entity_score, entity_status = 0.0, "candidate_entity_missing"
        elif key in targets:
            entity_score, entity_status = 1.0, "surface_match"
        else:
            entity_score, entity_status = -1.0, "different_resolved_company"
        active = {}
        if targets:
            active["entity"] = entity_score
        if required_years:
            active["period"] = period_score
        if target_terms:
            active["metric_lexical_proxy"] = lexical
        compatibility = sum(active.values()) / len(active) if active else 0.0
        base_rank, logit = bge_by_index[index]
        percentile = (len(chunks) - base_rank) / max(1, len(chunks) - 1)
        score = BGE_WEIGHT * percentile + METADATA_WEIGHT * compatibility
        scored.append({"index": index, "score": score})
        trace.append({"index": index, "bge_rank": base_rank, "bge_logit": logit,
                      "bge_rank_percentile": percentile, "final_score": score,
                      "metadata_compatibility": compatibility, "active_features": active,
                      "company": chunk.get("company"), "entity_status": entity_status,
                      "period_status": period_status, "local_years": sorted(relevant_years),
                      "text_years": sorted(text_years), "report_years": sorted(report_years),
                      "best_fragment_index": best, "best_fragment_preview": fragments[best][:400] if best is not None else "",
                      "metric_relevance": lexical})
    # Match the existing shadow evaluator's exact-score tie rule: original RRF
    # index. Keep real scores, rather than perturbing floats to break ties.
    scored.sort(key=lambda v: (-v["score"], v["index"]))
    return validate_order(scored, len(chunks)), {
        "version": VERSION, "weights": {"bge": BGE_WEIGHT, "metadata": METADATA_WEIGHT},
        "formula": "0.75 * BGE_rank_percentile + 0.25 * mean(active_metadata_features)",
        "tie_break": "original_RRF_index", "resolved_query_entities": sorted(targets),
        "required_years": sorted(required_years), "metric_query_terms": sorted(target_terms),
        "units": trace, "model_calls": 0,
        "limitations": "Entity surface matching and metric/year co-occurrence only; unknown is not contradiction; no verified entity/period/operand binding",
    }
