"""Deterministic question-intent and frozen-context alignment for shadow audits."""

from __future__ import annotations

import re
from typing import Any


_SOURCE_RE = re.compile(r"(?m)^Source:\s*(.+?)\s*\|\s*Page:\s*(\d+)\s*$")
_YEAR_RE = re.compile(r"\b(?:FY\s*)?((?:19|20)\d{2})\b", re.IGNORECASE)
_SHORT_FY_RE = re.compile(r"\bFY\s*['’]?(\d{2})\b", re.IGNORECASE)
_QUARTER_YEAR_RE = re.compile(r"(?<![A-Za-z0-9])Q([1-4])(?:\s+of)?[_\s-]*(?:FY[_\s-]*)?['’]?(\d{2,4})(?!\d)", re.IGNORECASE)
_YEAR_QUARTER_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})[_\s-]*Q([1-4])(?!\d)", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9&'-]*")
_STOP = {
    "a", "an", "and", "are", "as", "at", "based", "be", "between", "by", "calculate",
    "company", "did", "do", "does", "for", "from", "has", "have", "how", "if", "in", "is",
    "it", "its", "many", "of", "on", "or", "please", "roughly", "state", "than", "that", "the",
    "then", "this", "to", "using", "was", "were", "what", "when", "which", "with", "year",
}


_CONCEPTS = (
    ("wages expense as percent of sales", re.compile(r"\b(?:wages expense|store payroll|payroll and benefits)\b", re.I), ("wages expense", "store payroll", "payroll and benefits", "wage investments")),
    ("interest coverage", re.compile(r"\binterest coverage(?: ratio)?\b", re.I), ("interest coverage", "adjusted ebit", "operating income", "interest expense")),
    ("gross margin", re.compile(r"\bgross margin\b", re.I), ("gross margin", "gross profit", "cost of products", "cost of sales", "total revenues")),
    ("operating cash flow ratio", re.compile(r"\boperating cash flow ratio\b", re.I), ("operating cash flow ratio", "cash from operations", "net cash provided by operating activities", "current liabilities")),
    ("capital intensity", re.compile(r"\bcapital[- ]intensive|capital intensity\b", re.I), ("capital-intensive", "capital intensity", "return on assets", "roa", "fixed assets", "property plant and equipment", "total assets")),
    ("cash flow activity", re.compile(r"\boperations\b.*\binvesting\b.*\bfinancing\b", re.I | re.S), ("operating activities", "investing activities", "financing activities", "net cash provided", "net cash used")),
    ("inventory turnover", re.compile(r"\binventory turnover\b|\bsold (?:its|the) inventory\b", re.I), ("inventory turnover", "inventory", "inventories", "cost of goods sold", "cost of sales", "cost of products sold")),
    ("EPS growth", re.compile(r"^(?=.*\beps\b)(?=.*\b(?:growth|accelerat\w*|decelerat\w*)\b)", re.I | re.S), ("adjusted eps", "earnings per share", "eps", "growth", "accelerate", "decelerate")),
    ("restructuring liability", re.compile(r"\brestructuring liabilit(?:y|ies)\b", re.I), ("restructuring liability", "restructuring liabilities", "employee liabilities", "restructuring")),
    ("store count", re.compile(r"\b(?:number of|total|count of)\s+[^?.]{0,30}\bstores?\b|\bstore count\b", re.I), ("number of stores", "store count", "total stores", "stores")),
    ("inventory balance drivers", re.compile(r"\b(?:merchandise inventories|inventory balance)\b", re.I), ("merchandise inventories", "inventory balance", "new stores", "inventory")),
    ("working capital", re.compile(r"\bworking capital\b", re.I), ("working capital", "current assets", "current liabilities")),
    ("acquisitions", re.compile(r"\b(?:companies acquired|acquired|acquisition)\b", re.I), ("companies acquired", "acquired", "acquisition", "business combination")),
    ("business separation", re.compile(r"\b(?:spinning off|spin[- ]?off|spinoff)\b", re.I), ("spinning off", "spin-off", "spinoff", "separation", "business segment")),
    ("quick ratio", re.compile(r"\bquick ratio\b", re.I), ("quick ratio", "cash equivalents", "receivables", "current liabilities")),
    ("current ratio", re.compile(r"\bcurrent ratio\b", re.I), ("current ratio", "current assets", "current liabilities")),
    ("operating margin", re.compile(r"\boperating margin\b", re.I), ("operating margin", "operating income", "revenue")),
    ("revenue growth", re.compile(r"^(?=.*\b(?:revenue|net sales)\b)(?=.*\b(?:growth|grew|increase|decrease)\b)", re.I | re.S), ("revenue growth", "net sales growth", "revenue", "net sales")),
)


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _words(value: str) -> set[str]:
    return {word.casefold().strip("'-") for word in _WORD_RE.findall(str(value or "")) if word.casefold() not in _STOP}


def _period_values(value: str) -> list[str]:
    text = str(value or "")
    quarter_spans = []
    periods = []
    for match in _QUARTER_YEAR_RE.finditer(text):
        quarter, year = match.groups()
        normalized_year = str(2000 + int(year)) if len(year) == 2 else year
        periods.append(f"{normalized_year}Q{quarter}")
        quarter_spans.append(match.span())
    for match in _YEAR_QUARTER_RE.finditer(text):
        year, quarter = match.groups()
        periods.append(f"{year}Q{quarter}")
        quarter_spans.append(match.span())
    masked = list(text)
    for start, end in quarter_spans:
        masked[start:end] = " " * (end - start)
    remaining = "".join(masked)
    periods.extend(match.group(1) for match in _YEAR_RE.finditer(remaining))
    periods.extend(str(2000 + int(match.group(1))) for match in _SHORT_FY_RE.finditer(remaining))
    return list(dict.fromkeys(periods))


def _entity_candidates(question: str) -> list[dict[str, Any]]:
    patterns = (
        (r"\b([A-Z][A-Za-z0-9&.]*?(?:\s+[A-Z][A-Za-z0-9&.]*){0,2})['’]s\b", "possessive"),
        (r"^(?:Does|Did|Is|Was|Were|Has|Have)\s+([A-Z][A-Za-z0-9&.]*(?:\s+[A-Z][A-Za-z0-9&.]*){0,2})\b", "subject"),
        (r"\bfor\s+([A-Z][A-Za-z0-9&.]*(?:\s+[A-Z][A-Za-z0-9&.]*){0,2})(?:\?|\s+in\b|\s+as\b)", "for_clause"),
        (r"\bhas\s+([A-Z][A-Za-z0-9&.]*(?:\s+[A-Z][A-Za-z0-9&.]*){0,2})\s+(?:sold|reported)\b", "verb_subject"),
        (r"\bby\s+([A-Z][A-Za-z0-9&.]*(?:\s+[A-Z][A-Za-z0-9&.]*){0,2})(?:\s+mentioned|\s+in\b|\?|$)", "by_clause"),
        (r",\s*(?:is|does|did|was|were)\s+([A-Z][A-Za-z0-9&.]*(?:\s+[A-Z][A-Za-z0-9&.]*){0,2})\b", "post_temporal_subject"),
        (r"\bnumber of\s+([A-Z][A-Za-z0-9&.]*(?:\s+[A-Z][A-Za-z0-9&.]*){0,2})\s+stores?\b", "count_subject"),
    )
    for pattern, source in patterns:
        match = re.search(pattern, str(question or ""))
        if match:
            value = re.sub(r"^(?:Did|Does|Is|Was|Were|Has|Have)\s+", "", match.group(1).strip())
            return [{"value": value, "source": source, "confidence": 0.9}]
    return []


def _metric_candidates(question: str) -> list[dict[str, Any]]:
    lowered = str(question or "").casefold()
    matched = []
    for concept, pattern, aliases in _CONCEPTS:
        if pattern.search(lowered):
            matched.append({"value": concept, "aliases": list(aliases), "source": "concept_lexicon", "confidence": 0.9})
    if matched:
        return matched[:1]
    content = [word for word in _WORD_RE.findall(question) if word.casefold() not in _STOP]
    fallback = " ".join(content[-4:]).casefold()
    return [{"value": fallback, "aliases": [fallback], "source": "question_fallback", "confidence": 0.45}] if fallback else []


def _calculation_type(question: str) -> str:
    lowered = str(question or "").casefold()
    if re.search(r"\b(?:calculate|ratio|turnover|how many times|defined as)\b", lowered):
        return "calculation"
    if re.search(r"\b(?:among|which brought|three main|most|least)\b", lowered):
        return "selection"
    if re.search(r"\b(?:increase|decrease|change|improv|accelerat|decelerat|between)\b", lowered):
        return "comparison"
    if re.search(r"^(?:is|does|did|was|were)\b|\b(?:useful|relevant|capital-intensive)\b", lowered):
        return "judgment"
    return "lookup"


def extract_question_intent_v1(question: str) -> dict[str, Any]:
    return {
        "entity_candidates": _entity_candidates(question),
        "period_candidates": [{"value": value, "source": "question", "confidence": 1.0} for value in _period_values(question)],
        "metric_candidates": _metric_candidates(question),
        "calculation_type": _calculation_type(question),
    }


def build_frozen_context_chunks_v1(evidence: str, page_metadata: list[dict] | None = None) -> list[dict[str, Any]]:
    metadata: dict[tuple[str, int], list[dict]] = {}
    for item in page_metadata or []:
        metadata.setdefault((str(item.get("filename") or ""), int(item.get("page_number") or 0)), []).append(item)
    matches = list(_SOURCE_RE.finditer(str(evidence or "")))
    chunks = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(evidence)
        document, page = match.group(1).strip(), int(match.group(2))
        candidates = metadata.get((document, page), [])
        item = candidates.pop(0) if candidates else {}
        chunks.append({
            "chunk_id": str(item.get("chunk_id") or f"{document}::p{page}::context::{index}"),
            "document": document, "page": page, "company": str(item.get("company") or ""),
            "report_year": item.get("report_year"), "text": evidence[match.end() : end].strip(),
        })
    return chunks


def _entity_match(intent: dict, chunk: dict) -> tuple[bool, dict]:
    candidates = intent["entity_candidates"]
    if not candidates:
        return True, {"required": False, "matched": None}
    target = _compact(candidates[0]["value"])
    company = _compact(chunk.get("company") or "")
    matched = bool(target and company and (target in company or company in target))
    if not matched and target and len(target) <= 5 and company:
        iterator = iter(company)
        matched = all(character in iterator for character in target)
    if not matched:
        matched = bool(target and target in _compact(chunk.get("text", "")[:1200]))
    return matched, {"required": True, "target": candidates[0]["value"], "observed_company": chunk.get("company")}


def _period_match(intent: dict, chunk: dict) -> tuple[bool, dict]:
    required = {item["value"] for item in intent["period_candidates"]}
    if not required:
        return True, {"required": False, "matched": []}
    observed = set(_period_values(chunk.get("text", ""))) | set(_period_values(chunk.get("document", "")))
    report_year = chunk.get("report_year")
    if report_year:
        observed.add(str(report_year))
    matched = sorted(required & observed)
    return bool(matched), {"required": True, "targets": sorted(required), "observed": sorted(observed), "matched": matched}


def _phrase_match(text: str, phrase: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", text, re.IGNORECASE))


def _metric_match(intent: dict, chunk: dict) -> tuple[bool, dict]:
    text = str(chunk.get("text") or "")
    matches = []
    best_overlap = 0.0
    text_words = _words(text)
    for metric in intent["metric_candidates"]:
        aliases = [metric["value"], *metric.get("aliases", [])]
        phrase_hits = [alias for alias in aliases if _phrase_match(text, alias)]
        metric_words = _words(metric["value"])
        overlap = len(metric_words & text_words) / max(1, len(metric_words))
        best_overlap = max(best_overlap, overlap)
        if phrase_hits or overlap >= 0.6:
            matches.append({"metric": metric["value"], "aliases": phrase_hits, "overlap": round(overlap, 4)})
    return bool(matches), {"matches": matches, "best_overlap": round(best_overlap, 4)}


def align_context_chunk_v1(intent: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    entity_match, entity_trace = _entity_match(intent, chunk)
    period_match, period_trace = _period_match(intent, chunk)
    metric_match, metric_trace = _metric_match(intent, chunk)
    components = ((entity_match, 0.35), (period_match, 0.25), (metric_match, 0.40))
    score = sum(weight for matched, weight in components if matched)
    return {
        "chunk_id": chunk["chunk_id"], "document": chunk["document"], "page": chunk["page"],
        "entity_match": entity_match, "period_match": period_match, "metric_match": metric_match,
        "alignment_score": round(score, 4), "aligned": entity_match and period_match and metric_match,
        "trace": {"entity": entity_trace, "period": period_trace, "metric": metric_trace},
    }


def classify_context_alignment_v1(intent: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    alignments = [align_context_chunk_v1(intent, chunk) for chunk in chunks]
    aligned = [item for item in alignments if item["aligned"]]
    metric_related = [item for item in alignments if item["metric_match"]]
    if aligned:
        category = "C_evidence_aligned_but_reasoning_failure"
        reason = "At least one frozen context chunk matches entity, period, and metric intent."
    elif metric_related:
        category = "B_evidence_present_but_misaligned"
        reason = "Metric-related evidence exists, but no chunk also matches the required entity and period."
    else:
        category = "A_evidence_absent"
        reason = "No frozen context chunk matches the extracted metric intent."
    return {
        "aligned_evidence_present": bool(aligned), "classification": category, "reason": reason,
        "aligned_chunk_ids": [item["chunk_id"] for item in aligned], "chunk_alignments": alignments,
    }
