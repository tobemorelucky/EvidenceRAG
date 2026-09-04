"""Gold-free answerability proxies for offline Evidence Unit ranking only.

No metric dictionary, company rules, model calls, or production integration.
Scores describe observable support; they do not certify answerability.
"""

from __future__ import annotations

import re

from backend.evidence_packing_v1 import _rank, _terms

WEIGHTS = {
    "query_lexical_relevance": 0.35,
    "numeric_presence": 0.10,
    "period_consistency": 0.15,
    "entity_consistency": 0.10,
    "answer_type_compatibility": 0.15,
    "evidence_completeness": 0.15,
}
YEAR_RE = re.compile(r"(?<![A-Za-z0-9])(?:FY\s*[-/]?\s*)?((?:19|20)\d{2})(?!\d)", re.I)
NUMBER_RE = re.compile(r"(?<![\w])[-+]?\d[\d,]*(?:\.\d+)?%?")
TASK_WORDS = {
    "calculate", "compute", "compare", "comparison", "explain", "identify",
    "list", "much", "many", "percentage", "percent", "change", "difference",
    "highest", "lowest", "largest", "smallest", "most", "least", "between",
}


def answer_type(query: str) -> str:
    """Generic task-language detection, not financial metric classification."""
    words = _terms(query)
    if words & {"compare", "comparison", "difference", "change", "higher", "lower", "increase", "decrease"}:
        return "comparison"
    if words & {"highest", "lowest", "largest", "smallest", "most", "least"} or re.search(r"\bwhich\b", query, re.I):
        return "selection"
    if re.search(r"\bwhy\b|\bexplain\b", query, re.I):
        return "explanation"
    if re.search(r"\bhow (?:much|many)\b|\b(?:calculate|compute|percentage|percent|amount|ratio|rate)\b", query, re.I):
        return "numeric"
    return "lookup"


def numbers(text: str) -> set[str]:
    values = set()
    for match in NUMBER_RE.finditer(text):
        value = match.group().replace(",", "").rstrip(".")
        if not YEAR_RE.fullmatch(value):
            values.add(value)
    return values


def _fragments(text: str) -> list[str]:
    # Preserve decimal numbers, table rows, and original answer evidence.
    return [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z])", text) if part.strip()]


def score_answerability(query: str, unit: dict) -> dict:
    """Read existing metadata and text only. Unknown is distinct from conflict."""
    text = str(unit.get("source_text") or "")
    entity_words = _terms(unit.get("entity"))
    query_words = _terms(YEAR_RE.sub(" ", query))
    # Matching the document's entity alone is not evidence of metric support.
    target_words = query_words - entity_words - TASK_WORDS
    if not target_words:
        target_words = query_words - TASK_WORDS
    parts = _fragments(text)
    overlaps = [len(_terms(part) & target_words) / max(1, len(target_words)) for part in parts]
    global_overlap = len(_terms(text) & target_words) / max(1, len(target_words))
    local_overlap = max(overlaps, default=0.0)
    lexical = (global_overlap + local_overlap) / 2
    local_numeric = max((overlap for part, overlap in zip(parts, overlaps) if numbers(part)), default=0.0)
    local_pair = max((overlap for part, overlap in zip(parts, overlaps) if len(numbers(part)) >= 2), default=0.0)

    query_years = set(YEAR_RE.findall(query))
    raw_years = set(YEAR_RE.findall(text))
    meta_periods = unit.get("period") or []
    meta_years = set(YEAR_RE.findall(" ".join(map(str, meta_periods)) if isinstance(meta_periods, list) else str(meta_periods)))
    if not query_years:
        period_score, period_status = 0.5, "not_requested"
    elif raw_years:
        period_score = len(query_years & raw_years) / len(query_years)
        period_status = "text_match" if period_score == 1 else "text_partial" if period_score else "text_mismatch"
    elif meta_years:
        # Metadata-only dates are hints, not verified row-period alignment.
        period_score = 0.5
        period_status = "metadata_only_match" if query_years & meta_years else "metadata_only_mismatch"
    else:
        period_score, period_status = 0.5, "unknown"

    if entity_words and entity_words <= query_words:
        entity_score, entity_status = (1.0, "query_metadata_text_match") if entity_words <= _terms(text) else (0.5, "query_metadata_only")
    else:
        entity_score, entity_status = 0.5, "unspecified_or_unresolved"

    task = answer_type(query)
    table = unit.get("source_type") == "table"
    header = bool(re.search(r"(?im)^\s*(?:header|columns?)\s*:", text))
    row = bool(re.search(r"(?im)^\s*(?:row|selected rows?|target rows?)\s*:", text)) or bool(unit.get("metric") and unit.get("value"))
    narrative = any(len(_terms(part)) >= 6 for part in parts)
    if task == "numeric":
        compatibility = local_numeric
    elif task == "comparison":
        compatibility = local_pair * (period_score if len(query_years) > 1 else 1.0)
    elif task == "selection":
        compatibility = local_overlap * ((float(header and row) + float(narrative)) / 2 if table else float(narrative))
    elif task == "explanation":
        compatibility = local_overlap * float(narrative)
    else:
        compatibility = local_overlap

    structure = float(header and row) if table else float(narrative)
    provenance = (bool(unit.get("document_id")) + bool(unit.get("page_id"))) / 2
    local_support = local_numeric if task in {"numeric", "comparison"} else local_overlap
    completeness = (structure + provenance + local_support) / 3 if text.strip() else 0.0
    features = {
        "query_lexical_relevance": lexical,
        "numeric_presence": float(bool(numbers(text))),
        "period_consistency": period_score,
        "entity_consistency": entity_score,
        "answer_type_compatibility": compatibility,
        "evidence_completeness": completeness,
    }
    score = sum(WEIGHTS[name] * features[name] for name in WEIGHTS) if text.strip() else 0.0
    return {
        "score": round(score, 8),
        "features": {name: round(value, 8) for name, value in features.items()},
        "answer_type": task,
        "period_status": period_status,
        "entity_status": entity_status,
        "query_periods": sorted(query_years),
        "text_periods": sorted(raw_years),
        "metadata_periods": sorted(meta_years),
        "local_numeric_support": round(local_numeric, 8),
        "local_pair_support": round(local_pair, 8),
        "table_header_present": header,
        "table_row_present": row,
    }


def rank_answerability(query: str, units: list[dict]) -> list[dict]:
    """Return a sidecar ranking; never mutate Evidence Unit fields or metadata."""
    ranks = [_rank(unit) for unit in units]
    if len(set(ranks)) != len(ranks) or any(rank < 1 for rank in ranks):
        raise ValueError("Unique positive frozen ranks are required")
    scored = [{"original_rank": _rank(unit), **score_answerability(query, unit)} for unit in units]
    scored.sort(key=lambda item: (-item["score"], item["original_rank"]))
    return [dict(item, rank=index) for index, item in enumerate(scored, 1)]
