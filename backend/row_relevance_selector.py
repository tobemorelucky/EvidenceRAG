"""Deterministic relevance ranking for rows in trusted financial tables."""

from __future__ import annotations

from collections import Counter
import math
import re


FINANCIAL_METRIC_SYNONYMS = {
    "quick ratio": (
        "cash",
        "cash equivalents",
        "accounts receivable",
        "receivables",
        "inventory",
        "inventories",
        "prepaid expenses",
        "current assets",
        "current liabilities",
    ),
    "gross margin": ("gross profit", "revenue", "net sales", "cost of sales", "cost of revenue"),
    "inventory turnover": (
        "inventory",
        "inventories",
        "cost of goods sold",
        "cost of sales",
        "cost of revenue",
    ),
    "operating margin": ("operating income", "operating profit", "revenue", "net sales"),
    "working capital": ("current assets", "current liabilities"),
    "current ratio": ("current assets", "current liabilities"),
    "return on assets": ("net income", "total assets", "average assets"),
    "return on equity": ("net income", "shareholders equity", "stockholders equity", "average equity"),
    "free cash flow": ("operating cash flow", "cash provided by operating activities", "capital expenditures"),
}

_TOKEN_RE = re.compile(r"[a-z][a-z0-9]+|\d{2,4}", re.IGNORECASE)
_STOP_WORDS = {
    "and", "are", "as", "at", "based", "between", "by", "company", "did", "does", "for",
    "from", "had", "has", "how", "in", "into", "its", "most", "much", "of", "on", "please",
    "that", "the", "this", "to", "was", "were", "what", "when", "which", "with", "would", "year",
}


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _tokens(value: object) -> list[str]:
    return [token for token in _TOKEN_RE.findall(_normalize(value)) if token not in _STOP_WORDS]


def _row_text(row: dict, headers: list[str]) -> str:
    parts = []
    for header in headers or list(row):
        if str(header).startswith("_"):
            continue
        value = str(row.get(header, "") or "").strip()
        if value:
            parts.append(f"{header}: {value}")
    return "; ".join(parts)


def _expanded_query(question: str) -> tuple[list[str], list[str]]:
    normalized_question = _normalize(question)
    phrases = []
    terms = _tokens(question)
    for metric, synonyms in FINANCIAL_METRIC_SYNONYMS.items():
        if metric not in normalized_question:
            continue
        phrases.extend((metric, *synonyms))
        for phrase in synonyms:
            terms.extend(_tokens(phrase))
    return list(dict.fromkeys(terms)), list(dict.fromkeys(phrases))


def select_relevant_rows(
    question: str,
    table_title: str,
    headers: list[str],
    rows: list[dict],
    *,
    max_rows: int = 10,
) -> tuple[list[dict], dict]:
    """Rank table rows with BM25, lexical overlap, and metric synonyms."""
    candidates = []
    for index, row in enumerate(rows or []):
        if not isinstance(row, dict):
            continue
        text = _row_text(row, headers)
        if text:
            candidates.append((index, row, text, _tokens(text)))
    if not candidates:
        return [], {
            "method": "bm25_lexical_finance_synonyms",
            "expanded_phrases": [],
            "candidates": [],
        }

    query_terms, expanded_phrases = _expanded_query(question)
    query_counts = Counter(query_terms)
    document_frequency = Counter()
    for _, _, _, tokens in candidates:
        document_frequency.update(set(tokens))
    average_length = sum(len(item[3]) for item in candidates) / len(candidates)
    title_header_terms = set(_tokens(f"{table_title} {' '.join(headers or [])}"))
    query_context_overlap = sorted(set(query_terms) & title_header_terms)

    ranked = []
    candidate_count = len(candidates)
    for index, row, text, tokens in candidates:
        frequencies = Counter(tokens)
        bm25 = 0.0
        for term, query_frequency in query_counts.items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            document_frequency_for_term = document_frequency[term]
            inverse_frequency = math.log(1 + (candidate_count - document_frequency_for_term + 0.5) /
                                         (document_frequency_for_term + 0.5))
            denominator = frequency + 1.5 * (0.25 + 0.75 * len(tokens) / max(1.0, average_length))
            bm25 += query_frequency * inverse_frequency * frequency * 2.5 / denominator
        row_terms = set(tokens)
        overlap_terms = sorted(set(query_terms) & row_terms)
        normalized_row = _normalize(text)
        phrase_hits = [phrase for phrase in expanded_phrases if _normalize(phrase) in normalized_row]
        score = bm25 + len(overlap_terms) * 0.35 + len(phrase_hits) * 1.25
        ranked.append({
            "row_index": index,
            "row": row,
            "text": text,
            "score": round(score, 6),
            "bm25_score": round(bm25, 6),
            "matched_terms": overlap_terms,
            "matched_phrases": phrase_hits,
        })

    relevant = [item for item in ranked if item["score"] > 0]
    selected = sorted(relevant or ranked, key=lambda item: (-item["score"], item["row_index"]))[:max_rows]
    return selected, {
        "method": "bm25_lexical_finance_synonyms",
        "expanded_phrases": expanded_phrases,
        "query_context_overlap": query_context_overlap,
        "candidate_count": len(ranked),
        "selected_count": len(selected),
        "candidates": [
            {
                "row_index": item["row_index"],
                "score": item["score"],
                "bm25_score": item["bm25_score"],
                "matched_terms": item["matched_terms"],
                "matched_phrases": item["matched_phrases"],
                "selected": item in selected,
            }
            for item in sorted(ranked, key=lambda item: (-item["score"], item["row_index"]))
        ],
    }
