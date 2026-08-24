"""Build compact, source-preserving evidence units for answer generation."""

import os
import re
from typing import Dict, List

try:
    from query_parser import FIELD_ALIASES, matches_company_text
except ImportError:  # Package import in unit tests.
    from backend.query_parser import FIELD_ALIASES, matches_company_text


_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9&'.-]*|(?:19|20)\d{2}|Q[1-4]|\d+(?:\.\d+)?%?")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_STOP_TERMS = {
    "about", "after", "based", "between", "company", "does", "from", "have", "into",
    "million", "please", "question", "report", "the", "their", "this", "using", "what",
    "when", "whether", "which", "with", "would", "year",
}


def _parse_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _query_terms(question: str) -> set[str]:
    return {
        token.lower().strip(".'-")
        for token in _TOKEN_PATTERN.findall(question or "")
        if len(token.strip(".'-")) >= 3 and token.lower().strip(".'-") not in _STOP_TERMS
    }


def _split_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw_line in re.split(r"[\r\n]+", text or ""):
        normalized = re.sub(r"\s+", " ", raw_line).strip()
        if not normalized:
            continue
        if len(normalized) <= 500:
            lines.append(normalized)
            continue
        sentences = [item.strip() for item in _SENTENCE_BOUNDARY.split(normalized) if item.strip()]
        lines.extend(sentences or [normalized])
    return lines


def _required_aliases(task_spec: Dict[str, object]) -> List[str]:
    aliases: List[str] = []
    for field in task_spec.get("required_fields") or []:
        aliases.extend(FIELD_ALIASES.get(str(field), []))
    return list(dict.fromkeys(alias.lower() for alias in aliases if alias))


def _calculation_values(calculation: Dict[str, object] | None) -> set[str]:
    values: set[str] = set()
    for operand in (calculation or {}).get("operands", {}).values():
        raw = operand.get("value") if isinstance(operand, dict) else None
        for value in raw if isinstance(raw, list) else [raw]:
            normalized = str(value or "").replace(",", "").strip()
            if normalized:
                values.add(normalized)
    return values


def _line_score(
    line: str,
    *,
    query_terms: set[str],
    aliases: List[str],
    periods: set[str],
    calculation_values: set[str],
    calculation_task: bool,
) -> float:
    lowered = line.lower()
    tokens = {token.lower().strip(".'-") for token in _TOKEN_PATTERN.findall(line)}
    alias_hits = sum(alias in lowered for alias in aliases)
    period_hits = sum(period.lower() in lowered for period in periods)
    value_hits = sum(value in line.replace(",", "") for value in calculation_values)
    score = alias_hits * 18 + period_hits * 7 + value_hits * 12 + len(tokens & query_terms) * 3
    if calculation_task and re.search(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?", line):
        score += 2
    if re.search(r"\b(?:in millions|in thousands|year ended|at december|quarter ended)\b", lowered):
        score += 4
    return float(score)


def _compact_document(
    document: dict,
    *,
    query_terms: set[str],
    aliases: List[str],
    periods: set[str],
    calculation_values: set[str],
    calculation_task: bool,
    max_chars: int,
) -> tuple[str, float]:
    text = str(document.get("text") or document.get("page_text") or "")
    lines = _split_lines(text)
    if not lines:
        return "", 0.0
    scored = [
        (
            _line_score(
                line,
                query_terms=query_terms,
                aliases=aliases,
                periods=periods,
                calculation_values=calculation_values,
                calculation_task=calculation_task,
            ),
            index,
        )
        for index, line in enumerate(lines)
    ]
    positive = [(score, index) for score, index in scored if score > 0]
    ranked = sorted(positive or scored, key=lambda item: (item[0], -item[1]), reverse=True)
    structured_task = bool(aliases or periods or calculation_task)
    target_count = min(10 if structured_task else 4, len(ranked))
    target_indices = [index for _, index in ranked[:target_count]]
    selected_indices: set[int] = set()
    selected_chars = 0
    before, after = (2, 6) if structured_task else (1, 1)
    for index in target_indices:
        window = range(max(0, index - before), min(len(lines), index + after + 1))
        new_indices = [item for item in window if item not in selected_indices]
        new_chars = sum(len(lines[item]) + 1 for item in new_indices)
        if selected_indices and selected_chars + new_chars > max_chars:
            continue
        selected_indices.update(new_indices)
        selected_chars += new_chars
    for index, line in enumerate(lines[:12]):
        if re.search(r"\b(?:in millions|in thousands|year ended|at december|quarter ended)\b", line.lower()):
            if index not in selected_indices and selected_chars + len(line) + 1 <= max_chars:
                selected_indices.add(index)
                selected_chars += len(line) + 1

    selected: List[str] = []
    total = 0
    for index in sorted(selected_indices):
        line = lines[index]
        separator = 1 if selected else 0
        if total + separator + len(line) > max_chars:
            remaining = max_chars - total - separator
            if remaining >= 80:
                selected.append(line[:remaining].rstrip() + "…")
            break
        selected.append(line)
        total += separator + len(line)
    return "\n".join(selected), max((score for score, _ in ranked), default=0.0)


def build_compact_evidence(
    question: str,
    documents: List[dict],
    task_spec: Dict[str, object],
    calculation: Dict[str, object] | None = None,
) -> tuple[str, dict]:
    """Return compact cited evidence and trace metadata without mutating retrieval docs."""
    enabled = os.getenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    original_chars = sum(len(str(doc.get("text") or doc.get("page_text") or "")) for doc in documents)
    if not enabled or not documents:
        return "", {
            "answer_context_compressed": False,
            "answer_context_original_chars": original_chars,
            "answer_context_chars": 0,
            "answer_context_unit_count": 0,
            "answer_context_pages": [],
        }

    max_units = _parse_int("RAG_ANSWER_MAX_EVIDENCE_UNITS", 10, 1)
    rank_reserved = min(
        _parse_int("RAG_ANSWER_RANK_RESERVED_UNITS", 6, 0),
        max(0, max_units - 1),
    )
    max_context_chars = _parse_int("RAG_ANSWER_MAX_CONTEXT_CHARS", 24000, 2000)
    max_unit_chars = _parse_int("RAG_ANSWER_MAX_UNIT_CHARS", 2000, 400)
    query_terms = _query_terms(question)
    aliases = _required_aliases(task_spec)
    periods = {str(value) for value in task_spec.get("required_periods") or []}
    values = _calculation_values(calculation)
    calculation_task = task_spec.get("task_type") == "calculation"

    company = str(task_spec.get("company") or "")
    company_documents = [
        document for document in documents
        if not company or matches_company_text(
            "\n".join(
                [
                    str(document.get("filename") or ""),
                    str(document.get("doc_name") or ""),
                    str(document.get("text") or document.get("page_text") or ""),
                ]
            ),
            company,
        )
    ]
    source_documents = company_documents or documents
    grouped_pages: List[tuple[int, dict]] = []
    page_indexes: dict[tuple[str, object], int] = {}
    page_texts: dict[tuple[str, object], List[str]] = {}
    for rank, document in enumerate(source_documents):
        page_key = (str(document.get("filename") or ""), document.get("page_number"))
        text = str(document.get("text") or document.get("page_text") or "").strip()
        if page_key not in page_indexes:
            page_indexes[page_key] = len(grouped_pages)
            grouped_pages.append((rank, dict(document)))
            page_texts[page_key] = []
        if text and text not in page_texts[page_key]:
            page_texts[page_key].append(text)

    candidates: List[tuple[float, int, dict, str]] = []
    for rank, document in grouped_pages:
        page_key = (str(document.get("filename") or ""), document.get("page_number"))
        document = {**document, "text": "\n".join(page_texts.get(page_key, []))}
        snippet, score = _compact_document(
            document,
            query_terms=query_terms,
            aliases=aliases,
            periods=periods,
            calculation_values=values,
            calculation_task=calculation_task,
            max_chars=max_unit_chars,
        )
        if snippet:
            candidates.append((score, rank, document, snippet))

    ranked_by_retrieval = sorted(candidates, key=lambda item: item[1])
    selected = ranked_by_retrieval[:rank_reserved]
    selected_pages = {
        (str(item[2].get("filename") or ""), item[2].get("page_number"))
        for item in selected
    }
    remaining = [
        item for item in candidates
        if (str(item[2].get("filename") or ""), item[2].get("page_number")) not in selected_pages
    ]
    positive_remaining = [item for item in remaining if item[0] > 0]
    selection_pool = positive_remaining if (selected or positive_remaining) else remaining
    selected.extend(
        sorted(selection_pool, key=lambda item: (item[0], -item[1]), reverse=True)[: max_units - len(selected)]
    )
    blocks: List[str] = []
    pages: List[dict] = []
    used_chars = 0
    for _, _, document, snippet in selected:
        filename = str(document.get("filename") or "Unknown")
        page = document.get("page_number", "N/A")
        header = f"Source: {filename} | Page: {page}\n"
        remaining = max_context_chars - used_chars - len(header)
        if remaining < 100:
            break
        body = snippet[:remaining]
        block = header + body
        blocks.append(block)
        used_chars += len(block)
        pages.append({"filename": filename, "page_number": page})

    evidence = "\n\n---\n\n".join(blocks)
    compressed_chars = len(evidence)
    return evidence, {
        "answer_context_compressed": bool(evidence),
        "answer_context_original_chars": original_chars,
        "answer_context_chars": compressed_chars,
        "answer_context_unit_count": len(blocks),
        "answer_context_pages": pages,
        "answer_context_company_filtered_count": max(0, len(documents) - len(source_documents)),
        "answer_context_reduction_ratio": round(1 - compressed_chars / original_chars, 4) if original_chars else 0.0,
    }
