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
_FINANCIAL_NUMBER = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")
_TABLE_HEADER = re.compile(
    r"\b(?:consolidated|in millions|in thousands|year ended|years ended|at december|quarter ended|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
_STOP_TERMS = {
    "about", "after", "based", "between", "company", "does", "from", "have", "into",
    "million", "please", "question", "report", "the", "their", "this", "using", "what",
    "when", "whether", "which", "with", "would", "year",
}
_QUERY_CONCEPT_VARIANTS = (
    {"acquire", "acquired", "acquiring", "acquisition", "acquisitions", "combination", "combinations"},
    {"drive", "driven", "driver", "drivers", "drove", "factor", "factors"},
    {"forecast", "forecasting", "plan", "plans", "planned", "expect", "expects", "expected"},
    {"perform", "performed", "performance", "growth", "grew", "increase", "increased"},
)


def _parse_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _query_terms(question: str) -> set[str]:
    terms = {
        token.lower().strip(".'-")
        for token in _TOKEN_PATTERN.findall(question or "")
        if len(token.strip(".'-")) >= 3 and token.lower().strip(".'-") not in _STOP_TERMS
    }
    for variants in _QUERY_CONCEPT_VARIANTS:
        if terms & variants:
            terms.update(variants)
    return terms


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


def _fields_with_numeric_rows(text: str, task_spec: Dict[str, object]) -> set[str]:
    matched: set[str] = set()
    for field in task_spec.get("required_fields") or []:
        for line in _split_lines(text):
            lowered = line.lower()
            for alias in FIELD_ALIASES.get(str(field), []):
                position = lowered.find(alias.lower())
                if position < 0:
                    continue
                tail = line[position + len(alias) :]
                numbers = _FINANCIAL_NUMBER.findall(tail)
                has_financial_value = any(
                    not (
                        re.fullmatch(r"\d{4}", value)
                        and 1900 <= int(value) <= 2100
                    )
                    for value in numbers
                )
                if has_financial_value:
                    matched.add(str(field))
                    break
            if str(field) in matched:
                break
    return matched


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
    numeric_table_task: bool,
) -> float:
    lowered = line.lower()
    tokens = {token.lower().strip(".'-") for token in _TOKEN_PATTERN.findall(line)}
    alias_hits = sum(alias in lowered for alias in aliases)
    period_hits = int(any(period.lower() in lowered for period in periods))
    value_hits = sum(value in line.replace(",", "") for value in calculation_values)
    score = alias_hits * 18 + period_hits * 7 + value_hits * 12 + len(tokens & query_terms) * 3
    # Action words carry more intent than ubiquitous company/report terms on
    # long filing pages. This keeps transaction and driver sentences from
    # being displaced by headers, glossaries, or boilerplate.
    for variants in _QUERY_CONCEPT_VARIANTS:
        if query_terms & variants and tokens & variants:
            score += 12
    if calculation_task and re.search(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?", line):
        score += 2
    if numeric_table_task and ("%" in line or len(_FINANCIAL_NUMBER.findall(line)) >= 2):
        score += 6
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
    enumeration_task: bool,
    numeric_table_task: bool,
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
                numeric_table_task=numeric_table_task,
            ),
            index,
        )
        for index, line in enumerate(lines)
    ]
    positive = [(score, index) for score, index in scored if score > 0]
    ranked = sorted(positive or scored, key=lambda item: (item[0], -item[1]), reverse=True)
    structured_task = bool(aliases or periods or calculation_task)
    target_count = min(10 if structured_task or enumeration_task else 4, len(ranked))
    target_indices = [index for _, index in ranked[:target_count]]
    selected_indices: set[int] = set()
    selected_chars = 0
    if structured_task:
        before, after = 2, 6
    elif enumeration_task:
        before, after = 3, 10
    else:
        before, after = 1, 1
    for index in target_indices:
        window = range(max(0, index - before), min(len(lines), index + after + 1))
        new_indices = [item for item in window if item not in selected_indices]
        new_chars = sum(len(lines[item]) + 1 for item in new_indices)
        if selected_indices and selected_chars + new_chars > max_chars:
            continue
        selected_indices.update(new_indices)
        selected_chars += new_chars
    if calculation_task and target_indices:
        for target_index in target_indices:
            for index in range(max(0, target_index - 24), target_index):
                line = lines[index]
                if not (_TABLE_HEADER.search(line) or re.fullmatch(r"(?:19|20)\d{2}", line)):
                    continue
                if index not in selected_indices and selected_chars + len(line) + 1 <= max_chars:
                    selected_indices.add(index)
                    selected_chars += len(line) + 1
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
            "answer_context_retained_required_fields": [],
            "answer_context_missing_required_fields": list(task_spec.get("required_fields") or []),
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
    enumeration_task = bool(re.search(
        r"\b(?:what are|which are|list|identify|name|major|main|drivers?|factors?|categories|acquisitions?)\b",
        question or "",
        re.IGNORECASE,
    ))
    numeric_table_task = task_spec.get("task_type") in {"calculation", "comparison", "selection"} or bool(re.search(
        r"\b(?:best|worst|performed|performance|growth|highest|lowest|rank|ratio|margin|rate)\b",
        question or "",
        re.IGNORECASE,
    ))

    company = str(task_spec.get("company") or "")
    filename_company_documents = [
        document for document in documents
        if company and matches_company_text(
            "\n".join(
                [
                    str(document.get("filename") or ""),
                    str(document.get("doc_name") or ""),
                ]
            ),
            company,
        )
    ]
    content_company_documents = [
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
    # A matching filename is stronger document identity evidence than a body
    # mention (for example "$3 million" must not identify an AES page as 3M).
    source_documents = filename_company_documents or content_company_documents or documents
    company_filtered_count = max(0, len(documents) - len(source_documents))
    scope_source_count = len(source_documents)
    question_lower = (question or "").lower()
    domestic_requested = bool(re.search(r"\b(?:domestic|usa|u\.s\.)\b", question_lower))
    international_requested = bool(re.search(r"\b(?:international|foreign|outside the u\.s\.)\b", question_lower))
    if domestic_requested and not international_requested:
        scoped = [
            document for document in source_documents
            if not re.search(
                r"\binternational segment\b",
                str(document.get("text") or document.get("page_text") or "")[:1200],
                re.IGNORECASE,
            )
        ]
        source_documents = scoped or source_documents
    elif international_requested and not domestic_requested:
        scoped = [
            document for document in source_documents
            if not re.search(
                r"\bdomestic segment\b",
                str(document.get("text") or document.get("page_text") or "")[:1200],
                re.IGNORECASE,
            )
        ]
        source_documents = scoped or source_documents
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

    candidates: List[tuple[float, int, dict, str, set[str]]] = []
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
            enumeration_task=enumeration_task,
            numeric_table_task=numeric_table_task,
            max_chars=max_unit_chars,
        )
        if snippet:
            candidates.append((score, rank, document, snippet, _fields_with_numeric_rows(document["text"], task_spec)))

    ranked_by_retrieval = sorted(candidates, key=lambda item: item[1])
    selected = []
    selected_pages: set[tuple[str, object]] = set()
    for field in task_spec.get("required_fields") or []:
        field_candidates = [item for item in candidates if str(field) in item[4]]
        if not field_candidates:
            continue
        best = max(field_candidates, key=lambda item: (item[0], -item[1]))
        page_key = (str(best[2].get("filename") or ""), best[2].get("page_number"))
        if page_key not in selected_pages and len(selected) < max_units:
            selected.append(best)
            selected_pages.add(page_key)
    # Enumeration and numeric-table questions are especially vulnerable to
    # glossary, contents, and nearby-but-irrelevant pages occupying the answer
    # budget. Preserve a small retrieval-order safety net, then let task-aware
    # evidence scores choose the remaining pages.
    effective_rank_reserved = min(rank_reserved, 3) if (enumeration_task or numeric_table_task) else rank_reserved
    reserved_target = min(max_units, max(effective_rank_reserved, len(selected)))
    for item in ranked_by_retrieval:
        if len(selected) >= reserved_target:
            break
        page_key = (str(item[2].get("filename") or ""), item[2].get("page_number"))
        if page_key not in selected_pages:
            selected.append(item)
            selected_pages.add(page_key)
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
    retained_required_fields: set[str] = set()
    for _, _, document, snippet, field_hits in selected:
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
        retained_required_fields.update(field_hits)

    evidence = "\n\n---\n\n".join(blocks)
    compressed_chars = len(evidence)
    return evidence, {
        "answer_context_compressed": bool(evidence),
        "answer_context_original_chars": original_chars,
        "answer_context_chars": compressed_chars,
        "answer_context_unit_count": len(blocks),
        "answer_context_pages": pages,
        "answer_context_retained_required_fields": sorted(retained_required_fields),
        "answer_context_missing_required_fields": [
            str(field)
            for field in task_spec.get("required_fields") or []
            if str(field) not in retained_required_fields
        ],
        "answer_context_company_filtered_count": company_filtered_count,
        "answer_context_scope_filtered_count": max(0, scope_source_count - len(source_documents)),
        "answer_context_rank_reserved_units": effective_rank_reserved,
        "answer_context_reduction_ratio": round(1 - compressed_chars / original_chars, 4) if original_chars else 0.0,
    }
