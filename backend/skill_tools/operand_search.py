"""Bounded operand lookup and conservative financial-row value resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

from agent_tools import open_pages, select_pages
from skills.explicit_formula.schema import AtomicOperand, ResolvedOperand


_NUMBER = re.compile(r"\[\d{1,2}\]|\(?-?\$?\d[\d,]*(?:\.\d+)?\)?")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_FOOTNOTE_MARKER = re.compile(r"^(?:\(\d{1,2}\)|\[\d{1,2}\])$")
_GENERIC_FILE_TOKENS = {
    "annual", "earnings", "financial", "form", "quarterly", "report", "results",
    "10k", "10q", "20f", "pdf", "q1", "q2", "q3", "q4",
}


@dataclass(frozen=True)
class OperandSearchResult:
    documents: tuple[dict[str, Any], ...]
    calls: tuple[dict[str, Any], ...]
    dense_bm25_calls: int
    jina_calls: int = 0


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _filename_identity(filename: object) -> str:
    stem = Path(str(filename or "")).stem.casefold()
    tokens = [
        token for token in re.findall(r"[a-z0-9]+", stem)
        if token not in _GENERIC_FILE_TOKENS and not re.fullmatch(r"20\d{2}", token)
    ]
    return "".join(tokens)


def infer_target_filenames(question: str, documents: Iterable[dict[str, Any]]) -> list[str]:
    """Match question entities to existing metadata without a company registry."""
    normalized_question = _norm(question)
    matches: list[str] = []
    for document in documents:
        filename = str(document.get("filename") or "").strip()
        company = _norm(document.get("company"))
        identity = _filename_identity(filename)
        if not filename:
            continue
        confident = bool(company and len(company) >= 3 and company in normalized_question)
        confident = confident or bool(identity and len(identity) >= 3 and identity in normalized_question)
        if confident and filename not in matches:
            matches.append(filename)
    requested_years = set(re.findall(r"(?:FY\s*)?((?:19|20)\d{2})", question, re.IGNORECASE))
    year_matched = [
        filename for filename in matches
        if requested_years & set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", Path(filename).stem))
    ]
    return year_matched or matches


def document_matches_question_entity(question: str, filename: str, company: str = "") -> bool:
    normalized_question = _norm(question)
    identities = [_norm(company), _filename_identity(filename)]
    return any(item and len(item) >= 3 and item in normalized_question for item in identities)


def _unique_pages(documents: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, object]] = set()
    for document in documents:
        key = (str(document.get("filename") or ""), document.get("page_number"))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        result.append(document)
        if len(result) >= limit:
            break
    return result


def search_missing_operands(
    question: str,
    operands: list[AtomicOperand],
    baseline_documents: list[dict[str, Any]],
    candidate_documents: list[dict[str, Any]],
    *,
    max_queries: int = 4,
    pages_per_query: int = 12,
) -> OperandSearchResult:
    """Run at most one deterministic Dense+BM25 query per missing operand."""
    from rag_utils import retrieve_candidate_documents, retrieve_document_scoped_candidates

    all_existing = [*baseline_documents, *candidate_documents]
    scoped_filenames = infer_target_filenames(question, all_existing)
    discovered: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for operand in operands[: max(0, max_queries)]:
        period = operand.period
        identity_hint = ""
        if not scoped_filenames:
            # The original question is kept in the global query so the entity is
            # preserved without a benchmark/company registry.
            identity_hint = question.split("?")[0].strip()
        statement_hint = {
            "balance_sheet": "consolidated balance sheet",
            "income_statement": "consolidated statement of operations income",
            "cash_flow": "consolidated statement of cash flows",
        }.get(operand.statement_types[0] if operand.statement_types else "", "")
        alias_hint = " ".join(list(operand.aliases)[:3])
        query = " ".join(filter(None, [identity_hint, period, alias_hint, statement_hint])).strip()
        if scoped_filenames:
            chunks = retrieve_document_scoped_candidates(query, scoped_filenames, top_k=12)
            scope = "document_scoped"
        else:
            result = retrieve_candidate_documents(query, candidate_k=20)
            chunks = list(result.get("docs") or result.get("candidate_docs") or [])
            scope = "global_entity_operand"
        selected = _unique_pages(chunks, pages_per_query)
        opened = open_pages(select_pages(selected, limit=pages_per_query), limit=pages_per_query)
        # Preserve metadata/scores from the retrieved chunks when opening full pages.
        by_key = {(item.get("filename"), item.get("page_number")): item for item in opened}
        enriched = [
            {**item, **by_key.get((item.get("filename"), item.get("page_number")), {})}
            for item in selected
        ]
        discovered.extend(enriched)
        calls.append({
            "operand": operand.key,
            "query": query,
            "scope": scope,
            "filenames": scoped_filenames,
            "candidate_count": len(chunks),
            "opened_page_count": len(opened),
        })
    return OperandSearchResult(
        documents=tuple(_unique_pages(discovered, max(1, len(discovered)))),
        calls=tuple(calls),
        dense_bm25_calls=len(calls),
        jina_calls=0,
    )


def _number(value: str) -> Decimal | None:
    source = str(value or "").strip()
    negative = source.startswith("(") and source.endswith(")")
    source = source.replace("$", "").replace(",", "").strip("()[] ")
    try:
        number = Decimal(source)
    except InvalidOperation:
        return None
    return -number if negative and number > 0 else number


def _page_scale(text: str) -> str:
    head = text[:5000].casefold()
    if re.search(r"\b(?:in|amounts in)\s+billions\b", head):
        return "billions"
    if re.search(r"\b(?:in|amounts in)\s+millions\b", head):
        return "millions"
    if re.search(r"\b(?:in|amounts in)\s+thousands\b", head):
        return "thousands"
    return ""


def _page_currency(text: str) -> str:
    head = text[:5000]
    if re.search(r"\bUSD\b|U\.S\. dollars?|\$", head, re.IGNORECASE):
        return "USD"
    if "€" in head or re.search(r"\bEUR\b", head):
        return "EUR"
    if "£" in head or re.search(r"\bGBP\b", head):
        return "GBP"
    return ""


def _header_years(lines: list[str], row_index: int, value_count: int) -> list[str]:
    header = "\n".join(lines[max(0, row_index - 100):row_index])
    years = list(dict.fromkeys(_YEAR.findall(header)))
    if len(years) < value_count:
        return []
    return years[-value_count:]


def _nearby_header_years(lines: list[str], row_index: int) -> list[str]:
    """Return the closest compact year header immediately above a data row."""
    groups: list[list[str]] = []
    gap_after_year = 0
    for line in reversed(lines[max(0, row_index - 16):row_index]):
        years = _YEAR.findall(line)
        if years:
            groups.append(years)
            gap_after_year = 0
            continue
        if groups:
            gap_after_year += 1
            if gap_after_year >= 2:
                break
    ordered = []
    for group in reversed(groups):
        for year in group:
            if year not in ordered:
                ordered.append(year)
    return ordered


def _align_row_values(
    raw_values: list[str], lines: list[str], row_index: int
) -> tuple[list[str] | None, list[str]]:
    """Align row values to an explicit year header without treating notes as values.

    A leading parenthesized/bracketed integer is a footnote only when removing it
    produces exactly one value per nearby header year.  If an unexplained leading
    small integer leaves the same N+1 shape, the row is ambiguous and is rejected.
    This preserves a genuine value of 1 (including ``(1)``) when it occupies an
    actual year column.
    """
    header_years = _nearby_header_years(lines, row_index)
    if not header_years or len(raw_values) != len(header_years) + 1:
        return raw_values, header_years
    leading = raw_values[0].strip()
    if _FOOTNOTE_MARKER.fullmatch(leading):
        return raw_values[1:], header_years
    leading_number = _number(leading)
    if leading_number is not None and abs(leading_number) <= 9:
        return None, header_years
    return raw_values, header_years


def _scope_allowed(text: str, line: str) -> bool:
    head = text[:2500].casefold()
    line_lower = line.casefold()
    if "schedule i condensed financial information of parent" in head:
        return False
    if re.search(r"\bsegment\b", head) and "consolidated" not in head:
        return False
    if re.search(r"\b(?:segment|geographic)\b", line_lower) and "total" not in line_lower:
        return False
    return True


def _statement_types(text: str) -> set[str]:
    head = text[:3500].casefold()
    first_lines = "\n".join(text.splitlines()[:8]).casefold()
    if "notes to consolidated financial statements" in first_lines:
        # A notes table may contain a column such as "Affected Line Item in the
        # Consolidated Statements of Operations".  That is a cross-reference,
        # not the primary statement and cannot support an authoritative total.
        return set()
    result: set[str] = set()
    if re.search(r"consolidated (?:balance sheets?|statements? of financial position)", head):
        result.add("balance_sheet")
    if re.search(r"consolidated statements? of (?:operations|income|earnings)|consolidated statement of income", head):
        result.add("income_statement")
    if re.search(r"consolidated statements? of cash flows?", head):
        result.add("cash_flow")
    return result


def extract_operand_candidates(
    operand: AtomicOperand,
    documents: Iterable[dict[str, Any]],
    question: str,
    target_filenames: list[str] | None = None,
) -> list[ResolvedOperand]:
    """Extract period-aligned table-row values; narrative numbers are rejected."""
    results: list[tuple[int, ResolvedOperand]] = []
    target_filenames = target_filenames or []
    for document_rank, document in enumerate(documents):
        filename = str(document.get("filename") or "")
        if target_filenames and filename not in target_filenames:
            continue
        if not target_filenames and not document_matches_question_entity(
            question, filename, str(document.get("company") or "")
        ):
            continue
        text = str(document.get("text") or document.get("page_text") or "")
        if not text:
            continue
        page_statement_types = _statement_types(text)
        if operand.statement_types and not (set(operand.statement_types) & page_statement_types):
            continue
        lines = [re.sub(r"\s+", " ", item).strip() for item in re.split(r"[\r\n]+", text) if item.strip()]
        scale = _page_scale(text)
        currency = _page_currency(text)
        for row_index, line in enumerate(lines):
            lowered = line.casefold()
            alias = next((item for item in sorted(operand.aliases, key=len, reverse=True) if item.casefold() in lowered), "")
            if not alias or not _scope_allowed(text, line):
                continue
            position = lowered.find(alias.casefold())
            prefix_words = re.findall(r"[a-z]+", lowered[:position])
            if position > 55 or len(prefix_words) > 5:
                continue
            tail = line[position + len(alias):]
            raw_values = _NUMBER.findall(tail)
            raw_values, nearby_header_years = _align_row_values(raw_values, lines, row_index)
            if raw_values is None:
                continue
            values = [(raw, _number(raw)) for raw in raw_values]
            values = [(raw, value) for raw, value in values if value is not None]
            if not values:
                continue
            years = (
                nearby_header_years
                if len(nearby_header_years) == len(values)
                else _header_years(lines, row_index, len(values))
            )
            if operand.period:
                if operand.period in years:
                    index = years.index(operand.period)
                elif re.search(rf"\b{re.escape(operand.period)}\b", line):
                    non_year = [(raw, value) for raw, value in values if raw.strip("()$,") != operand.period]
                    if len(non_year) != 1:
                        continue
                    raw, value = non_year[0]
                    index = -1
                else:
                    continue
            elif len(values) == 1:
                index = 0
            else:
                continue
            if index >= 0:
                raw, value = values[index]
            if operand.cash_outflow_magnitude:
                value = abs(value)
            exact_prefix = bool(re.match(rf"^(?:total\s+|net\s+)?{re.escape(alias.casefold())}\b", lowered))
            score = 500 if exact_prefix else 300
            score += 150 if operand.period and operand.period in years else 0
            score += 100 if "consolidated" in text[:2500].casefold() else 0
            score += max(0, 50 - document_rank)
            if operand.concept == "net_income":
                if re.match(r"^net income(?:\s*\(loss\))?\s+(?:\$|\(?-?\d)", lowered):
                    score += 150
                if "attributable to" in lowered:
                    score -= 150
                if "per share" in lowered or "common stockholders" in lowered:
                    score -= 200
            confidence = min(0.99, score / 800)
            results.append((score, ResolvedOperand(
                key=operand.key,
                concept=operand.concept,
                period=operand.period,
                raw_value=raw,
                normalized_value=value,
                currency=currency,
                scale=scale,
                filename=filename,
                page_number=document.get("page_number"),
                source_text=line,
                confidence=round(confidence, 3),
            )))
    dedup: dict[tuple[str, object, str, str], tuple[int, ResolvedOperand]] = {}
    for score, item in results:
        key = (item.filename, item.page_number, item.period, format(item.normalized_value, "f"))
        if key not in dedup or score > dedup[key][0]:
            dedup[key] = (score, item)
    return [item for _, item in sorted(dedup.values(), key=lambda pair: pair[0], reverse=True)]


def resolve_unique_operand(candidates: list[ResolvedOperand]) -> tuple[ResolvedOperand | None, str]:
    if not candidates:
        return None, "operand_not_found"
    top = candidates[0]
    if top.confidence < 0.7:
        return None, "operand_confidence_too_low"
    scale_factors = {
        "thousands": Decimal("1000"),
        "millions": Decimal("1000000"),
        "billions": Decimal("1000000000"),
    }

    def equivalent(left: ResolvedOperand, right: ResolvedOperand) -> bool:
        left_value = left.normalized_value * scale_factors.get(left.scale, Decimal("1"))
        right_value = right.normalized_value * scale_factors.get(right.scale, Decimal("1"))
        difference = abs(left_value - right_value)
        denominator = max(abs(left_value), abs(right_value), Decimal("1"))
        return difference / denominator <= Decimal("0.0005")

    conflicts = [
        item for item in candidates[1:]
        if not equivalent(item, top)
        and item.confidence >= top.confidence - 0.04
    ]
    if conflicts:
        return None, "operand_value_ambiguous"
    return top, ""
