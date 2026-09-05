"""Offline search for missing operation operands in frozen RRF candidates."""

from __future__ import annotations

import re
from typing import Any


_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])\(?\s*[-+]?\s*[$€£]?\s*\d[\d,]*(?:\.\d+)?\s*%?\s*\)?(?![A-Za-z0-9])")
_CHANGE_IN_BALANCE_RE = re.compile(r"\b(?:increase|decrease|change)\s+in\s+inventor(?:y|ies)\b", re.IGNORECASE)


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _entity_match(required: str | None, chunk: dict[str, Any]) -> bool:
    if not required:
        return True
    target = _compact(required)
    company = _compact(chunk.get("company") or "")
    if target and company and (target in company or company in target):
        return True
    if target and len(target) <= 5 and company:
        iterator = iter(company)
        if all(character in iterator for character in target):
            return True
    return bool(target and target in _compact(str(chunk.get("text") or "")[:1200]))


def _period_match(explicit_periods: list[str], chunk: dict[str, Any]) -> bool:
    if not explicit_periods:
        return True
    report_year = str(chunk.get("report_year") or "")
    text = str(chunk.get("text") or "")
    for period in explicit_periods:
        year_match = re.match(r"((?:19|20)\d{2})", str(period))
        year = year_match.group(1) if year_match else str(period)
        if report_year == year or re.search(rf"\b(?:FY\s*)?{re.escape(year)}\b", text, re.IGNORECASE):
            return True
    return False


def _values(line: str) -> list[str]:
    result = []
    for match in _NUMBER_RE.finditer(line):
        raw = match.group(0).strip()
        cleaned = re.sub(r"[$€£,%()\s]", "", raw)
        try:
            number = float(cleaned)
        except ValueError:
            continue
        if "%" not in raw and number.is_integer() and 1900 <= abs(number) <= 2100:
            continue
        result.append(raw)
    return result


def _alias_match(line: str, alias: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])", line, re.IGNORECASE))


def _operand_lines(operand: dict[str, Any], text: str) -> list[dict[str, Any]]:
    matches = []
    for line_number, raw_line in enumerate(str(text or "").splitlines(), 1):
        line = re.sub(r"\s+", " ", raw_line).strip()
        aliases = [alias for alias in operand.get("aliases", []) if _alias_match(line, alias)]
        if not aliases:
            continue
        if operand.get("key") == "average_inventory" and _CHANGE_IN_BALANCE_RE.search(line):
            continue
        values = _values(line)
        if len(values) < int(operand.get("min_values") or 1):
            continue
        matches.append({"line_number": line_number, "text": line[:800], "matched_aliases": aliases, "values": values})
    return matches


def search_missing_operands_v1(
    schema: dict[str, Any],
    missing_operands: list[str],
    rrf_chunks: list[dict[str, Any]],
    context_documents: list[dict[str, Any]] | None = None,
    *,
    max_candidates_per_operand: int = 10,
) -> dict[str, Any]:
    """Search the unchanged Top120 set using operand, entity, and period constraints."""
    required = {item["key"]: item for item in schema.get("required_operands", [])}
    context_ids = {str(item.get("chunk_id") or "") for item in (context_documents or [])}
    context_sources = {
        (str(item.get("filename") or ""), int(item.get("page_number") or 0), str(item.get("content_hash") or ""))
        for item in (context_documents or [])
    }
    results: dict[str, list[dict[str, Any]]] = {}
    for key in missing_operands:
        operand = required.get(key)
        if operand is None:
            results[key] = []
            continue
        candidates = []
        for chunk in rrf_chunks:
            if not _entity_match(schema.get("entity_requirement"), chunk):
                continue
            if not _period_match(schema.get("period_requirement", {}).get("explicit_periods", []), chunk):
                continue
            line_matches = _operand_lines(operand, str(chunk.get("text") or ""))
            if not line_matches:
                continue
            chunk_id = str(chunk.get("chunk_id") or "")
            source_key = (str(chunk.get("filename") or ""), int(chunk.get("page_number") or 0), str(chunk.get("content_hash") or ""))
            already_in_context = bool(chunk_id and chunk_id in context_ids) or source_key in context_sources
            rank = int(chunk.get("rrf_rank") or 9999)
            alias_specificity = max(len(alias.split()) for match in line_matches for alias in match["matched_aliases"])
            score = 0.75 + min(0.15, alias_specificity * 0.03) + 0.10 / max(1, rank)
            candidates.append({
                "operand": key, "chunk_id": chunk_id, "document": str(chunk.get("filename") or ""),
                "page": int(chunk.get("page_number") or 0), "company": str(chunk.get("company") or ""),
                "report_year": chunk.get("report_year"), "rrf_rank": rank, "dense_rank": chunk.get("dense_rank"),
                "bm25_rank": chunk.get("bm25_rank"), "match_score": round(score, 4),
                "already_in_context": already_in_context, "matched_lines": line_matches[:3],
            })
        candidates.sort(key=lambda item: (item["already_in_context"], -item["match_score"], item["rrf_rank"]))
        results[key] = candidates[:max_candidates_per_operand]
    outside = {key: [item for item in values if not item["already_in_context"]] for key, values in results.items()}
    return {
        "found_candidates": results,
        "candidate_has_operand": {key: bool(values) for key, values in results.items()},
        "outside_context_candidate_has_operand": {key: bool(values) for key, values in outside.items()},
        "all_missing_operands_recoverable": bool(missing_operands) and all(outside.get(key) for key in missing_operands),
    }

