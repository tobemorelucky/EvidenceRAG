"""Independent Evidence Block v1 shadow pipeline.

Blocks are built from existing retrieval chunks and local page/table metadata.
The module is not imported by production orchestration and contains no company,
benchmark, or finance-metric rules.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any


_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%'-]*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "which", "who", "with", "would",
}


def _tokens(value: object) -> set[str]:
    return {
        token.casefold().removesuffix("'s")
        for token in _TOKEN_RE.findall(str(value or ""))
        if len(token) > 1 and token.casefold().removesuffix("'s") not in _STOPWORDS
    }


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _filename(value: object) -> str:
    return str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]


def _document_key(item: dict) -> str:
    return str(item.get("document_id") or "").strip() or _filename(
        item.get("filename") or item.get("doc_name")
    ).casefold()


def _page_key(item: dict) -> tuple[str, int]:
    return _document_key(item), _int(item.get("page_number"))


def _fallback_page_key(item: dict) -> tuple[str, int]:
    return _filename(item.get("filename") or item.get("doc_name")).casefold(), _int(item.get("page_number"))


def _chunk_id(item: dict) -> str:
    return str(item.get("chunk_id") or item.get("id") or "").strip()


def _section(item: dict, page: dict | None = None) -> str:
    explicit = next((str(item.get(name) or "").strip() for name in ("section_title", "section", "heading") if str(item.get(name) or "").strip()), "")
    if explicit:
        return explicit
    lines = [line.strip() for line in str((page or {}).get("page_text") or "").splitlines() if line.strip()]
    return lines[0] if lines else ""


def _jaccard(left: str, right: str) -> float:
    left_terms, right_terms = _tokens(left), _tokens(right)
    return len(left_terms & right_terms) / max(1, len(left_terms | right_terms))


def _coverage(query_terms: set[str], value: object) -> float:
    return len(query_terms & _tokens(value)) / max(1, len(query_terms))


def _nearby_window(page_text: str, chunk_text: str, query_terms: set[str], *, sentence_radius: int = 1) -> str:
    sentences = [item.strip() for item in _SENTENCE_RE.split(str(page_text or "")) if item.strip()]
    if not sentences:
        return ""
    needle = str(chunk_text or "").strip()[:120].casefold()
    anchor = next((index for index, sentence in enumerate(sentences) if needle and needle in sentence.casefold()), None)
    if anchor is None:
        anchor = max(
            range(len(sentences)),
            key=lambda index: (len(query_terms & _tokens(sentences[index])), -index),
        )
    start, end = max(0, anchor - sentence_radius), min(len(sentences), anchor + sentence_radius + 1)
    return " ".join(sentences[start:end])


def _row_label(row: object) -> str:
    if isinstance(row, dict):
        for name in ("row_label", "label", "name", "title"):
            if str(row.get(name) or "").strip():
                return str(row[name])
        values = row.get("cells") or row.get("values")
        if isinstance(values, list) and values:
            first = values[0]
            return str(first.get("text") if isinstance(first, dict) else first)
        return ""
    if isinstance(row, list) and row:
        first = row[0]
        return str(first.get("text") if isinstance(first, dict) else first)
    return str(row or "")


def _row_text(row: object) -> str:
    if isinstance(row, dict):
        values = row.get("cells") or row.get("values")
        if isinstance(values, list):
            return " | ".join(str(value.get("text") if isinstance(value, dict) else value) for value in values)
        return " | ".join(str(value) for value in row.values() if value not in (None, "", [], {}))
    if isinstance(row, list):
        return " | ".join(str(value.get("text") if isinstance(value, dict) else value) for value in row)
    return str(row or "")


def _target_rows(question: str, rows: list[object], *, limit: int = 5) -> list[str]:
    query_terms = _tokens(question)
    ranked = []
    for index, row in enumerate(rows):
        text = _row_text(row)
        overlap = len(query_terms & _tokens(f"{_row_label(row)} {text}"))
        ranked.append((overlap, -index, text))
    positive = [item for item in ranked if item[0] > 0]
    selected = sorted(positive or ranked[:3], reverse=True)[:limit]
    return [item[2] for item in selected if item[2].strip()]


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _render_text_block(section: str, chunk_texts: list[str], nearby: str) -> str:
    parts = ["[Text Block]"]
    if section:
        parts.append(f"Section: {section}")
    parts.append("Chunk text:\n" + "\n".join(dict.fromkeys(text for text in chunk_texts if text.strip())))
    if nearby and nearby not in chunk_texts:
        parts.append("Nearby sentence window:\n" + nearby)
    return "\n".join(parts)


def _render_table_block(table: dict, target_rows: list[str]) -> str:
    columns = table.get("columns") if isinstance(table.get("columns"), list) else []
    parts = ["[Table Block]"]
    title = str(table.get("title") or table.get("caption") or "").strip()
    if title:
        parts.append(f"Table title: {title}")
    if columns:
        parts.append("Header: " + " | ".join(str(column) for column in columns))
    if target_rows:
        parts.append("Target rows:\n" + "\n".join(target_rows))
    if table.get("unit"):
        parts.append(f"Unit: {table['unit']}")
    if table.get("scale"):
        parts.append(f"Scale: {table['scale']}")
    return "\n".join(parts)


def build_evidence_blocks_v1(
    question: str,
    chunk_candidates: list[dict],
    *,
    page_metadata: list[dict],
    table_metadata: list[dict] | None = None,
) -> list[dict]:
    """Build text/merged-chunk and table blocks from Top-K chunks."""
    query_terms = _tokens(question)
    query_years = set(_YEAR_RE.findall(question))
    pages_by_key = {_page_key(page): page for page in page_metadata}
    pages_by_fallback = {_fallback_page_key(page): page for page in page_metadata}
    chunks_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rank, chunk in enumerate(chunk_candidates, 1):
        page = pages_by_key.get(_page_key(chunk)) or pages_by_fallback.get(_fallback_page_key(chunk), {})
        chunks_by_page[_page_key(page or chunk)].append({
            **chunk,
            "_rank": _int(chunk.get("merged_rank")) or rank,
            "_section": _section(chunk, page),
            "_page": page,
        })

    blocks = []
    for page_key, page_chunks in chunks_by_page.items():
        remaining = sorted(page_chunks, key=lambda item: item["_rank"])
        while remaining:
            anchor = remaining.pop(0)
            merged = [anchor]
            keep = []
            for candidate in remaining:
                same_section = candidate["_section"].casefold() == anchor["_section"].casefold()
                if same_section and _jaccard(str(anchor.get("text") or ""), str(candidate.get("text") or "")) >= 0.35:
                    merged.append(candidate)
                else:
                    keep.append(candidate)
            remaining = keep
            page = anchor["_page"] or anchor
            texts = [str(item.get("text") or "").strip() for item in merged if str(item.get("text") or "").strip()]
            nearby = _nearby_window(str(page.get("page_text") or ""), texts[0] if texts else "", query_terms)
            content = _render_text_block(anchor["_section"], texts, nearby)
            rank = min(item["_rank"] for item in merged)
            blocks.append({
                "block_id": "text:" + ":".join([
                    str(page.get("page_id") or f"{page_key[0]}:{page_key[1]}"),
                    ",".join(_chunk_id(item) or str(item["_rank"]) for item in merged),
                ]),
                "block_type": "text" if len(merged) == 1 else "chunk_merge",
                "document_id": page.get("document_id") or anchor.get("document_id"),
                "source_pages": [{
                    "page_id": page.get("page_id"),
                    "filename": page.get("filename") or anchor.get("filename"),
                    "page_number": _int(page.get("page_number") or anchor.get("page_number")),
                }],
                "source_chunk_ids": [_chunk_id(item) for item in merged if _chunk_id(item)],
                "section_title": anchor["_section"],
                "content": content,
                "best_chunk_rank": rank,
                "chunk_support": len(merged),
                "lexical_raw": _coverage(query_terms, content),
                "structure_raw": _coverage(query_terms, anchor["_section"]),
                "year_raw": len(query_years & set(_YEAR_RE.findall(content))) / len(query_years) if query_years else 0.0,
            })

    candidate_page_keys = set(chunks_by_page)
    for table in table_metadata or []:
        page = pages_by_key.get(_page_key(table)) or pages_by_fallback.get(_fallback_page_key(table), {})
        page_key = _page_key(page or table)
        if page_key not in candidate_page_keys:
            continue
        rows = table.get("rows") if isinstance(table.get("rows"), list) else []
        columns = table.get("columns") if isinstance(table.get("columns"), list) else []
        if not rows or not columns:
            continue
        target_rows = _target_rows(question, rows)
        content = _render_table_block(table, target_rows)
        source_ranks = [item["_rank"] for item in chunks_by_page[page_key]]
        blocks.append({
            "block_id": f"table:{table.get('table_id') or page_key}",
            "block_type": "table",
            "document_id": table.get("document_id") or page.get("document_id"),
            "source_pages": [{
                "page_id": table.get("page_id") or page.get("page_id"),
                "filename": table.get("filename") or page.get("filename"),
                "page_number": _int(table.get("page_number") or page.get("page_number")),
            }],
            "source_chunk_ids": [_chunk_id(item) for item in chunks_by_page[page_key] if _chunk_id(item)],
            "table_id": table.get("table_id"),
            "table_title": table.get("title") or table.get("caption"),
            "target_rows": target_rows,
            "unit": table.get("unit"),
            "scale": table.get("scale"),
            "content": content,
            "best_chunk_rank": min(source_ranks, default=len(chunk_candidates) + 1),
            "chunk_support": len(chunks_by_page[page_key]),
            "lexical_raw": _coverage(query_terms, content),
            "structure_raw": _coverage(query_terms, "\n".join([
                str(table.get("title") or table.get("caption") or ""),
                " ".join(str(column) for column in columns),
            ])),
            "year_raw": len(query_years & set(_YEAR_RE.findall(content))) / len(query_years) if query_years else 0.0,
        })

    rank_raw = [1.0 / math.log2(block["best_chunk_rank"] + 2.0) for block in blocks]
    support_raw = [math.log2(block["chunk_support"] + 1.0) for block in blocks]
    for block, rank_score, support_score in zip(blocks, _normalize(rank_raw), _normalize(support_raw)):
        components = {
            "chunk_rank": rank_score,
            "query_overlap": block.pop("lexical_raw"),
            "title_header_overlap": block.pop("structure_raw"),
            "year_match": block.pop("year_raw"),
            "chunk_support": support_score,
        }
        block["score_components"] = {name: round(value, 8) for name, value in components.items()}
        block["block_score"] = round(sum(components.values()) / len(components), 8)
    blocks.sort(key=lambda item: (-item["block_score"], item["best_chunk_rank"], item["block_id"]))
    for rank, block in enumerate(blocks, 1):
        block["block_rank"] = rank
    return blocks


def select_evidence_blocks_v1(
    question: str,
    chunk_candidates: list[dict],
    *,
    page_metadata: list[dict],
    table_metadata: list[dict] | None = None,
    max_blocks: int = 12,
    max_context_chars: int = 28000,
) -> tuple[list[dict], str, dict[str, Any]]:
    blocks = build_evidence_blocks_v1(
        question, chunk_candidates, page_metadata=page_metadata, table_metadata=table_metadata,
    )
    selected, rendered = [], []
    used_chars = 0
    for block in blocks:
        source = block["source_pages"][0]
        value = (
            f"Source: {source.get('filename') or ''}, internal page {source.get('page_number', 0)}\n"
            f"Block ID: {block['block_id']}\n{block['content']}"
        )
        separator = 2 if rendered else 0
        if used_chars + separator + len(value) > max_context_chars:
            continue
        selected.append(block)
        rendered.append(value)
        used_chars += separator + len(value)
        if len(selected) >= max_blocks:
            break
    context = "\n\n".join(rendered)
    trace = {
        "selector": "evidence_block_v1",
        "candidate_block_count": len(blocks),
        "selected_block_count": len(selected),
        "context_chars": len(context),
        "max_blocks": max_blocks,
        "max_context_chars": max_context_chars,
        "block_scores": [
            {
                "block_id": block["block_id"],
                "block_type": block["block_type"],
                "block_rank": block["block_rank"],
                "block_score": block["block_score"],
                "score_components": block["score_components"],
                "source_pages": block["source_pages"],
                "source_chunk_ids": block["source_chunk_ids"],
                "selected": block in selected,
                "content_chars": len(block["content"]),
            }
            for block in blocks
        ],
        "selected_blocks": [block["block_id"] for block in selected],
    }
    return selected, context, trace
