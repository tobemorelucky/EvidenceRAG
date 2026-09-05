"""Shadow-only evidence packages built from frozen Jina chunks."""

from __future__ import annotations

import hashlib
import re
from copy import deepcopy


_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def _value(item: dict, key: str, default=""):
    value = item.get(key)
    if value not in (None, ""):
        return value
    return (item.get("metadata") or {}).get(key, default)


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _rank(chunk: dict) -> int:
    try:
        return int(_value(chunk, "jina_rank", _value(chunk, "rank", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def _chunk_record(chunk: dict) -> dict:
    return {
        "chunk_id": _clean(_value(chunk, "chunk_id")),
        "document_id": _clean(_value(chunk, "document_id")),
        "page_id": _clean(_value(chunk, "page_id")),
        "filename": _clean(_value(chunk, "filename")),
        "page_number": int(_value(chunk, "page_number", 0) or 0),
        "text": str(_value(chunk, "text") or ""),
        "jina_rank": _rank(chunk),
        "company": _clean(_value(chunk, "company")),
        "report_year": _clean(_value(chunk, "report_year")),
        "table_title": _clean(_value(chunk, "table_title")),
        "location": _clean(_value(chunk, "location")),
    }


def _valid_table(table: dict) -> bool:
    return bool(
        _clean(table.get("table_id"))
        and _clean(table.get("document_id"))
        and _clean(table.get("page_id"))
        and table.get("columns")
        and table.get("rows")
    )


def _table_record(table: dict) -> dict:
    return {
        "table_id": _clean(table.get("table_id")),
        "document_id": _clean(table.get("document_id")),
        "page_id": _clean(table.get("page_id")),
        "filename": _clean(table.get("filename")),
        "page_number": int(table.get("page_number") or 0),
        "title": _clean(table.get("title") or table.get("caption")),
        "header": [_clean(value) for value in table.get("columns") or []],
        "rows": deepcopy(list(table.get("rows") or [])),
        "unit": _clean(table.get("unit")),
        "scale": _clean(table.get("scale")),
    }


def _metadata(chunks: list[dict], tables: list[dict]) -> dict:
    entity = next((chunk["company"] for chunk in chunks if chunk["company"]), "")
    period = next((chunk["report_year"] for chunk in chunks if chunk["report_year"] not in ("", "0")), "")
    if not period:
        years = _YEAR_RE.findall(" ".join(chunk["text"][:500] for chunk in chunks))
        period = years[0] if years else ""
    metric = next((table["title"] for table in tables if table["title"]), "")
    if not metric:
        metric = next((chunk["table_title"] for chunk in chunks if chunk["table_title"]), "")
    if not metric:
        metric = next((chunk["location"] for chunk in chunks if chunk["location"]), "")
    return {"entity": entity, "period": period, "metric": metric}


def build_evidence_packages_v1(jina_chunks: list[dict], tables: list[dict]) -> list[dict]:
    """Group frozen Jina chunks by exact document/page identity.

    Every package anchor is the best-ranked Jina input chunk on that page.
    Tables with a different document or page are never attached.
    """
    chunks = [_chunk_record(chunk) for chunk in deepcopy(jina_chunks or [])]
    if any(not chunk["chunk_id"] or not chunk["document_id"] or not chunk["page_id"] or chunk["jina_rank"] <= 0 for chunk in chunks):
        raise ValueError("Every package input must be an identified, ranked Jina chunk")
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Jina chunk IDs must be unique")
    chunks.sort(key=lambda chunk: (chunk["jina_rank"], chunk["chunk_id"]))

    tables_by_key: dict[tuple[str, str], list[dict]] = {}
    for table in tables or []:
        if not _valid_table(table):
            continue
        record = _table_record(table)
        tables_by_key.setdefault((record["document_id"], record["page_id"]), []).append(record)
    for page_tables in tables_by_key.values():
        page_tables.sort(key=lambda table: table["table_id"])

    grouped: dict[tuple[str, str], list[dict]] = {}
    for chunk in chunks:
        grouped.setdefault((chunk["document_id"], chunk["page_id"]), []).append(chunk)
    packages = []
    for (document_id, page_id), page_chunks in grouped.items():
        anchor = page_chunks[0]
        related_tables = tables_by_key.get((document_id, page_id), [])
        identity = f"{anchor['chunk_id']}|{document_id}|{page_id}"
        packages.append({
            "package_id": "package_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24],
            "anchor_chunk_id": anchor["chunk_id"],
            "anchor_jina_rank": anchor["jina_rank"],
            "document_id": document_id,
            "page_id": page_id,
            "filename": anchor["filename"],
            "page_number": anchor["page_number"],
            "text_chunks": page_chunks,
            "related_tables": deepcopy(related_tables),
            "metadata": _metadata(page_chunks, related_tables),
        })
    return sorted(packages, key=lambda package: (package["anchor_jina_rank"], package["package_id"]))


def _row_text(row: object, header: list[str]) -> str:
    if isinstance(row, dict):
        raw = _clean(row.get("_raw_line"))
        if raw:
            return raw
        keys = header or [str(key) for key in row if not str(key).startswith("_")]
        return " | ".join(
            f"{key}: {_clean(row.get(key))}" for key in keys if _clean(row.get(key))
        )
    if isinstance(row, (list, tuple)):
        return " | ".join(_clean(value) for value in row if _clean(value))
    return _clean(row)


def _table_text(table: dict) -> str:
    lines = [f"[Related Table] {table['table_id']}"]
    if table["title"]:
        lines.append(f"Title: {table['title']}")
    lines.append("Header: " + " | ".join(table["header"]))
    if table["unit"] or table["scale"]:
        lines.append("Unit: " + " ".join(value for value in (table["unit"], table["scale"]) if value))
    lines.extend(
        f"Row: {text}" for row in table["rows"] if (text := _row_text(row, table["header"]))
    )
    return "\n".join(lines)


def render_evidence_packages_v1(packages: list[dict], *, max_chars: int = 28000) -> tuple[str, dict]:
    """Render packages under a fixed budget without changing package content."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    parts = []
    remaining = max_chars
    included_chunks = []
    dropped_chunks = []
    included_tables = []
    table_chars = 0
    package_sources = []

    def add(text: str) -> bool:
        nonlocal remaining
        separator = 2 if parts else 0
        if len(text) + separator > remaining:
            return False
        parts.append(text)
        remaining -= len(text) + separator
        return True

    for package in packages:
        source = {
            "package_id": package["package_id"],
            "anchor_chunk_id": package["anchor_chunk_id"],
            "document_id": package["document_id"],
            "page_id": package["page_id"],
            "included_chunk_ids": [],
            "included_table_ids": [],
        }
        chunks = package["text_chunks"]
        anchor = chunks[0]
        anchor_text = (
            f"[Evidence Package {package['package_id']}]\n"
            f"Source: {anchor['filename']} | Page: {anchor['page_number']}\n{anchor['text']}"
        )
        if not add(anchor_text):
            dropped_chunks.extend({
                "chunk_id": chunk["chunk_id"],
                "jina_rank": chunk["jina_rank"],
                "reason": "anchor_context_budget" if chunk is anchor else "package_anchor_not_selected",
            } for chunk in chunks)
            continue
        included_chunks.append(anchor["chunk_id"])
        source["included_chunk_ids"].append(anchor["chunk_id"])
        for chunk in chunks[1:]:
            rendered = (
                f"[Evidence Package {package['package_id']}]\n"
                f"Source: {chunk['filename']} | Page: {chunk['page_number']}\n{chunk['text']}"
            )
            if add(rendered):
                included_chunks.append(chunk["chunk_id"])
                source["included_chunk_ids"].append(chunk["chunk_id"])
            else:
                dropped_chunks.append({
                    "chunk_id": chunk["chunk_id"],
                    "jina_rank": chunk["jina_rank"],
                    "reason": "context_budget",
                })
        for table in package["related_tables"]:
            rendered = _table_text(table)
            if add(rendered):
                included_tables.append(table["table_id"])
                source["included_table_ids"].append(table["table_id"])
                table_chars += len(rendered)
        if source["included_chunk_ids"] or source["included_table_ids"]:
            package_sources.append(source)
    context = "\n\n".join(parts)
    return context, {
        "context_chars": len(context),
        "included_chunk_ids": included_chunks,
        "dropped_chunks": dropped_chunks,
        "included_table_ids": included_tables,
        "table_chars": table_chars,
        "package_sources": package_sources,
    }
