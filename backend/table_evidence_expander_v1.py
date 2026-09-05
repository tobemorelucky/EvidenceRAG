"""Shadow-only structural table expansion for already selected chunks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


MODES = {"direct_table_id", "page_table_fallback"}


def _value(chunk: dict, key: str, default: Any = "") -> Any:
    value = chunk.get(key)
    if value not in (None, ""):
        return value
    metadata = chunk.get("metadata") or {}
    return metadata.get(key, default)


def _rank(chunk: dict, fallback: int) -> int:
    for key in ("rank", "jina_rank", "rrf_rank"):
        try:
            value = int(_value(chunk, key, 0) or 0)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return fallback


def _chunk_unit(chunk: dict, rank: int) -> dict:
    return {
        "source_type": "chunk",
        "chunk_id": str(_value(chunk, "chunk_id") or ""),
        "document_id": str(_value(chunk, "document_id") or ""),
        "page_id": str(_value(chunk, "page_id") or ""),
        "filename": str(_value(chunk, "filename") or ""),
        "page_number": int(_value(chunk, "page_number", 0) or 0),
        "text": str(_value(chunk, "text") or ""),
        "rank": rank,
        "metadata": deepcopy(chunk.get("metadata") or {}),
    }


def _row_text(row: object, columns: list[str]) -> str:
    if isinstance(row, dict):
        keys = columns or [str(key) for key in row if not str(key).startswith("_")]
        return " | ".join(
            f"{key}: {str(row.get(key, '') or '').strip()}"
            for key in keys
            if str(row.get(key, "") or "").strip()
        )
    if isinstance(row, (list, tuple)):
        return " | ".join(str(value or "").strip() for value in row if str(value or "").strip())
    return str(row or "").strip()


def _table_parts(table: dict, max_chars: int) -> tuple[str, str, list[str]]:
    title = str(table.get("title") or table.get("caption") or "").strip()
    columns = [str(value or "").strip() for value in (table.get("columns") or [])]
    unit = str(table.get("unit") or "").strip()
    scale = str(table.get("scale") or "").strip()
    prefix_lines = ["[Table Expansion]", f"Table ID: {table.get('table_id', '')}"]
    if title:
        prefix_lines.append(f"Title: {title}")
    prefix_lines.append(f"Header: {' | '.join(columns)}")
    if unit:
        prefix_lines.append(f"Unit: {unit}")
    if scale:
        prefix_lines.append(f"Scale: {scale}")
    prefix = "\n".join(prefix_lines)
    rows = [value for row in (table.get("rows") or []) if (value := _row_text(row, columns))]
    selected_rows = []
    current = prefix
    for row in rows:
        candidate = f"{current}\nRow: {row}"
        if len(candidate) > max_chars:
            break
        selected_rows.append(row)
        current = candidate
    if rows and not selected_rows and len(prefix) < max_chars:
        room = max_chars - len(prefix) - len("\nRow: ")
        if room > 0:
            fragment = rows[0][:room]
            selected_rows.append(fragment)
            current = f"{prefix}\nRow: {fragment}"
    return prefix, current, selected_rows


def _format_chunk(unit: dict) -> str:
    return f"Source: {unit['filename']} | Page: {unit['page_number']}\n{unit['text']}"


def _format_table(unit: dict) -> str:
    return f"Source: {unit['filename']} | Page: {unit['page_number']}\n{unit['text']}"


def _render(units: list[dict]) -> str:
    return "\n\n".join(
        _format_chunk(unit) if unit["source_type"] == "chunk" else _format_table(unit)
        for unit in units
    )


def _valid_table(table: dict) -> bool:
    return bool(
        str(table.get("table_id") or "").strip()
        and str(table.get("document_id") or "").strip()
        and str(table.get("page_id") or "").strip()
        and (table.get("columns") or [])
        and (table.get("rows") or [])
    )


def expand_table_evidence_v1(
    retrieved_chunks: list[dict],
    *,
    table_store,
    mode: str = "direct_table_id",
    max_tables: int = 1,
    max_table_chars: int = 4000,
    max_context_chars: int = 28000,
    original_context: str | None = None,
) -> tuple[list[dict], str, dict]:
    """Expand exact table/page associations without modifying the input chunks."""
    if mode not in MODES:
        raise ValueError(f"Unsupported TABLE_EXPANSION_MODE: {mode}")
    chunks = [_chunk_unit(chunk, _rank(chunk, index)) for index, chunk in enumerate(retrieved_chunks or [], 1)]
    direct_ids = list(dict.fromkeys(
        str(_value(chunk, "table_id") or "").strip()
        for chunk in retrieved_chunks or []
        if str(_value(chunk, "table_id") or "").strip()
    ))
    direct_tables = table_store.get_tables_by_ids(direct_ids) if direct_ids else []
    tables_by_id = {str(table.get("table_id") or ""): table for table in direct_tables if _valid_table(table)}

    candidates: list[tuple[dict, dict, str]] = []
    association_mismatches = 0
    for source, chunk in zip(retrieved_chunks or [], chunks):
        table_id = str(_value(source, "table_id") or "").strip()
        table = tables_by_id.get(table_id)
        if table:
            if table.get("document_id") == chunk["document_id"] and table.get("page_id") == chunk["page_id"]:
                candidates.append((chunk, table, "direct_table_id"))
            else:
                association_mismatches += 1

    page_lookup_count = 0
    if mode == "page_table_fallback":
        page_ids = list(dict.fromkeys(chunk["page_id"] for chunk in chunks if chunk["page_id"]))
        page_tables = table_store.get_tables_by_page_ids(page_ids) if page_ids else []
        page_lookup_count = len(page_ids)
        tables_by_page: dict[str, list[dict]] = {}
        for table in page_tables:
            if _valid_table(table):
                tables_by_page.setdefault(str(table["page_id"]), []).append(table)
        existing = {table["table_id"] for _, table, _ in candidates}
        for chunk in chunks:
            for table in tables_by_page.get(chunk["page_id"], []):
                if table["table_id"] in existing:
                    continue
                if table["document_id"] != chunk["document_id"] or table["page_id"] != chunk["page_id"]:
                    association_mismatches += 1
                    continue
                candidates.append((chunk, table, "page_table_fallback"))
                existing.add(table["table_id"])

    candidates.sort(key=lambda item: (item[0]["rank"], str(item[1]["table_id"])))
    candidates = candidates[: max(0, int(max_tables))]
    table_units = []
    anchor_ids = set()
    for anchor, table, association_method in candidates:
        prefix, text, rows = _table_parts(table, max(1, int(max_table_chars)))
        anchor_ids.add(anchor["chunk_id"])
        table_units.append({
            "source_type": "table",
            "table_id": str(table["table_id"]),
            "document_id": str(table["document_id"]),
            "page_id": str(table["page_id"]),
            "filename": str(table.get("filename") or anchor["filename"]),
            "page_number": int(table.get("page_number") or anchor["page_number"]),
            "title": str(table.get("title") or table.get("caption") or ""),
            "header": [str(value or "") for value in (table.get("columns") or [])],
            "rows": rows,
            "unit": str(table.get("unit") or ""),
            "scale": str(table.get("scale") or ""),
            "text": text,
            "protected_prefix": prefix,
            "anchor_chunk_ids": [anchor["chunk_id"]],
            "association_method": association_method,
            "rank": anchor["rank"],
        })

    before_context = original_context if original_context is not None else _render(chunks)
    if not table_units:
        return chunks, before_context, {
            "mode": mode,
            "direct_table_id_count": len(direct_ids),
            "page_lookup_count": page_lookup_count,
            "expanded_table_count": 0,
            "expanded_table_ids": [],
            "expansion_before_chars": len(before_context),
            "expansion_after_chars": len(before_context),
            "removed_units": [],
            "association_mismatch_count": association_mismatches,
        }

    selected_chunks = list(chunks)
    units = selected_chunks + table_units
    removed = []
    while len(_render(units)) > max_context_chars:
        removable = [unit for unit in selected_chunks if unit["chunk_id"] not in anchor_ids]
        if not removable:
            break
        victim = max(removable, key=lambda unit: (unit["rank"], len(unit["text"])))
        selected_chunks.remove(victim)
        units.remove(victim)
        removed.append({
            "unit_id": victim["chunk_id"],
            "source_type": "chunk",
            "rank": victim["rank"],
            "reason": "low_rank_non_table_budget_replacement",
        })

    for table_unit in table_units:
        if len(_render(units)) <= max_context_chars:
            break
        overflow = len(_render(units)) - max_context_chars
        prefix = table_unit["protected_prefix"]
        keep = max(len(prefix), len(table_unit["text"]) - overflow)
        if keep < len(table_unit["text"]):
            table_unit["text"] = table_unit["text"][:keep]
            table_unit["rows_truncated"] = True

    context = _render(units)
    if len(context) > max_context_chars:
        # The protected anchor plus table structure cannot fit. Keep the untouched
        # baseline instead of corrupting either protected unit.
        return chunks, before_context, {
            "mode": mode,
            "direct_table_id_count": len(direct_ids),
            "page_lookup_count": page_lookup_count,
            "expanded_table_count": 0,
            "expanded_table_ids": [],
            "expansion_before_chars": len(before_context),
            "expansion_after_chars": len(before_context),
            "removed_units": [],
            "association_mismatch_count": association_mismatches,
            "budget_rejection": "protected_anchor_and_table_structure_do_not_fit",
        }
    return units, context, {
        "mode": mode,
        "direct_table_id_count": len(direct_ids),
        "page_lookup_count": page_lookup_count,
        "expanded_table_count": len(table_units),
        "expanded_table_ids": [unit["table_id"] for unit in table_units],
        "expansion_before_chars": len(before_context),
        "expansion_after_chars": len(context),
        "removed_units": removed,
        "association_mismatch_count": association_mismatches,
    }
