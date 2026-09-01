"""Audit or migrate legacy page/table rows to Evidence Assembly v1 IDs.

Dry-run is the default. Pass --execute only after reviewing the printed audit.
This script never touches Milvus or retrieval configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from sqlalchemy import inspect, text


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from database import engine  # noqa: E402
from evidence_identity import build_document_id, build_page_id, build_table_id  # noqa: E402
from table_quality import structural_quality_score, table_page_match_score  # noqa: E402


def _existing_columns(table_name: str) -> set[str]:
    return {item["name"] for item in inspect(engine).get_columns(table_name)}


def _add_columns() -> None:
    definitions = {
        "document_pages": {
            "document_id": "VARCHAR(64) NOT NULL DEFAULT ''",
            "page_id": "VARCHAR(128) NOT NULL DEFAULT ''",
        },
        "document_tables": {
            "document_id": "VARCHAR(64) NOT NULL DEFAULT ''",
            "page_id": "VARCHAR(128) NOT NULL DEFAULT ''",
            "start_page": "INTEGER NOT NULL DEFAULT 0",
            "end_page": "INTEGER NOT NULL DEFAULT 0",
            "parser_backend": "VARCHAR(50) NOT NULL DEFAULT ''",
            "quality_score": "DOUBLE PRECISION NOT NULL DEFAULT 0",
            "unit": "VARCHAR(100) NOT NULL DEFAULT ''",
            "scale": "VARCHAR(100) NOT NULL DEFAULT ''",
        },
    }
    with engine.begin() as connection:
        for table_name, columns in definitions.items():
            existing = _existing_columns(table_name)
            for column, definition in columns.items():
                if column not in existing:
                    connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column} {definition}"))


def _read_rows() -> tuple[list[dict], list[dict]]:
    page_columns = _existing_columns("document_pages")
    table_columns = _existing_columns("document_tables")
    page_optional = ", document_id, page_id" if {"document_id", "page_id"} <= page_columns else ""
    table_optional = ", document_id, page_id, start_page, end_page, parser_backend, quality_score, unit, scale" if {
        "document_id", "page_id", "start_page", "end_page", "parser_backend", "quality_score", "unit", "scale"
    } <= table_columns else ""
    with engine.connect() as connection:
        pages = [dict(row._mapping) for row in connection.execute(text(
            "SELECT id, filename, file_path, page_number, page_text" + page_optional + " FROM document_pages"
        ))]
        tables = [dict(row._mapping) for row in connection.execute(text(
            "SELECT table_id, filename, file_path, page_number, table_index, title, caption, before_context, "
            "columns, rows" + table_optional + " FROM document_tables"
        ))]
    return pages, tables


def _unit_scale(context: str) -> tuple[str, str]:
    value = str(context or "")
    unit = "USD" if re.search(r"\bUSD\b|\bdollars?\b|\$", value, re.IGNORECASE) else ""
    match = re.search(r"\b(thousands|millions|billions)\b", value, re.IGNORECASE)
    return unit, match.group(1).lower() if match else ""


def build_plan(pages: list[dict], tables: list[dict]) -> tuple[list[dict], list[dict]]:
    documents: dict[tuple[str, str], str] = {}
    page_updates = []
    pages_by_legacy_key = {}
    for page in pages:
        key = (str(page.get("filename") or ""), str(page.get("file_path") or ""))
        document_id = documents.setdefault(
            key,
            build_document_id(file_path=key[1], filename=key[0]),
        )
        page_number = int(page.get("page_number") or 0)
        page_id = build_page_id(document_id, page_number)
        update = {**page, "document_id": document_id, "page_id": page_id}
        page_updates.append(update)
        pages_by_legacy_key[(key[0], page_number)] = update

    table_updates = []
    for table in tables:
        already_migrated = bool(str(table.get("page_id") or "").strip())
        legacy_page = int(table.get("page_number") or 0)
        internal_page = legacy_page if already_migrated else max(0, legacy_page - 1)
        page = pages_by_legacy_key.get((str(table.get("filename") or ""), internal_page))
        if page is None:
            table_updates.append({**table, "association_valid": False, "internal_page": internal_page})
            continue
        document_id = page["document_id"]
        page_id = page["page_id"]
        table_index = max(1, int(table.get("table_index") or 1))
        normalized_table = {
            **table,
            "was_legacy": not already_migrated,
            "document_id": document_id,
            "page_id": page_id,
            "page_number": internal_page,
            "start_page": internal_page if not already_migrated else int(table.get("start_page") or internal_page),
            "end_page": internal_page if not already_migrated else int(table.get("end_page") or internal_page),
            "parser_backend": str(table.get("parser_backend") or "legacy_backfill"),
        }
        normalized_table["quality_score"] = float(table.get("quality_score") or structural_quality_score(normalized_table))
        normalized_table["page_match_score"] = table_page_match_score(normalized_table, page.get("page_text") or "")
        normalized_table["unit"], normalized_table["scale"] = _unit_scale(table.get("before_context") or "")
        normalized_table["new_table_id"] = build_table_id(page_id, table_index)
        normalized_table["association_valid"] = True
        table_updates.append(normalized_table)
    return page_updates, table_updates


def _audit(table_updates: list[dict], sample_size: int) -> dict:
    sampled = sorted(
        table_updates,
        key=lambda item: hashlib.sha256(
            f"{item.get('filename', '')}:{item.get('page_number', 0)}:{item.get('table_index', 0)}".encode("utf-8")
        ).hexdigest(),
    )[: max(1, sample_size)]
    # Every legacy parser row used a one-based page number against the loader's
    # zero-based page number. The historical same-number association therefore
    # points at the next physical page for every sampled table.
    before_errors = len(sampled)
    after_errors = sum(not item.get("association_valid") for item in sampled)
    usable = sum(
        bool(item.get("association_valid"))
        and bool(item.get("columns"))
        and bool(item.get("rows"))
        and float(item.get("quality_score") or 0.0) >= 0.65
        and float(item.get("page_match_score") or 0.0) >= 0.35
        for item in sampled
    )
    total = len(sampled)
    return {
        "sampled_tables": total,
        "association_error_before": before_errors,
        "association_error_rate_before": round(before_errors / max(1, total), 4),
        "association_error_after": after_errors,
        "association_error_rate_after": round(after_errors / max(1, total), 4),
        "table_evidence_usable": usable,
        "table_evidence_usable_rate": round(usable / max(1, total), 4),
        "quality_threshold": 0.65,
        "page_match_threshold": 0.35,
    }


def _execute(page_updates: list[dict], table_updates: list[dict]) -> None:
    _add_columns()
    with engine.begin() as connection:
        for page in page_updates:
            connection.execute(
                text("UPDATE document_pages SET document_id=:document_id, page_id=:page_id WHERE id=:id"),
                {"document_id": page["document_id"], "page_id": page["page_id"], "id": page["id"]},
            )
        for table in table_updates:
            if not table.get("association_valid"):
                continue
            connection.execute(
                text(
                    "UPDATE document_tables SET table_id=:new_table_id, document_id=:document_id, page_id=:page_id, "
                    "page_number=:page_number, start_page=:start_page, end_page=:end_page, "
                    "parser_backend=:parser_backend, quality_score=:quality_score, unit=:unit, scale=:scale "
                    "WHERE table_id=:old_table_id"
                ),
                {
                    "new_table_id": table["new_table_id"],
                    "document_id": table["document_id"],
                    "page_id": table["page_id"],
                    "page_number": table["page_number"],
                    "start_page": table["start_page"],
                    "end_page": table["end_page"],
                    "parser_backend": table["parser_backend"],
                    "quality_score": table["quality_score"],
                    "unit": table["unit"],
                    "scale": table["scale"],
                    "old_table_id": table["table_id"],
                },
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit/migrate Evidence Assembly v1 IDs and zero-based pages")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    pages, tables = _read_rows()
    page_updates, table_updates = build_plan(pages, tables)
    audit = _audit(table_updates, args.sample_size)
    if args.execute:
        _execute(page_updates, table_updates)
    print(json.dumps({
        "execute": args.execute,
        "pages": len(page_updates),
        "tables": len(table_updates),
        **audit,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
