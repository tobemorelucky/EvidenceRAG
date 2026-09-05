"""Build Evidence Fact Store v1 from PostgreSQL TableStore without APIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from database import SessionLocal  # noqa: E402
from evidence_fact_store_v1 import build_fact_index  # noqa: E402
from models import DocumentPage, DocumentTable  # noqa: E402
from table_store import TableStore  # noqa: E402


DEFAULT_OUTPUT = ROOT / "reports" / "fact_store_v1.json"


def load_tables() -> list[dict]:
    db = SessionLocal()
    try:
        pages = {
            row.page_id: {"company": row.company, "filename": row.filename}
            for row in db.query(DocumentPage.page_id, DocumentPage.company, DocumentPage.filename).all()
        }
        records = db.query(DocumentTable).order_by(
            DocumentTable.document_id.asc(),
            DocumentTable.page_number.asc(),
            DocumentTable.table_index.asc(),
        ).all()
        tables = []
        for record in records:
            table = TableStore._to_dict(record)
            page = pages.get(record.page_id, {})
            table["entity"] = str(page.get("company") or table.get("doc_name") or table.get("filename") or "")
            tables.append(table)
        return tables
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--quality-threshold", type=float, default=0.65)
    args = parser.parse_args()
    if not 0 <= args.quality_threshold <= 1:
        parser.error("--quality-threshold must be between 0 and 1")

    tables = load_tables()
    facts, stats = build_fact_index(tables, quality_threshold=args.quality_threshold)
    payload = {
        "schema": "evidence_fact_store_v1",
        "source": "postgresql.document_tables",
        "external_calls": {"llm": 0, "jina": 0, "judge": 0, "langsmith": 0},
        "stats": stats,
        "facts": facts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    print(f"Fact store: {args.output}", flush=True)


if __name__ == "__main__":
    main()
