from datetime import datetime
from typing import List
from sqlalchemy import inspect, text, tuple_

try:
    from database import SessionLocal, engine
    from evidence_identity import build_document_id, build_page_id, build_table_id
    from models import DocumentTable
except ModuleNotFoundError:
    from backend.database import SessionLocal, engine
    from backend.evidence_identity import build_document_id, build_page_id, build_table_id
    from backend.models import DocumentTable


class TableStore:
    """Store structured table records in PostgreSQL."""

    def __init__(self):
        self._ensure_schema()

    @staticmethod
    def _ensure_schema() -> None:
        try:
            bound_engine = getattr(SessionLocal, "kw", {}).get("bind") or engine
            inspector = inspect(bound_engine)
            if "document_tables" not in inspector.get_table_names():
                return
            columns = {item["name"] for item in inspector.get_columns("document_tables")}
            alter_sql = {
                "document_id": "ALTER TABLE document_tables ADD COLUMN document_id VARCHAR(64) NOT NULL DEFAULT ''",
                "page_id": "ALTER TABLE document_tables ADD COLUMN page_id VARCHAR(128) NOT NULL DEFAULT ''",
                "start_page": "ALTER TABLE document_tables ADD COLUMN start_page INTEGER NOT NULL DEFAULT 0",
                "end_page": "ALTER TABLE document_tables ADD COLUMN end_page INTEGER NOT NULL DEFAULT 0",
                "parser_backend": "ALTER TABLE document_tables ADD COLUMN parser_backend VARCHAR(50) NOT NULL DEFAULT ''",
                "quality_score": "ALTER TABLE document_tables ADD COLUMN quality_score DOUBLE PRECISION NOT NULL DEFAULT 0",
                "unit": "ALTER TABLE document_tables ADD COLUMN unit VARCHAR(100) NOT NULL DEFAULT ''",
                "scale": "ALTER TABLE document_tables ADD COLUMN scale VARCHAR(100) NOT NULL DEFAULT ''",
            }
            for name, sql in alter_sql.items():
                if name in columns:
                    continue
                with bound_engine.begin() as connection:
                    connection.execute(text(sql))
        except Exception:
            return

    @staticmethod
    def _normalize_string(value) -> str:
        return "" if value is None else str(value)

    @staticmethod
    def _normalize_list(value) -> list:
        return value if isinstance(value, list) else []

    @staticmethod
    def _normalize_int(value) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _to_dict(cls, item: DocumentTable) -> dict:
        return {
            "table_id": item.table_id,
            "document_id": item.document_id,
            "page_id": item.page_id,
            "filename": item.filename,
            "doc_name": item.doc_name,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "start_page": item.start_page,
            "end_page": item.end_page,
            "table_index": item.table_index,
            "parser_backend": item.parser_backend,
            "quality_score": item.quality_score,
            "title": item.title,
            "caption": item.caption,
            "before_context": item.before_context,
            "after_context": item.after_context,
            "columns": list(item.columns or []),
            "rows": list(item.rows or []),
            "html": item.html,
            "csv_text": item.csv_text,
            "unit": item.unit,
            "scale": item.scale,
        }

    def upsert_tables(self, tables: List[dict]) -> int:
        if not tables:
            return 0

        db = SessionLocal()
        upserted = 0
        try:
            for table in tables:
                filename = self._normalize_string(table.get("filename")).strip()
                if not filename:
                    continue
                page_number = self._normalize_int(table.get("page_number"))
                document_id = self._normalize_string(table.get("document_id")).strip() or build_document_id(
                    file_path=self._normalize_string(table.get("file_path")), filename=filename
                )
                page_id = self._normalize_string(table.get("page_id")).strip() or build_page_id(document_id, page_number)
                table_index = max(1, self._normalize_int(table.get("table_index")))
                supplied_table_id = self._normalize_string(table.get("table_id")).strip()
                if "table_id" in table and not supplied_table_id:
                    continue
                table_id = supplied_table_id or build_table_id(page_id, table_index)

                record = db.query(DocumentTable).filter(DocumentTable.table_id == table_id).first()
                normalized_unit = self._normalize_string(table.get("normalized_unit")).strip()
                unit = self._normalize_string(table.get("unit") or normalized_unit).strip()
                scale = self._normalize_string(table.get("scale")).strip()
                before_context = self._normalize_string(table.get("before_context"))
                if normalized_unit and normalized_unit not in before_context:
                    before_context = f"{normalized_unit}\n{before_context}".strip()
                payload = {
                    "document_id": document_id,
                    "page_id": page_id,
                    "filename": filename,
                    "doc_name": self._normalize_string(table.get("doc_name")),
                    "file_type": self._normalize_string(table.get("file_type")),
                    "file_path": self._normalize_string(table.get("file_path")),
                    "page_number": page_number,
                    "start_page": self._normalize_int(table.get("start_page", page_number)),
                    "end_page": self._normalize_int(table.get("end_page", page_number)),
                    "table_index": table_index,
                    "parser_backend": self._normalize_string(table.get("parser_backend")),
                    "quality_score": float(table.get("quality_score") or 0.0),
                    "title": self._normalize_string(table.get("normalized_title") or table.get("title")),
                    "caption": self._normalize_string(table.get("caption")),
                    "before_context": before_context,
                    "after_context": self._normalize_string(table.get("after_context")),
                    "columns": self._normalize_list(table.get("normalized_columns") or table.get("columns")),
                    "rows": self._normalize_list(table.get("normalized_rows") or table.get("rows")),
                    "html": self._normalize_string(table.get("html")),
                    "csv_text": self._normalize_string(table.get("csv_text")),
                    "unit": unit,
                    "scale": scale,
                    "updated_at": datetime.utcnow(),
                }

                if record:
                    for key, value in payload.items():
                        setattr(record, key, value)
                else:
                    db.add(DocumentTable(table_id=table_id, **payload))
                upserted += 1

            db.commit()
        finally:
            db.close()

        return upserted

    def get_tables_by_ids(self, table_ids: List[str]) -> List[dict]:
        if not table_ids:
            return []

        normalized_ids = []
        seen = set()
        for table_id in table_ids:
            key = self._normalize_string(table_id).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            normalized_ids.append(key)

        if not normalized_ids:
            return []

        db = SessionLocal()
        try:
            rows = db.query(DocumentTable).filter(DocumentTable.table_id.in_(normalized_ids)).all()
            by_id = {row.table_id: self._to_dict(row) for row in rows}
            return [by_id[table_id] for table_id in normalized_ids if table_id in by_id]
        finally:
            db.close()

    def get_tables_by_filename(self, filename: str) -> List[dict]:
        normalized_filename = self._normalize_string(filename).strip()
        if not normalized_filename:
            return []

        db = SessionLocal()
        try:
            rows = (
                db.query(DocumentTable)
                .filter(DocumentTable.filename == normalized_filename)
                .order_by(DocumentTable.page_number.asc(), DocumentTable.table_index.asc())
                .all()
            )
            return [self._to_dict(row) for row in rows]
        finally:
            db.close()

    def get_tables_by_page_keys(self, keys: List[tuple[str, int]]) -> List[dict]:
        """Read tables attached to an explicit page set for rerank representations."""
        normalized = list(dict.fromkeys(
            (self._normalize_string(filename).strip(), self._normalize_int(page_number))
            for filename, page_number in keys
            if self._normalize_string(filename).strip()
        ))
        if not normalized:
            return []
        db = SessionLocal()
        try:
            rows = db.query(DocumentTable).filter(
                tuple_(DocumentTable.filename, DocumentTable.page_number).in_(normalized)
            ).order_by(DocumentTable.filename.asc(), DocumentTable.page_number.asc(), DocumentTable.table_index.asc()).all()
            return [self._to_dict(row) for row in rows]
        finally:
            db.close()

    def get_tables_by_page_ids(self, page_ids: List[str]) -> List[dict]:
        normalized = list(dict.fromkeys(
            self._normalize_string(page_id).strip() for page_id in page_ids
            if self._normalize_string(page_id).strip()
        ))
        if not normalized:
            return []
        db = SessionLocal()
        try:
            rows = (
                db.query(DocumentTable)
                .filter(DocumentTable.page_id.in_(normalized))
                .order_by(DocumentTable.page_id.asc(), DocumentTable.table_index.asc())
                .all()
            )
            return [self._to_dict(row) for row in rows]
        finally:
            db.close()

    def delete_by_filename(self, filename: str) -> int:
        normalized_filename = self._normalize_string(filename).strip()
        if not normalized_filename:
            return 0

        db = SessionLocal()
        try:
            deleted = db.query(DocumentTable).filter(DocumentTable.filename == normalized_filename).count()
            if deleted > 0:
                db.query(DocumentTable).filter(DocumentTable.filename == normalized_filename).delete(synchronize_session=False)
                db.commit()
            return deleted
        finally:
            db.close()
