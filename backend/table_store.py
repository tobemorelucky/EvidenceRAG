from datetime import datetime
from typing import List

try:
    from database import SessionLocal
    from models import DocumentTable
except ModuleNotFoundError:
    from backend.database import SessionLocal
    from backend.models import DocumentTable


class TableStore:
    """Store structured table records in PostgreSQL."""

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
            "filename": item.filename,
            "doc_name": item.doc_name,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "table_index": item.table_index,
            "title": item.title,
            "caption": item.caption,
            "before_context": item.before_context,
            "after_context": item.after_context,
            "columns": list(item.columns or []),
            "rows": list(item.rows or []),
            "html": item.html,
            "csv_text": item.csv_text,
        }

    def upsert_tables(self, tables: List[dict]) -> int:
        if not tables:
            return 0

        db = SessionLocal()
        upserted = 0
        try:
            for table in tables:
                table_id = self._normalize_string(table.get("table_id")).strip()
                filename = self._normalize_string(table.get("filename")).strip()
                if not table_id or not filename:
                    continue

                record = db.query(DocumentTable).filter(DocumentTable.table_id == table_id).first()
                normalized_unit = self._normalize_string(table.get("normalized_unit")).strip()
                before_context = self._normalize_string(table.get("before_context"))
                if normalized_unit and normalized_unit not in before_context:
                    before_context = f"{normalized_unit}\n{before_context}".strip()
                payload = {
                    "filename": filename,
                    "doc_name": self._normalize_string(table.get("doc_name")),
                    "file_type": self._normalize_string(table.get("file_type")),
                    "file_path": self._normalize_string(table.get("file_path")),
                    "page_number": self._normalize_int(table.get("page_number")),
                    "table_index": self._normalize_int(table.get("table_index")),
                    "title": self._normalize_string(table.get("normalized_title") or table.get("title")),
                    "caption": self._normalize_string(table.get("caption")),
                    "before_context": before_context,
                    "after_context": self._normalize_string(table.get("after_context")),
                    "columns": self._normalize_list(table.get("normalized_columns") or table.get("columns")),
                    "rows": self._normalize_list(table.get("normalized_rows") or table.get("rows")),
                    "html": self._normalize_string(table.get("html")),
                    "csv_text": self._normalize_string(table.get("csv_text")),
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
