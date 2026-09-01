from datetime import datetime
import os
from typing import List

from cache import cache
from database import SessionLocal, engine
from embedding import embedding_service
from finance_rag_features import build_embedding_cache_key, compute_page_features
from models import DocumentPage
from evidence_identity import build_document_id, build_page_id
from sqlalchemy import inspect, text, tuple_
from text_sanitizer import sanitize_text


class DocumentPageStore:
    """Store per-page aggregated text for page-level FinanceBench retrieval."""

    def __init__(self):
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        try:
            bound_engine = getattr(SessionLocal, "kw", {}).get("bind") or engine
            inspector = inspect(bound_engine)
            if "document_pages" not in inspector.get_table_names():
                return
            columns = {item["name"] for item in inspector.get_columns("document_pages")}
            alter_sql = {
                "document_id": "ALTER TABLE document_pages ADD COLUMN document_id VARCHAR(64) NOT NULL DEFAULT ''",
                "page_id": "ALTER TABLE document_pages ADD COLUMN page_id VARCHAR(128) NOT NULL DEFAULT ''",
                "embedding_cache_key": "ALTER TABLE document_pages ADD COLUMN embedding_cache_key VARCHAR(255) NOT NULL DEFAULT ''",
                "page_dense_embedding": "ALTER TABLE document_pages ADD COLUMN page_dense_embedding JSON NOT NULL DEFAULT '[]'",
                "page_tokens": "ALTER TABLE document_pages ADD COLUMN page_tokens JSON NOT NULL DEFAULT '[]'",
                "page_numbers": "ALTER TABLE document_pages ADD COLUMN page_numbers JSON NOT NULL DEFAULT '[]'",
                "page_years": "ALTER TABLE document_pages ADD COLUMN page_years JSON NOT NULL DEFAULT '[]'",
                "page_metric_tokens": "ALTER TABLE document_pages ADD COLUMN page_metric_tokens JSON NOT NULL DEFAULT '[]'",
                "company": "ALTER TABLE document_pages ADD COLUMN company VARCHAR(255) NOT NULL DEFAULT ''",
                "report_year": "ALTER TABLE document_pages ADD COLUMN report_year INTEGER NOT NULL DEFAULT 0",
                "financial_document_type": "ALTER TABLE document_pages ADD COLUMN financial_document_type VARCHAR(50) NOT NULL DEFAULT ''",
                "location": "ALTER TABLE document_pages ADD COLUMN location VARCHAR(255) NOT NULL DEFAULT ''",
                "content_hash": "ALTER TABLE document_pages ADD COLUMN content_hash VARCHAR(64) NOT NULL DEFAULT ''",
            }
            for name, sql in alter_sql.items():
                if name in columns:
                    continue
                with bound_engine.begin() as conn:
                    conn.execute(text(sql))
        except Exception:
            return

    @staticmethod
    def _to_dict(item: DocumentPage) -> dict:
        return {
            "document_id": item.document_id,
            "page_id": item.page_id,
            "doc_name": item.doc_name,
            "filename": item.filename,
            "file_type": item.file_type,
            "file_path": item.file_path,
            "page_number": item.page_number,
            "company": item.company,
            "report_year": item.report_year,
            "financial_document_type": item.financial_document_type,
            "location": item.location,
            "content_hash": item.content_hash,
            "page_text": item.page_text,
            "table_text": item.table_text,
            "chunk_ids": list(item.chunk_ids or []),
            "embedding_cache_key": item.embedding_cache_key,
            "page_dense_embedding": list(item.page_dense_embedding or []),
            "page_tokens": list(item.page_tokens or []),
            "page_numbers": list(item.page_numbers or []),
            "page_years": list(item.page_years or []),
            "page_metric_tokens": list(item.page_metric_tokens or []),
        }

    @staticmethod
    def _cache_key(filename: str, page_number: int) -> str:
        return f"document_page:{filename}:{page_number}"

    @staticmethod
    def _page_embedding_text(page_text: str) -> str:
        """Keep full page text in storage but cap only the vector input length."""
        max_chars = max(1000, int(os.getenv("PAGE_EMBEDDING_MAX_CHARS", "3000")))
        if len(page_text) <= max_chars:
            return page_text
        half = max_chars // 2
        return f"{page_text[:half]}\n[page text truncated for embedding]\n{page_text[-half:]}"

    def upsert_pages(self, pages: List[dict]) -> int:
        if not pages:
            return 0

        normalized_pages = []
        page_texts = []
        for page in pages:
            filename = (page.get("filename") or "").strip()
            if not filename:
                continue
            page_number = int(page.get("page_number", 0) or 0)
            document_id = str(page.get("document_id") or "").strip() or build_document_id(
                file_path=page.get("file_path", ""), filename=filename
            )
            page_id = str(page.get("page_id") or "").strip() or build_page_id(document_id, page_number)
            page_text = sanitize_text(page.get("page_text", ""))
            table_text = sanitize_text(page.get("table_text", ""))
            features = compute_page_features(page_text, table_text)
            normalized_pages.append(
                {
                    "document_id": document_id,
                    "page_id": page_id,
                    "doc_name": page.get("doc_name", ""),
                    "filename": filename,
                    "file_type": page.get("file_type", ""),
                    "file_path": page.get("file_path", ""),
                    "page_number": page_number,
                    "company": page.get("company", ""),
                    "report_year": int(page.get("report_year", 0) or 0),
                    "financial_document_type": page.get("financial_document_type", ""),
                    "location": page.get("location", f"page:{page_number}"),
                    "content_hash": page.get("content_hash", ""),
                    "page_text": page_text,
                    "table_text": table_text,
                    "chunk_ids": list(page.get("chunk_ids") or []),
                    "embedding_cache_key": build_embedding_cache_key(filename, page_number, page_text),
                    **features,
                }
            )
            page_texts.append(page_text)

        if not normalized_pages:
            return 0

        page_batch_size = max(1, int(os.getenv("PAGE_EMBEDDING_BATCH_SIZE", "16")))
        page_embeddings = embedding_service.get_embeddings(
            [self._page_embedding_text(text) for text in page_texts],
            batch_size=page_batch_size,
        )

        db = SessionLocal()
        upserted = 0
        try:
            for page, page_embedding in zip(normalized_pages, page_embeddings):
                filename = page["filename"]
                page_number = page["page_number"]
                record = (
                    db.query(DocumentPage)
                    .filter(DocumentPage.page_id == page["page_id"])
                    .first()
                )
                if record is None:
                    record = (
                        db.query(DocumentPage)
                        .filter(DocumentPage.filename == filename, DocumentPage.page_number == page_number)
                        .first()
                    )
                payload = {
                    "document_id": page["document_id"],
                    "page_id": page["page_id"],
                    "doc_name": page["doc_name"],
                    "company": page["company"],
                    "report_year": page["report_year"],
                    "financial_document_type": page["financial_document_type"],
                    "location": page["location"],
                    "content_hash": page["content_hash"],
                    "file_type": page["file_type"],
                    "file_path": page["file_path"],
                    "page_text": page["page_text"],
                    "table_text": page["table_text"],
                    "chunk_ids": page["chunk_ids"],
                    "embedding_cache_key": page["embedding_cache_key"],
                    "page_dense_embedding": list(page_embedding or []),
                    "page_tokens": page["page_tokens"],
                    "page_numbers": page["page_numbers"],
                    "page_years": page["page_years"],
                    "page_metric_tokens": page["page_metric_tokens"],
                    "updated_at": datetime.utcnow(),
                }
                cache_payload = {
                    "document_id": payload["document_id"],
                    "page_id": payload["page_id"],
                    "doc_name": payload["doc_name"],
                    "filename": filename,
                    "file_type": payload["file_type"],
                    "file_path": payload["file_path"],
                    "page_number": page_number,
                    "company": payload["company"],
                    "report_year": payload["report_year"],
                    "financial_document_type": payload["financial_document_type"],
                    "location": payload["location"],
                    "content_hash": payload["content_hash"],
                    "page_text": payload["page_text"],
                    "table_text": payload["table_text"],
                    "chunk_ids": payload["chunk_ids"],
                    "embedding_cache_key": payload["embedding_cache_key"],
                    "page_dense_embedding": payload["page_dense_embedding"],
                    "page_tokens": payload["page_tokens"],
                    "page_numbers": payload["page_numbers"],
                    "page_years": payload["page_years"],
                    "page_metric_tokens": payload["page_metric_tokens"],
                }
                if record:
                    for key, value in payload.items():
                        setattr(record, key, value)
                else:
                    db.add(DocumentPage(filename=filename, page_number=page_number, **payload))

                cache.set_json(self._cache_key(filename, page_number), cache_payload)
                upserted += 1

            db.commit()
        finally:
            db.close()

        return upserted

    def insert_preembedded_pages(self, pages: List[dict], embeddings: List[list[float]]) -> int:
        """Fast path for a freshly rebuilt collection.

        The benchmark rebuild has already removed all derived page rows.  Using
        per-row SELECT/UPDATE and eagerly warming Redis in that situation adds
        thousands of network round trips without improving retrieval.  Runtime
        uploads continue to use :meth:`upsert_pages`.
        """
        if len(pages) != len(embeddings):
            raise ValueError("pages and embeddings must have the same length")
        if not pages:
            return 0

        now = datetime.utcnow()
        mappings = []
        for page, embedding in zip(pages, embeddings):
            filename = (page.get("filename") or "").strip()
            if not filename:
                continue
            page_text = sanitize_text(page.get("page_text", ""))
            table_text = sanitize_text(page.get("table_text", ""))
            page_number = int(page.get("page_number", 0) or 0)
            document_id = str(page.get("document_id") or "").strip() or build_document_id(
                file_path=page.get("file_path", ""), filename=filename
            )
            mappings.append(
                {
                    "document_id": document_id,
                    "page_id": str(page.get("page_id") or "").strip() or build_page_id(document_id, page_number),
                    "doc_name": page.get("doc_name", ""),
                    "filename": filename,
                    "file_type": page.get("file_type", ""),
                    "file_path": page.get("file_path", ""),
                    "page_number": page_number,
                    "company": page.get("company", ""),
                    "report_year": int(page.get("report_year", 0) or 0),
                    "financial_document_type": page.get("financial_document_type", ""),
                    "location": page.get("location", ""),
                    "content_hash": page.get("content_hash", ""),
                    "page_text": page_text,
                    "table_text": table_text,
                    "chunk_ids": list(page.get("chunk_ids") or []),
                    "embedding_cache_key": build_embedding_cache_key(
                        filename, int(page.get("page_number", 0) or 0), page_text
                    ),
                    "page_dense_embedding": list(embedding or []),
                    **compute_page_features(page_text, table_text),
                    "updated_at": now,
                }
            )
        if not mappings:
            return 0
        db = SessionLocal()
        try:
            db.bulk_insert_mappings(DocumentPage, mappings)
            db.commit()
        finally:
            db.close()
        return len(mappings)

    def get_pages_by_filenames(self, filenames: List[str], *, warm_cache: bool = True) -> List[dict]:
        normalized = [item.strip() for item in filenames if item and item.strip()]
        if not normalized:
            return []

        db = SessionLocal()
        try:
            rows = db.query(DocumentPage).filter(DocumentPage.filename.in_(normalized)).all()
            results = [self._to_dict(row) for row in rows]
            if warm_cache:
                for payload in results:
                    cache.set_json(
                        self._cache_key(payload["filename"], int(payload.get("page_number", 0) or 0)),
                        payload,
                    )
            return sorted(results, key=lambda item: ((item.get("filename") or "").lower(), int(item.get("page_number", 0) or 0)))
        finally:
            db.close()

    def get_pages_by_keys(self, keys: List[tuple[str, int]]) -> List[dict]:
        """Read only the requested pages instead of loading whole documents."""
        normalized = list(dict.fromkeys(
            (str(filename).strip(), int(page_number))
            for filename, page_number in keys
            if str(filename).strip()
        ))
        if not normalized:
            return []
        db = SessionLocal()
        try:
            rows = db.query(DocumentPage).filter(
                tuple_(DocumentPage.filename, DocumentPage.page_number).in_(normalized)
            ).all()
            by_key = {(row.filename, row.page_number): self._to_dict(row) for row in rows}
            return [by_key[key] for key in normalized if key in by_key]
        finally:
            db.close()

    def get_pages_by_ids(self, page_ids: List[str]) -> List[dict]:
        normalized = list(dict.fromkeys(
            str(page_id).strip() for page_id in page_ids if str(page_id).strip()
        ))
        if not normalized:
            return []
        db = SessionLocal()
        try:
            rows = db.query(DocumentPage).filter(DocumentPage.page_id.in_(normalized)).all()
            by_id = {row.page_id: self._to_dict(row) for row in rows}
            return [by_id[page_id] for page_id in normalized if page_id in by_id]
        finally:
            db.close()

    def delete_by_filename(self, filename: str) -> int:
        if not filename:
            return 0

        db = SessionLocal()
        try:
            rows = db.query(DocumentPage).filter(DocumentPage.filename == filename).all()
            deleted = len(rows)
            if deleted > 0:
                db.query(DocumentPage).filter(DocumentPage.filename == filename).delete(synchronize_session=False)
                db.commit()
                for row in rows:
                    cache.delete(self._cache_key(row.filename, row.page_number))
            return deleted
        finally:
            db.close()
