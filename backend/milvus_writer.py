"""文档向量化并写入 Milvus - 支持密集+稀疏向量"""
import os

from embedding import EmbeddingService, embedding_service as _default_embedding_service
from milvus_client import MilvusManager
from text_sanitizer import sanitize_text

_DEFAULT_MILVUS_TEXT_MAX_LENGTH = 7500
_TRUNCATION_SUFFIX = " ... [truncated]"


class MilvusWriter:
    """文档向量化并写入 Milvus 服务 - 支持混合检索"""

    def __init__(self, embedding_service: EmbeddingService = None, milvus_manager: MilvusManager = None):
        self.embedding_service = embedding_service or _default_embedding_service
        self.milvus_manager = milvus_manager or MilvusManager()

    @staticmethod
    def _get_text_max_length() -> int:
        raw_value = os.getenv("MILVUS_TEXT_MAX_LENGTH")
        try:
            limit = int(raw_value) if raw_value is not None else _DEFAULT_MILVUS_TEXT_MAX_LENGTH
        except (TypeError, ValueError):
            limit = _DEFAULT_MILVUS_TEXT_MAX_LENGTH
        return limit if limit > 0 else _DEFAULT_MILVUS_TEXT_MAX_LENGTH

    @classmethod
    def _sanitize_and_trim_text(cls, text: str) -> str:
        sanitized = sanitize_text(text or "")
        limit = cls._get_text_max_length()
        if len(sanitized) <= limit:
            return sanitized
        if limit <= len(_TRUNCATION_SUFFIX):
            return _TRUNCATION_SUFFIX[:limit]
        return sanitized[: limit - len(_TRUNCATION_SUFFIX)] + _TRUNCATION_SUFFIX

    def write_documents(
        self,
        documents: list[dict],
        batch_size: int = 50,
        progress_callback=None,
        embeddings: list[list[float]] | None = None,
    ):
        """
        批量写入文档到 Milvus（同时生成密集和稀疏向量）
        :param documents: 文档列表
        :param batch_size: 批次大小
        """
        if not documents:
            return
        if embeddings is not None and len(embeddings) != len(documents):
            raise ValueError("documents and embeddings must have the same length")

        self.milvus_manager.init_collection()

        sanitized_documents = [{**doc, "text": self._sanitize_and_trim_text(doc.get("text", ""))} for doc in documents]
        all_texts = [doc["text"] for doc in sanitized_documents]
        uses_builtin_bm25 = bool(getattr(self.milvus_manager, "uses_builtin_bm25", False))
        if not uses_builtin_bm25:
            self.embedding_service.increment_add_documents(all_texts)

        total = len(sanitized_documents)
        for i in range(0, total, batch_size):
            batch = sanitized_documents[i:i + batch_size]
            texts = [doc["text"] for doc in batch]
            
            if embeddings is not None:
                dense_embeddings = embeddings[i:i + batch_size]
                sparse_embeddings = [None] * len(batch)
            elif uses_builtin_bm25:
                dense_embeddings = self.embedding_service.get_embeddings(texts)
                sparse_embeddings = [None] * len(texts)
            else:
                dense_embeddings, sparse_embeddings = self.embedding_service.get_all_embeddings(texts)

            insert_data = [
                ({
                    "dense_embedding": dense_emb,
                    "text": doc["text"],
                    "filename": doc.get("filename", ""),
                    "file_type": doc.get("file_type", ""),
                    "file_path": doc.get("file_path", ""),
                    "page_number": doc.get("page_number", 0),
                    "chunk_idx": doc.get("chunk_idx", 0),
                    "company": doc.get("company", ""),
                    "report_year": doc.get("report_year", 0),
                    "financial_document_type": doc.get("financial_document_type", ""),
                    "location": doc.get("location", ""),
                    "content_hash": doc.get("content_hash", ""),
                    "evidence_type": doc.get("evidence_type", "text_chunk") or "text_chunk",
                    "table_id": doc.get("table_id", ""),
                    "row_id": doc.get("row_id", ""),
                    "table_title": doc.get("table_title", ""),
                    "chunk_id": doc.get("chunk_id", ""),
                    "parent_chunk_id": doc.get("parent_chunk_id", ""),
                    "root_chunk_id": doc.get("root_chunk_id", ""),
                    "chunk_level": doc.get("chunk_level", 0),
                } | ({"sparse_embedding": sparse_emb} if not uses_builtin_bm25 else {}))
                for doc, dense_emb, sparse_emb in zip(batch, dense_embeddings, sparse_embeddings)
            ]

            self.milvus_manager.insert(insert_data)

            # 每个批次写入后更新进度，前端据此展示“向量化入库 xx%”。
            if progress_callback:
                processed = min(i + batch_size, total)
                progress_callback(processed, total)
