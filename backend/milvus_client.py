"""Milvus 客户端 - 支持密集向量+稀疏向量混合检索"""
import logging
import os
import socket
import threading
from typing import Callable, TypeVar

from dotenv import load_dotenv
from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

try:
    from pymilvus import Function, FunctionType
except ImportError:  # pragma: no cover - compatibility with older clients
    Function = None
    FunctionType = None

load_dotenv()

# Milvus 单次 query 的 limit 上限（超出会报 invalid max query result window）
QUERY_MAX_LIMIT = 16384
T = TypeVar("T")
logger = logging.getLogger(__name__)


class MilvusManager:
    """Milvus 连接和集合管理 - 支持混合检索"""

    def __init__(self):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.collection_name = os.getenv("MILVUS_COLLECTION", "embeddings_collection")
        self.configured_search_ef = max(1, int(os.getenv("MILVUS_SEARCH_EF", "128")))
        self.sparse_mode = os.getenv("MILVUS_SPARSE_MODE", "milvus_bm25").strip().lower()
        self.uses_builtin_bm25 = self.sparse_mode == "milvus_bm25" and Function is not None
        self.uri = f"http://{self.host}:{self.port}"
        self.connect_timeout = max(1.0, float(os.getenv("MILVUS_CONNECT_TIMEOUT_SECONDS", "5")))
        self.client = None
        self._client_lock = threading.RLock()

    def _resolve_effective_ef(self, search_k: int) -> int:
        search_k = max(1, int(search_k))
        effective_ef = max(self.configured_search_ef, search_k * 2)
        if effective_ef <= search_k:
            effective_ef = search_k + 1
        if effective_ef != self.configured_search_ef:
            logger.info(
                "Milvus HNSW ef auto-adjusted from %s to %s because search_k=%s",
                self.configured_search_ef,
                effective_ef,
                search_k,
            )
        return effective_ef

    def _log_search_params(
        self,
        *,
        retrieval_type: str,
        candidate_k: int | None,
        final_top_k: int | None,
        actual_search_k: int,
        final_limit: int,
        metric_type: str,
        effective_ef: int,
    ) -> None:
        logger.info(
            (
                "Milvus search params retrieval_type=%s collection=%s metric_type=%s "
                "candidate_k=%s final_top_k=%s actual_search_k=%s final_limit=%s "
                "configured_ef=%s effective_ef=%s"
            ),
            retrieval_type,
            self.collection_name,
            metric_type,
            candidate_k,
            final_top_k,
            actual_search_k,
            final_limit,
            self.configured_search_ef,
            effective_ef,
        )

    def _get_client(self) -> MilvusClient:
        # Lazy-create client to avoid blocking app import/startup when Milvus is temporarily unavailable.
        with self._client_lock:
            if self.client is None:
                try:
                    with socket.create_connection((self.host, int(self.port)), timeout=self.connect_timeout):
                        pass
                except OSError as exc:
                    raise ConnectionError(
                        f"Milvus is unavailable at {self.host}:{self.port}. "
                        "Start the Milvus service before running retrieval."
                    ) from exc
                self.client = MilvusClient(uri=self.uri, timeout=self.connect_timeout)
            return self.client

    @staticmethod
    def _is_closed_channel_error(exc: Exception) -> bool:
        return isinstance(exc, ValueError) and "closed channel" in str(exc).lower()

    @staticmethod
    def _close_client(client) -> None:
        close = getattr(client, "close", None)
        if not callable(close):
            return
        try:
            close()
        except Exception:
            pass

    def _reset_client(self, failed_client=None) -> None:
        with self._client_lock:
            if self.client is None:
                return
            if failed_client is not None and self.client is not failed_client:
                return
            client = self.client
            self.client = None

        self._close_client(client)

    def _run_with_reconnect(self, operation: Callable[[MilvusClient], T]) -> T:
        client = self._get_client()
        try:
            return operation(client)
        except Exception as exc:
            if not self._is_closed_channel_error(exc):
                raise

            self._reset_client(client)
            return operation(self._get_client())

    def init_collection(self, dense_dim: int | None = None):
        """
        初始化 Milvus 集合 - 同时支持密集向量和稀疏向量
        :param dense_dim: 密集向量维度；默认读环境变量 DENSE_EMBEDDING_DIM（本地 BAAI/bge-m3 为 1024）
        """
        if dense_dim is None:
            dense_dim = int(os.getenv("DENSE_EMBEDDING_DIM", "1024"))
        def _init(client: MilvusClient) -> None:
            if not client.has_collection(self.collection_name):
                schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
                
                # 主键
                schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
                
                # 密集向量（来自 embedding 模型）
                schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
                
                # 文本和元数据字段
                schema.add_field(
                    "text",
                    DataType.VARCHAR,
                    max_length=8192,
                    enable_analyzer=self.uses_builtin_bm25,
                    enable_match=self.uses_builtin_bm25,
                    analyzer_params={"type": "standard"} if self.uses_builtin_bm25 else None,
                )
                schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
                if self.uses_builtin_bm25:
                    schema.add_function(
                        Function(
                            name="text_bm25",
                            function_type=FunctionType.BM25,
                            input_field_names=["text"],
                            output_field_names=["sparse_embedding"],
                        )
                    )
                schema.add_field("filename", DataType.VARCHAR, max_length=255)
                schema.add_field("file_type", DataType.VARCHAR, max_length=50)
                schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
                schema.add_field("page_number", DataType.INT64)
                schema.add_field("chunk_idx", DataType.INT64)
                schema.add_field("company", DataType.VARCHAR, max_length=255)
                schema.add_field("report_year", DataType.INT64)
                schema.add_field("financial_document_type", DataType.VARCHAR, max_length=50)
                schema.add_field("location", DataType.VARCHAR, max_length=255)
                schema.add_field("content_hash", DataType.VARCHAR, max_length=64)

                # Auto-merging 所需层级字段
                schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
                schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
                schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
                schema.add_field("chunk_level", DataType.INT64)

                # 表格证据元数据字段。已有 collection 不会自动增补这些字段；
                # 如开发环境需要启用，请手动重建 collection，代码不会自动删除用户数据。
                schema.add_field("evidence_type", DataType.VARCHAR, max_length=50)
                schema.add_field("table_id", DataType.VARCHAR, max_length=512)
                schema.add_field("row_id", DataType.VARCHAR, max_length=512)
                schema.add_field("table_title", DataType.VARCHAR, max_length=1024)

                # 为两种向量分别创建索引
                index_params = client.prepare_index_params()
                
                # 密集向量索引 - 使用 HNSW（更适合混合检索）
                index_params.add_index(
                    field_name="dense_embedding",
                    index_type="HNSW",
                    metric_type="IP",
                    params={"M": 16, "efConstruction": 256}
                )
                
                # 稀疏向量索引
                index_params.add_index(
                    field_name="sparse_embedding",
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="BM25" if self.uses_builtin_bm25 else "IP",
                    params={"drop_ratio_build": 0.2}
                )

                client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    index_params=index_params
                )
            else:
                logger.info(
                    "Milvus collection %s already exists; new table evidence fields are not auto-added. "
                    "If you need evidence_type/table_id/row_id/table_title in development, rebuild the collection manually.",
                    self.collection_name,
                )

        self._run_with_reconnect(_init)

    def insert(self, data: list[dict]):
        """插入数据到 Milvus"""
        return self._run_with_reconnect(lambda client: client.insert(self.collection_name, data))

    def query(
        self,
        filter_expr: str = "",
        output_fields: list[str] = None,
        limit: int = 10000,
        offset: int = 0,
    ):
        """查询数据。limit 不宜超过 QUERY_MAX_LIMIT。"""
        return self._run_with_reconnect(
            lambda client: client.query(
                collection_name=self.collection_name,
                filter=filter_expr,
                output_fields=output_fields or ["filename", "file_type"],
                limit=min(limit, QUERY_MAX_LIMIT),
                offset=offset,
            )
        )

    def query_all(self, filter_expr: str = "", output_fields: list[str] | None = None) -> list:
        """流式拉取匹配 filter 的全部行，避免 offset+limit 撞上 Milvus 查询窗口上限。"""
        fields = output_fields or ["filename", "file_type"]
        batch_size = min(1000, QUERY_MAX_LIMIT)

        def _query_all(client: MilvusClient) -> list:
            if not hasattr(client, "query_iterator"):
                return client.query(
                    collection_name=self.collection_name,
                    filter=filter_expr,
                    output_fields=fields,
                    limit=QUERY_MAX_LIMIT,
                )

            iterator = client.query_iterator(
                collection_name=self.collection_name,
                batch_size=batch_size,
                limit=-1,
                filter=filter_expr,
                output_fields=fields,
            )

            out: list = []
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    out.extend(batch)
                return out
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

        return self._run_with_reconnect(_query_all)

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """根据 chunk_id 批量查询分块（用于 Auto-merging 拉取父块）"""
        ids = [item for item in chunk_ids if item]
        if not ids:
            return []
        quoted_ids = ", ".join([f'"{item}"' for item in ids])
        filter_expr = f"chunk_id in [{quoted_ids}]"
        return self.query(
            filter_expr=filter_expr,
            output_fields=[
                "text",
                "filename",
                "file_type",
                "page_number",
                "chunk_id",
                "parent_chunk_id",
                "root_chunk_id",
                "chunk_level",
                "chunk_idx",
            ],
            limit=len(ids),
        )

    def hybrid_retrieve(
        self,
        dense_embedding: list[float],
        sparse_embedding: dict | None,
        top_k: int = 5,
        rrf_k: int = 60,     #可调节
        filter_expr: str = "",
        query_text: str | None = None,
    ) -> list[dict]:
        """
        混合检索 - 使用 RRF 融合密集向量和稀疏向量的检索结果
        
        :param dense_embedding: 密集向量
        :param sparse_embedding: 稀疏向量 {index: value, ...}
        :param top_k: 返回结果数量
        :param rrf_k: RRF 算法参数 k，默认60
        :return: 检索结果列表
        """
        output_fields = [
            "text",
            "filename",
            "file_type",
            "page_number",
            "evidence_type",
            "table_id",
            "row_id",
            "table_title",
            "chunk_id",
            "parent_chunk_id",
            "root_chunk_id",
            "chunk_level",
            "chunk_idx",
            "company",
            "report_year",
            "financial_document_type",
            "location",
            "content_hash",
        ]
        actual_search_k = max(1, top_k * 2)
        effective_ef = self._resolve_effective_ef(actual_search_k)
        self._log_search_params(
            retrieval_type="hybrid",
            candidate_k=top_k,
            final_top_k=top_k,
            actual_search_k=actual_search_k,
            final_limit=top_k,
            metric_type="IP",
            effective_ef=effective_ef,
        )
        
        # 密集向量搜索请求
        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": effective_ef}},
            limit=actual_search_k,  # 多取一些用于融合
            expr=filter_expr,
        )
        
        # 稀疏向量搜索请求
        if self.uses_builtin_bm25 and not query_text:
            raise ValueError("Milvus BM25 hybrid retrieval requires query_text")
        sparse_search = AnnSearchRequest(
            data=[query_text if self.uses_builtin_bm25 else sparse_embedding],
            anns_field="sparse_embedding",
            param={
                "metric_type": "BM25" if self.uses_builtin_bm25 else "IP",
                "params": {} if self.uses_builtin_bm25 else {"drop_ratio_search": 0.2},
            },
            limit=actual_search_k,
            expr=filter_expr,
        )
        
        # 使用 RRF 排序算法融合结果
        reranker = RRFRanker(k=rrf_k)
        
        results = self._run_with_reconnect(
            lambda client: client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_search, sparse_search],
                ranker=reranker,
                limit=top_k,
                output_fields=output_fields
            )
        )
        
        # 格式化返回结果
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("text", ""),
                    "filename": hit.get("filename", ""),
                    "file_type": hit.get("file_type", ""),
                    "page_number": hit.get("page_number", 0),
                    "evidence_type": hit.get("evidence_type", "text_chunk") or "text_chunk",
                    "table_id": hit.get("table_id", ""),
                    "row_id": hit.get("row_id", ""),
                    "table_title": hit.get("table_title", ""),
                    "chunk_id": hit.get("chunk_id", ""),
                    "parent_chunk_id": hit.get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("root_chunk_id", ""),
                    "chunk_level": hit.get("chunk_level", 0),
                    "chunk_idx": hit.get("chunk_idx", 0),
                    "company": hit.get("company", ""),
                    "report_year": hit.get("report_year", 0),
                    "financial_document_type": hit.get("financial_document_type", ""),
                    "location": hit.get("location", ""),
                    "content_hash": hit.get("content_hash", ""),
                    "score": hit.get("distance", 0.0)
                })
        
        return formatted_results

    def dense_retrieve(self, dense_embedding: list[float], top_k: int = 5, filter_expr: str = "") -> list[dict]:
        """
        仅使用密集向量检索（降级模式，用于稀疏向量不可用时）
        """
        actual_search_k = max(1, top_k)
        effective_ef = self._resolve_effective_ef(actual_search_k)
        self._log_search_params(
            retrieval_type="dense",
            candidate_k=top_k,
            final_top_k=top_k,
            actual_search_k=actual_search_k,
            final_limit=top_k,
            metric_type="IP",
            effective_ef=effective_ef,
        )
        results = self._run_with_reconnect(
            lambda client: client.search(
                collection_name=self.collection_name,
                data=[dense_embedding],
                anns_field="dense_embedding",
                search_params={"metric_type": "IP", "params": {"ef": effective_ef}},
                limit=top_k,
                output_fields=[
                    "text",
                    "filename",
                    "file_type",
                    "page_number",
                    "evidence_type",
                    "table_id",
                    "row_id",
                    "table_title",
                    "chunk_id",
                    "parent_chunk_id",
                    "root_chunk_id",
                    "chunk_level",
                    "chunk_idx",
                    "company",
                    "report_year",
                    "financial_document_type",
                    "location",
                    "content_hash",
                ],
                filter=filter_expr,
            )
        )
        
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "id": hit.get("id"),
                    "text": hit.get("entity", {}).get("text", ""),
                    "filename": hit.get("entity", {}).get("filename", ""),
                    "file_type": hit.get("entity", {}).get("file_type", ""),
                    "page_number": hit.get("entity", {}).get("page_number", 0),
                    "evidence_type": hit.get("entity", {}).get("evidence_type", "text_chunk") or "text_chunk",
                    "table_id": hit.get("entity", {}).get("table_id", ""),
                    "row_id": hit.get("entity", {}).get("row_id", ""),
                    "table_title": hit.get("entity", {}).get("table_title", ""),
                    "chunk_id": hit.get("entity", {}).get("chunk_id", ""),
                    "parent_chunk_id": hit.get("entity", {}).get("parent_chunk_id", ""),
                    "root_chunk_id": hit.get("entity", {}).get("root_chunk_id", ""),
                    "chunk_level": hit.get("entity", {}).get("chunk_level", 0),
                    "chunk_idx": hit.get("entity", {}).get("chunk_idx", 0),
                    "company": hit.get("entity", {}).get("company", ""),
                    "report_year": hit.get("entity", {}).get("report_year", 0),
                    "financial_document_type": hit.get("entity", {}).get("financial_document_type", ""),
                    "location": hit.get("entity", {}).get("location", ""),
                    "content_hash": hit.get("entity", {}).get("content_hash", ""),
                    "score": hit.get("distance", 0.0)
                })
        
        return formatted_results

    def delete(self, filter_expr: str):
        """删除数据"""
        return self._run_with_reconnect(
            lambda client: client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )
        )

    def has_collection(self) -> bool:
        """检查集合是否存在"""
        return self._run_with_reconnect(lambda client: client.has_collection(self.collection_name))

    def drop_collection(self):
        """删除集合（用于重建 schema）"""
        def _drop(client: MilvusClient) -> None:
            if client.has_collection(self.collection_name):
                client.drop_collection(self.collection_name)

        self._run_with_reconnect(_drop)
