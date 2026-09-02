"""Independent Milvus shadow index for semantic table discovery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time

from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient

try:
    from pymilvus import Function, FunctionType
except ImportError:  # pragma: no cover
    Function = None
    FunctionType = None


load_dotenv()

DEFAULT_TABLE_COLLECTION = "evidencerag_table_shadow_v1"
_NUMBER_RE = re.compile(r"(?:[$€£¥]\s*)?\(?-?\d[\d,]*(?:\.\d+)?%?\)?")


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _without_matrix_numbers(value: object) -> str:
    return _clean(_NUMBER_RE.sub(" ", str(value or "")))


def _row_label(row: dict, headers: list[str]) -> str:
    preferred = ("metric", "label", "item", "description", "line_item", "row_label")
    keys = list(row)
    normalized = {str(key).casefold(): key for key in keys}
    ordered = [normalized[name] for name in preferred if name in normalized]
    ordered.extend(key for key in headers if key in row and key not in ordered)
    ordered.extend(key for key in keys if key not in ordered and not str(key).startswith("_"))
    for key in ordered:
        value = _without_matrix_numbers(row.get(key))
        if value and re.search(r"[A-Za-z]", value):
            return value
    return ""


def build_table_document(table: dict, *, max_row_labels: int = 80) -> dict | None:
    """Build metadata and retrieval text without serializing the value matrix."""
    table_id = _clean(table.get("table_id"))
    document_id = _clean(table.get("document_id"))
    page_id = _clean(table.get("page_id"))
    if not table_id or not document_id or not page_id:
        return None
    title = _clean(table.get("title") or table.get("caption"))
    headers = [_clean(item) for item in (table.get("columns") or []) if _clean(item)]
    labels = []
    for row in table.get("rows") or []:
        if not isinstance(row, dict):
            continue
        label = _row_label(row, headers)
        if label and label.casefold() not in {item.casefold() for item in labels}:
            labels.append(label)
        if len(labels) >= max_row_labels:
            break
    nearby = _without_matrix_numbers(" ".join([
        _clean(table.get("before_context")),
        _clean(table.get("after_context")),
    ]))[:1600]
    semantic_description = _clean(
        "Financial table covering " + "; ".join(labels[:30])
        if labels else f"Financial table {title}"
    )
    search_text = "\n".join(
        part for part in (
            f"Table title: {title}" if title else "",
            f"Headers: {' | '.join(headers)}" if headers else "",
            f"Row labels: {'; '.join(labels)}" if labels else "",
            f"Unit and scale: {_clean(table.get('unit'))} {_clean(table.get('scale'))}".strip(),
            f"Semantic description: {semantic_description}",
            f"Nearby context: {nearby}" if nearby else "",
        ) if part
    )[:8192]
    if not search_text:
        return None
    return {
        "document_id": document_id,
        "page_id": page_id,
        "page_number": int(table.get("page_number") or 0),
        "table_id": table_id,
        "filename": _clean(table.get("filename")),
        "table_title": title[:2048],
        "headers": json.dumps(headers, ensure_ascii=False)[:4096],
        "row_labels": json.dumps(labels, ensure_ascii=False)[:8192],
        "unit": _clean(" ".join([_clean(table.get("unit")), _clean(table.get("scale"))]))[:200],
        "nearby_summary": nearby[:2048],
        "semantic_description": semantic_description[:4096],
        "search_text": search_text,
        "quality_score": float(table.get("quality_score") or 0.0),
        "content_hash": hashlib.sha256(search_text.encode("utf-8")).hexdigest(),
    }


def fuse_table_routes(dense: list[dict], bm25: list[dict], *, top_k: int = 30, rrf_k: int = 60) -> list[dict]:
    by_id: dict[str, dict] = {}
    scores: dict[str, float] = {}
    for route_name, items in (("dense", dense), ("bm25", bm25)):
        for rank, item in enumerate(items, 1):
            table_id = str(item.get("table_id") or "")
            if not table_id:
                continue
            by_id.setdefault(table_id, dict(item))
            by_id[table_id][f"{route_name}_rank"] = rank
            scores[table_id] = scores.get(table_id, 0.0) + 1.0 / (rrf_k + rank)
    ranked = sorted(by_id, key=lambda table_id: (-scores[table_id], table_id))[: max(1, top_k)]
    return [
        {
            **by_id[table_id],
            "rrf_score": round(scores[table_id], 8),
            "table_rank": rank,
        }
        for rank, table_id in enumerate(ranked, 1)
    ]


def merge_text_and_table_pages(
    text_pages: list[dict],
    table_results: list[dict],
    *,
    rrf_k: int = 60,
) -> list[dict]:
    """Page-level shadow fusion; does not mutate either source ranking."""
    pages: dict[tuple[str, int], dict] = {}
    scores: dict[tuple[str, int], float] = {}
    for route, items in (("text", text_pages), ("table", table_results)):
        seen = set()
        for rank, item in enumerate(items, 1):
            key = (_clean(item.get("filename")).casefold(), int(item.get("page_number") or 0))
            if not key[0] or key in seen:
                continue
            seen.add(key)
            pages.setdefault(key, {"filename": item.get("filename"), "page_number": key[1]})
            pages[key][f"{route}_rank"] = rank
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
    ranked = sorted(pages, key=lambda key: (-scores[key], key[0], key[1]))
    return [
        {
            **pages[key],
            "fusion_score": round(scores[key], 8),
            "candidate_rank": rank,
        }
        for rank, key in enumerate(ranked, 1)
    ]


class TableMilvusManager:
    """Own only the dedicated table shadow collection."""

    OUTPUT_FIELDS = [
        "document_id", "page_id", "page_number", "table_id", "filename", "table_title",
        "headers", "row_labels", "unit", "nearby_summary", "semantic_description",
        "search_text", "quality_score", "content_hash",
    ]

    def __init__(self, collection_name: str | None = None):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.uri = f"http://{self.host}:{self.port}"
        self.collection_name = collection_name or os.getenv("TABLE_MILVUS_COLLECTION", DEFAULT_TABLE_COLLECTION)
        self.connect_timeout = max(1.0, float(os.getenv("MILVUS_CONNECT_TIMEOUT_SECONDS", "5")))
        self.search_ef = max(64, int(os.getenv("TABLE_MILVUS_SEARCH_EF", "128")))
        self.client: MilvusClient | None = None

    def _client(self) -> MilvusClient:
        if self.client is None:
            with socket.create_connection((self.host, int(self.port)), timeout=self.connect_timeout):
                pass
            self.client = MilvusClient(uri=self.uri, timeout=self.connect_timeout)
        return self.client

    def has_collection(self) -> bool:
        return self._client().has_collection(self.collection_name)

    def init_collection(self, dense_dim: int = 1024) -> None:
        if self.has_collection():
            return
        if Function is None or FunctionType is None:
            raise RuntimeError("pymilvus BM25 Function support is required")
        client = self._client()
        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field(
            "search_text", DataType.VARCHAR, max_length=8192,
            enable_analyzer=True, enable_match=True, analyzer_params={"type": "standard"},
        )
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(
            name="table_text_bm25", function_type=FunctionType.BM25,
            input_field_names=["search_text"], output_field_names=["sparse_embedding"],
        ))
        for name, length in (
            ("document_id", 64), ("page_id", 128), ("table_id", 512), ("filename", 255),
            ("table_title", 2048), ("headers", 4096), ("row_labels", 8192), ("unit", 200),
            ("nearby_summary", 2048), ("semantic_description", 4096), ("content_hash", 64),
        ):
            schema.add_field(name, DataType.VARCHAR, max_length=length)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("quality_score", DataType.FLOAT)
        indexes = client.prepare_index_params()
        indexes.add_index(
            field_name="dense_embedding", index_type="HNSW", metric_type="IP",
            params={"M": 16, "efConstruction": 256},
        )
        indexes.add_index(
            field_name="sparse_embedding", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
            params={"drop_ratio_build": 0.2},
        )
        client.create_collection(collection_name=self.collection_name, schema=schema, index_params=indexes)

    def drop_collection(self) -> None:
        normalized = self.collection_name.casefold()
        if "table" not in normalized or "shadow" not in normalized:
            raise ValueError(f"refusing to drop non-shadow table collection: {self.collection_name}")
        if self.has_collection():
            self._client().drop_collection(self.collection_name)

    def insert(self, documents: list[dict]) -> None:
        if documents:
            self._client().insert(self.collection_name, documents)

    def flush(self) -> None:
        self._client().flush(self.collection_name)

    def count(self) -> int:
        result = self._client().query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        return int((result[0] if result else {}).get("count(*)") or 0)

    @staticmethod
    def _format(results) -> list[dict]:
        formatted = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                formatted.append({
                    "id": hit.get("id"),
                    **{field: entity.get(field, "") for field in TableMilvusManager.OUTPUT_FIELDS},
                    "page_number": int(entity.get("page_number") or 0),
                    "quality_score": float(entity.get("quality_score") or 0.0),
                    "score": float(hit.get("distance") or 0.0),
                })
        return formatted

    def dense_retrieve(self, dense_embedding: list[float], *, top_k: int = 30) -> list[dict]:
        results = self._client().search(
            collection_name=self.collection_name,
            data=[dense_embedding],
            anns_field="dense_embedding",
            search_params={"metric_type": "IP", "params": {"ef": max(self.search_ef, top_k * 2)}},
            limit=max(1, top_k),
            output_fields=self.OUTPUT_FIELDS,
        )
        return self._format(results)

    def bm25_retrieve(self, query_text: str, *, top_k: int = 30) -> list[dict]:
        results = self._client().search(
            collection_name=self.collection_name,
            data=[query_text],
            anns_field="sparse_embedding",
            search_params={"metric_type": "BM25", "params": {}},
            limit=max(1, top_k),
            output_fields=self.OUTPUT_FIELDS,
        )
        return self._format(results)

    def retrieve(self, question: str, dense_embedding: list[float], *, top_k: int = 30) -> dict:
        started = time.perf_counter()
        dense_started = time.perf_counter()
        dense = self.dense_retrieve(dense_embedding, top_k=top_k)
        dense_ms = (time.perf_counter() - dense_started) * 1000
        bm25_started = time.perf_counter()
        bm25 = self.bm25_retrieve(question, top_k=top_k)
        bm25_ms = (time.perf_counter() - bm25_started) * 1000
        fused = fuse_table_routes(dense, bm25, top_k=top_k)
        return {
            "dense": dense,
            "bm25": bm25,
            "fused": fused,
            "latency_ms": {
                "dense": round(dense_ms, 2),
                "bm25": round(bm25_ms, 2),
                "total": round((time.perf_counter() - started) * 1000, 2),
            },
        }
