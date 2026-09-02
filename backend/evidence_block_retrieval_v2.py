"""Offline Evidence Block Retrieval v2 shadow index.

This module is deliberately isolated from production retrieval.  It builds
query-independent text, table, and mixed blocks and owns a dedicated Milvus
collection.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
from collections import defaultdict

from dotenv import load_dotenv
from pymilvus import DataType, MilvusClient

try:
    from pymilvus import Function, FunctionType
except ImportError:  # pragma: no cover - compatibility with older clients
    Function = None
    FunctionType = None


load_dotenv()

DEFAULT_BLOCK_COLLECTION = "evidencerag_evidence_block_shadow_v2"
_SPACE_RE = re.compile(r"\s+")


def _clean(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip()


def _integer(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _block_id(source_type: str, page_id: str, identity: str) -> str:
    digest = hashlib.sha256(f"{source_type}|{page_id}|{identity}".encode("utf-8")).hexdigest()[:24]
    return f"block:{source_type}:{digest}"


def _row_text(row: object) -> str:
    if isinstance(row, dict):
        return " | ".join(_clean(value) for key, value in row.items() if not str(key).startswith("_") and _clean(value))
    if isinstance(row, list):
        return " | ".join(_clean(value) for value in row if _clean(value))
    return _clean(row)


def _table_text(table: dict, *, max_chars: int = 14000) -> str:
    title = _clean(table.get("title") or table.get("caption"))
    headers = [_clean(value) for value in table.get("columns") or [] if _clean(value)]
    rows = [_row_text(row) for row in table.get("rows") or []]
    rows = [row for row in rows if row]
    parts = ["[Table Block]"]
    if title:
        parts.append(f"Table title: {title}")
    if headers:
        parts.append("Header: " + " | ".join(headers))
    if rows:
        parts.append("Rows:\n" + "\n".join(rows))
    unit = _clean(table.get("unit"))
    scale = _clean(table.get("scale"))
    if unit or scale:
        parts.append(f"Unit/scale: {unit} {scale}".strip())
    return "\n".join(parts)[:max_chars]


def _metadata(page: dict, **extra: object) -> dict:
    return {
        "filename": _clean(page.get("filename")),
        "page_number": _integer(page.get("page_number")),
        "section": _clean(page.get("section") or page.get("section_title")),
        **extra,
    }


def build_evidence_blocks_v2(
    chunks: list[dict],
    *,
    pages: list[dict],
    tables: list[dict],
    max_adjacent_chunks: int = 3,
    max_text_chars: int = 14000,
) -> list[dict]:
    """Build query-independent blocks from the existing stores.

    Text blocks merge only consecutive chunk indexes on the same page.
    Table blocks preserve the existing TableStore rows.  Mixed blocks append
    only table-adjacent context already stored by the parser.
    """
    page_by_id = {_clean(page.get("page_id")): page for page in pages if _clean(page.get("page_id"))}
    page_by_key = {
        (_clean(page.get("filename")).casefold(), _integer(page.get("page_number"))): page
        for page in pages
    }
    chunks_by_page: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        page = page_by_key.get((_clean(chunk.get("filename")).casefold(), _integer(chunk.get("page_number"))))
        if not page:
            continue
        page_id = _clean(page.get("page_id"))
        if page_id and _clean(chunk.get("text")):
            chunks_by_page[page_id].append(chunk)

    blocks: list[dict] = []
    for page_id, page_chunks in chunks_by_page.items():
        page = page_by_id[page_id]
        ordered = sorted(page_chunks, key=lambda item: (_integer(item.get("chunk_idx")), _clean(item.get("chunk_id"))))
        group: list[dict] = []
        previous_index: int | None = None

        def emit_text_block(items: list[dict]) -> None:
            if not items:
                return
            chunk_ids = [_clean(item.get("chunk_id")) for item in items if _clean(item.get("chunk_id"))]
            body = "\n\n".join(dict.fromkeys(_clean(item.get("text")) for item in items if _clean(item.get("text"))))
            section = next((_clean(item.get("section") or item.get("section_title")) for item in items if _clean(item.get("section") or item.get("section_title"))), "")
            text = "\n".join(part for part in (
                "[Text Block]",
                f"Section: {section}" if section else "",
                body,
            ) if part)[:max_text_chars]
            identity = ",".join(chunk_ids) or f"{_integer(items[0].get('chunk_idx'))}:{len(items)}"
            blocks.append({
                "block_id": _block_id("text", page_id, identity),
                "document_id": _clean(page.get("document_id")),
                "page_id": page_id,
                "source_type": "text",
                "text": text,
                "metadata": _metadata(page, chunk_ids=chunk_ids, chunk_count=len(items), section=section),
            })

        for chunk in ordered:
            index = _integer(chunk.get("chunk_idx"))
            consecutive = previous_index is None or index <= previous_index + 1
            if group and (not consecutive or len(group) >= max(1, max_adjacent_chunks)):
                emit_text_block(group)
                group = []
            group.append(chunk)
            previous_index = index
        emit_text_block(group)

    for table in tables:
        page_id = _clean(table.get("page_id"))
        page = page_by_id.get(page_id) or page_by_key.get(
            (_clean(table.get("filename")).casefold(), _integer(table.get("page_number")))
        )
        if not page:
            continue
        page_id = _clean(page.get("page_id"))
        table_id = _clean(table.get("table_id"))
        table_text = _table_text(table, max_chars=max_text_chars)
        if not table_id or not table_text or not (table.get("columns") and table.get("rows")):
            continue
        common = {
            "table_id": table_id,
            "title": _clean(table.get("title") or table.get("caption")),
            "unit": _clean(table.get("unit")),
            "scale": _clean(table.get("scale")),
            "quality_score": float(table.get("quality_score") or 0.0),
        }
        blocks.append({
            "block_id": _block_id("table", page_id, table_id),
            "document_id": _clean(page.get("document_id")),
            "page_id": page_id,
            "source_type": "table",
            "text": table_text,
            "metadata": _metadata(page, **common),
        })
        nearby = "\n".join(part for part in (
            _clean(table.get("before_context")),
            _clean(table.get("after_context")),
        ) if part)
        if nearby:
            mixed_text = f"[Mixed Block]\n{nearby}\n\n{table_text}"[:max_text_chars]
            blocks.append({
                "block_id": _block_id("mixed", page_id, table_id),
                "document_id": _clean(page.get("document_id")),
                "page_id": page_id,
                "source_type": "mixed",
                "text": mixed_text,
                "metadata": _metadata(page, nearby_context=nearby[:3000], **common),
            })

    return sorted(blocks, key=lambda item: (item["document_id"], item["page_id"], item["source_type"], item["block_id"]))


def fuse_block_routes(dense: list[dict], bm25: list[dict], *, top_k: int = 30, rrf_k: int = 60) -> list[dict]:
    by_id: dict[str, dict] = {}
    scores: dict[str, float] = {}
    for route, items in (("dense", dense), ("bm25", bm25)):
        for rank, item in enumerate(items, 1):
            block_id = _clean(item.get("block_id"))
            if not block_id:
                continue
            by_id.setdefault(block_id, dict(item))
            by_id[block_id][f"{route}_rank"] = rank
            scores[block_id] = scores.get(block_id, 0.0) + 1.0 / (rrf_k + rank)
    ranked = sorted(by_id, key=lambda key: (-scores[key], key))[:max(1, top_k)]
    return [
        {**by_id[key], "rrf_score": round(scores[key], 8), "block_rank": rank}
        for rank, key in enumerate(ranked, 1)
    ]


class EvidenceBlockMilvusManager:
    """Own only the isolated Evidence Block v2 shadow collection."""

    OUTPUT_FIELDS = [
        "block_id", "document_id", "page_id", "source_type", "text", "metadata",
        "filename", "page_number", "content_hash",
    ]

    def __init__(self, collection_name: str | None = None):
        self.host = os.getenv("MILVUS_HOST", "localhost")
        self.port = os.getenv("MILVUS_PORT", "19530")
        self.uri = f"http://{self.host}:{self.port}"
        self.collection_name = collection_name or os.getenv("EVIDENCE_BLOCK_MILVUS_COLLECTION", DEFAULT_BLOCK_COLLECTION)
        self.connect_timeout = max(1.0, float(os.getenv("MILVUS_CONNECT_TIMEOUT_SECONDS", "5")))
        self.search_ef = max(64, int(os.getenv("EVIDENCE_BLOCK_SEARCH_EF", "128")))
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
            "text", DataType.VARCHAR, max_length=16000, enable_analyzer=True, enable_match=True,
            analyzer_params={"type": "standard"},
        )
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(Function(
            name="block_text_bm25", function_type=FunctionType.BM25,
            input_field_names=["text"], output_field_names=["sparse_embedding"],
        ))
        for name, length in (
            ("block_id", 128), ("document_id", 64), ("page_id", 128),
            ("source_type", 32), ("filename", 255), ("content_hash", 64),
        ):
            schema.add_field(name, DataType.VARCHAR, max_length=length)
        schema.add_field("page_number", DataType.INT64)
        schema.add_field("metadata", DataType.JSON)
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
        if "block" not in normalized or "shadow" not in normalized:
            raise ValueError(f"refusing to drop non-shadow block collection: {self.collection_name}")
        if self.has_collection():
            self._client().drop_collection(self.collection_name)

    def insert(self, documents: list[dict]) -> None:
        if documents:
            self._client().insert(self.collection_name, documents)

    def flush(self) -> None:
        self._client().flush(self.collection_name)

    def count(self) -> int:
        rows = self._client().query(collection_name=self.collection_name, filter="", output_fields=["count(*)"])
        return int((rows[0] if rows else {}).get("count(*)") or 0)

    @classmethod
    def _format(cls, results) -> list[dict]:
        output = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                output.append({
                    "id": hit.get("id"),
                    **{field: entity.get(field, "") for field in cls.OUTPUT_FIELDS},
                    "page_number": _integer(entity.get("page_number")),
                    "score": float(hit.get("distance") or 0.0),
                })
        return output

    def dense_retrieve(self, embedding: list[float], *, top_k: int = 30) -> list[dict]:
        results = self._client().search(
            collection_name=self.collection_name, data=[embedding], anns_field="dense_embedding",
            search_params={"metric_type": "IP", "params": {"ef": max(self.search_ef, top_k * 2)}},
            limit=max(1, top_k), output_fields=self.OUTPUT_FIELDS,
        )
        return self._format(results)

    def bm25_retrieve(self, question: str, *, top_k: int = 30) -> list[dict]:
        results = self._client().search(
            collection_name=self.collection_name, data=[question], anns_field="sparse_embedding",
            search_params={"metric_type": "BM25", "params": {}}, limit=max(1, top_k),
            output_fields=self.OUTPUT_FIELDS,
        )
        return self._format(results)

    def retrieve(self, question: str, embedding: list[float], *, top_k: int = 30) -> dict:
        started = time.perf_counter()
        dense = self.dense_retrieve(embedding, top_k=top_k)
        dense_ms = (time.perf_counter() - started) * 1000
        bm25_started = time.perf_counter()
        bm25 = self.bm25_retrieve(question, top_k=top_k)
        bm25_ms = (time.perf_counter() - bm25_started) * 1000
        fused = fuse_block_routes(dense, bm25, top_k=top_k)
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


def index_document(block: dict, dense_embedding: list[float]) -> dict:
    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    text = str(block.get("text") or "")[:16000]
    return {
        "block_id": _clean(block.get("block_id")),
        "document_id": _clean(block.get("document_id")),
        "page_id": _clean(block.get("page_id")),
        "source_type": _clean(block.get("source_type")),
        "text": text,
        "metadata": metadata,
        "filename": _clean(metadata.get("filename")),
        "page_number": _integer(metadata.get("page_number")),
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "dense_embedding": dense_embedding,
    }
