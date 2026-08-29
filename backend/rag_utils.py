from collections import defaultdict
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import logging
import os
import re
import threading
import time

import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from cache import cache
from document_page_store import DocumentPageStore
from embedding import embedding_service as _embedding_service
from local_reranker import LocalReranker
from runtime_profile import uses_clean_baseline_path
from finance_rag_features import (
    COMPANY_ALIASES,
    FINANCE_METRIC_HINTS,
    extract_keyword_tokens as _feature_extract_keyword_tokens,
    extract_metric_hints as _feature_extract_metric_hints,
    extract_numbers as _feature_extract_numbers,
    extract_years as _feature_extract_years,
    infer_doc_type,
    normalize_doc_name,
    parse_finance_query,
)
from query_parser import (
    FIELD_ALIASES,
    FIELD_STATEMENT_TYPES,
    assess_required_field_coverage,
    build_finance_query_rewrite,
    build_supplemental_field_query,
    infer_page_statement_types,
    infer_statement_types,
    infer_task_spec,
    match_required_fields_in_text,
    matches_company_text,
)
from milvus_client import MilvusManager
from parent_chunk_store import ParentChunkStore
from evidence_group_builder import (
    build_group_debug_payload as _build_group_debug_payload,
    format_evidence_group as _format_evidence_group,
)
from table_context_builder import (
    dedupe_table_ids as _dedupe_table_ids_for_context,
    build_table_context_preview as _build_table_context_preview,
    format_evidence_unit as _format_evidence_unit,
    truncate_table_context as _truncate_table_context,
)
from table_store import TableStore

load_dotenv()

logger = logging.getLogger(__name__)


def _retrieval_debug(message: str) -> None:
    if _parse_bool(os.getenv("RAG_RETRIEVAL_DEBUG"), False):
        print(f"[retrieval] {message}", flush=True)

ARK_API_KEY = os.getenv("ARK_API_KEY")
MODEL = os.getenv("MODEL")
BASE_URL = os.getenv("BASE_URL")
RERANK_MODEL = os.getenv("RERANK_MODEL")
RERANK_BINDING_HOST = os.getenv("RERANK_BINDING_HOST")
RERANK_API_KEY = os.getenv("RERANK_API_KEY")
RERANK_TIMEOUT_SECONDS = max(1.0, float(os.getenv("RERANK_TIMEOUT_SECONDS", "5")))
RERANK_REMOTE_CANDIDATE_K = max(1, int(os.getenv("RERANK_REMOTE_CANDIDATE_K", "16")))
RERANK_REMOTE_MAX_CHARS = max(256, int(os.getenv("RERANK_REMOTE_MAX_CHARS", "1600")))
RERANK_REMOTE_MIN_INTERVAL_SECONDS = max(0.0, float(os.getenv("RERANK_REMOTE_MIN_INTERVAL_SECONDS", "4")))
RERANK_REMOTE_MAX_ATTEMPTS = max(1, int(os.getenv("RERANK_REMOTE_MAX_ATTEMPTS", "2")))
RERANK_REMOTE_BACKOFF_SECONDS = max(0.0, float(os.getenv("RERANK_REMOTE_BACKOFF_SECONDS", "0.8")))
RERANK_CACHE_ENABLED = os.getenv("RERANK_CACHE_ENABLED", "true").lower() == "true"
RERANK_CACHE_TTL_SECONDS = max(60, int(os.getenv("RERANK_CACHE_TTL_SECONDS", "604800")))

AUTO_MERGE_ENABLED = os.getenv("AUTO_MERGE_ENABLED", "false").lower() == "true"
AUTO_MERGE_THRESHOLD = int(os.getenv("AUTO_MERGE_THRESHOLD", "2"))
LEAF_RETRIEVE_LEVEL = int(os.getenv("LEAF_RETRIEVE_LEVEL", "3"))

TRACE_OUTPUT_FIELDS = [
    "id",
    "text",
    "filename",
    "file_type",
    "page_number",
    "chunk_id",
    "parent_chunk_id",
    "root_chunk_id",
    "chunk_level",
    "chunk_idx",
]

# 全局初始化检索依赖（与 api 共用 embedding_service，保证 BM25 状态一致）
_milvus_manager = MilvusManager()
_parent_chunk_store = ParentChunkStore()
_document_page_store = DocumentPageStore()
_table_store = TableStore()
_local_reranker = LocalReranker()

_stepback_model = None
_rerank_request_lock = threading.Lock()
_last_remote_rerank_started_at = 0.0


def _get_hybrid_query_embeddings(query: str) -> tuple[list[float], dict | None]:
    """Avoid application-side sparse state when the collection uses Milvus BM25."""
    if getattr(_milvus_manager, "uses_builtin_bm25", False):
        return _embedding_service.get_embeddings([query])[0], None
    dense_embeddings, sparse_embeddings = _embedding_service.get_all_embeddings([query])
    return dense_embeddings[0], sparse_embeddings[0]


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _parse_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _parse_retrieval_mode(value: str | None) -> str:
    # Deprecated experimental FinanceBench path.
    # Kept for comparison and debugging only.
    # Do not use as the default production retrieval path.
    # New table-aware RAG work should be implemented behind TABLE_AWARE_* feature flags
    # and must preserve baseline behavior when TABLE_AWARE_RETRIEVAL=off.
    mode = (value or "baseline").strip().lower()
    if mode not in {"baseline", "finance_experimental"}:
        return "baseline"
    return mode


def _parse_table_aware_retrieval_mode(value: str | None) -> str:
    mode = (value or "off").strip().lower()
    if mode not in {"off", "force", "auto"}:
        return "off"
    return mode


def get_finance_rag_config() -> Dict[str, Any]:
    retrieval_mode = _parse_retrieval_mode(os.getenv("RAG_RETRIEVAL_MODE"))
    candidate_k = max(1, _parse_int(os.getenv("FINANCE_RAG_CANDIDATE_K"), 40))
    final_top_k = max(1, _parse_int(os.getenv("FINANCE_RAG_FINAL_TOP_K"), 5))
    experimental_two_stage = _parse_bool(os.getenv("FINANCE_RAG_TWO_STAGE_RETRIEVAL"), True)
    return {
        "retrieval_mode": retrieval_mode,
        "candidate_k": max(candidate_k, final_top_k),
        "final_top_k": final_top_k,
        "enable_step_back": _parse_bool(os.getenv("FINANCE_RAG_ENABLE_STEP_BACK"), False),
        "enable_page_merge": _parse_bool(os.getenv("FINANCE_RAG_ENABLE_PAGE_MERGE"), False),
        "adjacent_page_window": max(0, _parse_int(os.getenv("FINANCE_RAG_ADJACENT_PAGE_WINDOW"), 1)),
        "adjacent_chunk_window": max(0, _parse_int(os.getenv("FINANCE_RAG_ADJACENT_CHUNK_WINDOW"), 1)),
        "two_stage_retrieval": retrieval_mode == "finance_experimental" and experimental_two_stage,
        "doc_stage_top_n": max(1, _parse_int(os.getenv("FINANCE_RAG_DOC_STAGE_TOP_N"), 5)),
        "page_stage_top_n": max(1, _parse_int(os.getenv("FINANCE_RAG_PAGE_STAGE_TOP_N"), 10)),
        "max_evidence_pack_used": max(1, _parse_int(os.getenv("FINANCE_RAG_MAX_EVIDENCE_PACK_USED"), 5)),
        "min_evidence_pack_used": max(1, _parse_int(os.getenv("FINANCE_RAG_MIN_EVIDENCE_PACK_USED"), 3)),
        "max_page_text_chars": max(200, _parse_int(os.getenv("FINANCE_RAG_MAX_PAGE_TEXT_CHARS"), 2500)),
        "max_table_text_chars": max(200, _parse_int(os.getenv("FINANCE_RAG_MAX_TABLE_TEXT_CHARS"), 2500)),
        "w_dense": float(os.getenv("FINANCE_RAG_W_DENSE", "0.35")),
        "w_keyword": float(os.getenv("FINANCE_RAG_W_KEYWORD", "0.20")),
        "w_metric": float(os.getenv("FINANCE_RAG_W_METRIC", "0.15")),
        "w_number": float(os.getenv("FINANCE_RAG_W_NUMBER", "0.10")),
        "w_company": float(os.getenv("FINANCE_RAG_W_COMPANY", "0.15")),
        "w_year": float(os.getenv("FINANCE_RAG_W_YEAR", "0.10")),
        "w_doc_type": float(os.getenv("FINANCE_RAG_W_DOC_TYPE", "0.05")),
        "w_statement": float(os.getenv("FINANCE_RAG_W_STATEMENT", "0.18")),
        "w_required_fields": float(os.getenv("FINANCE_RAG_W_REQUIRED_FIELDS", "0.25")),
        "w_required_periods": float(os.getenv("FINANCE_RAG_W_REQUIRED_PERIODS", "0.20")),
        "w_selection_scope": float(os.getenv("FINANCE_RAG_W_SELECTION_SCOPE", "0.30")),
        "cover_toc_penalty": float(os.getenv("FINANCE_RAG_COVER_TOC_PENALTY", "0.40")),
    }


def get_table_aware_retrieval_config() -> Dict[str, Any]:
    return {
        "mode": _parse_table_aware_retrieval_mode(os.getenv("TABLE_AWARE_RETRIEVAL")),
        "top_k": max(1, _parse_int(os.getenv("TABLE_AWARE_EVIDENCE_TOP_K"), 5)),
        "max_candidate_docs": max(1, _parse_int(os.getenv("TABLE_AWARE_MAX_CANDIDATE_DOCS"), 3)),
        "max_tables": max(1, _parse_int(os.getenv("TABLE_AWARE_MAX_TABLES"), 3)),
        "max_rows": max(1, _parse_int(os.getenv("TABLE_AWARE_MAX_ROWS"), 8)),
        "max_context_chars": max(200, _parse_int(os.getenv("TABLE_AWARE_MAX_CONTEXT_CHARS"), 4000)),
        "global_fallback": _parse_bool(os.getenv("TABLE_AWARE_GLOBAL_FALLBACK"), False),
    }


def get_evidence_group_config() -> Dict[str, Any]:
    table_mode_enabled = _parse_table_aware_retrieval_mode(os.getenv("TABLE_AWARE_RETRIEVAL")) != "off"
    return {
        "enabled": _parse_bool(os.getenv("RAG_EVIDENCE_GROUPING_ENABLED"), table_mode_enabled),
        "max_groups": max(1, _parse_int(os.getenv("RAG_MAX_EVIDENCE_GROUPS"), 5)),
        "max_snippets_per_group": max(1, _parse_int(os.getenv("RAG_MAX_SNIPPETS_PER_GROUP"), 3)),
        "max_table_rows_per_group": max(1, _parse_int(os.getenv("RAG_MAX_TABLE_ROWS_PER_GROUP"), 5)),
        "max_chars_per_group": max(200, _parse_int(os.getenv("RAG_MAX_CHARS_PER_GROUP"), 1200)),
    }


_TABLE_QUERY_METRIC_HINTS = (
    "net sales",
    "revenue",
    "ebit",
    "ebitda",
    "eps",
    "ratio",
    "margin",
    "growth",
    "stores",
    "tax rate",
    "cash flow",
    "gross profit",
    "operating income",
    "cost of sales",
    "assets",
    "liabilities",
    "debt",
    "income",
)

_TABLE_QUERY_CONTACT_HINTS = (
    "number",
    "phone",
    "toll-free",
    "local number",
    "conference id",
    "contact",
    "email",
    "address",
)

_TABLE_QUERY_COMPARE_HINTS = (
    "what was",
    "how much",
    "compare",
    "growth",
    "percentage",
    "change",
)

_TABLE_QUERY_TABLE_HINTS = (
    "table",
    "row",
    "column",
    "data",
    "表格",
    "数据",
)

_QUERY_ANCHOR_STOPWORDS = {
    "what",
    "how",
    "when",
    "where",
    "why",
    "which",
    "who",
    "whom",
    "whose",
    "does",
    "did",
    "do",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "explain",
    "summarize",
    "describe",
    "tell",
    "show",
    "sales",
    "margin",
    "ratio",
    "growth",
    "income",
    "fiscal",
    "year",
    "company",
    "segment",
    "revenue",
    "ebit",
    "ebitda",
    "eps",
    "cash",
    "flow",
    "gross",
    "profit",
    "operating",
    "cost",
    "assets",
    "liabilities",
    "debt",
    "tax",
    "rate",
    "quick",
    "effective",
    "stores",
    "compare",
    "change",
    "increase",
    "decrease",
    "improving",
    "document",
    "table",
    "row",
    "column",
    "data",
}

_TABLE_NUMERIC_PATTERN = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")
_TABLE_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_TABLE_VALUE_PATTERN = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?%?\)?|[—-]")
_QUERY_ANCHOR_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9&'.-]*")


def _query_triggers_table_aware_retrieval(query: str) -> tuple[bool, list[str]]:
    text = (query or "").strip()
    lowered = text.lower()
    reasons: list[str] = []

    has_numeric = bool(_TABLE_NUMERIC_PATTERN.search(text) or _TABLE_YEAR_PATTERN.search(text))
    has_money_or_percent = any(token in lowered for token in ("$", "%", "million", "billion"))
    has_table_keyword = any(term in lowered for term in _TABLE_QUERY_TABLE_HINTS)
    has_metric_keyword = any(term in lowered for term in _TABLE_QUERY_METRIC_HINTS)
    has_contact_keyword = any(term in lowered for term in _TABLE_QUERY_CONTACT_HINTS)
    has_compare_keyword = any(term in lowered for term in _TABLE_QUERY_COMPARE_HINTS)

    if has_contact_keyword:
        reasons.append("query_contact_or_number")
    if has_metric_keyword and (has_numeric or has_money_or_percent or has_compare_keyword):
        reasons.append("query_metric_or_number")
    if has_compare_keyword and (has_metric_keyword or has_money_or_percent):
        reasons.append("query_compare_or_number")
    if has_table_keyword and (has_metric_keyword or has_numeric or has_money_or_percent):
        reasons.append("query_table_keyword")

    return bool(reasons), list(dict.fromkeys(reasons))


def extract_query_anchors(query: str) -> list[str]:
    text = (query or "").strip()
    if not text:
        return []

    anchors: list[str] = []
    seen = set()

    def _add_anchor(value: str) -> None:
        normalized = (value or "").strip()
        normalized = re.sub(r"[’']s$", "", normalized, flags=re.IGNORECASE)
        if not normalized:
            return
        lowered = normalized.lower()
        if lowered in _QUERY_ANCHOR_STOPWORDS:
            return
        if normalized.isdigit():
            return
        if lowered in seen:
            return
        seen.add(lowered)
        anchors.append(normalized)

    tokens = _QUERY_ANCHOR_TOKEN_PATTERN.findall(text)
    for token in tokens:
        normalized_token = re.sub(r"[’']s$", "", token, flags=re.IGNORECASE)
        lowered = normalized_token.lower()
        if lowered in _QUERY_ANCHOR_STOPWORDS:
            continue
        if normalized_token.isupper() and any(ch.isalpha() for ch in normalized_token):
            _add_anchor(normalized_token)
            continue
        if any(ch.isdigit() for ch in normalized_token) and any(ch.isalpha() for ch in normalized_token):
            _add_anchor(normalized_token)
            continue

    title_matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
    for match in title_matches:
        parts = [part for part in match.split() if part]
        if parts and all(part.lower() not in _QUERY_ANCHOR_STOPWORDS for part in parts):
            _add_anchor(match)

    for token in tokens:
        normalized_token = re.sub(r"[’']s$", "", token, flags=re.IGNORECASE)
        if normalized_token[:1].isupper() and normalized_token[1:].islower():
            _add_anchor(normalized_token)

    return anchors


def _count_numeric_values(text: str) -> int:
    return len(_TABLE_VALUE_PATTERN.findall(text or ""))


def _anchor_in_text(anchor: str, text: str) -> bool:
    candidate = (text or "").strip()
    if not anchor or not candidate:
        return False
    if anchor.isupper() or any(ch.isdigit() for ch in anchor):
        return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(anchor)}(?![A-Za-z0-9])", candidate))
    return anchor.lower() in candidate.lower()


def _doc_matches_query_anchors(doc: dict, query_anchors: List[str]) -> bool:
    if not query_anchors:
        return True
    haystack = "\n".join(
        [
            doc.get("text", "") or "",
            doc.get("table_title", "") or "",
        ]
    )
    return any(_anchor_in_text(anchor, haystack) for anchor in query_anchors)


def _build_anchor_matched_filename_stats(docs: List[dict], query_anchors: List[str]) -> dict[str, int]:
    stats: dict[str, int] = {}
    if not query_anchors:
        return stats
    for doc in docs or []:
        filename = (doc.get("filename") or "").strip()
        if not filename:
            continue
        if _doc_matches_query_anchors(doc, query_anchors):
            stats[filename] = stats.get(filename, 0) + 1
    return stats


def _apply_anchor_guard_to_docs(docs: List[dict], query_anchors: List[str]) -> tuple[List[dict], bool, int]:
    if not query_anchors:
        return list(docs or []), False, 0
    anchor_filename_hits = _build_anchor_matched_filename_stats(docs or [], query_anchors)
    filtered: list[dict] = []
    filtered_count = 0
    for doc in docs or []:
        filename = (doc.get("filename") or "").strip()
        direct_match = _doc_matches_query_anchors(doc, query_anchors)
        filename_match = anchor_filename_hits.get(filename, 0) >= 2
        if direct_match or filename_match:
            filtered.append(doc)
        else:
            filtered_count += 1
    if filtered:
        return filtered, True, filtered_count
    return list(docs or []), True, filtered_count


def _table_matches_anchor_supporting_docs(table: dict, support_docs: List[dict], query_anchors: List[str]) -> bool:
    if not query_anchors:
        return True
    filename = (table.get("filename") or "").strip()
    table_page = table.get("page_number")
    for doc in support_docs or []:
        doc_filename = (doc.get("filename") or "").strip()
        if filename and doc_filename and filename != doc_filename:
            continue
        doc_page = doc.get("page_number")
        if table_page not in (None, "") and doc_page not in (None, "") and doc_page != table_page:
            continue
        if _doc_matches_query_anchors(doc, query_anchors):
            return True
    return False


def _apply_anchor_guard_to_tables(
    tables: List[dict],
    hits: List[dict],
    retrieved_docs: List[dict],
    query_anchors: List[str],
) -> tuple[List[dict], List[dict], bool, int]:
    if not query_anchors:
        return list(tables or []), list(hits or []), False, 0

    anchor_filename_hits = _build_anchor_matched_filename_stats(retrieved_docs or [], query_anchors)
    filtered_tables: list[dict] = []
    allowed_table_ids: set[str] = set()
    filtered_count = 0

    for table in tables or []:
        table_id = (table.get("table_id") or "").strip()
        filename = (table.get("filename") or "").strip()
        direct_match = _table_matches_query_anchors(table, query_anchors)
        support_match = _table_matches_anchor_supporting_docs(table, hits or retrieved_docs or [], query_anchors)
        filename_match = anchor_filename_hits.get(filename, 0) >= 2 if filename else False
        if direct_match or support_match or filename_match:
            filtered_tables.append(table)
            if table_id:
                allowed_table_ids.add(table_id)
        else:
            filtered_count += 1

    if not filtered_tables:
        return [], [], True, filtered_count

    filtered_hits = [
        hit
        for hit in hits or []
        if (hit.get("table_id") or "").strip() in allowed_table_ids
        or _doc_matches_query_anchors(hit, query_anchors)
    ]
    return filtered_tables, filtered_hits, True, filtered_count


def _table_matches_query_anchors(table: dict, query_anchors: List[str]) -> bool:
    if not query_anchors:
        return True
    values = [
        table.get("title", "") or "",
        table.get("caption", "") or "",
        table.get("csv_text", "") or "",
    ]
    rows = table.get("rows") or []
    for row in rows[:8]:
        if isinstance(row, dict):
            values.extend(str(value) for value in row.values())
        elif isinstance(row, list):
            values.extend(str(value) for value in row)
    haystack = "\n".join(values)
    return any(_anchor_in_text(anchor, haystack) for anchor in query_anchors)


def _query_has_explicit_table_request(query: str) -> bool:
    lowered = (query or "").lower()
    return any(term in lowered for term in _TABLE_QUERY_TABLE_HINTS)


def _chunk_looks_table_like(text: str) -> bool:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not normalized.strip():
        return False

    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    if not lines:
        return False

    numeric_rich_lines = [line for line in lines if _count_numeric_values(line) >= 3]
    if len(numeric_rich_lines) >= 2:
        return True

    lowered = normalized.lower()
    if any(metric in lowered for metric in _TABLE_QUERY_METRIC_HINTS) and any(
        _count_numeric_values(line) >= 2 for line in lines
    ):
        return True

    delimited_lines = [
        line for line in lines if (line.count("|") >= 2 or line.count("  ") >= 3) and _count_numeric_values(line) >= 2
    ]
    if delimited_lines:
        return True

    return False


def _retrieved_docs_trigger_table_aware_retrieval(docs: List[dict]) -> tuple[bool, list[str]]:
    for doc in docs or []:
        if _chunk_looks_table_like(doc.get("text", "") or ""):
            return True, ["retrieved_table_like_chunk"]
    return False, []


def _extract_table_candidate_filenames(
    docs: List[dict],
    *,
    query_anchors: List[str] | None = None,
    max_files: int = 3,
) -> list[str]:
    ranked: dict[str, dict[str, Any]] = {}
    anchor_stats = _build_anchor_matched_filename_stats(docs, query_anchors or [])
    for rank, doc in enumerate(docs or [], 1):
        filename = (doc.get("filename") or "").strip()
        if not filename:
            continue
        item = ranked.setdefault(
            filename,
            {
                "filename": filename,
                "count": 0,
                "first_rank": rank,
                "best_score": _combined_score(doc),
                "anchor_hits": anchor_stats.get(filename, 0),
            },
        )
        item["count"] += 1
        item["first_rank"] = min(item["first_rank"], rank)
        item["best_score"] = max(item["best_score"], _combined_score(doc))

    ordered = sorted(
        ranked.values(),
        key=lambda item: (
            -int(item["anchor_hits"]),
            -int(item["count"]),
            int(item["first_rank"]),
            -float(item["best_score"]),
        ),
    )
    return [item["filename"] for item in ordered[: max(1, max_files)]]


def _build_table_evidence_filter_expr(candidate_filenames: List[str]) -> str:
    base_expr = 'evidence_type != "text_chunk"'
    filenames = [name for name in candidate_filenames if name]
    if not filenames:
        return base_expr
    quoted = ", ".join(f'"{_escape_milvus_string(name)}"' for name in filenames)
    return f"{base_expr} and filename in [{quoted}]"


def _build_text_chunk_filter_expr(extra_expr: str = "") -> str:
    text_expr = '(evidence_type == "text_chunk" or evidence_type == "")'
    if not extra_expr:
        return text_expr
    return f"({extra_expr}) and {text_expr}"


def _load_tables_by_filename(filenames: List[str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for filename in filenames or []:
        clean_name = (filename or "").strip()
        if not clean_name or clean_name in out:
            continue
        try:
            out[clean_name] = _table_store.get_tables_by_filename(clean_name) or []
        except Exception:
            logger.exception("failed to load tables for filename=%s", clean_name)
            out[clean_name] = []
    return out


def _build_attached_table_payload(table: dict, include_full: bool, skipped_reason: str = "") -> dict:
    return {
        "table": table,
        "include_full": include_full,
        "skipped_reason": skipped_reason or "",
    }


def _format_evidence_unit_docs(
    evidence_units: List[dict],
    *,
    max_rows: int,
    max_context_chars: int,
) -> tuple[list[dict], int]:
    if not evidence_units:
        return [], 0

    out: list[dict] = []
    total_chars = 0
    marker = "\n... table evidence truncated ..."

    for index, unit in enumerate(evidence_units, 1):
        formatted_text = _format_evidence_unit(
            unit,
            index=index,
            preview_rows=max_rows,
            preview_chars=max_context_chars,
        )
        remaining = max_context_chars - total_chars
        if remaining <= 0:
            break
        if len(formatted_text) > remaining:
            if remaining <= len(marker):
                break
            formatted_text = formatted_text[: remaining - len(marker)] + marker
        matched_chunk = unit.get("matched_chunk") or {}
        out.append(
            {
                **matched_chunk,
                "filename": unit.get("filename") or matched_chunk.get("filename", ""),
                "doc_name": matched_chunk.get("doc_name") or get_doc_name(unit.get("filename") or matched_chunk.get("filename", "")),
                "page_number": unit.get("page_number", matched_chunk.get("page_number", "")),
                "chunk_id": unit.get("chunk_id") or matched_chunk.get("chunk_id", ""),
                "type": "evidence_unit",
                "text": formatted_text,
                "evidence_type": "evidence_unit",
                "table_id": "",
                "row_id": "",
                "table_title": "",
            }
        )
        total_chars += len(formatted_text)
        if total_chars >= max_context_chars:
            break
    return out, total_chars


def _collect_planner_terms(meta: dict | None) -> set[str]:
    meta = meta or {}
    terms: set[str] = set()
    for field in (
        "planner_semantic_queries",
        "planner_evidence_field_queries",
        "planner_table_heading_queries",
        "planner_keyword_queries",
        "planner_must_keep_terms",
    ):
        for value in meta.get(field, []) or []:
            terms |= _extract_keyword_tokens(str(value))
            terms |= _extract_metric_hints(str(value))
            terms |= _fallback_query_terms(str(value))
    return terms


def _doc_text_overlap_score(text: str, *, query_terms: set[str], planner_terms: set[str], query_numbers: set[str], query_years: set[str]) -> float:
    text_terms = _extract_keyword_tokens(text) | _extract_metric_hints(text) | _fallback_query_terms(text)
    text_metrics = _extract_metric_hints(text)
    text_numbers = _extract_numbers(text) | set(_TABLE_NUMERIC_PATTERN.findall(text or ""))
    text_years = _extract_years(text) | set(_TABLE_YEAR_PATTERN.findall(text or ""))
    score = 0.0
    if query_terms:
        score += len(text_terms & query_terms) / max(1, len(query_terms))
    if planner_terms:
        score += 0.8 * (len((text_terms | text_metrics) & planner_terms) / max(1, len(planner_terms)))
    if query_numbers:
        score += 0.8 * (len(text_numbers & query_numbers) / max(1, len(query_numbers)))
    if query_years:
        score += 0.8 * (len(text_years & query_years) / max(1, len(query_years)))
    return score


def _table_row_preview_text(row: dict) -> str:
    parts = []
    for key, value in (row or {}).items():
        if str(key).startswith("_"):
            continue
        key_text = str(key).strip()
        value_text = str(value).strip()
        if key_text and value_text:
            parts.append(f"{key_text}: {value_text}")
    return "; ".join(parts)


def _select_relevant_table_rows(
    table: dict,
    *,
    query_terms: set[str],
    planner_terms: set[str],
    query_numbers: set[str],
    query_years: set[str],
    max_rows: int,
) -> list[dict]:
    selected: list[tuple[float, dict]] = []
    for row in table.get("rows") or []:
        if not isinstance(row, dict):
            continue
        row_text = _table_row_preview_text(row)
        if not row_text:
            continue
        overlap_score = _doc_text_overlap_score(
            row_text,
            query_terms=query_terms,
            planner_terms=planner_terms,
            query_numbers=query_numbers,
            query_years=query_years,
        )
        if overlap_score <= 0:
            continue
        selected.append(
            (
                overlap_score,
                {
                    "table_id": table.get("table_id", "") or "",
                    "columns": table.get("columns") or [],
                    "row_text": row_text,
                },
            )
        )
    selected.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in selected[:max_rows]]


def _expand_group_snippets(
    seed_chunks: List[dict],
    *,
    query_terms: set[str],
    planner_terms: set[str],
    query_numbers: set[str],
    query_years: set[str],
    max_snippets: int,
) -> list[str]:
    candidate_docs: list[dict] = []
    seen_chunk_keys = set()
    for doc in seed_chunks:
        filename = (doc.get("filename") or "").strip()
        page_number = _coerce_int(doc.get("page_number"))
        if filename and page_number is not None:
            for candidate in _fetch_neighbor_page_docs(filename, page_number, 0):
                key = _doc_key(candidate)
                if key not in seen_chunk_keys:
                    seen_chunk_keys.add(key)
                    candidate_docs.append(candidate)
        chunk_idx = _coerce_int(doc.get("chunk_idx"))
        if filename and chunk_idx is not None:
            for candidate in _fetch_neighbor_chunk_docs(filename, chunk_idx, 1):
                key = _doc_key(candidate)
                if key not in seen_chunk_keys:
                    seen_chunk_keys.add(key)
                    candidate_docs.append(candidate)

    parent_ids = []
    seen_parent_ids = set()
    for doc in seed_chunks:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id and parent_id not in seen_parent_ids:
            seen_parent_ids.add(parent_id)
            parent_ids.append(parent_id)
    if parent_ids:
        for candidate in _parent_chunk_store.get_documents_by_ids(parent_ids):
            key = _doc_key(candidate)
            if key not in seen_chunk_keys:
                seen_chunk_keys.add(key)
                candidate_docs.append(candidate)

    seed_texts = {(doc.get("text") or "").strip() for doc in seed_chunks if (doc.get("text") or "").strip()}
    scored: list[tuple[float, str]] = []
    for candidate in candidate_docs:
        text = (candidate.get("text") or "").strip()
        if not text or text in seed_texts or _looks_like_cover_or_toc_chunk(candidate, ""):
            continue
        overlap_score = _doc_text_overlap_score(
            text,
            query_terms=query_terms,
            planner_terms=planner_terms,
            query_numbers=query_numbers,
            query_years=query_years,
        )
        if overlap_score <= 0:
            continue
        scored.append((overlap_score + _combined_score(candidate), text))
    scored.sort(key=lambda item: item[0], reverse=True)
    out: list[str] = []
    seen_text = set()
    for _, text in scored:
        if text in seen_text:
            continue
        seen_text.add(text)
        out.append(text)
        if len(out) >= max_snippets:
            break
    return out


def _score_evidence_group(
    group: dict,
    *,
    query_terms: set[str],
    planner_terms: set[str],
    query_numbers: set[str],
    query_years: set[str],
    query_anchors: List[str],
) -> float:
    seed_chunks = group.get("seed_chunks") or []
    combined_text = "\n".join((chunk.get("text") or "") for chunk in seed_chunks)
    score = sum(_combined_score(chunk) for chunk in seed_chunks)
    score += _doc_text_overlap_score(
        combined_text,
        query_terms=query_terms,
        planner_terms=planner_terms,
        query_numbers=query_numbers,
        query_years=query_years,
    )
    score += 0.15 * len(group.get("matched_queries") or [])
    score += 0.1 * len(group.get("planner_sources") or [])
    if group.get("attached_table_ids"):
        score += 0.4
    if query_anchors and any(_doc_matches_query_anchors(chunk, query_anchors) for chunk in seed_chunks):
        score += 0.5
    if _looks_like_cover_or_toc_chunk({"text": combined_text, "page_number": group.get("page_number")}, ""):
        score -= 1.0
    return score


def _format_evidence_group_docs(
    groups: List[dict],
    *,
    max_chars_per_group: int,
) -> tuple[list[dict], int]:
    out: list[dict] = []
    total_chars = 0
    for index, group in enumerate(groups, 1):
        formatted_text = _format_evidence_group(
            group,
            index=index,
            preview_chars=max_chars_per_group,
        )
        matched_chunk = (group.get("seed_chunks") or [{}])[0] or {}
        out.append(
            {
                **matched_chunk,
                "filename": group.get("filename") or matched_chunk.get("filename", ""),
                "doc_name": matched_chunk.get("doc_name") or get_doc_name(group.get("filename") or matched_chunk.get("filename", "")),
                "page_number": group.get("page_number", matched_chunk.get("page_number", "")),
                "chunk_id": group.get("chunk_id") or matched_chunk.get("chunk_id", ""),
                "type": "evidence_group",
                "text": formatted_text,
                "evidence_type": "evidence_group",
                "table_id": "",
                "row_id": "",
                "table_title": "",
            }
        )
        total_chars += len(formatted_text)
    return out, total_chars


def _build_evidence_groups(
    query: str,
    context_docs: List[dict],
    final_retrieved_docs: List[dict],
    retrieval_meta: dict | None,
) -> tuple[list[dict] | None, dict]:
    table_config = get_table_aware_retrieval_config()
    group_config = get_evidence_group_config()
    query_anchors = extract_query_anchors(query)
    query_terms = _extract_keyword_tokens(query) | _extract_metric_hints(query) | _fallback_query_terms(query)
    query_numbers = _extract_numbers(query)
    query_years = _extract_years(query)
    planner_terms = _collect_planner_terms(retrieval_meta)
    source_docs = list(context_docs or [])
    combined_docs = _deduplicate_docs(source_docs + list(final_retrieved_docs or []))
    candidate_filenames = _extract_table_candidate_filenames(
        combined_docs,
        query_anchors=query_anchors,
        max_files=table_config["max_candidate_docs"],
    )
    should_enable, auto_triggered, trigger_reasons = _should_enable_table_aware_retrieval(
        query,
        combined_docs,
        table_config["mode"],
    )
    skipped_reasons: list[str] = []
    base_meta = {
        "table_aware_retrieval_mode": table_config["mode"],
        "table_aware_auto_triggered": auto_triggered,
        "table_aware_trigger_reason": trigger_reasons,
        "table_context_source": "none",
        "table_candidate_filenames": candidate_filenames,
        "table_candidate_pages": _extract_table_candidate_pages(source_docs),
        "table_ids": [],
        "table_context_skipped_reasons": skipped_reasons,
        "evidence_group_count": 0,
        "selected_evidence_group_count": 0,
        "evidence_groups_debug": [],
        "selected_evidence_groups": [],
        "group_scores": [],
        "expanded_snippet_count": 0,
        "relevant_table_row_count": 0,
        "dropped_group_reasons": [],
        "final_evidence_pack_source": "chunk_rerank_fallback",
        "evidence_unit_count": 0,
        "evidence_units_with_tables": 0,
        "table_attached_count": 0,
        "table_attach_reasons": [],
    }
    if table_config["mode"] == "off" or not group_config["enabled"]:
        return None, base_meta
    if not should_enable and table_config["mode"] != "force":
        _append_skip_reason(skipped_reasons, "non_table_query")
        return None, base_meta

    table_cache = _load_tables_by_filename([doc.get("filename") or "" for doc in combined_docs] + candidate_filenames)
    groups_by_key: dict[tuple, dict] = {}
    for doc in source_docs:
        filename = (doc.get("filename") or "").strip()
        page_number = _coerce_int(doc.get("page_number"))
        if not filename or page_number is None:
            continue
        key = (filename, page_number)
        group = groups_by_key.setdefault(
            key,
            {
                "seed_chunks": [],
                "filename": filename,
                "page_number": page_number,
                "parent_chunk_ids": [],
                "matched_queries": [],
                "planner_sources": [],
                "group_score": 0.0,
                "expanded_snippets": [],
                "relevant_table_rows": [],
                "attached_table_ids": [],
                "table_attach_reason": "none",
            },
        )
        group["seed_chunks"].append(doc)
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id and parent_id not in group["parent_chunk_ids"]:
            group["parent_chunk_ids"].append(parent_id)
        for item in doc.get("planner_queries", []) or []:
            if item not in group["matched_queries"]:
                group["matched_queries"].append(item)
        for item in doc.get("planner_sources", []) or []:
            if item not in group["planner_sources"]:
                group["planner_sources"].append(item)

    groups = list(groups_by_key.values())
    dropped_group_reasons: list[dict] = []
    selected_table_ids: list[str] = []
    any_table_attached = False
    for group in groups:
        seed_chunks = group.get("seed_chunks") or []
        chunk_table_like = any(_chunk_looks_table_like(doc.get("text", "") or "") for doc in seed_chunks)
        explicit_table_request = _query_has_explicit_table_request(query)
        attach_reason = "chunk_table_like" if chunk_table_like else ("query_explicit_table_request" if explicit_table_request else "none")
        group["matched_snippets"] = [(doc.get("text") or "").strip() for doc in seed_chunks if (doc.get("text") or "").strip()][:group_config["max_snippets_per_group"]]
        group["expanded_snippets"] = _expand_group_snippets(
            seed_chunks,
            query_terms=query_terms,
            planner_terms=planner_terms,
            query_numbers=query_numbers,
            query_years=query_years,
            max_snippets=group_config["max_snippets_per_group"],
        )
        same_page_tables = [
            table for table in (table_cache.get(group["filename"]) or [])
            if _coerce_int(table.get("page_number")) == group["page_number"]
        ]
        if same_page_tables:
            filtered_tables, _, _, _ = _apply_anchor_guard_to_tables(
                same_page_tables,
                seed_chunks,
                combined_docs,
                query_anchors,
            )
            for table in filtered_tables[: table_config["max_tables"]]:
                is_full, rejected_reason = _table_quality_guard(table)
                if not is_full:
                    dropped_group_reasons.append(
                        {
                            "filename": group["filename"],
                            "page_number": group["page_number"],
                            "reason": rejected_reason or "table_quality_rejected",
                        }
                    )
                    continue
                rows = _select_relevant_table_rows(
                    table,
                    query_terms=query_terms,
                    planner_terms=planner_terms,
                    query_numbers=query_numbers,
                    query_years=query_years,
                    max_rows=group_config["max_table_rows_per_group"],
                )
                if not rows:
                    continue
                group["relevant_table_rows"].extend(rows)
                table_id = (table.get("table_id") or "").strip()
                if table_id and table_id not in group["attached_table_ids"]:
                    group["attached_table_ids"].append(table_id)
                    selected_table_ids.append(table_id)
                any_table_attached = True
        if group["attached_table_ids"] and attach_reason == "none":
            attach_reason = "same_page_high_quality_table"
        group["table_attach_reason"] = attach_reason
        group["group_score"] = _score_evidence_group(
            group,
            query_terms=query_terms,
            planner_terms=planner_terms,
            query_numbers=query_numbers,
            query_years=query_years,
            query_anchors=query_anchors,
        )

    if not any_table_attached and (table_config["mode"] == "force" or (table_config["mode"] == "auto" and should_enable)):
        if candidate_filenames or table_config.get("global_fallback"):
            filter_expr = _build_table_evidence_filter_expr(candidate_filenames)
            dense_embedding, sparse_embedding = _get_hybrid_query_embeddings(query)
            search_hits = _milvus_manager.hybrid_retrieve(
                dense_embedding=dense_embedding,
                sparse_embedding=sparse_embedding,
                top_k=table_config["top_k"],
                filter_expr=filter_expr,
                query_text=query,
            )
            search_table_ids = _dedupe_table_ids_for_context(search_hits)[: table_config["max_tables"]]
            candidate_tables = _table_store.get_tables_by_ids(search_table_ids) if search_table_ids else []
            if candidate_filenames:
                filename_set = set(candidate_filenames)
                candidate_tables = [table for table in candidate_tables if (table.get("filename") or "") in filename_set]
            filtered_tables, _, _, _ = _apply_anchor_guard_to_tables(
                candidate_tables,
                search_hits,
                combined_docs,
                query_anchors,
            )
            for table in filtered_tables:
                filename = (table.get("filename") or "").strip()
                matching_groups = [group for group in groups if group.get("filename") == filename]
                if not matching_groups:
                    continue
                rows = _select_relevant_table_rows(
                    table,
                    query_terms=query_terms,
                    planner_terms=planner_terms,
                    query_numbers=query_numbers,
                    query_years=query_years,
                    max_rows=group_config["max_table_rows_per_group"],
                )
                if not rows:
                    continue
                matching_group = max(matching_groups, key=lambda item: item.get("group_score", 0.0))
                matching_group["relevant_table_rows"].extend(rows)
                table_id = (table.get("table_id") or "").strip()
                if table_id and table_id not in matching_group["attached_table_ids"]:
                    matching_group["attached_table_ids"].append(table_id)
                    selected_table_ids.append(table_id)
                if matching_group.get("table_attach_reason") == "none":
                    matching_group["table_attach_reason"] = "same_page_high_quality_table"
                any_table_attached = True
            if any_table_attached:
                base_meta["table_context_source"] = "document_scoped_search"

    groups.sort(key=lambda item: float(item.get("group_score", 0.0) or 0.0), reverse=True)
    selected_groups = groups[: group_config["max_groups"]]
    formatted_docs, total_chars = _format_evidence_group_docs(
        selected_groups,
        max_chars_per_group=group_config["max_chars_per_group"],
    )

    if not formatted_docs:
        _append_skip_reason(skipped_reasons, "no_valid_table_candidates")
        return None, base_meta

    source = base_meta["table_context_source"]
    if source == "none":
        if any(group.get("attached_table_ids") for group in selected_groups):
            source = "same_page_table"
        else:
            _append_skip_reason(skipped_reasons, "no_valid_table_candidates")

    meta = {
        **base_meta,
        "table_context_source": source,
        "table_ids": list(dict.fromkeys(selected_table_ids)),
        "evidence_group_count": len(groups),
        "selected_evidence_group_count": len(selected_groups),
        "evidence_groups_debug": [_build_group_debug_payload(group) for group in groups],
        "selected_evidence_groups": [_build_group_debug_payload(group) for group in selected_groups],
        "group_scores": [
            {
                "filename": group.get("filename", ""),
                "page_number": group.get("page_number", ""),
                "group_score": group.get("group_score", 0.0),
            }
            for group in groups
        ],
        "expanded_snippet_count": sum(len(group.get("expanded_snippets") or []) for group in selected_groups),
        "relevant_table_row_count": sum(len(group.get("relevant_table_rows") or []) for group in selected_groups),
        "dropped_group_reasons": dropped_group_reasons,
        "final_evidence_pack_source": "evidence_groups",
        "table_context_table_count": sum(len(group.get("attached_table_ids") or []) for group in selected_groups),
        "table_context_char_count": total_chars,
        "evidence_unit_count": len(selected_groups),
        "evidence_units_with_tables": sum(1 for group in selected_groups if group.get("attached_table_ids")),
        "table_attached_count": sum(len(group.get("attached_table_ids") or []) for group in selected_groups),
        "table_attach_reasons": list(
            dict.fromkeys(
                reason for reason in (group.get("table_attach_reason", "none") for group in selected_groups) if reason and reason != "none"
            )
        ),
    }
    return formatted_docs, meta


def _build_evidence_units(
    query: str,
    context_docs: List[dict],
    final_retrieved_docs: List[dict],
) -> tuple[list[dict] | None, dict]:
    config = get_table_aware_retrieval_config()
    source_docs = list(context_docs or [])
    reference_docs = list(final_retrieved_docs or [])
    combined_docs = _deduplicate_docs(source_docs + reference_docs)
    query_anchors = extract_query_anchors(query)
    candidate_filenames = _extract_table_candidate_filenames(
        combined_docs,
        query_anchors=query_anchors,
        max_files=config["max_candidate_docs"],
    )
    candidate_pages = _extract_table_candidate_pages(source_docs)
    should_enable, auto_triggered, trigger_reasons = _should_enable_table_aware_retrieval(
        query,
        combined_docs,
        config["mode"],
    )
    query_triggered, _ = _query_triggers_table_aware_retrieval(query)
    skipped_reasons: list[str] = []
    base_meta = {
        "table_aware_retrieval_mode": config["mode"],
        "table_aware_auto_triggered": auto_triggered,
        "table_aware_trigger_reason": trigger_reasons,
        "query_anchors": query_anchors,
        "anchor_guard_applied": False,
        "anchor_filtered_count": 0,
        "table_context_source": "none",
        "table_evidence_hit_count": 0,
        "table_context_table_count": 0,
        "table_context_char_count": 0,
        "table_candidate_filenames": candidate_filenames,
        "table_candidate_pages": candidate_pages,
        "table_ids": [],
        "table_context_skipped_reasons": skipped_reasons,
        "evidence_unit_count": 0,
        "evidence_units_with_tables": 0,
        "table_attached_count": 0,
        "table_attach_reasons": [],
    }
    if config["mode"] == "off":
        return None, base_meta
    if not should_enable:
        _append_skip_reason(skipped_reasons, "non_table_query")
        return None, base_meta

    explicit_table_request = _query_has_explicit_table_request(query)
    anchor_filename_hits = _build_anchor_matched_filename_stats(combined_docs, query_anchors)
    table_cache = _load_tables_by_filename([doc.get("filename") or "" for doc in combined_docs])

    evidence_units: list[dict] = []
    attached_table_ids: list[str] = []
    attached_table_seen: set[str] = set()
    table_attach_reasons: list[str] = []
    anchor_guard_applied = False
    anchor_filtered_count = 0
    source = "none"

    for doc in source_docs:
        filename = (doc.get("filename") or "").strip()
        page_number = _coerce_int(doc.get("page_number"))
        unit = {
            "matched_chunk": doc,
            "filename": filename,
            "page_number": doc.get("page_number", ""),
            "chunk_id": doc.get("chunk_id", ""),
            "text": doc.get("text", "") or "",
            "attached_tables": [],
            "table_attach_reason": "none",
        }
        if filename and page_number is not None:
            same_page_tables = [
                table
                for table in (table_cache.get(filename) or [])
                if _coerce_int(table.get("page_number")) == page_number
            ]
            if same_page_tables:
                filtered_tables, _, table_anchor_applied, table_anchor_filtered = _apply_anchor_guard_to_tables(
                    same_page_tables,
                    [doc],
                    combined_docs,
                    query_anchors,
                )
                anchor_guard_applied = anchor_guard_applied or table_anchor_applied
                anchor_filtered_count += table_anchor_filtered
                same_page_tables = filtered_tables
                chunk_table_like = _chunk_looks_table_like(doc.get("text", "") or "")
                attach_reason = "none"
                if chunk_table_like:
                    attach_reason = "chunk_table_like"
                elif explicit_table_request:
                    attach_reason = "query_explicit_table_request"
                elif config["mode"] == "force":
                    attach_reason = "same_page_high_quality_table"

                if same_page_tables and attach_reason != "none":
                    for table in same_page_tables:
                        table_id = (table.get("table_id") or "").strip()
                        is_full, rejected_reason = _table_quality_guard(table)
                        unit["attached_tables"].append(
                            _build_attached_table_payload(
                                table,
                                include_full=is_full,
                                skipped_reason=rejected_reason or "table_quality_rejected",
                            )
                        )
                        if table_id and table_id not in attached_table_seen:
                            attached_table_seen.add(table_id)
                            attached_table_ids.append(table_id)
                    unit["table_attach_reason"] = attach_reason
                    source = source if source != "none" else "same_page_table"
                    if attach_reason not in table_attach_reasons:
                        table_attach_reasons.append(attach_reason)
        evidence_units.append(unit)

    if not attached_table_ids:
        if not candidate_filenames:
            _append_skip_reason(skipped_reasons, "no_candidate_documents")
        if config["mode"] == "force" or (config["mode"] == "auto" and query_triggered):
            allow_global_fallback = bool(config.get("global_fallback")) and not candidate_filenames
            if candidate_filenames or allow_global_fallback:
                filter_expr = _build_table_evidence_filter_expr(candidate_filenames)
                dense_embedding, sparse_embedding = _get_hybrid_query_embeddings(query)
                search_hits = _milvus_manager.hybrid_retrieve(
                    dense_embedding=dense_embedding,
                    sparse_embedding=sparse_embedding,
                    top_k=config["top_k"],
                    filter_expr=filter_expr,
                    query_text=query,
                )
                search_table_ids = _dedupe_table_ids_for_context(search_hits)[: config["max_tables"]]
                candidate_tables = _table_store.get_tables_by_ids(search_table_ids) if search_table_ids else []
                if candidate_filenames:
                    candidate_filename_set = set(candidate_filenames)
                    candidate_tables = [
                        table for table in candidate_tables if (table.get("filename") or "") in candidate_filename_set
                    ]
                filtered_tables, filtered_hits, table_anchor_applied, table_anchor_filtered = _apply_anchor_guard_to_tables(
                    candidate_tables,
                    search_hits,
                    combined_docs,
                    query_anchors,
                )
                anchor_guard_applied = anchor_guard_applied or table_anchor_applied
                anchor_filtered_count += table_anchor_filtered
                if filtered_tables:
                    source = "document_scoped_search" if candidate_filenames else "global_fallback"
                    evidence_by_filename: dict[str, list[dict]] = {}
                    for table in filtered_tables[: config["max_tables"]]:
                        filename = (table.get("filename") or "").strip()
                        evidence_by_filename.setdefault(filename, []).append(table)
                    for unit in evidence_units:
                        filename = (unit.get("filename") or "").strip()
                        unit_tables = evidence_by_filename.get(filename, [])
                        if not unit_tables:
                            continue
                        attach_reason = "query_explicit_table_request" if explicit_table_request else "same_page_high_quality_table"
                        for table in unit_tables:
                            table_id = (table.get("table_id") or "").strip()
                            is_full, rejected_reason = _table_quality_guard(table)
                            unit["attached_tables"].append(
                                _build_attached_table_payload(
                                    table,
                                    include_full=is_full,
                                    skipped_reason=rejected_reason or "table_quality_rejected",
                                )
                            )
                            if table_id and table_id not in attached_table_seen:
                                attached_table_seen.add(table_id)
                                attached_table_ids.append(table_id)
                        if unit_tables:
                            unit["table_attach_reason"] = attach_reason
                            if attach_reason not in table_attach_reasons:
                                table_attach_reasons.append(attach_reason)
                    if not attached_table_ids:
                        _append_skip_reason(skipped_reasons, "no_valid_table_candidates")
                    table_evidence_hit_count = len(filtered_hits)
                else:
                    _append_skip_reason(skipped_reasons, "anchor_guard_filtered" if table_anchor_filtered else "no_valid_table_candidates")
                    table_evidence_hit_count = 0
            else:
                _append_skip_reason(skipped_reasons, "no_candidate_documents")

    if not attached_table_ids:
        if not any(unit.get("attached_tables") for unit in evidence_units):
            _append_skip_reason(skipped_reasons, "no_valid_table_candidates")

    formatted_docs, total_chars = _format_evidence_unit_docs(
        evidence_units,
        max_rows=config["max_rows"],
        max_context_chars=config["max_context_chars"],
    )

    meta = {
        **base_meta,
        "anchor_guard_applied": anchor_guard_applied,
        "anchor_filtered_count": anchor_filtered_count,
        "table_context_source": source,
        "table_evidence_hit_count": len(attached_table_ids) if attached_table_ids else 0,
        "table_context_table_count": len(attached_table_seen),
        "table_context_char_count": total_chars,
        "table_ids": attached_table_ids,
        "evidence_unit_count": len(evidence_units),
        "evidence_units_with_tables": sum(1 for unit in evidence_units if unit.get("attached_tables")),
        "table_attached_count": sum(len(unit.get("attached_tables") or []) for unit in evidence_units),
        "table_attach_reasons": table_attach_reasons,
    }
    if not any(unit.get("attached_tables") for unit in evidence_units):
        return None, meta
    return formatted_docs or None, meta


def _is_retrieved_table_evidence(doc: dict) -> bool:
    evidence_type = (doc.get("evidence_type") or "").strip()
    chunk_id = (doc.get("chunk_id") or "").strip()
    table_id = (doc.get("table_id") or "").strip()
    return (
        evidence_type in {"table_summary", "table_row", "table_raw"}
        or bool(table_id)
        or "::table::" in chunk_id
    )


def _extract_retrieved_table_ids(docs: List[dict], max_tables: int) -> list[str]:
    out: list[str] = []
    seen = set()
    for doc in docs or []:
        if not _is_retrieved_table_evidence(doc):
            continue
        table_id = (doc.get("table_id") or "").strip()
        if not table_id or table_id in seen:
            continue
        seen.add(table_id)
        out.append(table_id)
        if len(out) >= max(1, max_tables):
            break
    return out


def _extract_table_candidate_pages(docs: List[dict], max_pages: int = 6) -> list[dict]:
    out: list[dict] = []
    seen = set()
    for doc in docs or []:
        if _is_retrieved_table_evidence(doc):
            continue
        filename = (doc.get("filename") or "").strip()
        page_number = _coerce_int(doc.get("page_number"))
        if not filename or page_number is None:
            continue
        if not _chunk_looks_table_like(doc.get("text", "") or ""):
            continue
        key = (filename, page_number)
        if key in seen:
            continue
        seen.add(key)
        out.append({"filename": filename, "page_number": page_number})
        if len(out) >= max(1, max_pages):
            break
    return out


def _fetch_tables_for_candidate_pages(candidate_pages: List[dict], max_tables: int) -> list[dict]:
    if not candidate_pages:
        return []
    page_map: dict[str, set[int]] = {}
    ordered_filenames: list[str] = []
    for item in candidate_pages:
        filename = (item.get("filename") or "").strip()
        page_number = _coerce_int(item.get("page_number"))
        if not filename or page_number is None:
            continue
        if filename not in page_map:
            page_map[filename] = set()
            ordered_filenames.append(filename)
        page_map[filename].add(page_number)

    out: list[dict] = []
    seen = set()
    for filename in ordered_filenames:
        tables = _table_store.get_tables_by_filename(filename)
        for table in tables:
            table_id = (table.get("table_id") or "").strip()
            page_number = _coerce_int(table.get("page_number"))
            if not table_id or table_id in seen or page_number not in page_map.get(filename, set()):
                continue
            seen.add(table_id)
            out.append(table)
            if len(out) >= max(1, max_tables):
                return out
    return out


def _build_same_page_table_hits(tables: List[dict], docs: List[dict]) -> list[dict]:
    hits: list[dict] = []
    table_by_page: dict[tuple[str, int], list[dict]] = {}
    for table in tables or []:
        filename = (table.get("filename") or "").strip()
        page_number = _coerce_int(table.get("page_number"))
        if not filename or page_number is None:
            continue
        table_by_page.setdefault((filename, page_number), []).append(table)

    for doc in docs or []:
        filename = (doc.get("filename") or "").strip()
        page_number = _coerce_int(doc.get("page_number"))
        if not filename or page_number is None:
            continue
        matched_tables = table_by_page.get((filename, page_number), [])
        for table in matched_tables:
            hits.append(
                {
                    "table_id": table.get("table_id", ""),
                    "score": _combined_score(doc),
                    "text": doc.get("text", "") or "",
                    "evidence_type": "same_page_table",
                    "row_id": "",
                    "page_number": page_number,
                }
            )
    return hits


def _is_sentence_like_cell(text: str) -> bool:
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if _count_numeric_values(cleaned) > 0:
        return False
    words = re.findall(r"[A-Za-z]+", cleaned)
    return len(words) >= 5


def _table_quality_guard(table: dict) -> tuple[bool, str]:
    columns = [str(item).strip() for item in (table.get("columns") or []) if str(item).strip()]
    rows = table.get("rows") or []
    title = ((table.get("title") or "") or (table.get("caption") or "")).strip()
    effective_col_count = len(columns)
    data_row_count = len(rows)

    if effective_col_count >= 8 and data_row_count <= 1:
        return False, "table_quality_rejected"

    sentence_like_columns = sum(1 for col in columns if _is_sentence_like_cell(col))
    if effective_col_count >= 4 and sentence_like_columns >= max(3, effective_col_count // 2):
        return False, "table_quality_rejected"

    numeric_cells = 0
    non_empty_cells = 0
    for row in rows[:8]:
        values = row.values() if isinstance(row, dict) else row
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            non_empty_cells += 1
            if _count_numeric_values(text) > 0:
                numeric_cells += 1
    if non_empty_cells > 0 and effective_col_count >= 3 and (numeric_cells / non_empty_cells) < 0.12:
        return False, "table_quality_rejected"

    if title and _is_sentence_like_cell(title) and effective_col_count >= 6 and data_row_count <= 2:
        return False, "table_quality_rejected"

    return True, ""


def _append_skip_reason(reasons: list[str], reason: str) -> None:
    if reason and reason not in reasons:
        reasons.append(reason)


def _should_enable_table_aware_retrieval(query: str, retrieved_docs: List[dict], mode: str) -> tuple[bool, bool, list[str]]:
    if mode == "off":
        return False, False, []
    if mode == "force":
        return True, True, ["force"]

    query_triggered, query_reasons = _query_triggers_table_aware_retrieval(query)
    if query_triggered:
        return True, True, query_reasons

    retrieved_triggered, retrieved_reasons = _retrieved_docs_trigger_table_aware_retrieval(retrieved_docs)
    if retrieved_triggered:
        return True, True, retrieved_reasons

    return False, False, []


def get_doc_name(filename: str) -> str:
    return normalize_doc_name(filename or "")


def _escape_milvus_string(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace('"', '\\"')


def _doc_key(doc: dict) -> tuple:
    return (
        doc.get("chunk_id") or "",
        doc.get("filename") or "",
        doc.get("page_number"),
        doc.get("chunk_idx"),
        doc.get("text") or "",
    )


def _deduplicate_docs(docs: List[dict]) -> List[dict]:
    out: List[dict] = []
    seen = set()
    for doc in docs:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _combined_score(doc: dict) -> float:
    for field in ("rerank_score", "score"):
        try:
            value = doc.get(field)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


_COVER_TEXT_HINTS = (
    "table of contents",
    "index to consolidated financial statements",
    "united states securities and exchange commission",
    "washington, d.c. 20549",
    "commission file number",
    "form 10-k",
    "form 10-q",
    "annual report pursuant to section 13 or 15(d)",
    "quarterly report pursuant to section 13 or 15(d)",
)

_COVER_QUERY_HINTS = (
    "cover",
    "title page",
    "filing date",
    "filed",
    "commission file number",
    "cik",
    "form 10-k",
    "form 10-q",
    "exchange commission",
)

def _query_targets_cover_metadata(query: str) -> bool:
    lowered = (query or "").strip().lower()
    return any(hint in lowered for hint in _COVER_QUERY_HINTS)


def _looks_like_cover_or_toc_chunk(doc: dict, query: str) -> bool:
    if _query_targets_cover_metadata(query):
        return False

    text = (doc.get("text") or "").lower()
    if not text:
        return False

    page_number = _coerce_int(doc.get("page_number"))
    cover_hint_hits = sum(1 for hint in _COVER_TEXT_HINTS if hint in text)

    if "table of contents" in text or "index to consolidated financial statements" in text:
        return True
    if cover_hint_hits >= 2:
        return True
    if page_number is not None and page_number <= 0 and cover_hint_hits >= 1:
        return True
    return False


def _filter_cover_or_toc_docs(query: str, docs: List[dict]) -> Tuple[List[dict], int]:
    filtered: List[dict] = []
    removed = 0
    for doc in docs:
        if _looks_like_cover_or_toc_chunk(doc, query):
            removed += 1
            continue
        filtered.append(doc)
    return filtered, removed


def _page_zero_count(docs: List[dict]) -> int:
    return sum(1 for doc in docs if (_coerce_int(doc.get("page_number")) or 0) <= 0)


def _page_distribution(docs: List[dict]) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    for doc in docs:
        page = doc.get("page_number")
        key = "" if page is None else str(page)
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _dot_product(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right)))


def _sparse_dot(left: dict, right: dict) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return float(sum(float(value) * float(right.get(key, 0.0)) for key, value in left.items()))


def _extract_numbers(text: str) -> set[str]:
    return _feature_extract_numbers(text)


def _extract_years(text: str) -> set[str]:
    return _feature_extract_years(text)


def _extract_keyword_tokens(text: str) -> set[str]:
    return _feature_extract_keyword_tokens(text)


def _extract_metric_hints(text: str) -> set[str]:
    return _feature_extract_metric_hints(text)


_FINANCE_OPERAND_HINTS = {
    "quick ratio": {"cash", "cash equivalents", "receivables", "current liabilities", "short-term investments"},
    "current ratio": {"current assets", "current liabilities"},
    "working capital ratio": {"current assets", "current liabilities"},
    "operating margin": {"operating income", "income from operations", "revenue", "net sales"},
    "gross margin": {"gross profit", "revenue", "net sales"},
    "fixed asset turnover": {"revenue", "net sales", "property plant equipment", "ppe", "pp&e"},
    "ppne": {"property plant equipment", "ppe", "pp&e", "net property"},
    "free cash flow": {"operating cash flow", "capital expenditures", "capex"},
}


def _expand_finance_operand_tokens(query: str) -> set[str]:
    lowered = (query or "").lower()
    expanded: set[str] = set()
    for metric, hints in _FINANCE_OPERAND_HINTS.items():
        if metric in lowered:
            for hint in hints:
                expanded.update(_extract_keyword_tokens(hint))
    return expanded


def _fallback_query_terms(text: str) -> set[str]:
    terms = set()
    for token in _QUERY_ANCHOR_TOKEN_PATTERN.findall(text or ""):
        lowered = token.lower()
        if lowered in _QUERY_ANCHOR_STOPWORDS or len(lowered) <= 2:
            continue
        terms.add(lowered)
    return terms


def _score_overlap(query_values: set[str], page_values: set[str]) -> float:
    if not query_values:
        return 0.0
    return len(query_values & page_values) / max(1, len(query_values))


def _safe_preview(text: str, limit: int) -> str:
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def _contains_company_alias(text: str, aliases: List[str]) -> bool:
    original = text or ""
    lowered = original.lower()
    for alias in aliases:
        if not alias:
            continue
        if alias.isupper() and len(alias) <= 4:
            import re as _re

            if _re.search(rf"(?<![A-Za-z0-9]){_re.escape(alias)}(?![A-Za-z0-9])", original):
                return True
            continue
        if alias.lower() in lowered:
            return True
    return False


def _company_match_score(query_parse: Dict[str, Any], filename: str, page_text: str, table_text: str = "") -> float:
    company = query_parse.get("company") or ""
    if not company:
        return 0.0
    haystack = "\n".join([filename or "", page_text or "", table_text or ""])
    return 1.0 if matches_company_text(haystack, company) else 0.0


def _year_match_score(query_parse: Dict[str, Any], filename: str, page_years: set[str], page_text: str) -> float:
    years = {str(year) for year in (query_parse.get("years") or [])}
    if not years:
        return 0.0
    filename_years = _extract_years(filename)
    text_years = page_years or _extract_years(page_text)
    if years & filename_years:
        return 1.0
    if years & text_years:
        return 0.5
    return 0.0


def _doc_type_match_score(query_parse: Dict[str, Any], filename: str, page_text: str) -> float:
    query_doc_type = query_parse.get("doc_type") or ""
    if not query_doc_type:
        return 0.0
    page_doc_type = infer_doc_type(f"{filename}\n{page_text}")
    return 1.0 if page_doc_type == query_doc_type else 0.0


def _normalize_component_scores(items: List[dict], field_name: str, target_name: str) -> None:
    values = [float(item.get(field_name, 0.0) or 0.0) for item in items]
    max_value = max(values) if values else 0.0
    for item in items:
        raw = float(item.get(field_name, 0.0) or 0.0)
        item[target_name] = raw / max_value if max_value > 0 else 0.0


def _estimate_prompt_chars(entries: List[dict]) -> int:
    return sum(len((entry.get("text") or "").strip()) for entry in entries)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _build_filename_filter(filenames: List[str]) -> str:
    clauses = [
        f'filename == "{_escape_milvus_string(filename)}"'
        for filename in filenames
        if filename
    ]
    if not clauses:
        return f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    return f"({' or '.join(clauses)}) and chunk_level == {LEAF_RETRIEVE_LEVEL}"


def _get_rerank_endpoint() -> str:
    if not RERANK_BINDING_HOST:
        return ""
    host = RERANK_BINDING_HOST.strip().rstrip("/")
    return host if host.endswith("/v1/rerank") else f"{host}/v1/rerank"


def _rerank_candidate_hash(doc: dict) -> str:
    existing = str(doc.get("content_hash") or doc.get("embedding_cache_key") or "").strip()
    if existing:
        return existing
    identity = {
        "chunk_id": doc.get("chunk_id") or "",
        "filename": doc.get("filename") or "",
        "page_number": doc.get("page_number"),
        "chunk_idx": doc.get("chunk_idx"),
        "text": doc.get("text") or "",
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_rerank_cache_key(query: str, model: str, docs: List[dict], top_n: int) -> str:
    identity = {
        "version": 1,
        "query": str(query or "").strip(),
        "model": str(model or "").strip(),
        "top_n": int(top_n),
        "candidate_hashes": [_rerank_candidate_hash(doc) for doc in docs],
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"rerank:v1:{hashlib.sha256(encoded).hexdigest()}"


def _validate_remote_rerank_results(payload: Any, candidate_count: int, expected_count: int) -> List[dict]:
    if not isinstance(payload, dict):
        raise ValueError("invalid_response_payload")
    items = payload.get("results")
    if not isinstance(items, list):
        raise ValueError("missing_rerank_results")
    if len(items) != expected_count:
        raise ValueError(f"result_count_mismatch:{len(items)}!={expected_count}")
    normalized: List[dict] = []
    seen_indices: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("invalid_rerank_result")
        index = item.get("index")
        score = item.get("relevance_score")
        if not isinstance(index, int) or not 0 <= index < candidate_count:
            raise ValueError(f"invalid_rerank_index:{index}")
        if index in seen_indices:
            raise ValueError(f"duplicate_rerank_index:{index}")
        if not isinstance(score, (int, float)):
            raise ValueError(f"invalid_rerank_score:{score}")
        seen_indices.add(index)
        normalized.append({"index": index, "relevance_score": float(score)})
    return normalized


def _sync_rerank_trace(meta: Dict[str, Any]) -> None:
    meta["rerank_trace"] = {
        "remote_attempt_count": int(meta.get("remote_attempt_count") or 0),
        "remote_success": bool(meta.get("remote_success")),
        "fallback_used": bool(meta.get("rerank_fallback_used")),
        "fallback_reason": str(meta.get("rerank_fallback_reason") or ""),
        "rerank_cache_hit": bool(meta.get("rerank_cache_hit")),
    }


def _remote_rerank_from_items(remote_docs: List[dict], items: List[dict], top_k: int) -> List[dict]:
    reranked: List[dict] = []
    for item in items:
        doc = dict(remote_docs[item["index"]])
        doc["rerank_score"] = item["relevance_score"]
        reranked.append(doc)
    return reranked[:top_k]


def _merge_to_parent_level(docs: List[dict], threshold: int = 2) -> Tuple[List[dict], int]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if parent_id:
            groups[parent_id].append(doc)

    merge_parent_ids = [parent_id for parent_id, children in groups.items() if len(children) >= threshold]
    if not merge_parent_ids:
        return docs, 0

    parent_docs = _parent_chunk_store.get_documents_by_ids(merge_parent_ids)
    parent_map = {item.get("chunk_id", ""): item for item in parent_docs if item.get("chunk_id")}

    merged_docs: List[dict] = []
    merged_count = 0
    for doc in docs:
        parent_id = (doc.get("parent_chunk_id") or "").strip()
        if not parent_id or parent_id not in parent_map:
            merged_docs.append(doc)
            continue
        parent_doc = dict(parent_map[parent_id])
        score = doc.get("score")
        if score is not None:
            parent_doc["score"] = max(float(parent_doc.get("score", score)), float(score))
        rerank_score = doc.get("rerank_score")
        if rerank_score is not None:
            parent_doc["rerank_score"] = max(float(parent_doc.get("rerank_score", rerank_score)), float(rerank_score))
        parent_doc["merged_from_children"] = True
        parent_doc["merged_child_count"] = len(groups[parent_id])
        merged_docs.append(parent_doc)
        merged_count += 1

    return _deduplicate_docs(merged_docs), merged_count


def _auto_merge_documents(docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    auto_merge_enabled = AUTO_MERGE_ENABLED and not uses_clean_baseline_path()
    if not auto_merge_enabled or not docs:
        return docs[:top_k], {
            "auto_merge_enabled": auto_merge_enabled,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }

    merged_docs, merged_count_l3_l2 = _merge_to_parent_level(docs, threshold=AUTO_MERGE_THRESHOLD)
    merged_docs, merged_count_l2_l1 = _merge_to_parent_level(merged_docs, threshold=AUTO_MERGE_THRESHOLD)

    merged_docs.sort(key=_combined_score, reverse=True)
    merged_docs = merged_docs[:top_k]

    replaced_count = merged_count_l3_l2 + merged_count_l2_l1
    return merged_docs, {
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": replaced_count > 0,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": replaced_count,
        "auto_merge_steps": int(merged_count_l3_l2 > 0) + int(merged_count_l2_l1 > 0),
    }


def _rerank_documents(query: str, docs: List[dict], top_k: int) -> Tuple[List[dict], Dict[str, Any]]:
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(docs, 1)]
    meta: Dict[str, Any] = {
        "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
        "remote_rerank_candidate_count": 0,
        "remote_rerank_anchor_count": 0,
        "remote_rerank_input_chars": 0,
        "remote_rerank_rate_limited": False,
        "remote_attempt_count": 0,
        "remote_success": False,
        "remote_errors": [],
        "rerank_cache_hit": False,
        "rerank_fallback_used": False,
        "rerank_fallback_reason": "",
        "rerank_provider": "none",
        "local_rerank_enabled": _local_reranker.enabled,
        "local_rerank_applied": False,
        "local_rerank_error": None,
    }
    _sync_rerank_trace(meta)
    if not docs_with_rank:
        return docs_with_rank[:top_k], meta
    if meta["rerank_enabled"]:
        anchor_docs = []
        covered_fields: set[str] = set()
        covered_statements: set[str] = set()
        for doc in docs_with_rank:
            if not (
                doc.get("required_field_anchor")
                or doc.get("statement_type_anchor")
                or doc.get("selection_scope_anchor")
            ):
                continue
            anchor_docs.append(doc)
            covered_fields.update((doc.get("matched_required_fields") or {}).keys())
            covered_statements.update(doc.get("statement_types") or [])
            if len(anchor_docs) >= min(6, RERANK_REMOTE_CANDIDATE_K):
                break
        for doc in docs_with_rank:
            if len(anchor_docs) >= min(6, RERANK_REMOTE_CANDIDATE_K):
                break
            fields = set((doc.get("matched_required_fields") or {}).keys())
            statements = set(doc.get("statement_types") or [])
            adds_field = bool(fields - covered_fields)
            adds_statement = bool(doc.get("statement_type_anchor") and statements - covered_statements)
            if doc.get("selection_scope_anchor") or adds_field or adds_statement:
                anchor_docs.append(doc)
                covered_fields.update(fields)
                covered_statements.update(statements)
        remote_docs = _deduplicate_docs(anchor_docs + docs_with_rank)[:RERANK_REMOTE_CANDIDATE_K]
        meta["remote_rerank_anchor_count"] = sum(
            1
            for doc in remote_docs
            if doc.get("required_field_anchor")
            or doc.get("statement_type_anchor")
            or doc.get("matched_required_fields")
            or doc.get("selection_scope_anchor")
        )
        remote_texts = [str(doc.get("text", "") or "")[:RERANK_REMOTE_MAX_CHARS] for doc in remote_docs]
        meta["remote_rerank_candidate_count"] = len(remote_docs)
        meta["remote_rerank_input_chars"] = sum(len(text) for text in remote_texts)
        payload = {
            "model": RERANK_MODEL,
            "query": query,
            "documents": remote_texts,
            "top_n": min(top_k, len(remote_docs)),
            "return_documents": False,
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {RERANK_API_KEY}"}
        expected_count = int(payload["top_n"])
        cache_key = _build_rerank_cache_key(query, str(RERANK_MODEL), remote_docs, expected_count)
        cached_items = cache.get_json(cache_key) if RERANK_CACHE_ENABLED else None
        if cached_items is not None:
            try:
                items = _validate_remote_rerank_results(
                    {"results": cached_items},
                    candidate_count=len(remote_docs),
                    expected_count=expected_count,
                )
                meta["rerank_cache_hit"] = True
                meta["remote_success"] = True
                meta["rerank_applied"] = True
                meta["rerank_provider"] = "remote_cache"
                _sync_rerank_trace(meta)
                return _remote_rerank_from_items(remote_docs, items, top_k), meta
            except (KeyError, ValueError, TypeError) as exc:
                meta["remote_errors"].append(f"cache_invalid:{exc}")
                cache.delete(cache_key)

        retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
        for attempt in range(1, RERANK_REMOTE_MAX_ATTEMPTS + 1):
            if attempt > 1 and RERANK_REMOTE_BACKOFF_SECONDS > 0:
                time.sleep(RERANK_REMOTE_BACKOFF_SECONDS * (2 ** (attempt - 2)))
            status_code = None
            try:
                global _last_remote_rerank_started_at
                with _rerank_request_lock:
                    remaining = RERANK_REMOTE_MIN_INTERVAL_SECONDS - (time.monotonic() - _last_remote_rerank_started_at)
                    if remaining > 0:
                        time.sleep(remaining)
                    _last_remote_rerank_started_at = time.monotonic()
                    response = requests.post(
                        meta["rerank_endpoint"],
                        headers=headers,
                        json=payload,
                        timeout=RERANK_TIMEOUT_SECONDS,
                    )
                status_code = response.status_code
                meta["remote_attempt_count"] = attempt
                if response.status_code >= 400:
                    error = f"HTTP {response.status_code}: {response.text}"
                    meta["remote_rerank_rate_limited"] = response.status_code == 429
                    raise RuntimeError(error)
                items = _validate_remote_rerank_results(
                    response.json(),
                    candidate_count=len(remote_docs),
                    expected_count=expected_count,
                )
                if RERANK_CACHE_ENABLED:
                    cache.set_json(cache_key, items, ttl=RERANK_CACHE_TTL_SECONDS)
                meta["remote_success"] = True
                meta["rerank_applied"] = True
                meta["rerank_provider"] = "remote"
                meta["rerank_error"] = None
                _sync_rerank_trace(meta)
                return _remote_rerank_from_items(remote_docs, items, top_k), meta
            except Exception as exc:
                meta["remote_attempt_count"] = attempt
                error = f"{type(exc).__name__}: {exc}"
                meta["rerank_error"] = error
                meta["remote_errors"].append(error)
                if isinstance(status_code, int) and status_code >= 400 and status_code not in retryable_statuses:
                    break
        _sync_rerank_trace(meta)

    if _local_reranker.enabled:
        try:
            reranked = _local_reranker.rerank(query, docs_with_rank, top_k)
            if reranked:
                meta["local_rerank_applied"] = True
                meta["rerank_applied"] = True
                meta["rerank_model"] = str(_local_reranker.model_path)
                meta["rerank_endpoint"] = "local_cross_encoder"
                meta["rerank_provider"] = "local_fallback" if meta["rerank_error"] else "local"
                if meta["rerank_error"]:
                    meta["rerank_fallback_used"] = True
                    meta["rerank_fallback_reason"] = str(meta["rerank_error"])
                _sync_rerank_trace(meta)
                return reranked, meta
            meta["local_rerank_error"] = "empty_rerank_results"
        except (FileNotFoundError, OSError, RuntimeError, ValueError, TypeError) as exc:
            meta["local_rerank_error"] = str(exc)
    _sync_rerank_trace(meta)
    return docs_with_rank[:top_k], meta


def _select_doc_stage_documents(candidate_docs: List[dict], doc_stage_top_n: int, query_parse: Dict[str, Any]) -> List[dict]:
    stats: Dict[str, dict] = {}
    target_years = {str(year) for year in (query_parse.get("years") or [])}
    target_doc_type = (query_parse.get("doc_types") or [query_parse.get("doc_type") or ""])[0]
    target_company = query_parse.get("company") or ""
    for doc in candidate_docs:
        filename = doc.get("filename") or ""
        if not filename:
            continue
        score = _combined_score(doc)
        filename_text = filename
        company_match = 1.0 if target_company and matches_company_text(filename_text, target_company) else 0.0
        year_match = 1.0 if target_years and (_extract_years(filename_text) & target_years) else 0.0
        doc_type_match = 1.0 if target_doc_type and infer_doc_type(filename_text) == target_doc_type else 0.0
        item = stats.setdefault(
            filename,
            {
                "filename": filename,
                "doc_name": get_doc_name(filename),
                "best_score": score,
                "hit_count": 0,
                "company_match_score": company_match,
                "year_match_score": year_match,
                "doc_type_match_score": doc_type_match,
            },
        )
        item["hit_count"] += 1
        item["best_score"] = max(item["best_score"], score)
        item["company_match_score"] = max(item["company_match_score"], company_match)
        item["year_match_score"] = max(item["year_match_score"], year_match)
        item["doc_type_match_score"] = max(item["doc_type_match_score"], doc_type_match)
        item["doc_stage_score"] = (
            item["best_score"]
            + 0.95 * item["company_match_score"]
            + 0.45 * item["year_match_score"]
            + 0.25 * item["doc_type_match_score"]
        )

    ranked = sorted(
        stats.values(),
        key=lambda item: (
            item.get("doc_stage_score", item["best_score"]),
            item["hit_count"],
            item["best_score"],
            item["filename"],
        ),
        reverse=True,
    )
    if target_company:
        company_ranked = [item for item in ranked if item.get("company_match_score", 0.0) > 0]
        if company_ranked:
            ranked = company_ranked + [item for item in ranked if item.get("company_match_score", 0.0) <= 0]
    if target_company and target_years:
        company_year_ranked = [
            item for item in ranked
            if item.get("company_match_score", 0.0) > 0 and item.get("year_match_score", 0.0) > 0
        ]
        if company_year_ranked:
            ranked = company_year_ranked + [
                item for item in ranked
                if not (item.get("company_match_score", 0.0) > 0 and item.get("year_match_score", 0.0) > 0)
            ]
    return ranked[:doc_stage_top_n]


def _retrieve_leaf_chunks(
    query: str,
    *,
    top_k: int,
    filter_expr: str,
    retrieval_scope: str,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "retrieval_scope": retrieval_scope,
        "retrieval_mode": "failed",
        "candidate_count": 0,
        "retrieve_error": None,
    }
    try:
        embedding_started = time.perf_counter()
        _retrieval_debug(f"scope={retrieval_scope} embedding starting")
        dense_embedding, sparse_embedding = _get_hybrid_query_embeddings(query)
        meta["query_embedding_ms"] = round((time.perf_counter() - embedding_started) * 1000, 2)
        _retrieval_debug(f"scope={retrieval_scope} embedding_ms={meta['query_embedding_ms']}")
        milvus_started = time.perf_counter()
        _retrieval_debug(f"scope={retrieval_scope} milvus starting")
        retrieved = _milvus_manager.hybrid_retrieve(
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=top_k,
            filter_expr=filter_expr,
            query_text=query,
        )
        meta["milvus_retrieval_ms"] = round((time.perf_counter() - milvus_started) * 1000, 2)
        meta["retrieval_mode"] = "hybrid"
        meta["candidate_count"] = len(retrieved)
        return {"docs": retrieved, "meta": meta}
    except Exception as exc:
        meta["retrieve_error"] = f"hybrid:{exc}"
        try:
            dense_embeddings = _embedding_service.get_embeddings([query])
            dense_embedding = dense_embeddings[0]
            retrieved = _milvus_manager.dense_retrieve(
                dense_embedding=dense_embedding,
                top_k=top_k,
                filter_expr=filter_expr,
            )
            meta["retrieval_mode"] = "dense_fallback"
            meta["candidate_count"] = len(retrieved)
            return {"docs": retrieved, "meta": meta}
        except Exception as dense_exc:
            meta["retrieve_error"] = f"{meta['retrieve_error']}; dense:{dense_exc}"
            return {"docs": [], "meta": meta}


def _planner_query_limit(candidate_k: int) -> int:
    if candidate_k <= 4:
        return candidate_k
    return max(3, min(candidate_k, candidate_k // 2))


def _is_query_planner_enabled() -> bool:
    return _parse_bool(os.getenv("RAG_QUERY_PLANNER_ENABLED"), False)


def _is_anchor_guard_enabled() -> bool:
    """Anchor filtering is an experimental precision heuristic, not a baseline."""
    return _parse_bool(os.getenv("RAG_ANCHOR_GUARD_ENABLED"), False)


def _is_cover_filter_enabled() -> bool:
    """Cover/TOC filtering is a precision experiment and can remove valid filings."""
    return _parse_bool(os.getenv("RAG_COVER_FILTER_ENABLED"), False)


def _is_page_first_enabled() -> bool:
    """Finance baseline: discover pages before choosing their evidence chunks."""
    return _parse_bool(os.getenv("RAG_PAGE_FIRST_ENABLED"), True)


def _is_field_aware_enabled() -> bool:
    """Enable the controlled anchor-window and evidence-coverage experiment."""
    return _parse_bool(os.getenv("RAG_FIELD_AWARE_ENABLED"), False)


def _is_page_level_fusion_enabled() -> bool:
    return _parse_bool(os.getenv("RAG_PAGE_LEVEL_FUSION"), True)


def _build_retrieval_routes(
    original_query: str,
    candidate_k: int,
    planner: Dict[str, Any],
    field_aware_query: str = "",
) -> list[dict]:
    routes: list[dict] = [
        {
            "label": "original",
            "query": original_query,
            "weight": 1.0,
            "top_k": candidate_k,
            "category": "original",
        }
    ]
    seen = {original_query.strip().lower()}
    def _append(items: list[str], label_prefix: str, weight: float, category: str) -> None:
        for index, item in enumerate(items or [], 1):
            query = (item or "").strip()
            lowered = query.lower()
            if not query or lowered in seen:
                continue
            seen.add(lowered)
            routes.append(
                {
                    "label": f"{label_prefix}_{index}",
                    "query": query,
                    "weight": weight,
                    "top_k": candidate_k,
                    "category": category,
                }
            )

    _append([field_aware_query], "finance_rewrite", 0.90, "finance_rewrite")
    return routes


def _rrf_fuse_retrieval_routes(
    route_results: list[dict],
    *,
    rrf_k: int = 60,
) -> list[dict]:
    fused: dict[tuple, dict] = {}
    for route in route_results:
        docs = route.get("docs") or []
        weight = float(route.get("weight", 1.0) or 1.0)
        label = route.get("label") or ""
        query = route.get("query") or ""
        for rank, doc in enumerate(docs, 1):
            key = _doc_key(doc)
            entry = fused.setdefault(
                key,
                {
                    **doc,
                    "score": 0.0,
                    "planner_sources": [],
                    "planner_queries": [],
                },
            )
            entry["score"] = float(entry.get("score", 0.0) or 0.0) + (weight / (rrf_k + rank))
            if label and label not in entry["planner_sources"]:
                entry["planner_sources"].append(label)
            if query and query not in entry["planner_queries"]:
                entry["planner_queries"].append(query)
            if _combined_score(doc) > _combined_score(entry):
                for field, value in doc.items():
                    if field in {"score", "planner_sources", "planner_queries"}:
                        continue
                    entry[field] = value
    ordered = sorted(fused.values(), key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    for rank, doc in enumerate(ordered, 1):
        doc["rrf_rank"] = rank
    return ordered


def _page_fusion_key(doc: dict) -> tuple[str, int] | None:
    filename = (doc.get("filename") or "").strip()
    page_number = _coerce_int(doc.get("page_number"))
    if not filename or page_number is None:
        return None
    return filename, page_number


def _fuse_retrieval_pages(
    route_results: list[dict],
    *,
    rrf_k: int = 60,
) -> list[dict]:
    pages: dict[tuple[str, int], dict] = {}
    for route in route_results:
        docs = route.get("docs") or []
        weight = float(route.get("weight", 1.0) or 1.0)
        label = route.get("label") or ""
        query = route.get("query") or ""
        for rank, doc in enumerate(docs, 1):
            key = _page_fusion_key(doc)
            if key is None:
                continue
            filename, page_number = key
            contribution = weight / (rrf_k + rank)
            entry = pages.setdefault(
                key,
                {
                    "filename": filename,
                    "page_number": page_number,
                    "page_score": 0.0,
                    "contributing_routes": [],
                    "matched_queries": [],
                    "top_chunks": [],
                    "page_text": [],
                },
            )
            entry["page_score"] += contribution
            if label and label not in entry["contributing_routes"]:
                entry["contributing_routes"].append(label)
            if query and query not in entry["matched_queries"]:
                entry["matched_queries"].append(query)
            if doc.get("text"):
                entry["page_text"].append(doc.get("text", "") or "")
            entry["top_chunks"].append(
                {
                    "chunk_id": doc.get("chunk_id", "") or "",
                    "text": doc.get("text", "") or "",
                    "score": _combined_score(doc),
                    "route": label,
                    "query": query,
                }
            )
    ordered = sorted(pages.values(), key=lambda item: float(item.get("page_score", 0.0) or 0.0), reverse=True)
    for entry in ordered:
        entry["top_chunks"] = sorted(
            entry.get("top_chunks", []),
            key=lambda item: float(item.get("score", 0.0) or 0.0),
            reverse=True,
        )[:3]
        entry["page_text"] = "\n".join(entry.get("page_text", [])[:5])
    return ordered


def _fused_page_matches_query_anchors(page_entry: dict, query_anchors: List[str]) -> bool:
    if not query_anchors:
        return True
    haystack_parts = [
        page_entry.get("page_text", "") or "",
    ]
    for chunk in page_entry.get("top_chunks", []) or []:
        haystack_parts.append(chunk.get("text", "") or "")
    haystack = "\n".join(haystack_parts)
    return any(_anchor_in_text(anchor, haystack) for anchor in query_anchors)


def _apply_anchor_guard_to_fused_pages(
    fused_pages: list[dict],
    route_results: list[dict],
    query_anchors: List[str],
) -> tuple[list[dict], dict]:
    if not query_anchors:
        return fused_pages, {
            "page_anchor_filtered_count": 0,
            "fused_pages_after_anchor_guard": fused_pages,
            "page_anchor_guard_fallback_reason": "",
        }
    all_docs = []
    for route in route_results:
        all_docs.extend(route.get("docs") or [])
    anchor_filename_hits = _build_anchor_matched_filename_stats(all_docs, query_anchors)
    filtered: list[dict] = []
    filtered_count = 0
    for page in fused_pages:
        filename = (page.get("filename") or "").strip()
        if _fused_page_matches_query_anchors(page, query_anchors) or anchor_filename_hits.get(filename, 0) >= 2:
            filtered.append(page)
        else:
            filtered_count += 1
    if filtered:
        return filtered, {
            "page_anchor_filtered_count": filtered_count,
            "fused_pages_after_anchor_guard": filtered,
            "page_anchor_guard_fallback_reason": "",
        }
    return fused_pages, {
        "page_anchor_filtered_count": filtered_count,
        "fused_pages_after_anchor_guard": fused_pages,
        "page_anchor_guard_fallback_reason": "anchor_guard_empty_fallback",
    }


def _apply_page_level_fusion(
    fused_docs: list[dict],
    route_results: list[dict],
    *,
    candidate_k: int,
) -> tuple[list[dict], list[dict], dict]:
    fused_pages = _fuse_retrieval_pages(route_results)
    if not fused_pages:
        return fused_docs[:candidate_k], [], {
            "page_anchor_filtered_count": 0,
            "fused_pages_after_anchor_guard": [],
            "page_anchor_guard_fallback_reason": "",
        }

    query_anchors = []
    for route in route_results:
        if route.get("category") == "original":
            query_anchors = extract_query_anchors(route.get("query") or "")
            break
    guarded_pages, guard_meta = _apply_anchor_guard_to_fused_pages(
        fused_pages,
        route_results,
        query_anchors,
    )

    page_rank_map = {
        (entry["filename"], entry["page_number"]): index
        for index, entry in enumerate(guarded_pages, 1)
    }
    page_score_map = {
        (entry["filename"], entry["page_number"]): float(entry.get("page_score", 0.0) or 0.0)
        for entry in guarded_pages
    }
    page_meta_map = {
        (entry["filename"], entry["page_number"]): entry
        for entry in guarded_pages
    }

    prioritized_docs: list[dict] = []
    for doc in fused_docs:
        page_key = _page_fusion_key(doc)
        if page_key is None:
            prioritized_docs.append(doc)
            continue
        page_meta = page_meta_map.get(page_key, {})
        prioritized_docs.append(
            {
                **doc,
                "page_fused_score": page_score_map.get(page_key, 0.0),
                "page_fused_rank": page_rank_map.get(page_key, 10**9),
                "page_contributing_routes": list(page_meta.get("contributing_routes", [])),
                "page_matched_queries": list(page_meta.get("matched_queries", [])),
            }
        )

    prioritized_docs.sort(
        key=lambda item: (
            int(item.get("page_fused_rank", 10**9)),
            -float(item.get("page_fused_score", 0.0) or 0.0),
            -float(item.get("score", 0.0) or 0.0),
            -float(item.get("rerank_score", 0.0) or 0.0),
        )
    )
    return prioritized_docs[:candidate_k], guarded_pages, guard_meta


def _select_docs_from_fused_pages(
    docs: List[dict],
    *,
    final_top_k: int,
    max_per_page: int = 2,
) -> tuple[List[dict], List[dict], bool]:
    page_docs = [doc for doc in docs or [] if doc.get("page_fused_rank") is not None]
    if not page_docs:
        return [], [], False

    ordered_docs = sorted(
        page_docs,
        key=lambda item: (
            int(item.get("page_fused_rank", 10**9)),
            -float(item.get("page_fused_score", 0.0) or 0.0),
            -_combined_score(item),
        ),
    )
    selected: list[dict] = []
    page_counts: dict[tuple[str, int], int] = {}
    selected_pages: list[dict] = []
    seen_pages: set[tuple[str, int]] = set()
    for doc in ordered_docs:
        page_key = _page_fusion_key(doc)
        if page_key is None:
            continue
        if page_counts.get(page_key, 0) >= max_per_page:
            continue
        selected.append(doc)
        page_counts[page_key] = page_counts.get(page_key, 0) + 1
        if page_key not in seen_pages:
            seen_pages.add(page_key)
            selected_pages.append(
                {
                    "filename": page_key[0],
                    "page_number": page_key[1],
                    "page_score": float(doc.get("page_fused_score", 0.0) or 0.0),
                    "contributing_routes": doc.get("page_contributing_routes", []) or [],
                    "matched_queries": doc.get("page_matched_queries", []) or [],
                }
            )
        if len(selected) >= final_top_k:
            break
    return selected, selected_pages, bool(selected_pages)


def _should_trigger_two_stage_fallback(page_stage_candidates: List[dict], final_top_k: int, query: str) -> bool:
    if len(page_stage_candidates) < final_top_k:
        return True
    inspection_window = page_stage_candidates[: min(len(page_stage_candidates), max(3, final_top_k // 2))]
    if not inspection_window:
        return True

    page_zero_count = _page_zero_count(inspection_window)
    cover_like_count = sum(1 for doc in inspection_window if _looks_like_cover_or_toc_chunk(doc, query))
    return page_zero_count >= max(2, (len(inspection_window) + 1) // 2) or cover_like_count >= max(
        1,
        len(inspection_window) // 2,
    )


def _fetch_document_pages(selected_docs: List[dict]) -> List[dict]:
    return _document_page_store.get_pages_by_filenames([doc.get("filename") or "" for doc in selected_docs])


def _score_candidate_pages(
    query: str,
    page_records: List[dict],
    candidate_docs: List[dict],
    query_parse: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[List[dict], Dict[str, Any]]:
    if not page_records:
        return [], {
            "page_embedding_cache_hit_count": 0,
            "page_embedding_cache_miss_count": 0,
            "query_time_page_embedding_executed": False,
        }

    initial_page_score_map: Dict[tuple, float] = {}
    for doc in candidate_docs:
        key = (doc.get("filename") or "", _coerce_int(doc.get("page_number")) or 0)
        initial_page_score_map[key] = max(initial_page_score_map.get(key, 0.0), _combined_score(doc))

    query_embedding = _embedding_service.get_embeddings([query])[0]
    query_numbers = _extract_numbers(query)
    query_years = _extract_years(query)
    query_metrics = _extract_metric_hints(query)
    query_tokens = _extract_keyword_tokens(query) | _expand_finance_operand_tokens(query)
    target_company = query_parse.get("company") or ""
    target_years = set(str(year) for year in (query_parse.get("years") or []))
    target_doc_types = set(query_parse.get("doc_types") or ([] if not query_parse.get("doc_type") else [query_parse.get("doc_type")]))
    target_statement_types = set(query_parse.get("statement_types") or [])
    required_fields = list(query_parse.get("required_fields") or [])
    required_periods = [str(item).lower() for item in (query_parse.get("required_periods") or [])]

    page_details: List[dict] = []
    for page in page_records:
        page_text = page.get("page_text", "") or ""
        table_text = page.get("table_text", "") or ""
        dense_embedding = page.get("page_dense_embedding") or []
        combined_text = "\n".join([page_text, table_text]).strip()
        table_tokens = _extract_keyword_tokens(table_text)
        table_numbers = _extract_numbers(table_text)
        table_years = _extract_years(table_text)
        table_metrics = _extract_metric_hints(table_text)
        page_tokens = set(page.get("page_tokens") or _extract_keyword_tokens(combined_text))
        page_numbers = set(page.get("page_numbers") or _extract_numbers(combined_text))
        page_years = set(page.get("page_years") or _extract_years(combined_text))
        page_metrics = set(page.get("page_metric_tokens") or _extract_metric_hints(combined_text))
        company_match_score = _company_match_score(query_parse, page.get("filename", ""), page_text, table_text)
        year_match_score = _year_match_score(query_parse, page.get("filename", ""), page_years, page_text)
        doc_type_match_score = _doc_type_match_score(query_parse, page.get("filename", ""), page_text)
        page_statement_types = infer_page_statement_types(combined_text)
        statement_type_match_score = (
            len(target_statement_types & set(page_statement_types)) / len(target_statement_types)
            if target_statement_types
            else 0.0
        )
        matched_required_fields = match_required_fields_in_text(required_fields, combined_text)
        required_field_match_score = (
            len(matched_required_fields) / len(required_fields)
            if required_fields
            else 0.0
        )
        period_scope_text = "\n".join([page.get("filename", "") or "", combined_text]).lower()
        matched_required_periods = [period for period in required_periods if period in period_scope_text]
        required_period_match_score = (
            len(matched_required_periods) / len(required_periods)
            if required_periods
            else 0.0
        )
        lowered_combined_text = combined_text.lower()
        selection_scope_score = 0.0
        if query_parse.get("task_type") == "selection":
            has_scope_heading = any(
                marker in lowered_combined_text
                for marker in ("segment results", "results by segment", "business segments", "geographic regions")
            )
            metric_occurrences = max(
                (lowered_combined_text.count(alias) for field in required_fields for alias in FIELD_ALIASES.get(field, [])),
                default=0,
            )
            if has_scope_heading and metric_occurrences >= 2:
                selection_scope_score = 1.0
        token_overlap = _score_overlap(query_tokens, page_tokens)
        metric_overlap = _score_overlap(query_metrics, page_metrics)
        number_overlap = max(_score_overlap(query_numbers, page_numbers), _score_overlap(query_years, page_years))
        table_metric_overlap = _score_overlap(query_metrics, table_metrics)
        table_number_overlap = max(_score_overlap(query_numbers, table_numbers), _score_overlap(query_years, table_years))
        table_keyword_overlap = _score_overlap(query_tokens, table_tokens)
        table_signal_score = 0.0
        if table_text.strip():
            table_signal_score = max(
                table_metric_overlap,
                table_number_overlap,
                table_keyword_overlap * 0.5,
            )
        initial_chunk_hit_score = initial_page_score_map.get(
            ((page.get("filename") or ""), _coerce_int(page.get("page_number")) or 0),
            0.0,
        )
        dense_raw = max(0.0, _dot_product(query_embedding, dense_embedding)) if dense_embedding else 0.0
        keyword_raw = token_overlap + metric_overlap + number_overlap
        cover_penalty = config["cover_toc_penalty"] if _is_cover_filter_enabled() and _looks_like_cover_or_toc_chunk(
            {"page_number": page.get("page_number"), "text": page_text},
            query,
        ) else 0.0
        penalty_reasons: List[str] = []
        filter_reason = ""
        if cover_penalty > 0:
            penalty_reasons.append("cover_or_toc")
        if target_company and company_match_score <= 0:
            penalty_reasons.append("company_mismatch")
        if target_years and year_match_score <= 0:
            penalty_reasons.append("year_mismatch")
        if target_doc_types and doc_type_match_score <= 0:
            penalty_reasons.append("doc_type_mismatch")
        if not dense_embedding:
            penalty_reasons.append("missing_page_dense_embedding")
        if len(page_text.strip()) < 40:
            penalty_reasons.append("short_text")

        page_details.append(
            {
                **page,
                "dense_raw_score": dense_raw,
                "keyword_raw_score": keyword_raw,
                "number_overlap_score": number_overlap,
                "metric_overlap_score": metric_overlap,
                "company_match_score": company_match_score,
                "year_match_score": year_match_score,
                "doc_type_match_score": doc_type_match_score,
                "statement_types": page_statement_types,
                "statement_type_match_score": statement_type_match_score,
                "matched_required_fields": matched_required_fields,
                "required_field_match_score": required_field_match_score,
                "matched_required_periods": matched_required_periods,
                "required_period_match_score": required_period_match_score,
                "selection_scope_score": selection_scope_score,
                "table_signal_score": table_signal_score,
                "has_table_text": bool(table_text.strip()),
                "initial_chunk_hit_score": initial_chunk_hit_score,
                "cover_toc_penalty": cover_penalty,
                "cover_like": cover_penalty > 0,
                "filter_reason": filter_reason,
                "penalty_reason": ", ".join(penalty_reasons),
                "has_dense_embedding": bool(dense_embedding),
            }
        )

    _normalize_component_scores(page_details, "dense_raw_score", "dense_score")
    _normalize_component_scores(page_details, "keyword_raw_score", "keyword_score")
    _normalize_component_scores(page_details, "initial_chunk_hit_score", "initial_chunk_hit_score_norm")

    scored_pages: List[dict] = []
    for page in page_details:
        final_score = (
            config["w_dense"] * page["dense_score"]
            + config["w_keyword"] * page["keyword_score"]
            + config["w_metric"] * page["metric_overlap_score"]
            + config["w_number"] * page["number_overlap_score"]
            + config["w_company"] * page["company_match_score"]
            + config["w_year"] * page["year_match_score"]
            + config["w_doc_type"] * page["doc_type_match_score"]
            + config["w_statement"] * page["statement_type_match_score"]
            + config["w_required_fields"] * page["required_field_match_score"]
            + config["w_required_periods"] * page["required_period_match_score"]
            + config["w_selection_scope"] * page["selection_scope_score"]
            + 0.08 * page["initial_chunk_hit_score_norm"]
            + 0.12 * page["table_signal_score"]
            - page["cover_toc_penalty"]
        )
        if target_company and page["company_match_score"] <= 0:
            final_score -= 0.85
        if target_years and page["year_match_score"] <= 0:
            final_score -= 0.45
        if target_doc_types and page["doc_type_match_score"] <= 0:
            final_score -= 0.18
        if not page["has_dense_embedding"]:
            final_score -= 0.35
            page["filter_reason"] = "missing_page_dense_embedding"
        if page["cover_like"]:
            page["filter_reason"] = "cover_or_toc" if not page["filter_reason"] else page["filter_reason"]

        scored_pages.append(
            {
                **page,
                "dense_score": round(page["dense_score"], 6),
                "keyword_score": round(page["keyword_score"], 6),
                "metric_overlap_score": round(page["metric_overlap_score"], 6),
                "number_overlap_score": round(page["number_overlap_score"], 6),
                "company_match_score": round(page["company_match_score"], 6),
                "year_match_score": round(page["year_match_score"], 6),
                "doc_type_match_score": round(page["doc_type_match_score"], 6),
                "statement_type_match_score": round(page["statement_type_match_score"], 6),
                "required_field_match_score": round(page["required_field_match_score"], 6),
                "required_period_match_score": round(page["required_period_match_score"], 6),
                "selection_scope_score": round(page["selection_scope_score"], 6),
                "table_signal_score": round(page["table_signal_score"], 6),
                "has_table_text": page["has_table_text"],
                "initial_chunk_hit_score": round(page["initial_chunk_hit_score_norm"], 6),
                "final_score": round(final_score, 6),
                "page_score": round(final_score, 6),
                "doc_name": page.get("doc_name") or get_doc_name(page.get("filename", "")),
                "filter_reason": page.get("filter_reason", ""),
                "penalty_reason": page.get("penalty_reason", ""),
            }
        )

    scored_pages.sort(key=lambda item: item.get("page_score", 0.0), reverse=True)
    hit_count = sum(1 for page in page_records if page.get("page_dense_embedding"))
    miss_count = max(0, len(page_records) - hit_count)
    return scored_pages, {
        "page_embedding_cache_hit_count": hit_count,
        "page_embedding_cache_miss_count": miss_count,
        "query_time_page_embedding_executed": False,
    }


def _build_evidence_pack(
    selected_pages: List[dict],
    all_page_records: List[dict],
    *,
    query_parse: Dict[str, Any],
    config: Dict[str, Any],
    final_top_k: int,
    enable_page_merge: bool,
    adjacent_page_window: int,
) -> Tuple[List[dict], Dict[str, Any]]:
    if not selected_pages:
        return [], {
            "page_merge_applied": False,
            "merged_chunk_count": 0,
            "final_context_chunk_count": 0,
            "final_evidence_pack_debug": [],
            "final_evidence_pack_used": [],
            "final_evidence_pack_debug_count": 0,
            "final_evidence_pack_used_count": 0,
            "dropped_evidence_count": 0,
            "dropped_reasons": {},
            "prompt_context_char_count_estimate": 0,
            "fallback_reason": "",
        }

    pages_by_key = {
        ((page.get("filename") or ""), _coerce_int(page.get("page_number")) or 0): page
        for page in all_page_records
    }
    evidence_pack_debug: List[dict] = []
    seen = set()

    def _append_entry(entry: dict) -> None:
        key = (
            entry.get("type") or "",
            entry.get("filename") or "",
            entry.get("page_number"),
            entry.get("chunk_id") or "",
            entry.get("text") or "",
        )
        if key in seen or not (entry.get("text") or "").strip():
            return
        seen.add(key)
        evidence_pack_debug.append(entry)

    max_context_entries = max(final_top_k * 3, 24)
    for page in selected_pages:
        filename = page.get("filename") or ""
        doc_name = page.get("doc_name") or get_doc_name(filename)
        page_number = _coerce_int(page.get("page_number")) or 0
        page_score = float(page.get("page_score", 0.0) or 0.0)

        _append_entry(
            {
                "filename": filename,
                "doc_name": doc_name,
                "page_number": page_number,
                "type": "page_text",
                "text": _safe_preview(page.get("page_text", "") or "", config["max_page_text_chars"]),
                "score": page_score,
                "rerank_score": page_score,
                "page_score": page_score,
                "source": f"{filename}#page={page_number}#page_text",
            }
        )

        table_text = (page.get("table_text") or "").strip()
        if table_text:
            _append_entry(
                {
                    "filename": filename,
                    "doc_name": doc_name,
                    "page_number": page_number,
                    "type": "table_text",
                    "text": _safe_preview(table_text, config["max_table_text_chars"]),
                    "score": page_score,
                    "rerank_score": page_score,
                    "page_score": page_score,
                    "source": f"{filename}#page={page_number}#table_text",
                }
            )

        page_window = adjacent_page_window if enable_page_merge else 0
        for target_page in range(page_number - page_window, page_number + page_window + 1):
            if target_page < 0:
                continue
            page_record = pages_by_key.get((filename, target_page))
            if not page_record:
                continue
            chunk_ids = [item for item in page_record.get("chunk_ids", []) if item]
            for chunk in _milvus_manager.get_chunks_by_ids(chunk_ids):
                _append_entry(
                    {
                        **chunk,
                        "doc_name": doc_name,
                        "page_number": _coerce_int(chunk.get("page_number")) or target_page,
                        "type": "chunk",
                        "text": _safe_preview(chunk.get("text", "") or "", 1200),
                        "score": page_score,
                        "rerank_score": page_score,
                        "page_score": page_score,
                        "source": f"{filename}#page={target_page}#chunk",
                    }
                )
                if len(evidence_pack_debug) >= max_context_entries:
                    break
            if len(evidence_pack_debug) >= max_context_entries:
                break
        if len(evidence_pack_debug) >= max_context_entries:
            break

    company_filtered = [
        item for item in evidence_pack_debug
        if not query_parse.get("company") or matches_company_text(
            "\n".join([item.get("filename", ""), item.get("doc_name", ""), item.get("text", "")]),
            query_parse.get("company") or "",
        )
    ]
    fallback_reason = ""
    working = company_filtered
    if not working:
        working = evidence_pack_debug
        if query_parse.get("company"):
            fallback_reason = "no_company_matched_evidence"

    seen_page_type = set()
    used_pack: List[dict] = []
    dropped_reasons: Dict[str, int] = defaultdict(int)
    target_years = set(str(year) for year in (query_parse.get("years") or []))
    target_doc_types = set(query_parse.get("doc_types") or [])

    def _entry_priority(entry: dict) -> tuple:
        filename = entry.get("filename", "") or ""
        text = entry.get("text", "") or ""
        entry_years = _extract_years("\n".join([filename, text]))
        year_match = 1 if (not target_years or bool(target_years & entry_years)) else 0
        company_match = 1 if (not query_parse.get("company") or matches_company_text("\n".join([filename, text]), query_parse.get("company") or "")) else 0
        doc_type_match = 1 if (not target_doc_types or infer_doc_type("\n".join([filename, text])) in target_doc_types) else 0
        table_bonus = 0
        if entry.get("type") == "table_text":
            table_bonus = 1
            table_bonus += len(_extract_metric_hints(text) & set(query_parse.get("metrics") or []))
            table_bonus += len(_extract_numbers(text) & set(query_parse.get("numbers") or []))
        metric_bonus = len(_extract_metric_hints(text) & set(query_parse.get("metrics") or []))
        number_bonus = len(_extract_numbers(text) & set(query_parse.get("numbers") or []))
        return (
            company_match,
            year_match,
            doc_type_match,
            metric_bonus,
            number_bonus,
            table_bonus,
            float(entry.get("page_score", entry.get("score", 0.0)) or 0.0),
        )

    for entry in sorted(working, key=_entry_priority, reverse=True):
        filename = entry.get("filename") or ""
        page_number = _coerce_int(entry.get("page_number")) or 0
        page_type_key = (filename, page_number, entry.get("type") or "")
        if page_type_key in seen_page_type:
            dropped_reasons["duplicate_page_type"] += 1
            continue
        if _looks_like_cover_or_toc_chunk({"page_number": page_number, "text": entry.get("text", "")}, query_parse.get("raw_question", "")):
            dropped_reasons["cover_or_toc"] += 1
            continue
        if query_parse.get("years"):
            entry_years = _extract_years("\n".join([filename, entry.get("text", "")]))
            if entry_years and not (set(query_parse.get("years") or []) & entry_years):
                dropped_reasons["year_mismatch"] += 1
                continue
        if len((entry.get("text") or "").strip()) < 30:
            dropped_reasons["too_short"] += 1
            continue
        seen_page_type.add(page_type_key)
        used_pack.append(entry)
        if len(used_pack) >= config["max_evidence_pack_used"]:
            break

    if len(used_pack) < config["min_evidence_pack_used"]:
        for entry in sorted(evidence_pack_debug, key=_entry_priority, reverse=True):
            filename = entry.get("filename") or ""
            page_number = _coerce_int(entry.get("page_number")) or 0
            page_type_key = (filename, page_number, entry.get("type") or "")
            if page_type_key in seen_page_type:
                continue
            seen_page_type.add(page_type_key)
            used_pack.append(entry)
            if len(used_pack) >= config["min_evidence_pack_used"]:
                break
        if len(used_pack) < config["min_evidence_pack_used"]:
            fallback_reason = fallback_reason or "insufficient_same_company_year_metric_evidence"

    prompt_chars = _estimate_prompt_chars(used_pack)
    merged_chunk_count = max(0, len(evidence_pack_debug[:max_context_entries]) - min(len(selected_pages), final_top_k))
    return used_pack, {
        "page_merge_applied": merged_chunk_count > 0,
        "merged_chunk_count": merged_chunk_count,
        "final_context_chunk_count": len(used_pack),
        "final_evidence_pack_debug": evidence_pack_debug[:max_context_entries],
        "final_evidence_pack_used": used_pack,
        "final_evidence_pack_debug_count": len(evidence_pack_debug[:max_context_entries]),
        "final_evidence_pack_used_count": len(used_pack),
        "dropped_evidence_count": max(0, len(evidence_pack_debug[:max_context_entries]) - len(used_pack)),
        "dropped_reasons": dict(dropped_reasons),
        "prompt_context_char_count_estimate": prompt_chars,
        "fallback_reason": fallback_reason,
    }


def _as_evidence_chunk(doc: dict, default_score: float | None = None) -> dict:
    filename = doc.get("filename", "") or ""
    page_number = _coerce_int(doc.get("page_number"))
    score = _combined_score(doc) if default_score is None else default_score
    return {
        **doc,
        "filename": filename,
        "doc_name": doc.get("doc_name") or get_doc_name(filename),
        "page_number": page_number if page_number is not None else doc.get("page_number", ""),
        "type": doc.get("type") or "chunk",
        "score": score,
        "rerank_score": doc.get("rerank_score", score),
        "source": doc.get("source") or f"{filename}#page={page_number if page_number is not None else 'unknown'}#chunk",
    }


def _run_two_stage_retrieval(
    query: str,
    candidate_docs: List[dict],
    *,
    query_parse: Dict[str, Any],
    config: Dict[str, Any],
    doc_stage_top_n: int,
    page_stage_top_n: int,
    final_top_k: int,
) -> Tuple[List[dict], Dict[str, Any]]:
    selected_docs = _select_doc_stage_documents(candidate_docs, doc_stage_top_n, query_parse)
    page_records = _fetch_document_pages(selected_docs)
    scored_pages, page_perf_meta = _score_candidate_pages(query, page_records, candidate_docs, query_parse, config)
    filtered_pages = [page for page in scored_pages if not page.get("cover_like") and page.get("filter_reason") != "missing_page_dense_embedding"]
    company_confident = float(query_parse.get("company_confidence") or 0.0) >= 0.8
    company_matched_pages = [page for page in filtered_pages if page.get("company_match_score", 0.0) > 0]
    if company_confident and company_matched_pages:
        filtered_pages = company_matched_pages
    year_matched_pages = [page for page in filtered_pages if page.get("year_match_score", 0.0) > 0]
    if company_confident and year_matched_pages:
        filtered_pages = year_matched_pages
    cover_page_filtered_count = max(0, len(scored_pages) - len(filtered_pages))
    selected_pages = (filtered_pages or scored_pages)[:page_stage_top_n]

    page_chunk_candidates: List[dict] = []
    for page in selected_pages:
        chunk_ids = [item for item in page.get("chunk_ids", []) if item]
        for chunk in _milvus_manager.get_chunks_by_ids(chunk_ids):
            page_chunk_candidates.append(
                {
                    **chunk,
                    "doc_name": page.get("doc_name") or get_doc_name(page.get("filename", "")),
                    "type": "chunk",
                    "score": page.get("page_score", 0.0),
                    "rerank_score": page.get("page_score", 0.0),
                }
            )

    page_stage_candidates = _deduplicate_docs(page_chunk_candidates)
    fallback_used = _should_trigger_two_stage_fallback(page_stage_candidates, final_top_k, query)
    logger.info(
        (
            "finance_rag_two_stage selected_docs=%s page_records=%s selected_pages=%s "
            "page_stage_candidates=%s cover_page_filtered_count=%s fallback_used=%s"
        ),
        [item.get("filename") for item in selected_docs],
        len(page_records),
        [(item.get("filename"), item.get("page_number")) for item in selected_pages],
        len(page_stage_candidates),
        cover_page_filtered_count,
        fallback_used,
    )
    meta = {
        "doc_stage_selected_docs": selected_docs,
        "selected_docs": selected_docs,
        "selected_pages": [
            {
                "filename": page.get("filename"),
                "doc_name": page.get("doc_name"),
                "page_number": page.get("page_number"),
                "page_score": page.get("page_score"),
                "company_match_score": page.get("company_match_score"),
                "year_match_score": page.get("year_match_score"),
                "doc_type_match_score": page.get("doc_type_match_score"),
                "statement_types": page.get("statement_types", []),
                "statement_type_match_score": page.get("statement_type_match_score"),
                "matched_required_fields": page.get("matched_required_fields", {}),
                "required_field_match_score": page.get("required_field_match_score"),
                "matched_required_periods": page.get("matched_required_periods", []),
                "required_period_match_score": page.get("required_period_match_score"),
                "filter_reason": page.get("filter_reason"),
                "penalty_reason": page.get("penalty_reason"),
            }
            for page in selected_pages
        ],
        "selected_page_records": selected_pages,
        "page_scores": [
            {
                "filename": page.get("filename"),
                "doc_name": page.get("doc_name"),
                "page_number": page.get("page_number"),
                "page_score": page.get("page_score"),
                "dense_score": page.get("dense_score"),
                "keyword_score": page.get("keyword_score"),
                "metric_overlap_score": page.get("metric_overlap_score"),
                "number_overlap_score": page.get("number_overlap_score"),
                "company_match_score": page.get("company_match_score"),
                "year_match_score": page.get("year_match_score"),
                "doc_type_match_score": page.get("doc_type_match_score"),
                "statement_types": page.get("statement_types", []),
                "statement_type_match_score": page.get("statement_type_match_score"),
                "matched_required_fields": page.get("matched_required_fields", {}),
                "required_field_match_score": page.get("required_field_match_score"),
                "matched_required_periods": page.get("matched_required_periods", []),
                "required_period_match_score": page.get("required_period_match_score"),
                "table_signal_score": page.get("table_signal_score"),
                "has_table_text": page.get("has_table_text"),
                "initial_chunk_hit_score": page.get("initial_chunk_hit_score"),
                "cover_toc_penalty": page.get("cover_toc_penalty"),
                "final_score": page.get("final_score"),
                "filter_reason": page.get("filter_reason"),
                "penalty_reason": page.get("penalty_reason"),
            }
            for page in scored_pages[: max(page_stage_top_n * 2, final_top_k)]
        ],
        "cover_page_filtered_count": cover_page_filtered_count,
        "page_stage_candidate_count": len(page_stage_candidates),
        "page_stage_candidates": page_stage_candidates,
        "fallback_used": fallback_used,
        "all_page_records": page_records,
        **page_perf_meta,
    }
    return page_stage_candidates, meta


def _build_page_first_candidates(
    query: str,
    chunk_candidates: List[dict],
    *,
    candidate_k: int,
    config: Dict[str, Any],
) -> tuple[List[dict], Dict[str, Any]]:
    """Promote pages first, then return the best chunk and its direct neighbors.

    The initial hybrid search is retained only for inexpensive document
    discovery.  This prevents a frequently mentioned but wrong chunk from
    crowding out every other page before page-level financial signals are used.
    """
    if not chunk_candidates:
        return [], {"page_first_enabled": True, "page_first_fallback": "no_chunk_candidates"}
    query_parse = parse_finance_query(query)
    query_parse.update(infer_task_spec(query))
    query_parse["statement_types"] = infer_statement_types(query, query_parse.get("required_fields") or [])
    query_parse["raw_question"] = query
    selected_docs = _select_doc_stage_documents(
        chunk_candidates,
        doc_stage_top_n=config["doc_stage_top_n"],
        query_parse=query_parse,
    )
    pages = _document_page_store.get_pages_by_filenames(
        [item.get("filename") or "" for item in selected_docs],
        warm_cache=False,
    )
    scored_pages, page_meta = _score_candidate_pages(query, pages, chunk_candidates, query_parse, config)
    if _is_cover_filter_enabled():
        scored_pages = [page for page in scored_pages if not page.get("cover_like")]
    page_limit = max(config["page_stage_top_n"], candidate_k // 4)
    priority_pages: list[dict] = []
    for field in query_parse.get("required_fields") or []:
        preferred_statements = set(FIELD_STATEMENT_TYPES.get(field, []))
        aliases = FIELD_ALIASES.get(field, [])
        matches = [page for page in scored_pages if field in (page.get("matched_required_fields") or {})]

        def _field_anchor_score(page: dict) -> tuple[int, int, int, float]:
            matched_alias = (page.get("matched_required_fields") or {}).get(field, "")
            try:
                alias_quality = len(aliases) - aliases.index(matched_alias)
            except ValueError:
                alias_quality = 0
            statement_quality = int(
                bool(preferred_statements & set(page.get("statement_types") or []))
            )
            company_quality = int(
                not query_parse.get("company") or float(page.get("company_match_score", 0.0) or 0.0) > 0
            )
            return company_quality, alias_quality, statement_quality, float(page.get("page_score", 0.0) or 0.0)

        match = max(matches, key=_field_anchor_score) if matches else None
        if match:
            priority_pages.append({**match, "required_field_anchor": field})
    for statement_type in query_parse.get("statement_types") or []:
        match = next(
            (page for page in scored_pages if statement_type in (page.get("statement_types") or [])),
            None,
        )
        if match:
            priority_pages.append({**match, "statement_type_anchor": statement_type})
    if query_parse.get("task_type") == "selection":
        match = next((page for page in scored_pages if page.get("selection_scope_score", 0.0) > 0), None)
        if match:
            priority_pages.append({**match, "selection_scope_anchor": True})
    selected_pages = []
    selected_page_keys = set()
    for page in priority_pages + scored_pages:
        key = ((page.get("filename") or ""), _coerce_int(page.get("page_number")) or 0)
        if key in selected_page_keys:
            continue
        selected_page_keys.add(key)
        selected_pages.append(page)
        if len(selected_pages) >= page_limit:
            break
    neighbor_window = max(0, _parse_int(os.getenv("RAG_PAGE_NEIGHBOR_WINDOW"), 0))
    if neighbor_window and selected_pages:
        pages_by_key = {
            ((page.get("filename") or ""), _coerce_int(page.get("page_number")) or 0): page
            for page in pages
        }
        expanded_pages = {
            ((page.get("filename") or ""), _coerce_int(page.get("page_number")) or 0): dict(page)
            for page in selected_pages
        }
        for anchor in selected_pages[:5]:
            filename = anchor.get("filename") or ""
            anchor_page = _coerce_int(anchor.get("page_number")) or 0
            for delta in range(1, neighbor_window + 1):
                for page_number in (anchor_page - delta, anchor_page + delta):
                    neighbor = pages_by_key.get((filename, page_number))
                    if not neighbor:
                        continue
                    neighbor_score = float(anchor.get("page_score", 0.0) or 0.0) - 0.02 * delta
                    key = (filename, page_number)
                    current = expanded_pages.get(key)
                    if current is None or neighbor_score > float(current.get("page_score", 0.0) or 0.0):
                        expanded_pages[key] = {
                            **neighbor,
                            "page_score": neighbor_score,
                            "neighbor_page": True,
                            "neighbor_anchor_page": anchor_page,
                        }
        selected_pages = sorted(
            expanded_pages.values(),
            key=lambda page: (
                bool(page.get("required_field_anchor")),
                bool(page.get("statement_type_anchor")),
                float(page.get("page_score", 0.0) or 0.0),
            ),
            reverse=True,
        )

    query_tokens = _extract_keyword_tokens(query) | _expand_finance_operand_tokens(query)
    query_metrics = _extract_metric_hints(query)
    selected_chunks: list[dict] = []
    page_chunks: dict[tuple[str, int], list[dict]] = defaultdict(list)
    all_chunk_ids = [chunk_id for page in selected_pages for chunk_id in page.get("chunk_ids", []) if chunk_id]
    for chunk in _milvus_manager.get_chunks_by_ids(list(dict.fromkeys(all_chunk_ids))):
        page_key = ((chunk.get("filename") or ""), _coerce_int(chunk.get("page_number")) or 0)
        page_chunks[page_key].append(chunk)
    for page in selected_pages:
        chunks = page_chunks.get(((page.get("filename") or ""), _coerce_int(page.get("page_number")) or 0), [])
        if not chunks:
            continue

        def chunk_priority(chunk: dict) -> float:
            text = chunk.get("text", "") or ""
            return (
                float(page.get("page_score", 0.0) or 0.0)
                + 0.20 * _score_overlap(query_tokens, _extract_keyword_tokens(text))
                + 0.15 * _score_overlap(query_metrics, _extract_metric_hints(text))
            )

        ordered_chunks = sorted(
            chunks,
            key=lambda chunk: (_coerce_int(chunk.get("chunk_idx")) if _coerce_int(chunk.get("chunk_idx")) is not None else 10**9, chunk.get("chunk_id", "")),
        )
        best_chunk = max(ordered_chunks, key=chunk_priority)
        anchor_index = ordered_chunks.index(best_chunk)
        anchor_window = 1 if _is_field_aware_enabled() else 0
        window_chunks = ordered_chunks[max(0, anchor_index - anchor_window) : anchor_index + anchor_window + 1]
        window_chunks.sort(key=lambda chunk: (0 if chunk is best_chunk else 1, abs(ordered_chunks.index(chunk) - anchor_index)))
        for chunk in window_chunks:
            anchor_distance = abs(ordered_chunks.index(chunk) - anchor_index)
            score = chunk_priority(chunk) - 0.01 * anchor_distance
            selected_chunks.append(
                {
                    **chunk,
                    "doc_name": page.get("doc_name") or get_doc_name(page.get("filename", "")),
                    "page_score": page.get("page_score", 0.0),
                    "score": score,
                    "rerank_score": score,
                    "source": "page_first_anchor" if _is_field_aware_enabled() else "page_first",
                    "anchor_chunk_id": best_chunk.get("chunk_id", ""),
                    "anchor_chunk_offset": ordered_chunks.index(chunk) - anchor_index,
                    "required_field_anchor": page.get("required_field_anchor", ""),
                    "statement_type_anchor": page.get("statement_type_anchor", ""),
                    "matched_required_fields": page.get("matched_required_fields", {}),
                    "statement_types": page.get("statement_types", []),
                    "selection_scope_anchor": bool(page.get("selection_scope_anchor")),
                }
            )

    selected_chunks.sort(key=_combined_score, reverse=True)
    selected_keys = {_doc_key(item) for item in selected_chunks}
    fallback = [item for item in chunk_candidates if _doc_key(item) not in selected_keys]
    return (selected_chunks + fallback)[:candidate_k], {
        "page_first_enabled": True,
        "page_first_selected_documents": [item.get("filename", "") for item in selected_docs],
        "page_first_selected_pages": [
            {
                "filename": item.get("filename", ""),
                "page_number": item.get("page_number"),
                "page_score": item.get("page_score", 0.0),
                "neighbor_page": bool(item.get("neighbor_page")),
                "required_field_anchor": item.get("required_field_anchor", ""),
                "statement_type_anchor": item.get("statement_type_anchor", ""),
            }
            for item in selected_pages
        ],
        "page_first_page_count": len(pages),
        "page_first_chunk_count": len(selected_chunks),
        "page_first_anchor_window": 1 if _is_field_aware_enabled() else 0,
        **page_meta,
    }


def _get_stepback_model():
    global _stepback_model
    if not ARK_API_KEY or not MODEL:
        return None
    if _stepback_model is None:
        _stepback_model = init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=ARK_API_KEY,
            base_url=BASE_URL,
            temperature=0.2,
        )
    return _stepback_model


def _generate_step_back_question(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请为财务/年报问答生成一个更适合检索的补充问题，只能做轻量改写，不能泛化丢失原始约束。\n"
        "必须保留并突出：公司名、年份、季度、报表名称、指标名、金额单位、百分比、日期、问题中的专有名词。\n"
        "不要改写成通用财务原理问题，不要加入中文解释，不要删除原问题中的数字和限定词。\n"
        "只输出一句补充检索问题。\n"
        f"原始问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def _answer_step_back_question(step_back_question: str) -> str:
    model = _get_stepback_model()
    if not model or not step_back_question:
        return ""
    prompt = (
        "请仅用一句话补充该问题涉及的同义表达或财务表述，帮助检索，不要输出通用原理解释，"
        "不要引入原问题之外的新公司、新年份或新指标。\n"
        f"补充问题：{step_back_question}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def generate_hypothetical_document(query: str) -> str:
    model = _get_stepback_model()
    if not model:
        return ""
    prompt = (
        "请基于用户问题生成一段‘假设性文档’，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def step_back_expand(query: str) -> dict:
    step_back_question = _generate_step_back_question(query)
    step_back_answer = _answer_step_back_question(step_back_question)
    parts = [query]
    if step_back_question:
        parts.append(step_back_question)
    if step_back_answer:
        parts.append(step_back_answer)
    expanded_query = "\n".join(part for part in parts if part).strip() or query
    return {
        "step_back_question": step_back_question,
        "step_back_answer": step_back_answer,
        "expanded_query": expanded_query,
    }


def retrieve_candidate_documents(query: str, candidate_k: int | None = None) -> Dict[str, Any]:
    config = get_finance_rag_config()
    candidate_k = max(1, candidate_k or config["candidate_k"])
    logger.info(
        "finance_rag_candidate_retrieve candidate_k=%s final_top_k=%s query_length=%s",
        candidate_k,
        config["final_top_k"],
        len(query or ""),
    )
    filter_expr = _build_text_chunk_filter_expr(f"chunk_level == {LEAF_RETRIEVE_LEVEL}")
    base_meta: Dict[str, Any] = {
        "retrieval_mode": "failed",
        "candidate_k": candidate_k,
        "leaf_retrieve_level": LEAF_RETRIEVE_LEVEL,
        "candidate_count": 0,
        "query_planner_enabled": False,
        "planner_intent": "",
        "planner_must_keep_terms": [],
        "planner_dense_queries": [],
        "planner_semantic_queries": [],
        "planner_evidence_field_queries": [],
        "planner_table_heading_queries": [],
        "planner_keyword_queries": [],
        "planner_table_queries": [],
        "planner_validation_dropped_queries": [],
        "planner_parse_error": "",
        "per_query_retrieval_counts": [],
        "rrf_fused_candidate_count": 0,
        "page_level_fusion_enabled": False,
        "fused_page_count": 0,
        "fused_top_pages": [],
        "fused_pages_after_anchor_guard": [],
        "page_anchor_filtered_count": 0,
        "page_anchor_guard_fallback_reason": "",
        "page_contributing_routes": {},
        "page_first_enabled": False,
        "page_first_selected_documents": [],
        "page_first_selected_pages": [],
        "page_first_page_count": 0,
        "page_first_chunk_count": 0,
        "retrieve_error": None,
        "rerank_enabled": bool(RERANK_MODEL and RERANK_API_KEY and RERANK_BINDING_HOST),
        "rerank_applied": False,
        "rerank_model": RERANK_MODEL,
        "rerank_endpoint": _get_rerank_endpoint(),
        "rerank_error": None,
        "auto_merge_enabled": AUTO_MERGE_ENABLED,
        "auto_merge_applied": False,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
        "auto_merge_replaced_chunks": 0,
        "auto_merge_steps": 0,
    }
    started_at = time.perf_counter()
    planner_enabled = False
    planner = {
        "enabled": False,
        "intent": "",
        "must_keep_terms": [],
        "semantic_queries": [],
        "evidence_field_queries": [],
        "table_heading_queries": [],
        "keyword_queries": [],
        "planner_validation_dropped_queries": [],
        "expected_evidence_type": "",
        "constraints": [],
        "parse_error": "",
    }
    field_aware_query = build_finance_query_rewrite(query) if _is_field_aware_enabled() else ""
    routes = _build_retrieval_routes(query, candidate_k, planner, field_aware_query)
    route_results: list[dict] = []
    per_query_retrieval_counts: list[dict] = []
    last_meta: Dict[str, Any] = {}
    for route in routes:
        _retrieval_debug(f"route={route['label']} starting")
        retrieved = _retrieve_leaf_chunks(
            route["query"],
            top_k=route["top_k"],
            filter_expr=filter_expr,
            retrieval_scope=f"planned:{route['label']}",
        )
        route_results.append({**route, "docs": retrieved.get("docs", []), "meta": retrieved.get("meta", {})})
        per_query_retrieval_counts.append(
            {
                "label": route["label"],
                "category": route["category"],
                "query": route["query"],
                "count": len(retrieved.get("docs", []) or []),
                "retrieval_mode": (retrieved.get("meta", {}) or {}).get("retrieval_mode", "failed"),
            }
        )
        last_meta = retrieved.get("meta", {}) or {}
        _retrieval_debug(
            f"route={route['label']} candidates={len(retrieved.get('docs', []) or [])} "
            f"embedding_ms={(retrieved.get('meta', {}) or {}).get('query_embedding_ms', '')} "
            f"milvus_ms={(retrieved.get('meta', {}) or {}).get('milvus_retrieval_ms', '')}"
        )

    fused_docs_all = _rrf_fuse_retrieval_routes(route_results)
    page_level_fusion_enabled = planner_enabled and _is_page_level_fusion_enabled()
    if page_level_fusion_enabled:
        fused_docs, fused_pages, page_guard_meta = _apply_page_level_fusion(
            fused_docs_all,
            route_results,
            candidate_k=candidate_k,
        )
    else:
        fused_docs = fused_docs_all[:candidate_k]
        fused_pages = []
        page_guard_meta = {
            "page_anchor_filtered_count": 0,
            "fused_pages_after_anchor_guard": [],
            "page_anchor_guard_fallback_reason": "",
        }
    page_first_meta: Dict[str, Any] = {}
    if _is_page_first_enabled() and fused_docs_all:
        try:
            _retrieval_debug(f"page_first starting fused_candidates={len(fused_docs_all)}")
            fused_docs, page_first_meta = _build_page_first_candidates(
                query,
                fused_docs_all,
                candidate_k=candidate_k,
                config=config,
            )
            _retrieval_debug(f"page_first candidates={len(fused_docs)}")
        except Exception as exc:
            logger.exception("page-first candidate selection failed; using chunk baseline")
            page_first_meta = {
                "page_first_enabled": True,
                "page_first_fallback": f"{type(exc).__name__}: {exc}",
            }
    meta = {
        **base_meta,
        **last_meta,
        **page_first_meta,
        "query_planner_enabled": planner_enabled,
        "planner_intent": planner.get("intent", ""),
        "planner_must_keep_terms": planner.get("must_keep_terms", []) or [],
        "planner_dense_queries": planner.get("semantic_queries", []) or [],
        "planner_semantic_queries": planner.get("semantic_queries", []) or [],
        "planner_evidence_field_queries": planner.get("evidence_field_queries", []) or [],
        "planner_table_heading_queries": planner.get("table_heading_queries", []) or [],
        "planner_keyword_queries": planner.get("keyword_queries", []) or [],
        "planner_table_queries": planner.get("table_heading_queries", []) or [],
        "planner_validation_dropped_queries": planner.get("planner_validation_dropped_queries", []) or [],
            "planner_parse_error": planner.get("parse_error", ""),
            "field_aware_query": field_aware_query,
        "per_query_retrieval_counts": per_query_retrieval_counts,
        "rrf_fused_candidate_count": len(fused_docs_all),
        "page_level_fusion_enabled": page_level_fusion_enabled,
        "fused_page_count": len(fused_pages),
        "fused_top_pages": [
            {
                "filename": entry.get("filename", "") or "",
                "page_number": entry.get("page_number", ""),
                "page_score": entry.get("page_score", 0.0),
                "matched_queries": entry.get("matched_queries", []) or [],
                "contributing_routes": entry.get("contributing_routes", []) or [],
            }
            for entry in fused_pages[:candidate_k]
        ],
        "fused_pages_after_anchor_guard": [
            {
                "filename": entry.get("filename", "") or "",
                "page_number": entry.get("page_number", ""),
                "page_score": entry.get("page_score", 0.0),
                "matched_queries": entry.get("matched_queries", []) or [],
                "contributing_routes": entry.get("contributing_routes", []) or [],
            }
            for entry in (page_guard_meta.get("fused_pages_after_anchor_guard", []) or [])[:candidate_k]
        ],
        "page_anchor_filtered_count": page_guard_meta.get("page_anchor_filtered_count", 0),
        "page_anchor_guard_fallback_reason": page_guard_meta.get("page_anchor_guard_fallback_reason", ""),
        "page_contributing_routes": {
            f"{entry.get('filename', '')}#page={entry.get('page_number', '')}": entry.get("contributing_routes", []) or []
            for entry in fused_pages[:candidate_k]
        },
        "candidate_count": len(fused_docs),
        "retrieval_mode": "planned_rrf" if len(route_results) > 1 else last_meta.get("retrieval_mode", "failed"),
    }
    meta["latency_breakdown"] = {
        "initial_retrieval_ms": round((time.perf_counter() - started_at) * 1000, 2),
    }
    return {"docs": fused_docs, "meta": meta}


def _fetch_neighbor_page_docs(filename: str, page_number: int, page_window: int) -> List[dict]:
    if page_window < 0:
        return []
    if not hasattr(_milvus_manager, "query_all"):
        return []
    out: List[dict] = []
    escaped_filename = _escape_milvus_string(filename)
    for target_page in range(page_number - page_window, page_number + page_window + 1):
        if target_page < 0:
            continue
        filter_expr = (
            f'filename == "{escaped_filename}" and '
            f"page_number == {target_page} and "
            f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
        )
        filter_expr = _build_text_chunk_filter_expr(filter_expr)
        out.extend(_milvus_manager.query_all(filter_expr=filter_expr, output_fields=TRACE_OUTPUT_FIELDS))
    return out


def _fetch_neighbor_chunk_docs(filename: str, chunk_idx: int, chunk_window: int) -> List[dict]:
    if chunk_window <= 0:
        return []
    if not hasattr(_milvus_manager, "query_all"):
        return []
    out: List[dict] = []
    escaped_filename = _escape_milvus_string(filename)
    for target_idx in range(chunk_idx - chunk_window, chunk_idx + chunk_window + 1):
        if target_idx < 0:
            continue
        filter_expr = (
            f'filename == "{escaped_filename}" and '
            f"chunk_idx == {target_idx} and "
            f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
        )
        filter_expr = _build_text_chunk_filter_expr(filter_expr)
        out.extend(_milvus_manager.query_all(filter_expr=filter_expr, output_fields=TRACE_OUTPUT_FIELDS))
    return out


def _merge_context_chunks(
    final_docs: List[dict],
    *,
    enable_page_merge: bool,
    adjacent_page_window: int,
    adjacent_chunk_window: int,
    final_top_k: int,
) -> Tuple[List[dict], Dict[str, Any]]:
    if not final_docs:
        return [], {
            "page_merge_applied": False,
            "merged_chunk_count": 0,
            "final_context_chunk_count": 0,
        }
    if not enable_page_merge:
        return final_docs[:final_top_k], {
            "page_merge_applied": False,
            "merged_chunk_count": 0,
            "final_context_chunk_count": len(final_docs[:final_top_k]),
        }

    extra_docs: List[dict] = []
    seen_pages = set()
    seen_chunks = set()

    for doc in final_docs:
        filename = doc.get("filename") or ""
        if not filename:
            continue

        page_number = _coerce_int(doc.get("page_number"))
        if page_number is not None:
            page_key = (filename, page_number)
            if page_key not in seen_pages:
                extra_docs.extend(_fetch_neighbor_page_docs(filename, page_number, adjacent_page_window))
                seen_pages.add(page_key)

        chunk_idx = _coerce_int(doc.get("chunk_idx"))
        if chunk_idx is not None and adjacent_chunk_window > 0:
            chunk_key = (filename, chunk_idx)
            if chunk_key not in seen_chunks:
                extra_docs.extend(_fetch_neighbor_chunk_docs(filename, chunk_idx, adjacent_chunk_window))
                seen_chunks.add(chunk_key)

    context_docs = _deduplicate_docs(final_docs + extra_docs)
    max_context_chunks = max(final_top_k, min(40, final_top_k * 3))
    context_docs = context_docs[:max_context_chunks]

    merged_chunk_count = max(0, len(context_docs) - len(final_docs))
    return context_docs, {
        "page_merge_applied": merged_chunk_count > 0,
        "merged_chunk_count": merged_chunk_count,
        "final_context_chunk_count": len(context_docs),
    }


def _build_table_context_doc(query: str, retrieved_docs: List[dict] | None = None) -> Tuple[dict | None, Dict[str, Any]]:
    config = get_table_aware_retrieval_config()
    retrieved_docs = list(retrieved_docs or [])
    query_anchors = extract_query_anchors(query)
    query_triggered, query_trigger_reasons = _query_triggers_table_aware_retrieval(query)
    candidate_filenames = _extract_table_candidate_filenames(
        retrieved_docs,
        query_anchors=query_anchors,
        max_files=config["max_candidate_docs"],
    )
    candidate_pages = _extract_table_candidate_pages(retrieved_docs)
    should_enable, auto_triggered, trigger_reasons = _should_enable_table_aware_retrieval(
        query,
        retrieved_docs,
        config["mode"],
    )
    skipped_reasons: list[str] = []
    base_meta = {
        "table_aware_retrieval_mode": config["mode"],
        "table_aware_auto_triggered": auto_triggered,
        "table_aware_trigger_reason": trigger_reasons,
        "query_anchors": query_anchors,
        "anchor_guard_applied": False,
        "anchor_filtered_count": 0,
        "table_context_source": "none",
        "table_evidence_hit_count": 0,
        "table_context_table_count": 0,
        "table_context_char_count": 0,
        "table_candidate_filenames": candidate_filenames,
        "table_candidate_pages": candidate_pages,
        "table_ids": [],
        "table_context_skipped_reasons": skipped_reasons,
    }
    if not should_enable:
        _append_skip_reason(skipped_reasons, "non_table_query")
        return None, base_meta

    try:
        source = "none"
        hits: list[dict] = []
        tables: list[dict] = []
        table_ids: list[str] = []
        table_evidence_hit_count = 0
        anchor_guard_applied = False
        anchor_filtered_count = 0
        full_table_ids: set[str] = set()
        skipped_table_reasons: dict[str, str] = {}

        retrieved_table_ids = _extract_retrieved_table_ids(retrieved_docs, config["max_tables"])
        if retrieved_table_ids:
            candidate_tables = _table_store.get_tables_by_ids(retrieved_table_ids)
            tables = candidate_tables[: config["max_tables"]]
            table_ids = [table.get("table_id", "") for table in tables if table.get("table_id")]
            if table_ids:
                selected_table_ids = set(table_ids)
                hits = [doc for doc in retrieved_docs if (doc.get("table_id") or "") in selected_table_ids]
                table_evidence_hit_count = len(hits)
                source = "retrieved_table_chunk"

        if not table_ids and not candidate_pages:
            _append_skip_reason(skipped_reasons, "no_table_like_chunks")

        if not table_ids and candidate_pages:
            candidate_tables = _fetch_tables_for_candidate_pages(candidate_pages, config["max_tables"])
            if candidate_tables:
                tables = candidate_tables[: config["max_tables"]]
                table_ids = [table.get("table_id", "") for table in tables if table.get("table_id")]
                hits = _build_same_page_table_hits(tables, retrieved_docs)
                table_evidence_hit_count = len(hits)
                source = "same_page_table"
            else:
                _append_skip_reason(skipped_reasons, "no_valid_same_page_tables")

        if not table_ids and (config["mode"] == "force" or (config["mode"] == "auto" and query_triggered)):
            filter_expr = _build_table_evidence_filter_expr(candidate_filenames)
            dense_embedding, sparse_embedding = _get_hybrid_query_embeddings(query)
            search_hits = _milvus_manager.hybrid_retrieve(
                dense_embedding=dense_embedding,
                sparse_embedding=sparse_embedding,
                top_k=config["top_k"],
                filter_expr=filter_expr,
                query_text=query,
            )
            search_table_ids = _dedupe_table_ids_for_context(search_hits)[: config["max_tables"]]
            selected_hits = [hit for hit in search_hits if (hit.get("table_id") or "") in set(search_table_ids)]
            candidate_tables = _table_store.get_tables_by_ids(search_table_ids) if search_table_ids else []
            if candidate_filenames:
                candidate_tables = [
                    table for table in candidate_tables if (table.get("filename") or "") in set(candidate_filenames)
                ]
            if candidate_tables:
                tables = candidate_tables[: config["max_tables"]]
                table_ids = [table.get("table_id", "") for table in tables if table.get("table_id")]
                hits = [hit for hit in selected_hits if (hit.get("table_id") or "") in set(table_ids)]
                table_evidence_hit_count = len(search_hits)
                source = "document_scoped_search" if candidate_filenames else "global_fallback"
            else:
                _append_skip_reason(skipped_reasons, "no_valid_table_evidence")
        elif not table_ids and not candidate_filenames:
            _append_skip_reason(skipped_reasons, "no_candidate_documents")

        if tables:
            tables, hits, anchor_guard_applied, anchor_filtered_count = _apply_anchor_guard_to_tables(
                tables,
                hits,
                retrieved_docs,
                query_anchors,
            )
            if not tables:
                source = "none"
                table_ids = []
                table_evidence_hit_count = 0
                full_table_ids.clear()
                skipped_table_reasons.clear()
                _append_skip_reason(skipped_reasons, "anchor_guard_filtered")
                _append_skip_reason(skipped_reasons, "no_valid_table_candidates")
            else:
                selected_table_ids = {(table.get("table_id") or "").strip() for table in tables if table.get("table_id")}
                table_ids = [table.get("table_id", "") for table in tables if table.get("table_id")]
                hits = [hit for hit in hits if not selected_table_ids or (hit.get("table_id") or "").strip() in selected_table_ids]
                table_evidence_hit_count = len(hits)

        for table in tables:
            table_id = (table.get("table_id") or "").strip()
            if not table_id:
                continue
            is_full, rejected_reason = _table_quality_guard(table)
            if is_full:
                full_table_ids.add(table_id)
            else:
                skipped_table_reasons[table_id] = rejected_reason or "table_quality_rejected"
                _append_skip_reason(skipped_reasons, "table_quality_rejected")

        table_context = _build_table_context_preview(
            hits,
            tables,
            preview_rows=config["max_rows"],
            preview_chars=config["max_context_chars"],
            full_table_ids=full_table_ids,
            skipped_table_reasons=skipped_table_reasons,
        )
        table_context = _truncate_table_context(table_context, config["max_context_chars"])
        meta = {
            **base_meta,
            "anchor_guard_applied": anchor_guard_applied,
            "anchor_filtered_count": anchor_filtered_count,
            "table_context_source": source,
            "table_evidence_hit_count": table_evidence_hit_count,
            "table_context_table_count": len(tables),
            "table_context_char_count": len(table_context),
            "table_ids": table_ids,
        }
        if not table_context:
            return None, meta
        return (
            {
                "filename": "StructuredTableEvidence",
                "doc_name": "StructuredTableEvidence",
                "file_type": "TABLE_CONTEXT",
                "page_number": "",
                "chunk_id": "structured_table_context",
                "parent_chunk_id": "",
                "root_chunk_id": "",
                "chunk_level": LEAF_RETRIEVE_LEVEL,
                "chunk_idx": 0,
                "type": "table_context",
                "text": f"Additional structured table evidence:\n\n{table_context}",
                "evidence_type": "table_context",
                "table_id": "",
                "row_id": "",
                "table_title": "",
            },
            meta,
        )
    except Exception:
        logger.exception("table-aware retrieval failed query=%s", query)
        return None, base_meta


def finalize_retrieved_documents(
    query: str,
    candidate_docs: List[dict],
    *,
    final_top_k: int | None = None,
    enable_page_merge: bool | None = None,
    adjacent_page_window: int | None = None,
    adjacent_chunk_window: int | None = None,
) -> Dict[str, Any]:
    config = get_finance_rag_config()
    retrieval_mode = config["retrieval_mode"]
    experimental_mode = retrieval_mode == "finance_experimental"
    final_top_k = max(1, final_top_k or config["final_top_k"])
    enable_page_merge = config["enable_page_merge"] if enable_page_merge is None else enable_page_merge
    adjacent_page_window = config["adjacent_page_window"] if adjacent_page_window is None else max(0, adjacent_page_window)
    adjacent_chunk_window = config["adjacent_chunk_window"] if adjacent_chunk_window is None else max(0, adjacent_chunk_window)
    doc_stage_top_n = config["doc_stage_top_n"]
    page_stage_top_n = max(config["page_stage_top_n"], final_top_k)
    logger.info(
        (
            "finance_rag_finalize retrieval_mode=%s candidate_k=%s final_top_k=%s candidate_docs=%s "
            "enable_page_merge=%s two_stage_retrieval=%s doc_stage_top_n=%s page_stage_top_n=%s"
        ),
        retrieval_mode,
        config["candidate_k"],
        final_top_k,
        len(candidate_docs),
        enable_page_merge,
        config["two_stage_retrieval"],
        doc_stage_top_n,
        page_stage_top_n,
    )

    deduped_candidates = _deduplicate_docs(candidate_docs)
    stage_two_meta: Dict[str, Any] = {
        "retrieval_mode": retrieval_mode,
        "two_stage_retrieval": config["two_stage_retrieval"],
        "query_anchors": extract_query_anchors(query),
        "anchor_guard_applied": False,
        "anchor_filtered_count": 0,
        "doc_stage_top_n": doc_stage_top_n,
        "page_stage_top_n": page_stage_top_n,
        "query_parse": {},
        "doc_stage_selected_docs": [],
        "selected_docs": [],
        "selected_pages": [],
        "page_scores": [],
        "final_evidence_pack": [],
        "final_evidence_pack_debug": [],
        "final_evidence_pack_used": [],
        "page_stage_candidates": [],
        "page_stage_candidate_count": 0,
        "cover_page_filtered_count": 0,
        "fallback_used": False,
        "fallback_reason": "",
        "page_fusion_used_for_final_context": False,
        "final_evidence_pack_source": "chunk_rerank_fallback",
    }
    rerank_meta: Dict[str, Any]
    if experimental_mode:
        t_parse = time.perf_counter()
        query_parse = parse_finance_query(query)
        query_parse.update(infer_task_spec(query))
        query_parse["statement_types"] = infer_statement_types(query, query_parse.get("required_fields") or [])
        query_parse["raw_question"] = query
        query_parse_ms = round((time.perf_counter() - t_parse) * 1000, 2)
        stage_two_meta["query_parse"] = query_parse
        initial_rerank_top_k = max(final_top_k * 2, page_stage_top_n)
        initial_reranked_docs, initial_rerank_meta = _rerank_documents(
            query=query,
            docs=deduped_candidates,
            top_k=initial_rerank_top_k,
        )
    else:
        query_parse_ms = 0.0
        initial_reranked_docs = []
        initial_rerank_meta = {}

    if experimental_mode and config["two_stage_retrieval"]:
        t_page = time.perf_counter()
        stage_two_candidates, stage_two_rerank_meta = _run_two_stage_retrieval(
            query,
            deduped_candidates,
            query_parse=query_parse,
            config=config,
            doc_stage_top_n=doc_stage_top_n,
            page_stage_top_n=page_stage_top_n,
            final_top_k=final_top_k,
        )
        page_rerank_ms = round((time.perf_counter() - t_page) * 1000, 2)
        stage_two_meta.update(stage_two_rerank_meta)
        selected_pages = stage_two_meta.get("selected_page_records", []) or []
        all_page_records = stage_two_meta.get("all_page_records", []) or []
        evidence_pack_used, page_merge_meta = _build_evidence_pack(
            selected_pages,
            all_page_records,
            query_parse=query_parse,
            config=config,
            final_top_k=final_top_k,
            enable_page_merge=enable_page_merge,
            adjacent_page_window=adjacent_page_window,
        )
        fallback_candidates = [_as_evidence_chunk(doc) for doc in initial_reranked_docs]
        combined_pack = _deduplicate_docs(
            evidence_pack_used + stage_two_candidates + fallback_candidates
        )
        if _is_anchor_guard_enabled():
            combined_pack, anchor_guard_applied, anchor_filtered_count = _apply_anchor_guard_to_docs(
                combined_pack,
                stage_two_meta.get("query_anchors", []) or [],
            )
            if evidence_pack_used:
                filtered_used_pack, _, _ = _apply_anchor_guard_to_docs(
                    evidence_pack_used,
                    stage_two_meta.get("query_anchors", []) or [],
                )
                evidence_pack_used = filtered_used_pack
        else:
            anchor_guard_applied, anchor_filtered_count = False, 0
        stage_two_meta["anchor_guard_applied"] = anchor_guard_applied
        stage_two_meta["anchor_filtered_count"] = anchor_filtered_count
        if len(evidence_pack_used) < final_top_k:
            stage_two_meta["fallback_used"] = True
            stage_two_meta["fallback_reason"] = stage_two_meta.get("fallback_reason") or "insufficient_used_evidence"
        context_docs = evidence_pack_used[: config["max_evidence_pack_used"]]
        final_docs = (context_docs + [doc for doc in combined_pack if doc not in context_docs])[:final_top_k]
        stage_two_meta["final_evidence_pack"] = page_merge_meta.get("final_evidence_pack_debug", [])
        stage_two_meta["final_evidence_pack_debug"] = page_merge_meta.get("final_evidence_pack_debug", [])
        stage_two_meta["final_evidence_pack_used"] = page_merge_meta.get("final_evidence_pack_used", [])
        page_merge_meta["final_context_chunk_count"] = len(context_docs)
        page_merge_meta["merged_chunk_count"] = max(0, len(page_merge_meta.get("final_evidence_pack_debug", [])) - len(context_docs))
        rerank_meta = {
            "rerank_enabled": True,
            "rerank_applied": True,
            "rerank_model": "page_hybrid_rerank",
            "rerank_endpoint": "local_page_hybrid_rerank",
            "rerank_error": None,
        }
        merge_meta = {
            "auto_merge_enabled": AUTO_MERGE_ENABLED,
            "auto_merge_applied": False,
            "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
            "auto_merge_replaced_chunks": 0,
            "auto_merge_steps": 0,
        }
    else:
        page_rerank_ms = 0.0
        if _is_cover_filter_enabled():
            filtered_candidates, cover_page_filtered_count = _filter_cover_or_toc_docs(query, deduped_candidates)
            rerank_input = filtered_candidates or deduped_candidates
            stage_two_meta["cover_page_filtered_count"] = cover_page_filtered_count if filtered_candidates else 0
        else:
            rerank_input = deduped_candidates
            stage_two_meta["cover_page_filtered_count"] = 0
        reranked_docs, rerank_meta = _rerank_documents(
            query=query,
            docs=rerank_input,
            top_k=max(final_top_k * 2, final_top_k),
        )
        if _is_anchor_guard_enabled():
            reranked_docs, anchor_guard_applied, anchor_filtered_count = _apply_anchor_guard_to_docs(
                reranked_docs,
                stage_two_meta.get("query_anchors", []) or [],
            )
        else:
            anchor_guard_applied, anchor_filtered_count = False, 0
        stage_two_meta["anchor_guard_applied"] = anchor_guard_applied
        stage_two_meta["anchor_filtered_count"] = anchor_filtered_count
        page_fusion_selected_docs, selected_pages, page_fusion_used = _select_docs_from_fused_pages(
            reranked_docs,
            final_top_k=final_top_k,
        )
        stage_two_meta["selected_pages"] = selected_pages
        if page_fusion_used:
            final_docs = page_fusion_selected_docs
            merge_meta = {
                "auto_merge_enabled": AUTO_MERGE_ENABLED,
                "auto_merge_applied": False,
                "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
                "auto_merge_replaced_chunks": 0,
                "auto_merge_steps": 0,
            }
            stage_two_meta["page_fusion_used_for_final_context"] = True
            stage_two_meta["final_evidence_pack_source"] = "page_fusion"
        else:
            final_docs, merge_meta = _auto_merge_documents(docs=reranked_docs, top_k=final_top_k)
        if len(final_docs) < final_top_k and len(reranked_docs) >= final_top_k:
            final_docs = reranked_docs[:final_top_k]
            merge_meta = {
                **merge_meta,
                "auto_merge_applied": False,
                "auto_merge_replaced_chunks": 0,
                "auto_merge_steps": 0,
            }
            stage_two_meta["selected_pages"] = []
            stage_two_meta["page_fusion_used_for_final_context"] = False
            stage_two_meta["final_evidence_pack_source"] = "chunk_rerank_fallback"
        context_docs, page_merge_meta = _merge_context_chunks(
            final_docs,
            enable_page_merge=enable_page_merge,
            adjacent_page_window=adjacent_page_window,
            adjacent_chunk_window=adjacent_chunk_window,
            final_top_k=final_top_k,
        )
        stage_two_meta["final_evidence_pack_used"] = final_docs[: config["max_evidence_pack_used"]]
        stage_two_meta["final_evidence_pack_debug"] = context_docs
        stage_two_meta["final_evidence_pack"] = context_docs

    meta = {
        **stage_two_meta,
        **(initial_rerank_meta if experimental_mode and config["two_stage_retrieval"] else {}),
        **rerank_meta,
        **merge_meta,
        **page_merge_meta,
        "final_top_k": final_top_k,
        "final_candidate_count": len(deduped_candidates),
        "latency_breakdown": {
            "query_parse_ms": query_parse_ms,
            "page_rerank_ms": page_rerank_ms,
            "final_evidence_pack_used_count": len(stage_two_meta.get("final_evidence_pack_used", []) or []),
            "prompt_context_char_count_estimate": _estimate_prompt_chars(stage_two_meta.get("final_evidence_pack_used", []) or context_docs),
            "page_embedding_cache_hit_count": stage_two_meta.get("page_embedding_cache_hit_count", 0),
            "page_embedding_cache_miss_count": stage_two_meta.get("page_embedding_cache_miss_count", 0),
            "query_time_page_embedding_executed": stage_two_meta.get("query_time_page_embedding_executed", False),
        },
    }
    if meta["latency_breakdown"]["query_time_page_embedding_executed"]:
        logger.warning("WARNING: query-time page embedding executed; this should be cached/precomputed.")
    logger.info(
        (
            "finance_rag_finalize_result retrieval_mode=%s final_retrieved_count=%s final_page_distribution=%s "
            "final_page_zero_count=%s cover_page_filtered_count=%s fallback_used=%s "
            "query_parse_ms=%s page_rerank_ms=%s pack_used=%s prompt_chars=%s cache_hit=%s cache_miss=%s"
        ),
        retrieval_mode,
        len(final_docs),
        _page_distribution(final_docs),
        _page_zero_count(final_docs),
        meta.get("cover_page_filtered_count", 0),
        meta.get("fallback_used", False),
        query_parse_ms,
        page_rerank_ms,
        len(stage_two_meta.get("final_evidence_pack_used", []) or []),
        meta["latency_breakdown"]["prompt_context_char_count_estimate"],
        meta["latency_breakdown"]["page_embedding_cache_hit_count"],
        meta["latency_breakdown"]["page_embedding_cache_miss_count"],
    )
    return {
        "final_retrieved_docs": final_docs,
        "context_docs": context_docs,
        "meta": meta,
    }


def retrieve_documents(
    query: str,
    top_k: int = 5,
    *,
    candidate_k: int | None = None,
    apply_page_merge: bool | None = None,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    table_aware_config = get_table_aware_retrieval_config()
    candidates = retrieve_candidate_documents(query, candidate_k=candidate_k)
    field_aware = _is_field_aware_enabled()
    finalized = finalize_retrieved_documents(
        query,
        candidates.get("docs", []),
        final_top_k=top_k,
        enable_page_merge=apply_page_merge or field_aware,
        adjacent_page_window=max(0, _parse_int(os.getenv("RAG_CONTEXT_PAGE_WINDOW"), 2)) if field_aware else None,
        adjacent_chunk_window=1 if field_aware else None,
    )
    context_docs = list(finalized.get("context_docs", []) or [])
    combined_meta = {**(candidates.get("meta", {}) or {}), **(finalized.get("meta", {}) or {})}
    clean_baseline = uses_clean_baseline_path()
    if clean_baseline:
        task_spec = {}
        initial_coverage = {"status": "not_evaluated", "answerable": None}
    else:
        task_spec = parse_finance_query(query)
        task_spec.update(infer_task_spec(query))
        task_spec["statement_types"] = infer_statement_types(query, task_spec.get("required_fields") or [])
        initial_coverage = assess_required_field_coverage(task_spec, context_docs)
    supplemental_meta = {
        "supplemental_search_enabled": (
            _parse_bool(os.getenv("RAG_SUPPLEMENTAL_SEARCH_ENABLED"), True)
            and not _parse_bool(os.getenv("SUPPLEMENTAL_FIND_ENABLED"), False)
        ),
        "supplemental_search_attempted": False,
        "supplemental_search_query": "",
        "supplemental_search_filenames": [],
        "supplemental_search_candidate_count": 0,
        "supplemental_search_error": "",
        "evidence_coverage_before_supplemental": initial_coverage,
    }
    combined_candidates = list(candidates.get("docs", []) or [])
    supplemental_query = "" if clean_baseline else build_supplemental_field_query(query, initial_coverage)
    if (
        field_aware
        and supplemental_meta["supplemental_search_enabled"]
        and initial_coverage.get("status") in {"partial", "insufficient"}
        and supplemental_query
    ):
        supplemental_meta["supplemental_search_attempted"] = True
        supplemental_meta["supplemental_search_query"] = supplemental_query
        scoped_docs = _select_doc_stage_documents(combined_candidates, 3, task_spec)
        scoped_filenames = list(dict.fromkeys(doc.get("filename") or "" for doc in scoped_docs if doc.get("filename")))
        supplemental_meta["supplemental_search_filenames"] = scoped_filenames
        if scoped_filenames:
            filter_expr = f'({_build_filename_filter(scoped_filenames)}) and (evidence_type == "text_chunk" or evidence_type == "")'
            try:
                supplemental = _retrieve_leaf_chunks(
                    supplemental_query,
                    top_k=max(1, _parse_int(os.getenv("RAG_SUPPLEMENTAL_CANDIDATE_K"), 12)),
                    filter_expr=filter_expr,
                    retrieval_scope="supplemental:required_fields",
                )
                supplemental_docs = list(supplemental.get("docs", []) or [])
                supplemental_meta["supplemental_search_candidate_count"] = len(supplemental_docs)
                if supplemental_docs:
                    combined_candidates = _deduplicate_docs(combined_candidates + supplemental_docs)
                    refreshed_candidates, supplemental_page_meta = _build_page_first_candidates(
                        query,
                        combined_candidates,
                        candidate_k=max(top_k, get_finance_rag_config()["candidate_k"]),
                        config=get_finance_rag_config(),
                    )
                    if refreshed_candidates:
                        combined_candidates = refreshed_candidates
                    supplemental_meta["supplemental_page_first_selected_pages"] = supplemental_page_meta.get(
                        "page_first_selected_pages", []
                    )
                    finalized = finalize_retrieved_documents(
                        query,
                        combined_candidates,
                        final_top_k=top_k,
                        enable_page_merge=True,
                        adjacent_page_window=max(0, _parse_int(os.getenv("RAG_CONTEXT_PAGE_WINDOW"), 2)),
                        adjacent_chunk_window=1,
                    )
                    context_docs = list(finalized.get("context_docs", []) or [])
                    combined_meta = {**combined_meta, **(finalized.get("meta", {}) or {})}
            except (RuntimeError, ValueError, TypeError, OSError) as exc:
                supplemental_meta["supplemental_search_error"] = f"{type(exc).__name__}: {exc}"
    if clean_baseline:
        table_context_meta = {
            "table_aware_retrieval_mode": "off",
            "evidence_group_count": 0,
            "selected_evidence_group_count": 0,
            "final_evidence_pack_source": combined_meta.get(
                "final_evidence_pack_source", "chunk_rerank_fallback"
            ),
        }
    else:
        evidence_group_docs, table_context_meta = _build_evidence_groups(
            query,
            context_docs,
            finalized.get("final_retrieved_docs", []) or [],
            combined_meta,
        )
        if table_aware_config["mode"] != "off" and evidence_group_docs:
            context_docs = evidence_group_docs
    meta = {**combined_meta, **table_context_meta, **supplemental_meta}
    meta["field_aware_enabled"] = _is_field_aware_enabled()
    meta["evidence_coverage"] = (
        {"status": "not_evaluated", "answerable": None}
        if clean_baseline else assess_required_field_coverage(task_spec, context_docs)
    )
    meta["evidence_coverage"]["supplemental_search_attempted"] = supplemental_meta["supplemental_search_attempted"]
    meta["latency_breakdown"] = {
        **(candidates.get("meta", {}).get("latency_breakdown", {}) or {}),
        **(finalized.get("meta", {}).get("latency_breakdown", {}) or {}),
        "prompt_context_char_count_estimate": _estimate_prompt_chars(context_docs),
        "total_retrieval_ms": round((time.perf_counter() - started_at) * 1000, 2),
    }
    return {
        "docs": context_docs,
        "candidate_docs": combined_candidates,
        "final_retrieved_docs": finalized.get("final_retrieved_docs", []),
        "context_docs": context_docs,
        "meta": meta,
    }


def retrieve_document_scoped_candidates(
    query: str,
    filenames: list[str],
    *,
    top_k: int = 12,
) -> list[dict[str, Any]]:
    """Run one leaf retrieval restricted to an already-selected document set."""
    scoped_filenames = list(dict.fromkeys(str(item).strip() for item in filenames if str(item).strip()))
    if not query.strip() or not scoped_filenames:
        return []
    filter_expr = (
        f'({_build_filename_filter(scoped_filenames)}) and '
        '(evidence_type == "text_chunk" or evidence_type == "")'
    )
    result = _retrieve_leaf_chunks(
        query,
        top_k=max(1, top_k),
        filter_expr=filter_expr,
        retrieval_scope="supplemental:document_scoped_once",
    )
    return list(result.get("docs") or [])


def debug_retrieval_pipeline(question: str, top_k: int = 10) -> Dict[str, Any]:
    result = retrieve_documents(question, top_k=top_k)
    meta = result.get("meta", {}) or {}

    def _trace_chunk(doc: dict, rank: int) -> dict:
        filename = doc.get("filename", "") or ""
        page_number = doc.get("page_number", "")
        return {
            "rank": rank,
            "filename": filename,
            "doc_name": doc.get("doc_name") or get_doc_name(filename),
            "page_number": page_number if page_number is not None else "",
            "chunk_id": doc.get("chunk_id", "") or "",
            "type": doc.get("type", "") or "",
            "text": doc.get("text", "") or "",
            "score": _safe_float(doc.get("score")),
            "rerank_score": _safe_float(doc.get("rerank_score")),
            "source": doc.get("source") or "",
        }

    return {
        "question": question,
        "query_parse": meta.get("query_parse", {}),
        "rag_trace": {
            "retrieval_mode": meta.get("retrieval_mode", "baseline"),
            "field_aware_enabled": _is_field_aware_enabled(),
            "evidence_coverage": meta.get("evidence_coverage", {}),
            "query_planner_enabled": meta.get("query_planner_enabled", False),
            "planner_intent": meta.get("planner_intent", ""),
            "planner_must_keep_terms": meta.get("planner_must_keep_terms", []) or [],
            "planner_dense_queries": meta.get("planner_dense_queries", []) or [],
            "planner_semantic_queries": meta.get("planner_semantic_queries", []) or [],
            "planner_evidence_field_queries": meta.get("planner_evidence_field_queries", []) or [],
            "planner_table_heading_queries": meta.get("planner_table_heading_queries", []) or [],
            "planner_keyword_queries": meta.get("planner_keyword_queries", []) or [],
            "planner_table_queries": meta.get("planner_table_queries", []) or [],
            "planner_validation_dropped_queries": meta.get("planner_validation_dropped_queries", []) or [],
            "planner_parse_error": meta.get("planner_parse_error", ""),
            "per_query_retrieval_counts": meta.get("per_query_retrieval_counts", []) or [],
            "rrf_fused_candidate_count": meta.get("rrf_fused_candidate_count", 0),
            "page_level_fusion_enabled": meta.get("page_level_fusion_enabled", False),
            "fused_page_count": meta.get("fused_page_count", 0),
            "fused_top_pages": meta.get("fused_top_pages", []) or [],
            "fused_pages_after_anchor_guard": meta.get("fused_pages_after_anchor_guard", []) or [],
            "page_anchor_filtered_count": meta.get("page_anchor_filtered_count", 0),
            "page_contributing_routes": meta.get("page_contributing_routes", {}) or {},
            "page_fusion_used_for_final_context": meta.get("page_fusion_used_for_final_context", False),
            "final_evidence_pack_source": meta.get("final_evidence_pack_source", "chunk_rerank_fallback"),
            "table_aware_retrieval_mode": meta.get("table_aware_retrieval_mode", "off"),
            "table_aware_auto_triggered": meta.get("table_aware_auto_triggered", False),
            "table_aware_trigger_reason": meta.get("table_aware_trigger_reason", []) or [],
            "query_anchors": meta.get("query_anchors", []) or [],
            "anchor_guard_applied": meta.get("anchor_guard_applied", False),
            "anchor_filtered_count": meta.get("anchor_filtered_count", 0),
            "table_context_source": meta.get("table_context_source", "none"),
            "table_evidence_hit_count": meta.get("table_evidence_hit_count", 0),
            "table_context_table_count": meta.get("table_context_table_count", 0),
            "table_context_char_count": meta.get("table_context_char_count", 0),
            "evidence_unit_count": meta.get("evidence_unit_count", 0),
            "evidence_units_with_tables": meta.get("evidence_units_with_tables", 0),
            "table_attached_count": meta.get("table_attached_count", 0),
            "table_attach_reasons": meta.get("table_attach_reasons", []) or [],
            "evidence_group_count": meta.get("evidence_group_count", 0),
            "selected_evidence_group_count": meta.get("selected_evidence_group_count", 0),
            "evidence_groups_debug": meta.get("evidence_groups_debug", []) or [],
            "selected_evidence_groups": meta.get("selected_evidence_groups", []) or [],
            "group_scores": meta.get("group_scores", []) or [],
            "expanded_snippet_count": meta.get("expanded_snippet_count", 0),
            "relevant_table_row_count": meta.get("relevant_table_row_count", 0),
            "dropped_group_reasons": meta.get("dropped_group_reasons", []) or [],
            "table_candidate_filenames": meta.get("table_candidate_filenames", []) or [],
            "table_candidate_pages": meta.get("table_candidate_pages", []) or [],
            "table_ids": meta.get("table_ids", []) or [],
            "table_context_skipped_reasons": meta.get("table_context_skipped_reasons", []) or [],
            "two_stage_retrieval": meta.get("two_stage_retrieval", False),
            "selected_docs": meta.get("selected_docs", []) or meta.get("doc_stage_selected_docs", []) or [],
            "selected_pages": meta.get("selected_pages", []) or [],
            "page_scores": meta.get("page_scores", []) or [],
            "final_retrieved_chunks": [
                _trace_chunk(doc, idx)
                for idx, doc in enumerate(result.get("final_retrieved_docs", []) or [], 1)
            ],
            "final_evidence_pack_debug": [
                _trace_chunk(doc, idx)
                for idx, doc in enumerate(meta.get("final_evidence_pack_debug", []) or [], 1)
            ],
            "final_evidence_pack_used": [
                _trace_chunk(doc, idx)
                for idx, doc in enumerate(meta.get("final_evidence_pack_used", []) or result.get("context_docs", []), 1)
            ],
            "latency_breakdown": meta.get("latency_breakdown", {}) or {},
        },
    }
