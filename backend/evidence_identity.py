"""Stable identifiers and the internal zero-based page-number contract."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_document_id(
    *,
    file_path: str = "",
    filename: str = "",
    content_digest: str = "",
) -> str:
    """Build a stable document identifier, preferring the PDF content digest."""
    digest = str(content_digest or "").strip().lower()
    path = Path(file_path) if file_path else None
    if not digest and path is not None and path.is_file():
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(block)
        digest = hasher.hexdigest()
    if not digest:
        digest = _digest(str(filename or "").strip().casefold().encode("utf-8"))
    return f"doc_{_digest(digest.encode('ascii', errors='ignore'))[:60]}"


def build_page_id(document_id: str, page_number: int) -> str:
    page = int(page_number)
    if page < 0:
        raise ValueError("internal page_number must be zero-based and non-negative")
    document = str(document_id or "").strip()
    if not document:
        raise ValueError("document_id is required")
    return f"{document}:page:{page:06d}"


def build_table_id(page_id: str, table_index: int) -> str:
    page = str(page_id or "").strip()
    if not page:
        raise ValueError("page_id is required")
    index = int(table_index)
    if index < 1:
        raise ValueError("table_index must be positive")
    return f"{page}:table:{index:04d}"


def external_page_to_internal(page_number: int, *, external_base: int = 1) -> int:
    """Convert an external parser page number exactly once at its boundary."""
    page = int(page_number) - int(external_base)
    if page < 0:
        return 0
    return page
