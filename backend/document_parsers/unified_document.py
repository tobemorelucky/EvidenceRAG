"""Neutral parser output converted to LangChain documents at the boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document


@dataclass(frozen=True, slots=True)
class ParsedUnit:
    """One addressable unit, such as a section, worksheet, or slide."""

    text: str
    location: str
    section: str = ""
    page_number: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Parser-independent representation of a source document."""

    source: str
    filename: str
    file_type: str
    parser_backend: str
    units: tuple[ParsedUnit, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def to_langchain_documents(parsed: ParsedDocument) -> list[Document]:
    """Convert neutral units without discarding parser-provided metadata."""

    documents: list[Document] = []
    for unit in parsed.units:
        text = (unit.text or "").strip()
        if not text:
            continue
        metadata = {
            **parsed.metadata,
            **unit.metadata,
            "source": parsed.source,
            "filename": parsed.filename,
            "file_type": parsed.file_type,
            "page": int(unit.page_number),
            "parser_backend": parsed.parser_backend,
            "location": unit.location,
            "section": unit.section,
        }
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def parsed_document(
    source: str | Path,
    *,
    file_type: str,
    parser_backend: str,
    units: Iterable[ParsedUnit],
    filename: str | None = None,
) -> ParsedDocument:
    """Small constructor that consistently normalizes source paths."""

    source_path = Path(source)
    return ParsedDocument(
        source=str(source_path),
        filename=filename or source_path.name,
        file_type=file_type,
        parser_backend=parser_backend,
        units=tuple(units),
    )
