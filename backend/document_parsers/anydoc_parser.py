"""Firecrawl AnyDoc adapter for Office documents and CSV files."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from .base import DocumentParser
from .unified_document import ParsedDocument, ParsedUnit, parsed_document

MarkdownConverter = Callable[[str], str]

_FILE_TYPES = {
    ".docx": "Word",
    ".xlsx": "Excel",
    ".pptx": "PowerPoint",
    ".csv": "CSV",
}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class AnyDocUnavailableError(RuntimeError):
    """Raised only when an AnyDoc-backed format is requested without AnyDoc."""


def _default_converter(path: str) -> str:
    try:
        from anydoc import to_markdown
    except ImportError as exc:  # pragma: no cover - depends on deployment extras
        raise AnyDocUnavailableError(
            "firecrawl-anydoc is required for DOCX/XLSX/PPTX/CSV ingestion"
        ) from exc
    return to_markdown(path, ocr="reject")


def _normalize_markdown(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict) and isinstance(value.get("markdown"), str):
        return value["markdown"]
    markdown = getattr(value, "markdown", None)
    if isinstance(markdown, str):
        return markdown
    raise TypeError(f"AnyDoc returned unsupported result type: {type(value).__name__}")


def _split_markdown(markdown: str, extension: str) -> list[ParsedUnit]:
    """Split on converter-emitted headings while retaining every source line."""

    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []
    for line in (markdown or "").splitlines():
        match = _HEADING_RE.match(line.strip())
        if match and current_lines:
            sections.append((current_title, current_lines))
            current_lines = []
        if match:
            current_title = match.group(2).strip()
        current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))

    units: list[ParsedUnit] = []
    for index, (section, lines) in enumerate(sections):
        text = "\n".join(lines).strip()
        if not text:
            continue
        if extension == ".pptx":
            location = f"slide:{index}"
        elif extension == ".xlsx":
            location = f"sheet:{section or index}"
        elif extension == ".csv":
            location = "table:0"
        else:
            location = f"section:{section or index}"
        units.append(
            ParsedUnit(
                text=text,
                page_number=index,
                location=location,
                section=section,
                metadata={"source_extension": extension},
            )
        )
    return units


class AnyDocParser(DocumentParser):
    """Convert supported files to Markdown-backed logical document units."""

    backend_name = "firecrawl-anydoc"
    supported_extensions = frozenset(_FILE_TYPES)

    def __init__(self, converter: MarkdownConverter | None = None):
        self._converter = converter or _default_converter

    def parse(self, file_path: str | Path, *, filename: str | None = None) -> ParsedDocument:
        path = Path(file_path)
        extension = path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"AnyDoc does not support this adapter extension: {extension}")
        markdown = _normalize_markdown(self._converter(str(path)))
        units = _split_markdown(markdown, extension)
        if not units:
            raise ValueError(f"AnyDoc produced no text for {filename or path.name}")
        return parsed_document(
            path,
            filename=filename,
            file_type=_FILE_TYPES[extension],
            parser_backend=self.backend_name,
            units=units,
        )

