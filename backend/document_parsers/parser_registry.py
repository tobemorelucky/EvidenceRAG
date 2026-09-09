"""Extension-based parser registry and LangChain loader compatibility shim."""

from __future__ import annotations

from pathlib import Path

from .anydoc_parser import AnyDocParser
from .base import DocumentParser
from .unified_document import to_langchain_documents


class _ParserLoaderAdapter:
    """Expose the existing loader ``load()`` contract to DocumentLoader."""

    def __init__(self, parser: DocumentParser, file_path: str, filename: str):
        self._parser = parser
        self._file_path = file_path
        self._filename = filename

    def load(self):
        parsed = self._parser.parse(self._file_path, filename=self._filename)
        return to_langchain_documents(parsed)


class ParserRegistry:
    def __init__(self, parsers: list[DocumentParser] | None = None):
        self._parsers = tuple(parsers or [AnyDocParser()])

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(extension for parser in self._parsers for extension in parser.supported_extensions)

    def parser_for(self, filename: str) -> DocumentParser | None:
        extension = Path(filename).suffix.lower()
        return next((parser for parser in self._parsers if extension in parser.supported_extensions), None)

    def loader_for(self, file_path: str, filename: str):
        parser = self.parser_for(filename)
        if parser is None:
            return None
        file_type = {
            ".docx": "Word",
            ".xlsx": "Excel",
            ".pptx": "PowerPoint",
            ".csv": "CSV",
        }[Path(filename).suffix.lower()]
        return file_type, _ParserLoaderAdapter(parser, file_path, filename)


parser_registry = ParserRegistry()
