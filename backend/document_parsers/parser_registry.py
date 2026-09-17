"""Extension-based parser registry and LangChain loader compatibility shim."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document

from .anydoc_parser import AnyDocParser
from .base import DocumentParser
from .ocr_parser import IMAGE_EXTENSIONS, OCRParser, needs_ocr, ocr_enabled
from .unified_document import to_langchain_documents


logger = logging.getLogger(__name__)


class _ParserLoaderAdapter:
    """Expose the existing loader ``load()`` contract to DocumentLoader."""

    def __init__(self, parser: DocumentParser, file_path: str, filename: str):
        self._parser = parser
        self._file_path = file_path
        self._filename = filename

    def load(self):
        parsed = self._parser.parse(self._file_path, filename=self._filename)
        return to_langchain_documents(parsed)


class _PdfOCRFallbackLoader:
    """Run the existing PDF loader first and OCR only its low-text pages."""

    def __init__(self, primary_loader, parser: OCRParser, file_path: str, filename: str):
        self._primary_loader = primary_loader
        self._parser = parser
        self._file_path = file_path
        self._filename = filename

    def load(self):
        documents = self._primary_loader.load()
        page_numbers = [
            int(document.metadata.get("page", index))
            for index, document in enumerate(documents)
        ]
        low_text_pages = {
            int(document.metadata.get("page", index))
            for index, document in enumerate(documents)
            if needs_ocr(document.page_content)
        }
        try:
            image_dominant_pages = self._parser.image_dominant_pdf_pages(self._file_path, page_numbers)
        except Exception:
            logger.exception("PDF image-dominance detection failed filename=%s", self._filename)
            image_dominant_pages = set()
        ocr_pages = sorted(low_text_pages | image_dominant_pages)
        if not ocr_pages:
            return documents
        try:
            parsed = self._parser.parse_pdf_pages(
                self._file_path,
                ocr_pages,
                filename=self._filename,
            )
        except Exception:
            # OCR is an optional fallback: a model/runtime failure must not
            # break the pre-existing text extraction path.
            logger.exception("PDF OCR fallback failed filename=%s", self._filename)
            return documents
        replacements = {
            int(document.metadata["page"]): document
            for document in to_langchain_documents(parsed)
        }
        merged = []
        for index, document in enumerate(documents):
            page_number = int(document.metadata.get("page", index))
            replacement = replacements.get(page_number)
            if replacement is None:
                merged.append(document)
                continue
            merged.append(
                Document(
                    page_content=replacement.page_content,
                    metadata={**document.metadata, **replacement.metadata},
                )
            )
        return merged


class ParserRegistry:
    def __init__(self, parsers: list[DocumentParser] | None = None):
        self._parsers = tuple(parsers or [AnyDocParser(), OCRParser()])

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
            **{extension: "Image" for extension in IMAGE_EXTENSIONS},
        }[Path(filename).suffix.lower()]
        return file_type, _ParserLoaderAdapter(parser, file_path, filename)

    def with_pdf_ocr_fallback(self, primary_loader, file_path: str, filename: str):
        if not ocr_enabled():
            return primary_loader
        parser = next((item for item in self._parsers if isinstance(item, OCRParser)), None)
        if parser is None:
            return primary_loader
        return _PdfOCRFallbackLoader(primary_loader, parser, file_path, filename)


parser_registry = ParserRegistry()
