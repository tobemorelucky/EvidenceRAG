"""Optional PaddleOCR parser for images and low-text PDF pages."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .base import DocumentParser
from .unified_document import ParsedDocument, ParsedUnit, parsed_document


IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})
OCR_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf"}


class OCRUnavailableError(RuntimeError):
    """Raised when OCR is requested but its optional runtime is unavailable."""


def ocr_enabled() -> bool:
    return os.getenv("OCR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}


def ocr_min_text_threshold() -> int:
    try:
        return max(0, int(os.getenv("OCR_MIN_TEXT_THRESHOLD", "100")))
    except ValueError:
        return 100


def meaningful_character_count(text: str) -> int:
    """Count visible language/number characters instead of PDF control noise."""

    return len(re.findall(r"[A-Za-z0-9\u3400-\u9fff]", text or ""))


def needs_ocr(text: str, threshold: int | None = None) -> bool:
    resolved = ocr_min_text_threshold() if threshold is None else max(0, int(threshold))
    return meaningful_character_count(text) < resolved


def _language() -> str:
    configured = os.getenv("OCR_LANG", "auto").strip().lower()
    # PaddleOCR's Chinese recognition model also recognizes English and digits,
    # making it the conservative local default for mixed-language documents.
    return "ch" if configured in {"", "auto", "zh", "zh-cn", "chinese"} else configured


def _default_engine_factory():
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:  # pragma: no cover - optional deployment dependency
        raise OCRUnavailableError(
            "PaddleOCR is not installed. Install the project's OCR optional dependency and a compatible inference engine."
        ) from exc
    return PaddleOCR(
        lang=_language(),
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )


def _as_payload(value: Any) -> Any:
    payload = getattr(value, "json", value)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        try:
            return __import__("json").loads(payload)
        except ValueError:
            return payload
    return payload


def _collect_texts(value: Any) -> list[str]:
    """Normalize PaddleOCR 3.x results while retaining 2.x compatibility."""

    value = _as_payload(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, dict):
        for key in ("rec_texts", "texts"):
            texts = value.get(key)
            if isinstance(texts, list):
                return [str(item).strip() for item in texts if str(item).strip()]
        if "res" in value:
            return _collect_texts(value["res"])
        collected: list[str] = []
        for nested in value.values():
            if isinstance(nested, (dict, list, tuple)):
                collected.extend(_collect_texts(nested))
        return collected
    if isinstance(value, (list, tuple)):
        # PaddleOCR 2.x leaf: [box, (text, confidence)].
        if (
            len(value) == 2
            and isinstance(value[1], (list, tuple))
            and value[1]
            and isinstance(value[1][0], str)
        ):
            text = value[1][0].strip()
            return [text] if text else []
        collected: list[str] = []
        for nested in value:
            collected.extend(_collect_texts(nested))
        return collected
    return []


def _render_pdf_pages(file_path: str | Path, page_numbers: Iterable[int]) -> dict[int, Any]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - required by the existing PDF path
        raise OCRUnavailableError("pypdfium2 is required to render scanned PDF pages") from exc

    requested = sorted(set(int(page) for page in page_numbers))
    pdf = pdfium.PdfDocument(str(file_path))
    try:
        rendered: dict[int, Any] = {}
        for page_number in requested:
            if page_number < 0 or page_number >= len(pdf):
                continue
            page = pdf[page_number]
            try:
                bitmap = page.render(scale=2.0)
                try:
                    rendered[page_number] = bitmap.to_pil().copy()
                finally:
                    close = getattr(bitmap, "close", None)
                    if callable(close):
                        close()
            finally:
                page.close()
        return rendered
    finally:
        pdf.close()


def _image_dominant_pdf_pages(file_path: str | Path, page_numbers: Iterable[int]) -> set[int]:
    """Detect pages where one embedded image covers most of the PDF canvas."""

    try:
        import pypdfium2 as pdfium
    except ImportError:
        return set()
    requested = sorted(set(int(page) for page in page_numbers))
    pdf = pdfium.PdfDocument(str(file_path))
    try:
        detected: set[int] = set()
        for page_number in requested:
            if page_number < 0 or page_number >= len(pdf):
                continue
            page = pdf[page_number]
            try:
                width, height = page.get_size()
                page_area = max(float(width) * float(height), 1.0)
                for image in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]):
                    left, bottom, right, top = image.get_bounds()
                    image_area = abs(float(right) - float(left)) * abs(float(top) - float(bottom))
                    if image_area / page_area >= 0.6:
                        detected.add(page_number)
                        break
            finally:
                page.close()
        return detected
    finally:
        pdf.close()


class OCRParser(DocumentParser):
    """PaddleOCR-backed parser with injectable boundaries for deterministic tests."""

    backend_name = "paddleocr"
    supported_extensions = OCR_EXTENSIONS

    def __init__(
        self,
        *,
        engine_factory: Callable[[], Any] | None = None,
        pdf_renderer: Callable[[str | Path, Iterable[int]], dict[int, Any]] | None = None,
        pdf_image_detector: Callable[[str | Path, Iterable[int]], set[int]] | None = None,
    ):
        self._engine_factory = engine_factory or _default_engine_factory
        self._pdf_renderer = pdf_renderer or _render_pdf_pages
        self._pdf_image_detector = pdf_image_detector or _image_dominant_pdf_pages
        self._engine: Any | None = None

    def _recognize(self, source: Any) -> str:
        if self._engine is None:
            self._engine = self._engine_factory()
        if hasattr(self._engine, "predict"):
            result = self._engine.predict(input=source)
        elif hasattr(self._engine, "ocr"):  # PaddleOCR 2.x compatibility
            result = self._engine.ocr(source, cls=True)
        elif callable(self._engine):
            result = self._engine(source)
        else:
            raise TypeError("Unsupported PaddleOCR engine interface")
        texts = _collect_texts(result)
        return "\n".join(texts).strip()

    def parse_pdf_pages(
        self,
        file_path: str | Path,
        page_numbers: Iterable[int],
        *,
        filename: str | None = None,
    ) -> ParsedDocument:
        path = Path(file_path)
        units: list[ParsedUnit] = []
        for page_number, image in self._pdf_renderer(path, page_numbers).items():
            text = self._recognize(image)
            if text:
                units.append(
                    ParsedUnit(
                        text=text,
                        page_number=int(page_number),
                        location=f"page:{int(page_number)}",
                        section="",
                        metadata={"ocr_fallback": True, "source_extension": ".pdf"},
                    )
                )
        return parsed_document(
            path,
            filename=filename,
            file_type="PDF",
            parser_backend=self.backend_name,
            units=units,
        )

    def image_dominant_pdf_pages(
        self,
        file_path: str | Path,
        page_numbers: Iterable[int],
    ) -> set[int]:
        return set(self._pdf_image_detector(file_path, page_numbers))

    def parse(self, file_path: str | Path, *, filename: str | None = None) -> ParsedDocument:
        if not ocr_enabled():
            raise RuntimeError("OCR ingestion is disabled; set OCR_ENABLED=true to enable it")
        path = Path(file_path)
        extension = path.suffix.lower()
        if extension not in self.supported_extensions:
            raise ValueError(f"PaddleOCR adapter does not support: {extension}")
        if extension == ".pdf":
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(str(path))
            try:
                page_numbers = range(len(pdf))
                return self.parse_pdf_pages(path, page_numbers, filename=filename)
            finally:
                pdf.close()
        text = self._recognize(str(path))
        units = (
            ParsedUnit(
                text=text,
                page_number=0,
                location="image:0",
                section="",
                metadata={"ocr_fallback": False, "source_extension": extension},
            ),
        ) if text else ()
        return parsed_document(
            path,
            filename=filename,
            file_type="Image",
            parser_backend=self.backend_name,
            units=units,
        )
