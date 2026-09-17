from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from PIL import Image, ImageDraw, ImageFont

from backend.document_loader import DocumentLoader
from backend.document_parsers.anydoc_parser import AnyDocParser
from backend.document_parsers.ocr_parser import OCRParser, needs_ocr
from backend.document_parsers.parser_registry import ParserRegistry
from backend.document_parsers.unified_document import to_langchain_documents


class FakePaddleOCR:
    def __init__(self, text: str):
        self.text = text
        self.inputs = []

    def predict(self, *, input):
        self.inputs.append(input)
        return [{"res": {"rec_texts": [self.text]}}]


class StaticLoader:
    def __init__(self, documents):
        self.documents = documents

    def load(self):
        return self.documents


def _font(size: int = 30):
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _image(path: Path, text: str) -> Path:
    image = Image.new("RGB", (900, 180), "white")
    ImageDraw.Draw(image).text((25, 60), text, fill="black", font=_font())
    image.save(path)
    return path


def _registry_with_ocr(parser: OCRParser) -> ParserRegistry:
    return ParserRegistry([AnyDocParser(converter=lambda _path: "# unused"), parser])


def test_english_image_ocr_returns_unified_and_langchain_documents(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OCR_ENABLED", "true")
    image_path = _image(tmp_path / "english.png", "Revenue was 125 million in 2025")
    engine = FakePaddleOCR("Revenue was 125 million in 2025")

    parsed = OCRParser(engine_factory=lambda: engine).parse(image_path)
    documents = to_langchain_documents(parsed)

    assert documents[0].page_content == "Revenue was 125 million in 2025"
    assert documents[0].metadata["parser_backend"] == "paddleocr"
    assert documents[0].metadata["page"] == 0
    assert documents[0].metadata["location"] == "image:0"
    assert engine.inputs == [str(image_path)]


def test_chinese_image_ocr_preserves_chinese_text(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OCR_ENABLED", "true")
    image_path = _image(tmp_path / "chinese.png", "二〇二五年营业收入一百二十五百万元")
    expected = "二〇二五年营业收入为一百二十五百万元"

    parsed = OCRParser(engine_factory=lambda: FakePaddleOCR(expected)).parse(image_path)

    assert parsed.units[0].text == expected
    assert parsed.parser_backend == "paddleocr"


def test_pdf_loader_ocr_replaces_only_low_text_page(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OCR_ENABLED", "true")
    monkeypatch.setenv("OCR_MIN_TEXT_THRESHOLD", "100")
    scan = _image(tmp_path / "scan.png", "Scanned current liabilities 200")
    scan_pdf = tmp_path / "scanned.pdf"
    Image.open(scan).save(scan_pdf, "PDF")
    original_text = "This is a native PDF page with enough searchable text. " * 5
    primary_documents = [
        Document(page_content=original_text, metadata={"source": str(scan_pdf), "page": 0}),
        Document(page_content="", metadata={"source": str(scan_pdf), "page": 1}),
    ]
    engine = FakePaddleOCR("Scanned current liabilities were 200 million")
    parser = OCRParser(
        engine_factory=lambda: engine,
        pdf_renderer=lambda _path, pages: {int(page): Image.new("RGB", (40, 40), "white") for page in pages},
        pdf_image_detector=lambda _path, _pages: set(),
    )
    wrapped = _registry_with_ocr(parser).with_pdf_ocr_fallback(
        StaticLoader(primary_documents), str(scan_pdf), scan_pdf.name
    )

    documents = wrapped.load()

    assert documents[0] is primary_documents[0]
    assert documents[0].page_content == original_text
    assert documents[1].page_content == "Scanned current liabilities were 200 million"
    assert documents[1].metadata["page"] == 1
    assert documents[1].metadata["parser_backend"] == "paddleocr"
    assert documents[1].metadata["ocr_fallback"] is True


def test_disabled_pdf_fallback_returns_original_loader(monkeypatch):
    monkeypatch.setenv("OCR_ENABLED", "false")
    primary = StaticLoader([])
    registry = _registry_with_ocr(OCRParser(engine_factory=lambda: FakePaddleOCR("unused")))

    assert registry.with_pdf_ocr_fallback(primary, "test.pdf", "test.pdf") is primary


def test_pdf_ocr_failure_preserves_original_documents(monkeypatch):
    monkeypatch.setenv("OCR_ENABLED", "true")
    original = [Document(page_content="", metadata={"source": "scan.pdf", "page": 0, "title": "scan"})]
    parser = OCRParser(
        engine_factory=lambda: (_ for _ in ()).throw(RuntimeError("model unavailable")),
        pdf_renderer=lambda _path, pages: {int(page): Image.new("RGB", (20, 20)) for page in pages},
        pdf_image_detector=lambda _path, _pages: {0},
    )
    wrapped = _registry_with_ocr(parser).with_pdf_ocr_fallback(StaticLoader(original), "scan.pdf", "scan.pdf")

    assert wrapped.load() == original


def test_ocr_image_continues_through_existing_chunk_pipeline(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OCR_ENABLED", "true")
    image_path = _image(tmp_path / "financial_scan.jpg", "Operating cash flow 300")
    parser = OCRParser(engine_factory=lambda: FakePaddleOCR("Operating cash flow was 300 million in 2025."))
    monkeypatch.setattr(
        "backend.document_loader.parser_registry",
        _registry_with_ocr(parser),
    )

    bundle = DocumentLoader(chunk_size=64, chunk_overlap=16, include_parent_chunks=False).load_document_bundle(
        str(image_path), image_path.name
    )

    assert bundle["chunks"]
    assert bundle["chunks"][0]["parser_backend"] == "paddleocr"
    assert bundle["chunks"][0]["location"] == "image:0"
    assert "Operating cash flow" in bundle["chunks"][0]["text"]
    assert bundle["chunks"][0]["document_id"]
    assert bundle["chunks"][0]["page_id"]


def test_low_text_detection_ignores_pdf_control_noise():
    assert needs_ocr("\x00 \n /Image /XObject", threshold=30) is True
    assert needs_ocr("Searchable financial statement text 2025 " * 5, threshold=30) is False
