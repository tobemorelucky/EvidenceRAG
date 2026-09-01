"""Document loading and hierarchical chunking utilities."""

import logging
import os
import re
import hashlib
from typing import Dict, List

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

try:
    from evidence_identity import build_document_id, build_page_id
    from table_parser import TableAwareParser
    from text_sanitizer import sanitize_text
except ModuleNotFoundError:
    from backend.evidence_identity import build_document_id, build_page_id
    from backend.table_parser import TableAwareParser
    from backend.text_sanitizer import sanitize_text

logger = logging.getLogger(__name__)


class _PdfiumLoader:
    """Fast local text-only PDF loader for indexing.

    pypdf is convenient but spends a disproportionate amount of benchmark
    rebuild time in Python.  PDFium performs the same page-text extraction in
    native code and returns the page metadata shape expected by the existing
    loader pipeline.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> list[Document]:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(self.file_path)
        try:
            documents = []
            for page_index in range(len(pdf)):
                page = pdf[page_index]
                try:
                    text = page.get_textpage().get_text_range()
                finally:
                    page.close()
                documents.append(Document(page_content=text, metadata={"source": self.file_path, "page": page_index}))
            return documents
        finally:
            pdf.close()


class DocumentLoader:
    """Load documents and split them into three chunk levels."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 128,
        include_parent_chunks: bool = True,
    ):
        # Keep the original constructor shape for compatibility.
        level_1_size = max(1600, chunk_size * 2)
        level_1_overlap = max(256, chunk_overlap * 2)
        level_2_size = max(800, chunk_size)
        level_2_overlap = max(128, chunk_overlap)
        level_3_size = max(768, chunk_size)
        level_3_overlap = max(128, chunk_overlap)

        separators = ["\n\n", "\n", ".", ";", ",", " ", ""]
        self._splitter_level_1 = RecursiveCharacterTextSplitter(
            chunk_size=level_1_size,
            chunk_overlap=level_1_overlap,
            add_start_index=True,
            separators=separators,
            length_function=self._token_count,
        )
        self._splitter_level_2 = RecursiveCharacterTextSplitter(
            chunk_size=level_2_size,
            chunk_overlap=level_2_overlap,
            add_start_index=True,
            separators=separators,
            length_function=self._token_count,
        )
        self._splitter_level_3 = RecursiveCharacterTextSplitter(
            chunk_size=level_3_size,
            chunk_overlap=level_3_overlap,
            add_start_index=True,
            separators=separators,
            length_function=self._token_count,
        )
        self._include_parent_chunks = include_parent_chunks
        self._table_parser = TableAwareParser()

    @staticmethod
    def _token_count(text: str) -> int:
        """Stable finance-aware token estimate used without a remote tokenizer."""
        return len(re.findall(r"[$€£¥]?\d[\d,]*(?:\.\d+)?%?|[A-Za-z]+(?:[-'][A-Za-z]+)*|[\u4e00-\u9fff]|\S", text or ""))

    @staticmethod
    def _financial_metadata(filename: str) -> dict:
        stem = os.path.splitext(filename)[0]
        year_match = re.search(r"(?:19|20)\d{2}", stem)
        upper = stem.upper()
        if "10K" in upper:
            document_type = "10-K"
        elif "10Q" in upper:
            document_type = "10-Q"
        elif "EARNINGS" in upper:
            document_type = "earnings"
        else:
            document_type = "financial_document"
        company = re.split(r"_(?:19|20)\d{2}", stem, maxsplit=1)[0]
        return {
            "company": company,
            "report_year": int(year_match.group()) if year_match else 0,
            "financial_document_type": document_type,
        }

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()

    @staticmethod
    def _table_ingestion_enabled() -> bool:
        return os.getenv("TABLE_AWARE_INGESTION", "false").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _build_chunk_id(filename: str, page_number: int, level: int, index: int) -> str:
        return f"{filename}::p{page_number}::l{level}::{index}"

    @staticmethod
    def _resolve_doc_type_and_loader(file_path: str, filename: str):
        file_lower = filename.lower()
        if file_lower.endswith(".pdf"):
            if os.getenv("PDF_TEXT_BACKEND", "pdfium").strip().lower() == "pdfium":
                try:
                    import pypdfium2  # noqa: F401

                    return "PDF", _PdfiumLoader(file_path)
                except ImportError:
                    logger.warning("pypdfium2 unavailable; falling back to PyPDFLoader")
            return "PDF", PyPDFLoader(file_path)
        if file_lower.endswith((".docx", ".doc")):
            return "Word", Docx2txtLoader(file_path)
        if file_lower.endswith((".xlsx", ".xls")):
            return "Excel", UnstructuredExcelLoader(file_path)
        if file_lower.endswith(".txt"):
            return "Text", TextLoader(file_path, autodetect_encoding=True)
        if file_lower.endswith(".md"):
            return "Markdown", TextLoader(file_path, autodetect_encoding=True)
        if file_lower.endswith(".csv"):
            return "CSV", CSVLoader(file_path, autodetect_encoding=True)
        raise ValueError(f"Unsupported file type: {filename}")

    @staticmethod
    def _resolve_page_number(metadata: Dict) -> int:
        if not metadata:
            return 1
        if metadata.get("page") is not None:
            return int(metadata.get("page") or 0)
        if metadata.get("row") is not None:
            return int(metadata.get("row") or 0) + 1
        return 1

    @staticmethod
    def _extract_table_text(text: str) -> str:
        lines = [line.strip() for line in sanitize_text(text).splitlines() if line.strip()]
        table_like = [
            line
            for line in lines
            if "|" in line
            or "\t" in line
            or re.search(r"\d", line) and re.search(r"\s{2,}", line)
        ]
        return sanitize_text("\n".join(table_like)).strip()

    def _split_page_to_three_levels(
        self,
        text: str,
        base_doc: Dict,
        page_global_chunk_idx: int,
    ) -> List[Dict]:
        text = sanitize_text(text).strip()
        if not text:
            return []

        root_chunks: List[Dict] = []
        page_number = int(base_doc.get("page_number", 0))
        filename = base_doc["filename"]

        # Finance retrieval uses page discovery followed by leaf-chunk search.
        # When auto-merging is disabled, producing levels 1/2 only duplicates
        # parsing and database work. Keep the hierarchy for legacy callers, but
        # let the benchmark builder generate the actual retrieval unit directly.
        if not self._include_parent_chunks:
            leaf_docs = self._splitter_level_3.create_documents([text], [base_doc])
            return [
                {
                    **base_doc,
                    "text": sanitize_text(leaf_doc.page_content).strip(),
                    "chunk_id": self._build_chunk_id(filename, page_number, 3, index),
                    "parent_chunk_id": "",
                    "root_chunk_id": "",
                    "chunk_level": 3,
                    "chunk_idx": page_global_chunk_idx + index,
                }
                for index, leaf_doc in enumerate(leaf_docs)
                if sanitize_text(leaf_doc.page_content).strip()
            ]

        level_1_docs = self._splitter_level_1.create_documents([text], [base_doc])
        level_1_counter = 0
        level_2_counter = 0
        level_3_counter = 0

        for level_1_doc in level_1_docs:
            level_1_text = sanitize_text(level_1_doc.page_content).strip()
            if not level_1_text:
                continue
            level_1_id = self._build_chunk_id(filename, page_number, 1, level_1_counter)
            level_1_counter += 1

            level_1_chunk = {
                **base_doc,
                "text": level_1_text,
                "chunk_id": level_1_id,
                "parent_chunk_id": "",
                "root_chunk_id": level_1_id,
                "chunk_level": 1,
                "chunk_idx": page_global_chunk_idx,
            }
            page_global_chunk_idx += 1
            root_chunks.append(level_1_chunk)

            level_2_docs = self._splitter_level_2.create_documents([level_1_text], [base_doc])
            for level_2_doc in level_2_docs:
                level_2_text = sanitize_text(level_2_doc.page_content).strip()
                if not level_2_text:
                    continue
                level_2_id = self._build_chunk_id(filename, page_number, 2, level_2_counter)
                level_2_counter += 1

                level_2_chunk = {
                    **base_doc,
                    "text": level_2_text,
                    "chunk_id": level_2_id,
                    "parent_chunk_id": level_1_id,
                    "root_chunk_id": level_1_id,
                    "chunk_level": 2,
                    "chunk_idx": page_global_chunk_idx,
                }
                page_global_chunk_idx += 1
                root_chunks.append(level_2_chunk)

                level_3_docs = self._splitter_level_3.create_documents([level_2_text], [base_doc])
                for level_3_doc in level_3_docs:
                    level_3_text = sanitize_text(level_3_doc.page_content).strip()
                    if not level_3_text:
                        continue
                    level_3_id = self._build_chunk_id(filename, page_number, 3, level_3_counter)
                    level_3_counter += 1
                    root_chunks.append(
                        {
                            **base_doc,
                            "text": level_3_text,
                            "chunk_id": level_3_id,
                            "parent_chunk_id": level_2_id,
                            "root_chunk_id": level_1_id,
                            "chunk_level": 3,
                            "chunk_idx": page_global_chunk_idx,
                        }
                    )
                    page_global_chunk_idx += 1

        return root_chunks

    def load_document_bundle(self, file_path: str, filename: str) -> dict:
        """Load one document and return hierarchical chunks plus page-level records."""
        doc_type, loader = self._resolve_doc_type_and_loader(file_path, filename)

        try:
            raw_docs = loader.load()
            document_id = build_document_id(file_path=file_path, filename=filename)
            documents = []
            pages = []
            page_global_chunk_idx = 0
            for doc in raw_docs:
                page_number = self._resolve_page_number(doc.metadata)
                page_text = sanitize_text(doc.page_content).strip()
                base_doc = {
                    "document_id": document_id,
                    "page_id": build_page_id(document_id, page_number),
                    "filename": filename,
                    "file_path": file_path,
                    "file_type": doc_type,
                    "page_number": page_number,
                    "location": f"page:{page_number}",
                    **self._financial_metadata(filename),
                }
                page_chunks = self._split_page_to_three_levels(
                    text=page_text,
                    base_doc=base_doc,
                    page_global_chunk_idx=page_global_chunk_idx,
                )
                page_global_chunk_idx += len(page_chunks)
                for chunk in page_chunks:
                    chunk["content_hash"] = self._content_hash(chunk.get("text", ""))
                documents.extend(page_chunks)
                pages.append(
                    {
                        "doc_name": os.path.splitext(filename)[0],
                        "document_id": document_id,
                        "page_id": build_page_id(document_id, page_number),
                        "filename": filename,
                        "file_type": doc_type,
                        "file_path": file_path,
                        "page_number": page_number,
                        "location": f"page:{page_number}",
                        "page_text": page_text,
                        "content_hash": self._content_hash(page_text),
                        **self._financial_metadata(filename),
                        "table_text": self._extract_table_text(page_text),
                        "chunk_ids": [
                            chunk.get("chunk_id", "")
                            for chunk in page_chunks
                            if int(chunk.get("chunk_level", 0) or 0) == 3 and chunk.get("chunk_id")
                        ],
                    }
                )
            tables = []
            if doc_type == "PDF" and self._table_ingestion_enabled():
                try:
                    tables = self._table_parser.extract_tables(file_path, filename)
                except Exception:
                    logger.exception("table parsing failed filename=%s", filename)
                    tables = []
            return {"chunks": documents, "pages": pages, "tables": tables}
        except Exception as e:
            raise Exception(f"Failed to process document: {str(e)}")

    def load_document(self, file_path: str, filename: str) -> list[dict]:
        """Load one document and split it into chunks."""
        return self.load_document_bundle(file_path, filename).get("chunks", [])

    def load_documents_from_folder(self, folder_path: str) -> list[dict]:
        """Load every supported document from a folder."""
        all_documents = []

        for filename in os.listdir(folder_path):
            file_lower = filename.lower()
            if not (
                file_lower.endswith(".pdf")
                or file_lower.endswith((".docx", ".doc"))
                or file_lower.endswith((".xlsx", ".xls"))
                or file_lower.endswith((".txt", ".md", ".csv"))
            ):
                continue

            file_path = os.path.join(folder_path, filename)
            try:
                documents = self.load_document(file_path, filename)
                all_documents.extend(documents)
            except Exception:
                continue

        return all_documents
