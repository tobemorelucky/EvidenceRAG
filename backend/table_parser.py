"""Optional table-aware PDF parser for future structured table ingestion."""

import csv
import io
import logging
import os
from typing import List

try:
    from evidence_identity import build_document_id, build_page_id, build_table_id, external_page_to_internal
    from table_config import get_table_aware_config
    from table_reconstructor import reconstruct_tables_from_words
    from table_quality import structural_quality_score
    from text_sanitizer import sanitize_text
except ModuleNotFoundError:
    from backend.evidence_identity import build_document_id, build_page_id, build_table_id, external_page_to_internal
    from backend.table_config import get_table_aware_config
    from backend.table_reconstructor import reconstruct_tables_from_words
    from backend.table_quality import structural_quality_score
    from backend.text_sanitizer import sanitize_text

logger = logging.getLogger(__name__)


class TableAwareParser:
    """Best-effort structured table extraction for PDF files."""

    @staticmethod
    def _build_table_id(filename: str, page_number: int, table_index: int) -> str:
        return f"{filename}::table::p{page_number}::{table_index}"

    @staticmethod
    def _load_docling():
        try:
            from docling.document_converter import DocumentConverter
        except Exception:
            return None
        return DocumentConverter

    @staticmethod
    def _load_docling_pdf_format_option():
        try:
            from docling.document_converter import PdfFormatOption
        except Exception:
            return None
        return PdfFormatOption

    @staticmethod
    def _load_docling_pipeline_options():
        try:
            from docling.datamodel.pipeline_options import PdfPipelineOptions
        except Exception:
            return None
        return PdfPipelineOptions

    @staticmethod
    def _load_pdfplumber():
        try:
            import pdfplumber
        except Exception:
            return None
        return pdfplumber

    @staticmethod
    def _normalize_cell(value) -> str:
        return sanitize_text("" if value is None else str(value)).strip()

    @classmethod
    def _normalize_matrix(cls, matrix) -> list[list[str]]:
        rows = []
        for row in matrix or []:
            normalized_row = [cls._normalize_cell(cell) for cell in (row or [])]
            if any(cell for cell in normalized_row):
                rows.append(normalized_row)
        return rows

    @staticmethod
    def _dedupe_columns(columns: list[str]) -> list[str]:
        deduped = []
        seen = {}
        for index, column in enumerate(columns, start=1):
            name = sanitize_text(column).strip() or f"column_{index}"
            count = seen.get(name, 0)
            if count:
                deduped.append(f"{name}_{count + 1}")
            else:
                deduped.append(name)
            seen[name] = count + 1
        return deduped

    @classmethod
    def _columns_and_rows_from_matrix(cls, matrix) -> tuple[list[str], list[dict]]:
        normalized = cls._normalize_matrix(matrix)
        if not normalized:
            return [], []

        width = max(len(row) for row in normalized)
        padded = [row + [""] * (width - len(row)) for row in normalized]
        if len(padded) >= 2 and any(cell for cell in padded[0]):
            columns = cls._dedupe_columns(padded[0])
            data_rows = padded[1:]
        else:
            columns = [f"column_{index}" for index in range(1, width + 1)]
            data_rows = padded

        rows = []
        for row in data_rows:
            row_payload = {columns[index]: row[index] for index in range(len(columns))}
            if any(value for value in row_payload.values()):
                rows.append(row_payload)
        return columns, rows

    @staticmethod
    def _build_csv_text(columns: list[str], rows: list[dict]) -> str:
        if not columns:
            return ""
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: sanitize_text(row.get(column, "")) for column in columns})
        return sanitize_text(buffer.getvalue()).strip()

    @staticmethod
    def _build_html(columns: list[str], rows: list[dict]) -> str:
        if not columns:
            return ""
        head = "".join(f"<th>{column}</th>" for column in columns)
        body_rows = []
        for row in rows:
            cells = "".join(f"<td>{sanitize_text(row.get(column, ''))}</td>" for column in columns)
            body_rows.append(f"<tr>{cells}</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"

    @staticmethod
    def _set_docling_option_if_present(options, names: list[str], value) -> bool:
        for name in names:
            if hasattr(options, name):
                setattr(options, name, value)
                return True
        return False

    def _build_docling_converter(self, *, docling_ocr: bool, timeout_seconds: int):
        document_converter_cls = self._load_docling()
        if document_converter_cls is None:
            return None

        pipeline_options_cls = self._load_docling_pipeline_options()
        pdf_format_option_cls = self._load_docling_pdf_format_option()
        pipeline_options = None
        ocr_control_applied = docling_ocr

        if pipeline_options_cls is not None:
            try:
                pipeline_options = pipeline_options_cls()
                ocr_configured = self._set_docling_option_if_present(pipeline_options, ["do_ocr", "ocr_enabled"], docling_ocr)
                ocr_control_applied = ocr_configured
                timeout_configured = self._set_docling_option_if_present(
                    pipeline_options,
                    ["document_timeout", "timeout", "timeout_seconds"],
                    timeout_seconds,
                )
                if not ocr_configured:
                    logger.warning("docling pipeline options do not expose OCR switch; OCR behavior may use library defaults")
                if not timeout_configured:
                    logger.warning("docling pipeline options do not expose timeout setting; timeout config was ignored")
            except Exception:
                logger.exception("failed to initialize docling pipeline options")
                pipeline_options = None

        if not docling_ocr and not ocr_control_applied:
            logger.warning("docling OCR disable setting could not be applied safely; skipping docling to avoid implicit OCR")
            return None

        if pdf_format_option_cls is not None and pipeline_options is not None:
            try:
                return document_converter_cls(format_options={"pdf": pdf_format_option_cls(pipeline_options=pipeline_options)})
            except Exception:
                logger.exception("failed to initialize docling converter with PdfFormatOption")

        if pipeline_options is not None:
            for kwargs in ({"pipeline_options": pipeline_options}, {"pdf_pipeline_options": pipeline_options}):
                try:
                    return document_converter_cls(**kwargs)
                except TypeError:
                    continue
                except Exception:
                    logger.exception("failed to initialize docling converter with pipeline options")
                    break

        try:
            return document_converter_cls()
        except Exception:
            logger.exception("failed to initialize docling converter")
            return None

    def _extract_with_docling(
        self,
        file_path: str,
        filename: str,
        *,
        docling_ocr: bool,
        timeout_seconds: int,
        max_pages: int | None,
    ) -> list[dict]:
        converter = self._build_docling_converter(docling_ocr=docling_ocr, timeout_seconds=timeout_seconds)
        if converter is None:
            return []

        try:
            result = converter.convert(file_path)
            document = getattr(result, "document", result)
            tables = []
            for raw_table in getattr(document, "tables", []) or []:
                page_number = int(getattr(raw_table, "page_no", 0) or getattr(raw_table, "page_number", 0) or 0)
                if max_pages is not None and page_number > max_pages:
                    continue
                table_index = len(tables) + 1
                data = getattr(raw_table, "data", None)
                if hasattr(data, "to_dict"):
                    data = data.to_dict(orient="records")
                if isinstance(data, list) and data and isinstance(data[0], dict):
                    columns = self._dedupe_columns([sanitize_text(item) for item in data[0].keys()])
                    rows = [
                        {column: self._normalize_cell(row.get(column, "")) for column in columns}
                        for row in data
                    ]
                else:
                    columns, rows = self._columns_and_rows_from_matrix(data)
                if not columns and not rows:
                    continue
                tables.append(
                    {
                        "table_id": self._build_table_id(filename, page_number, table_index),
                        "filename": filename,
                        "doc_name": os.path.splitext(filename)[0],
                        "file_type": "PDF",
                        "file_path": file_path,
                        "page_number": page_number,
                        "table_index": table_index,
                        "title": self._normalize_cell(getattr(raw_table, "title", "")),
                        "caption": self._normalize_cell(getattr(raw_table, "caption", "")),
                        "before_context": "",
                        "after_context": "",
                        "columns": columns,
                        "rows": rows,
                        "html": self._build_html(columns, rows),
                        "csv_text": self._build_csv_text(columns, rows),
                        "parser_backend": "docling",
                    }
                )
            return tables
        except Exception:
            logger.exception("docling table parsing failed filename=%s", filename)
            return []

    def _extract_with_pdfplumber(self, file_path: str, filename: str, *, max_pages: int | None) -> list[dict]:
        pdfplumber = self._load_pdfplumber()
        if pdfplumber is None:
            return []

        tables = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_number, page in enumerate(getattr(pdf, "pages", []) or [], start=1):
                    if max_pages is not None and page_number > max_pages:
                        break
                    page_tables = page.extract_tables() or []
                    for table_index, matrix in enumerate(page_tables, start=1):
                        columns, rows = self._columns_and_rows_from_matrix(matrix)
                        if not columns and not rows:
                            continue
                        tables.append(
                            {
                                "table_id": self._build_table_id(filename, page_number, table_index),
                                "filename": filename,
                                "doc_name": os.path.splitext(filename)[0],
                                "file_type": "PDF",
                                "file_path": file_path,
                                "page_number": page_number,
                                "table_index": table_index,
                                "title": "",
                                "caption": "",
                                "before_context": "",
                                "after_context": "",
                                "columns": columns,
                                "rows": rows,
                                "html": self._build_html(columns, rows),
                                "csv_text": self._build_csv_text(columns, rows),
                                "parser_backend": "pdfplumber",
                            }
                        )
            return tables
        except Exception:
            logger.exception("pdfplumber table parsing failed filename=%s", filename)
            return []

    def _extract_with_pdfplumber_words(self, file_path: str, filename: str, *, max_pages: int | None) -> list[dict]:
        return reconstruct_tables_from_words(file_path, filename, max_pages=max_pages)

    def _extract_with_pdfplumber_words_debug(
        self,
        file_path: str,
        filename: str,
        *,
        max_pages: int | None,
        include_rejected: bool,
    ) -> list[dict]:
        return reconstruct_tables_from_words(
            file_path,
            filename,
            max_pages=max_pages,
            include_rejected=include_rejected,
        )

    def extract_tables(
        self,
        file_path: str,
        filename: str,
        *,
        parser_backend: str | None = None,
        docling_ocr: bool | None = None,
        timeout_seconds: int | None = None,
        max_pages: int | None = None,
        include_rejected: bool = False,
    ) -> List[dict]:
        config = get_table_aware_config()
        if not config.table_aware_ingestion:
            return []
        if not (filename or "").lower().endswith(".pdf"):
            return []

        selected_backend = parser_backend or config.table_parser_backend
        if selected_backend not in {"auto", "pdfplumber", "pdfplumber_words", "docling"}:
            selected_backend = "auto"
        effective_docling_ocr = config.table_docling_ocr if docling_ocr is None else bool(docling_ocr)
        effective_timeout_seconds = config.table_docling_timeout_seconds if timeout_seconds is None else max(1, int(timeout_seconds))
        effective_max_pages = None if max_pages is None else max(1, int(max_pages))

        tables = []
        resolved_backend = "none"

        if selected_backend == "pdfplumber":
            tables = self._extract_with_pdfplumber(file_path, filename, max_pages=effective_max_pages)
            resolved_backend = "pdfplumber" if tables else "none"
        elif selected_backend == "pdfplumber_words":
            tables = self._extract_with_pdfplumber_words_debug(
                file_path,
                filename,
                max_pages=effective_max_pages,
                include_rejected=include_rejected,
            )
            resolved_backend = "pdfplumber_words" if tables else "none"
        elif selected_backend == "docling":
            tables = self._extract_with_docling(
                file_path,
                filename,
                docling_ocr=effective_docling_ocr,
                timeout_seconds=effective_timeout_seconds,
                max_pages=effective_max_pages,
            )
            resolved_backend = "docling" if tables else "none"
        else:
            tables = self._extract_with_docling(
                file_path,
                filename,
                docling_ocr=effective_docling_ocr,
                timeout_seconds=effective_timeout_seconds,
                max_pages=effective_max_pages,
            )
            resolved_backend = "docling" if tables else "none"
            if not tables:
                tables = self._extract_with_pdfplumber(file_path, filename, max_pages=effective_max_pages)
                resolved_backend = "pdfplumber" if tables else "none"

        logger.info(
            "table parsing completed filename=%s tables=%s parser=%s selected_backend=%s docling_ocr=%s timeout_seconds=%s max_pages=%s",
            filename,
            len(tables),
            resolved_backend,
            selected_backend,
            effective_docling_ocr,
            effective_timeout_seconds,
            effective_max_pages,
        )
        document_id = build_document_id(file_path=file_path, filename=filename)
        for position, table in enumerate(tables, start=1):
            internal_page = external_page_to_internal(table.get("page_number", 1), external_base=1)
            external_start = table.get("start_page", table.get("page_number", 1))
            external_end = table.get("end_page", table.get("page_number", 1))
            start_page = external_page_to_internal(external_start, external_base=1)
            end_page = max(start_page, external_page_to_internal(external_end, external_base=1))
            page_id = build_page_id(document_id, internal_page)
            table_index = max(1, int(table.get("table_index") or position))
            table["table_index"] = table_index
            table.update({
                "document_id": document_id,
                "page_id": page_id,
                "page_number": internal_page,
                "start_page": start_page,
                "end_page": end_page,
                "table_id": build_table_id(page_id, table_index),
                "parser_backend": table.get("parser_backend") or resolved_backend,
                "quality_score": float(table.get("quality_score") or structural_quality_score(table)),
            })
        return tables
