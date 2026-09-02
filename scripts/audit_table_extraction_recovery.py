"""Offline extraction-recovery audit for the six post-contract missing cases."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from database import SessionLocal  # noqa: E402
from models import DocumentPage, DocumentTable, ParentChunk  # noqa: E402
from table_reconstructor import _extract_page_tables  # noqa: E402


DEFAULT_ASSOCIATION_AUDIT = ROOT / "reports" / "table_association_audit_diagnostic14.json"
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_JSON = ROOT / "reports" / "table_extraction_recovery_audit6.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "table_extraction_recovery_audit6.md"
DEFAULT_ARTIFACT_DIR = ROOT / "output" / "pdf" / "table_extraction_recovery_audit6"
NUMBER_RE = re.compile(r"(?:[$€£]\s*)?\(?-?\d[\d,]*(?:\.\d+)?%?\)?")
SPACE_RE = re.compile(r"\s+")
CLASS_LABELS = {
    "A": "parser漏检",
    "B": "复杂布局",
    "C": "OCR/图片",
    "D": "表格结构已丢失但文本存在",
}


def _clean(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "")).strip()


def _filename(value: Any) -> str:
    name = _clean(value).replace("\\", "/").rsplit("/", 1)[-1]
    if name and not name.casefold().endswith(".pdf"):
        name += ".pdf"
    return name.casefold()


def _preview(value: Any, limit: int = 600) -> str:
    text = _clean(value)
    return text[:limit] + ("…" if len(text) > limit else "")


def _table_like_lines(text: str) -> int:
    return sum(len(NUMBER_RE.findall(line)) >= 2 for line in str(text or "").splitlines())


def _tokens(value: str) -> set[str]:
    return {token.casefold().replace(",", "") for token in re.findall(r"[a-z0-9][a-z0-9,.%-]*", value or "", re.I)}


def _token_recall(needle: str, haystack: str) -> float:
    wanted = _tokens(needle)
    if not wanted:
        return 0.0
    return round(len(wanted & _tokens(haystack)) / len(wanted), 4)


def _prose_shape(text: str) -> dict:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    word_counts = [len(re.findall(r"[a-z]+", line, re.I)) for line in lines]
    average_words = sum(word_counts) / max(1, len(word_counts))
    return {
        "evidence_line_count": len(lines),
        "evidence_average_words_per_line": round(average_words, 2),
        "evidence_prose_like": len(lines) <= 15 and average_words >= 5.0,
    }


def classify_recovery(signals: dict) -> tuple[str, str]:
    """Classify a missing stored table from deterministic extraction signals."""
    if int(signals.get("image_count") or 0) > 0 and int(signals.get("raw_text_chars") or 0) < 150:
        return "C", "page is image-dominant and exposes too little extractable text for the current non-OCR parser"
    if signals.get("evidence_prose_like") and float(signals.get("evidence_text_recall_in_page") or 0.0) >= 0.60:
        return "D", "benchmark evidence is prose and is retained by text extraction; rejected table candidates are unrelated page regions"
    accepted = int(signals.get("accepted_word_candidates") or 0)
    rejected = int(signals.get("rejected_word_candidates") or 0)
    if accepted > 0:
        return "A", "current parser can produce an accepted candidate, but no table is stored for this page"
    if rejected > 0 or int(signals.get("max_effective_columns") or 0) >= 8:
        return "B", "parser sees candidate regions but rejects or fragments them because of layout complexity"
    if int(signals.get("table_like_text_lines") or 0) >= 3:
        return "D", "table values survive in extracted text, but no structured candidate survives"
    if int(signals.get("image_count") or 0) > 0:
        return "C", "page contains images and no usable structured/text table candidate"
    return "D", "structured candidate is absent; only unstructured page text remains"


def select_failure_pages(payload: dict) -> list[dict]:
    failures = []
    for case in payload.get("cases") or []:
        for audit in case.get("gold_page_audits") or []:
            post = audit.get("post_alignment") or {}
            if post.get("status") != "still_missing_after_page_boundary_fix":
                continue
            failures.append({
                "financebench_id": case.get("financebench_id"),
                "question": case.get("question"),
                "filename": _filename(case.get("gold_document")),
                "page_number": int(audit.get("benchmark_evidence_page_number")),
                "previous_classification": post.get("classification"),
            })
    return failures


def _jsonable_candidate(candidate: dict) -> dict:
    return {
        "table_id": candidate.get("table_id"),
        "page_number_external": candidate.get("page_number"),
        "table_index": candidate.get("table_index"),
        "accepted": bool(candidate.get("accepted")),
        "reject_reason": candidate.get("reject_reason") or "",
        "quality_score": candidate.get("quality_score"),
        "effective_col_count": candidate.get("effective_col_count"),
        "data_row_count": candidate.get("data_row_count"),
        "numeric_cell_ratio": candidate.get("numeric_cell_ratio"),
        "non_empty_cell_ratio": candidate.get("non_empty_cell_ratio"),
        "columns": candidate.get("columns") or [],
        "rows": candidate.get("rows") or [],
        "raw_matrix": candidate.get("raw_matrix") or [],
        "raw_lines": candidate.get("raw_lines") or [],
    }


def _stored_table(row: DocumentTable) -> dict:
    return {
        "table_id": row.table_id,
        "page_id": row.page_id,
        "page_number": row.page_number,
        "title": row.title,
        "columns": row.columns,
        "rows": row.rows,
        "parser_backend": row.parser_backend,
        "quality_score": row.quality_score,
    }


def _resolve_pdf(page: DocumentPage) -> Path:
    candidates = [Path(str(page.file_path or "")), ROOT / "data" / "documents" / page.filename]
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"PDF not found for {page.filename}")
    return path


def _render_page(pdf_path: Path, page_number: int, output: Path) -> None:
    executable = shutil.which("pdftoppm")
    if not executable:
        raise RuntimeError("pdftoppm is required to render audit screenshots")
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.with_suffix("")
    # pdftoppm uses 1-based command-line page positions; this is a renderer
    # boundary only, not a FinanceBench gold-page conversion.
    position = page_number + 1
    completed = subprocess.run(
        [executable, "-f", str(position), "-l", str(position), "-singlefile", "-r", "144", "-png", str(pdf_path), str(prefix)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"pdftoppm failed: {completed.stderr.strip()}")


def _load_database_catalog(filenames: set[str]):
    db = SessionLocal()
    try:
        pages = {
            (_filename(row.filename), int(row.page_number)): row
            for row in db.query(DocumentPage).all()
            if _filename(row.filename) in filenames
        }
        tables: dict[tuple[str, int], list[DocumentTable]] = {}
        for row in db.query(DocumentTable).all():
            key = (_filename(row.filename), int(row.page_number))
            if key[0] in filenames:
                tables.setdefault(key, []).append(row)
        chunks: dict[tuple[str, int], list[ParentChunk]] = {}
        for row in db.query(ParentChunk).all():
            key = (_filename(row.filename), int(row.page_number))
            if key[0] in filenames:
                chunks.setdefault(key, []).append(row)
        return pages, tables, chunks
    finally:
        db.close()


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Architecture - Table Extraction Recovery Audit", "",
        "> 离线审计6个页码修正后仍缺TableStore结构的案例。未调用LLM/Jina，未修改生产pipeline。", "",
        "## 汇总", "",
        f"- Cases: `{summary['cases']}`; unique PDF pages: `{summary['unique_pdf_pages']}`。",
        f"- Classification: `{summary['classification_counts']}`。",
        f"- Current parser accepted candidates: `{summary['accepted_candidate_pages']}` pages。",
        f"- Rejected-only candidate pages: `{summary['rejected_only_pages']}` pages。", "",
        "## 分类口径", "",
        "- **A parser漏检**：当前parser能产生accepted候选，但该页TableStore仍为空。",
        "- **B 复杂布局**：parser找到区域但因列数、密度、段落化或碎片化被拒绝。",
        "- **C OCR/图片**：页面图片主导且可提取文本不足。",
        "- **D 结构已丢失但文本存在**：文本仍包含数值行，但没有结构候选。", "",
        "## 案例", "",
    ]
    for item in payload["records"]:
        relative_image = Path("..") / Path(item["artifacts"]["screenshot"]).relative_to(ROOT)
        raw_text_link = (Path("..") / Path(item["artifacts"]["raw_text"]).relative_to(ROOT)).as_posix()
        parser_output_link = (Path("..") / Path(item["artifacts"]["parser_output"]).relative_to(ROOT)).as_posix()
        candidates_link = (Path("..") / Path(item["artifacts"]["table_candidates"]).relative_to(ROOT)).as_posix()
        lines.extend([
            f"### {item['financebench_id']} - {item['classification']} {item['classification_label']}", "",
            f"- Question: {item['question']}",
            f"- Source: `{item['filename']}`, internal page `{item['page_number']}`, page ID `{item['page_id']}`",
            f"- Reason: {item['classification_reason']}",
            f"- Signals: `{item['signals']}`",
            f"- Parser reject reasons: `{item['reject_reasons']}`",
            f"- Artifacts: [raw text]({raw_text_link}) | "
            f"[parser output]({parser_output_link}) | "
            f"[table candidates]({candidates_link})",
            "", f"![{item['financebench_id']} page {item['page_number']}]({relative_image.as_posix()})", "",
            f"Text preview: {item['text_preview']}", "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--association-audit", type=Path, default=DEFAULT_ASSOCIATION_AUDIT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()

    association = json.loads(args.association_audit.read_text(encoding="utf-8"))
    with args.dataset.open(encoding="utf-8-sig", newline="") as handle:
        dataset_by_id = {row["financebench_id"]: row for row in csv.DictReader(handle)}
    failures = select_failure_pages(association)
    if len(failures) != 6:
        raise RuntimeError(f"expected 6 post-contract extraction failures, found {len(failures)}")
    filenames = {item["filename"] for item in failures}
    pages, stored_tables, chunks = _load_database_catalog(filenames)
    records = []
    cache: dict[tuple[str, int], dict] = {}

    for index, failure in enumerate(failures, 1):
        key = (failure["filename"], failure["page_number"])
        page_record = pages.get(key)
        if page_record is None:
            raise RuntimeError(f"DocumentPage missing: {key}")
        pdf_path = _resolve_pdf(page_record)
        case_dir = args.artifact_dir / f"{failure['financebench_id']}_p{failure['page_number']:06d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        if key not in cache:
            reader = PdfReader(str(pdf_path))
            raw_text = reader.pages[failure["page_number"]].extract_text() or ""
            with pdfplumber.open(pdf_path) as pdf:
                pdf_page = pdf.pages[failure["page_number"]]
                candidates = [
                    _jsonable_candidate(item)
                    for item in _extract_page_tables(
                        pdf_page,
                        page_record.filename,
                        failure["page_number"] + 1,
                        include_rejected=True,
                    )
                ]
                layout = {
                    "word_count": len(pdf_page.extract_words() or []),
                    "image_count": len(pdf_page.images or []),
                    "line_count": len(pdf_page.lines or []),
                    "rect_count": len(pdf_page.rects or []),
                    "width": float(pdf_page.width),
                    "height": float(pdf_page.height),
                }
            cache[key] = {"raw_text": raw_text, "candidates": candidates, "layout": layout}

        extracted = cache[key]
        candidates = extracted["candidates"]
        page_chunks = sorted(chunks.get(key, []), key=lambda row: (row.chunk_level, row.chunk_idx))
        chunk_text = "\n\n".join(row.text for row in page_chunks)
        accepted = [item for item in candidates if item["accepted"]]
        rejected = [item for item in candidates if not item["accepted"]]
        dataset_row = dataset_by_id.get(failure["financebench_id"]) or {}
        evidence_items = json.loads(dataset_row.get("evidence") or "[]")
        evidence_text = "\n".join(
            str(item.get("evidence_text") or "")
            for item in evidence_items
            if _filename(item.get("doc_name")) == failure["filename"]
            and int(item.get("evidence_page_num") or 0) == failure["page_number"]
        )
        signals = {
            "stored_table_count": len(stored_tables.get(key, [])),
            "accepted_word_candidates": len(accepted),
            "rejected_word_candidates": len(rejected),
            "max_effective_columns": max((int(item.get("effective_col_count") or 0) for item in candidates), default=0),
            "table_like_text_lines": _table_like_lines(page_record.page_text),
            "raw_text_chars": len(extracted["raw_text"]),
            "document_page_text_chars": len(page_record.page_text or ""),
            "chunk_count": len(page_chunks),
            "evidence_text_recall_in_page": _token_recall(evidence_text, page_record.page_text),
            **_prose_shape(evidence_text),
            **extracted["layout"],
        }
        category, reason = classify_recovery(signals)
        screenshot = case_dir / "screenshot.png"
        _render_page(pdf_path, failure["page_number"], screenshot)

        raw_text_path = case_dir / "raw_text.txt"
        parser_output_path = case_dir / "current_parser_output.json"
        candidates_path = case_dir / "table_candidates.json"
        raw_text_path.write_text(
            "[PDF text]\n" + extracted["raw_text"] + "\n\n[DocumentPage text]\n" +
            (page_record.page_text or "") + "\n\n[ParentChunk text]\n" + chunk_text,
            encoding="utf-8",
        )
        parser_output_path.write_text(json.dumps({
            "table_store": [_stored_table(row) for row in stored_tables.get(key, [])],
            "layout": extracted["layout"],
            "accepted_candidates": accepted,
            "rejected_candidates": rejected,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        candidates_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        record = {
            **failure,
            "page_id": page_record.page_id,
            "classification": category,
            "classification_label": CLASS_LABELS[category],
            "classification_reason": reason,
            "signals": signals,
            "reject_reasons": dict(Counter(item["reject_reason"] or "unspecified" for item in rejected)),
            "text_preview": _preview(page_record.page_text),
            "benchmark_evidence_text": evidence_text,
            "artifacts": {
                "screenshot": str(screenshot),
                "raw_text": str(raw_text_path),
                "parser_output": str(parser_output_path),
                "table_candidates": str(candidates_path),
            },
        }
        records.append(record)
        print(f"[{index:02d}/6] {failure['financebench_id']} p{failure['page_number']}: {category} {CLASS_LABELS[category]}", flush=True)

    summary = {
        "cases": len(records),
        "unique_pdf_pages": len({(item["filename"], item["page_number"]) for item in records}),
        "classification_counts": dict(Counter(item["classification"] for item in records)),
        "accepted_candidate_pages": sum(item["signals"]["accepted_word_candidates"] > 0 for item in records),
        "rejected_only_pages": sum(
            item["signals"]["accepted_word_candidates"] == 0 and item["signals"]["rejected_word_candidates"] > 0
            for item in records
        ),
    }
    payload = {
        "profile": "evidence_architecture_table_extraction_recovery_audit6",
        "scope": "offline PDF/TableStore/text extraction; no production writes or external API calls",
        "summary": summary,
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {args.output_json}\nMarkdown: {args.output_markdown}\nArtifacts: {args.artifact_dir}", flush=True)


if __name__ == "__main__":
    main()
