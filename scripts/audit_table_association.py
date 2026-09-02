"""Offline audit for diagnostic30 gold pages missing structured tables.

Reads FinanceBench evidence metadata, PDF pages, PostgreSQL DocumentPage /
DocumentTable / ParentChunk rows.  It never writes the database and never calls
retrieval, reranking, LLM, or judge services.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf.generic import ContentStream


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from database import SessionLocal  # noqa: E402
from evidence_identity import build_page_id  # noqa: E402
from models import DocumentPage, DocumentTable, ParentChunk  # noqa: E402
from table_quality import table_page_match_score  # noqa: E402


DEFAULT_DIAGNOSTIC = ROOT / "reports" / "table_aware_retrieval_v1_diagnostic30.json"
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_JSON = ROOT / "reports" / "table_association_audit_diagnostic14.json"
DEFAULT_MARKDOWN = ROOT / "docs" / "table_association_audit_diagnostic14.md"
NUMBER_RE = re.compile(r"(?:[$€£]\s*)?\(?-?\d[\d,]*(?:\.\d+)?%?\)?")
SPACE_RE = re.compile(r"\s+")

CLASS_LABELS = {
    "A": "parser漏抽",
    "B": "page id mismatch",
    "C": "table被转为text",
    "D": "gold page不是实际table页",
    "E": "无法判断",
}


def _clean(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "")).strip()


def _filename(value: Any) -> str:
    name = _clean(value).replace("\\", "/").rsplit("/", 1)[-1]
    if name and not name.casefold().endswith(".pdf"):
        name += ".pdf"
    return name.casefold()


def _preview(value: Any, limit: int = 700) -> str:
    text = _clean(value)
    return text[:limit] + ("…" if len(text) > limit else "")


def _md_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def table_like_line_count(text: str) -> int:
    """Count text rows with at least two numeric cells; diagnostic signal only."""
    return sum(len(NUMBER_RE.findall(line)) >= 2 for line in str(text or "").splitlines())


def _tokens(value: str) -> set[str]:
    return {token.casefold().replace(",", "") for token in re.findall(r"[a-z0-9][a-z0-9,.%-]*", value or "", re.I)}


def _text_similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 4)


def classify_page(signals: dict) -> tuple[str, str]:
    """Classify only when deterministic evidence supports the category."""
    if signals.get("benchmark_boundary_mismatch"):
        return "B", "FinanceBench evidence text matches the unshifted internal page, not the page produced by subtracting one"
    if signals.get("intrinsic_identity_mismatch") or signals.get("identity_mismatch") or signals.get("relocated_table_match"):
        return "B", "page/document identity is inconsistent or a nearby table matches this page better than its assigned page"
    if not signals.get("pdf_available"):
        return "E", "PDF page could not be opened; association cannot be verified"
    if int(signals.get("pdf_table_count") or 0) > 0:
        return "A", "PDF drawing operations and numeric rows indicate a table on the gold page, but TableStore has no table"
    if signals.get("stored_table_text") or int(signals.get("table_like_text_lines") or 0) >= 3:
        return "C", "structured table is absent, while page/chunk text retains table-like rows"
    if int(signals.get("nearby_table_count") or 0) > 0:
        return "D", "gold page has no table signal and structured tables occur only on nearby pages"
    return "E", "no decisive table geometry, text-table, identity, or nearby-page signal"


def select_cases(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return all missing-table records and the 14 adjacent-table audit cases."""
    missing = [record for record in records if not record.get("gold_table_ids")]
    focused = [
        record for record in missing
        if int((record.get("gold_table_diagnostics") or {}).get("adjacent_table_count") or 0) > 0
    ]
    return missing, focused


def _table_payload(row: DocumentTable) -> dict:
    return {
        "table_id": row.table_id,
        "document_id": row.document_id,
        "page_id": row.page_id,
        "page_number": int(row.page_number),
        "start_page": int(row.start_page),
        "end_page": int(row.end_page),
        "title": _preview(row.title, 240),
        "header": [_preview(item, 100) for item in (row.columns or [])],
        "quality_score": round(float(row.quality_score or 0.0), 4),
        "parser_backend": row.parser_backend,
        "rows": len(row.rows or []),
    }


class AuditCatalog:
    def __init__(self, filenames: set[str]):
        db = SessionLocal()
        try:
            pages = db.query(DocumentPage).all()
            tables = db.query(DocumentTable).all()
            chunks = db.query(ParentChunk).all()
        finally:
            db.close()
        self.pages = {
            (_filename(row.filename), int(row.page_number)): row
            for row in pages if _filename(row.filename) in filenames
        }
        self.tables: dict[tuple[str, int], list[DocumentTable]] = defaultdict(list)
        for row in tables:
            key = (_filename(row.filename), int(row.page_number))
            if key[0] in filenames:
                self.tables[key].append(row)
        self.chunks: dict[tuple[str, int], list[ParentChunk]] = defaultdict(list)
        for row in chunks:
            key = (_filename(row.filename), int(row.page_number))
            if key[0] in filenames:
                self.chunks[key].append(row)


class PdfCatalog:
    def __init__(self, catalog: AuditCatalog):
        self.paths: dict[str, Path] = {}
        for (filename, _), page in catalog.pages.items():
            candidates = [Path(str(page.file_path or "")), ROOT / "data" / "documents" / page.filename]
            self.paths[filename] = next((path for path in candidates if path.is_file()), candidates[-1])
        self._cache: dict[str, dict[int, dict]] = defaultdict(dict)
        self._readers: dict[str, PdfReader] = {}

    def inspect(self, filename: str, page_number: int) -> dict:
        if page_number in self._cache[filename]:
            return self._cache[filename][page_number]
        path = self.paths.get(filename)
        result = {
            "available": False,
            "path": str(path or ""),
            "text": "",
            "table_count": 0,
            "vector_edge_count": 0,
            "numeric_row_count": 0,
            "error": "",
        }
        if path is None or not path.is_file():
            result["error"] = "PDF file not found"
            self._cache[filename][page_number] = result
            return result
        try:
            reader = self._readers.get(filename)
            if reader is None:
                reader = PdfReader(str(path))
                self._readers[filename] = reader
            if page_number < 0 or page_number >= len(reader.pages):
                raise IndexError(f"page {page_number} outside PDF range 0..{len(reader.pages) - 1}")
            page = reader.pages[page_number]
            text = page.extract_text() or ""
            content = ContentStream(page.get_contents(), reader)
            vector_edges = sum(operator in (b"re", b"l") for _, operator in content.operations)
            numeric_rows = table_like_line_count(text)
            result.update({
                "available": True,
                "text": text,
                # A lightweight geometry proxy avoids invoking the full
                # table parser during an association-only audit.
                "table_count": int(vector_edges >= 4 and numeric_rows >= 3),
                "vector_edge_count": vector_edges,
                "numeric_row_count": numeric_rows,
            })
        except Exception as exc:  # audit must preserve a per-page failure rather than abort all cases
            result["error"] = f"{type(exc).__name__}: {exc}"
        self._cache[filename][page_number] = result
        return result

    def close(self) -> None:
        for reader in self._readers.values():
            stream = getattr(reader, "stream", None)
            close = getattr(stream, "close", None)
            if close:
                close()


def _identity_mismatch(page: DocumentPage | None, tables: list[DocumentTable]) -> tuple[bool, list[str]]:
    reasons = []
    if page is None:
        return True, ["DocumentPage missing for filename/page_number"]
    expected = build_page_id(page.document_id, page.page_number) if page.document_id else ""
    if not page.document_id:
        reasons.append("DocumentPage.document_id empty")
    if not page.page_id or page.page_id != expected:
        reasons.append(f"DocumentPage.page_id={page.page_id!r}, expected={expected!r}")
    for table in tables:
        if table.document_id != page.document_id or table.page_id != page.page_id:
            reasons.append(f"table {table.table_id} disagrees with DocumentPage identity")
    return bool(reasons), reasons


def _relocated_table_match(
    gold_page: DocumentPage | None,
    nearby_tables: list[DocumentTable],
    pages: dict[tuple[str, int], DocumentPage],
    filename: str,
) -> tuple[bool, list[dict]]:
    if gold_page is None:
        return False, []
    matches = []
    for table in nearby_tables:
        assigned = pages.get((filename, int(table.page_number)))
        payload = {
            "title": table.title,
            "caption": table.caption,
            "columns": table.columns,
            "rows": table.rows,
        }
        gold_score = table_page_match_score(payload, gold_page.page_text)
        assigned_score = table_page_match_score(payload, assigned.page_text if assigned else "")
        if gold_score >= 0.35 and gold_score >= assigned_score + 0.15:
            matches.append({
                "table_id": table.table_id,
                "assigned_page": int(table.page_number),
                "gold_page_match_score": gold_score,
                "assigned_page_match_score": assigned_score,
            })
    return bool(matches), matches


def audit_gold_page(
    *,
    filename: str,
    gold_page_number: int,
    benchmark_page_number: int | None,
    benchmark_evidence_text: str,
    catalog: AuditCatalog,
    pdfs: PdfCatalog,
) -> dict:
    gold_key = (filename, gold_page_number)
    gold_page = catalog.pages.get(gold_key)
    gold_tables = catalog.tables.get(gold_key, [])
    neighborhood = []
    nearby_tables: list[DocumentTable] = []
    for page_number in range(max(0, gold_page_number - 2), gold_page_number + 3):
        key = (filename, page_number)
        page = catalog.pages.get(key)
        tables = catalog.tables.get(key, [])
        chunks = catalog.chunks.get(key, [])
        if page_number != gold_page_number:
            nearby_tables.extend(tables)
        raw_pdf = pdfs.inspect(filename, page_number)
        chunk_text = "\n".join(row.text for row in sorted(chunks, key=lambda row: (row.chunk_level, row.chunk_idx)))
        neighborhood.append({
            "offset": page_number - gold_page_number,
            "page_number": page_number,
            "document_page_found": page is not None,
            "document_id": page.document_id if page else "",
            "page_id": page.page_id if page else "",
            "table_count": len(tables),
            "tables": [_table_payload(table) for table in tables],
            "chunk_count": len(chunks),
            "chunk_ids": [row.chunk_id for row in chunks],
            "page_text_preview": _preview(page.page_text if page else ""),
            "table_text_preview": _preview(page.table_text if page else "", 400),
            "chunk_text_preview": _preview(chunk_text, 400),
            "pdf_page_available": raw_pdf["available"],
            "pdf_table_count": raw_pdf["table_count"],
            "pdf_vector_edge_count": raw_pdf["vector_edge_count"],
            "pdf_numeric_row_count": raw_pdf["numeric_row_count"],
            "pdf_text_preview": _preview(raw_pdf["text"]),
            "pdf_error": raw_pdf["error"],
        })

    raw_gold = pdfs.inspect(filename, gold_page_number)
    chunk_text = "\n".join(row.text for row in catalog.chunks.get(gold_key, []))
    mismatch, mismatch_reasons = _identity_mismatch(gold_page, gold_tables)
    intrinsic_mismatch = mismatch
    benchmark_page = (
        catalog.pages.get((filename, benchmark_page_number))
        if benchmark_page_number is not None else None
    )
    converted_similarity = _text_similarity(
        benchmark_evidence_text,
        gold_page.page_text if gold_page else "",
    )
    benchmark_similarity = _text_similarity(
        benchmark_evidence_text,
        benchmark_page.page_text if benchmark_page else "",
    )
    benchmark_boundary_mismatch = bool(
        benchmark_page_number is not None
        and benchmark_page_number != gold_page_number
        and benchmark_page is not None
        and benchmark_similarity >= 0.35
        and benchmark_similarity >= converted_similarity + 0.15
    )
    if benchmark_boundary_mismatch:
        mismatch = True
        mismatch_reasons.append(
            "benchmark evidence text matches internal page "
            f"{benchmark_page_number} ({benchmark_similarity}) better than converted page "
            f"{gold_page_number} ({converted_similarity})"
        )
    relocated, relocation_matches = _relocated_table_match(gold_page, nearby_tables, catalog.pages, filename)
    combined_text = "\n".join((gold_page.page_text if gold_page else "", chunk_text))
    signals = {
        "identity_mismatch": mismatch,
        "intrinsic_identity_mismatch": intrinsic_mismatch,
        "benchmark_boundary_mismatch": benchmark_boundary_mismatch,
        "relocated_table_match": relocated,
        "pdf_available": raw_gold["available"],
        "pdf_table_count": raw_gold["table_count"],
        "stored_table_text": bool(_clean(gold_page.table_text if gold_page else "")),
        "table_like_text_lines": table_like_line_count(combined_text),
        "nearby_table_count": len(nearby_tables),
    }
    category, reason = classify_page(signals)
    post_alignment = {"status": "not_applicable", "classification": "", "classification_label": "", "reason": ""}
    if benchmark_boundary_mismatch and benchmark_page_number is not None:
        aligned_key = (filename, benchmark_page_number)
        aligned_tables = catalog.tables.get(aligned_key, [])
        if aligned_tables:
            post_alignment = {
                "status": "resolved_by_page_boundary",
                "classification": "",
                "classification_label": "",
                "reason": f"unshifted internal page contains {len(aligned_tables)} TableStore table(s)",
            }
        else:
            aligned_raw = pdfs.inspect(filename, benchmark_page_number)
            aligned_chunks = "\n".join(row.text for row in catalog.chunks.get(aligned_key, []))
            aligned_mismatch, _ = _identity_mismatch(benchmark_page, aligned_tables)
            aligned_nearby_count = sum(
                len(catalog.tables.get((filename, page), []))
                for page in range(max(0, benchmark_page_number - 2), benchmark_page_number + 3)
                if page != benchmark_page_number
            )
            aligned_signals = {
                "intrinsic_identity_mismatch": aligned_mismatch,
                "pdf_available": aligned_raw["available"],
                "pdf_table_count": aligned_raw["table_count"],
                "stored_table_text": bool(_clean(benchmark_page.table_text if benchmark_page else "")),
                "table_like_text_lines": table_like_line_count(
                    "\n".join((benchmark_page.page_text if benchmark_page else "", aligned_chunks))
                ),
                "nearby_table_count": aligned_nearby_count,
            }
            aligned_category, aligned_reason = classify_page(aligned_signals)
            post_alignment = {
                "status": "still_missing_after_page_boundary_fix",
                "classification": aligned_category,
                "classification_label": CLASS_LABELS[aligned_category],
                "reason": aligned_reason,
                "signals": aligned_signals,
            }
    return {
        "gold_page": gold_page_number,
        "benchmark_evidence_page_number": benchmark_page_number,
        "benchmark_evidence_similarity": {
            "converted_internal_page": converted_similarity,
            "raw_benchmark_number_as_internal_page": benchmark_similarity,
        },
        "benchmark_page_table_count": len(
            catalog.tables.get((filename, benchmark_page_number), [])
        ) if benchmark_page_number is not None else 0,
        "gold_document_id": gold_page.document_id if gold_page else "",
        "gold_page_id": gold_page.page_id if gold_page else "",
        "classification": category,
        "classification_label": CLASS_LABELS[category],
        "classification_reason": reason,
        "signals": signals,
        "identity_mismatch_reasons": mismatch_reasons,
        "relocation_matches": relocation_matches,
        "post_alignment": post_alignment,
        "nearby_pages": neighborhood,
    }


def _case_classification(page_audits: list[dict]) -> str:
    # Prefer the most actionable deterministic finding for multi-gold-page cases.
    for category in ("B", "A", "C", "D", "E"):
        if any(item["classification"] == category for item in page_audits):
            return category
    return "E"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Architecture v5 — Table Association Audit", "",
        "> 离线只读审计。未调用 Dense/BM25/RRF/Jina/LLM/Judge，未修改数据库或生产 pipeline。页码均为内部 0-based。", "",
        "## 范围与结论", "",
        f"- Diagnostic30 中 gold page 无 TableStore 表的问题：`{summary['all_missing_question_count']}` 题。",
        f"- 本报告按要求审计其中相邻 ±1 页存在表的固定案例：`{summary['focused_case_count']}` 题。",
        f"- 涉及 gold pages：`{summary['gold_page_audit_count']}` 页。",
        f"- 题级分类：`{summary['case_classification_counts']}`。",
        f"- 页级分类：`{summary['page_classification_counts']}`。", "",
        f"- benchmark减1边界错配：`{summary['benchmark_boundary_mismatch_pages']}/{summary['gold_page_audit_count']}` 页。",
        f"- PostgreSQL内部ID契约异常：`{summary['intrinsic_identity_mismatch_pages']}/{summary['gold_page_audit_count']}` 页。",
        f"- 原始benchmark页号对应内部页存在TableStore表：`{summary['raw_benchmark_page_has_table_pages']}/{summary['gold_page_audit_count']}` 页。", "",
        f"- 修正页码后状态：`{summary['post_alignment_status_counts']}`。", "",
        f"- 题级修正结果：完全恢复 `{summary['post_alignment_case_counts']['fully_resolved']}`，部分恢复 `{summary['post_alignment_case_counts']['partially_resolved']}`，仍全部缺失 `{summary['post_alignment_case_counts']['not_resolved']}`。", "",
        "## 分类定义", "",
        "- **A parser漏抽**：PDF页同时存在足够的矢量边界与多数字行，但TableStore无表。该信号不调用完整table parser。",
        "- **B page id mismatch**：DocumentPage/Table的ID契约不一致，或FinanceBench证据文本与未减1页显著更匹配，或附近表结构与gold页文本的匹配显著高于其绑定页。",
        "- **C table被转为text**：未检测到可用几何表，但page `table_text` 或chunks保留明显表格行。",
        "- **D gold page不是实际table页**：gold页没有表格信号，而结构表只存在于附近页面。",
        "- **E 无法判断**：PDF不可读取、信号冲突或证据不足。", "",
        "## 汇总表", "",
        "| ID | Document | Gold pages | 分类 | 附近表数 | PDF表数 |", "|---|---|---|---|---:|---:|",
    ]
    for case in payload["cases"]:
        pages = ", ".join(str(item["gold_page"]) for item in case["gold_page_audits"])
        nearby = sum(item["signals"]["nearby_table_count"] for item in case["gold_page_audits"])
        pdf_tables = sum(item["signals"]["pdf_table_count"] for item in case["gold_page_audits"])
        lines.append(
            f"| `{case['financebench_id']}` | `{case['gold_document']}` | {pages} | "
            f"{case['classification']} {case['classification_label']} | {nearby} | {pdf_tables} |"
        )
    lines.extend(["", "## 逐题证据", ""])
    for case in payload["cases"]:
        lines.extend([
            f"### {case['financebench_id']} — {case['classification']} {case['classification_label']}", "",
            f"- Question: {case['question']}",
            f"- Gold document: `{case['gold_document']}`",
        ])
        for audit in case["gold_page_audits"]:
            lines.extend([
                f"- Gold page `{audit['gold_page']}`: **{audit['classification']} {audit['classification_label']}** — {audit['classification_reason']}",
                f"- Benchmark evidence_page_num: `{audit['benchmark_evidence_page_number']}`; text similarity converted/raw: `{audit['benchmark_evidence_similarity']}`",
                f"- Gold page ID: `{audit['gold_page_id'] or '(missing)'}`",
                f"- Signals: `{audit['signals']}`",
                f"- Post-alignment: `{audit['post_alignment']}`",
            ])
            if audit["identity_mismatch_reasons"]:
                lines.append(f"- Identity details: `{audit['identity_mismatch_reasons']}`")
            if audit["relocation_matches"]:
                lines.append(f"- Relocation matches: `{audit['relocation_matches']}`")
            lines.extend(["", "| Offset | Page | Table count | Page ID | Table title / header | Page text preview |", "|---:|---:|---:|---|---|---|"])
            for page in audit["nearby_pages"]:
                table_summary = "<br>".join(
                    f"`{_md_cell(table['table_id'])}`<br>{_md_cell(table['title'])}<br>Header: {_md_cell(table['header'])}"
                    for table in page["tables"]
                ) or "—"
                preview = _md_cell(page["page_text_preview"] or page["pdf_text_preview"]) or "—"
                lines.append(
                    f"| {page['offset']:+d} | {page['page_number']} | {page['table_count']} | "
                    f"`{page['page_id'] or '(missing)'}` | {table_summary} | {preview} |"
                )
            lines.append("")
    lines.extend([
        "## 下一步判断", "",
        "- A类应检查table parser的质量门控与未落库原因，但本报告不修改parser。",
        "- B类应优先修复page/table identity或页码边界；在修复前不能建立新的table索引。",
        "- C类说明信息仍在文本链路，可继续作为page-text evidence；结构恢复应离线验证后再开发。",
        "- D类不能通过强制给gold页绑定相邻表解决，应保留page-text并核验benchmark evidence语义。",
        "- E类需要人工查看PDF截图，不能用规则自动定性。", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    with args.dataset.open(encoding="utf-8-sig", newline="") as handle:
        dataset_by_id = {row["financebench_id"]: row for row in csv.DictReader(handle)}
    all_missing, focused = select_cases(diagnostic.get("records") or [])
    if len(focused) != 14:
        raise RuntimeError(f"expected fixed adjacent-table diagnostic set of 14, found {len(focused)}")
    filenames = {_filename(name) for record in focused for name, _ in record.get("gold_pages", [])}
    catalog = AuditCatalog(filenames)
    pdfs = PdfCatalog(catalog)
    cases = []
    try:
        for index, record in enumerate(focused, 1):
            dataset_row = dataset_by_id.get(str(record.get("financebench_id"))) or {}
            benchmark_evidence = json.loads(dataset_row.get("evidence") or "[]")
            evidence_by_internal_gold: dict[tuple[str, int], dict] = {}
            for evidence in benchmark_evidence:
                raw_page = int(evidence.get("evidence_page_num") or 0)
                evidence_by_internal_gold[(_filename(evidence.get("doc_name")), raw_page)] = evidence
            by_document: dict[str, list[int]] = defaultdict(list)
            for evidence in benchmark_evidence:
                by_document[_filename(evidence.get("doc_name"))].append(
                    int(evidence.get("evidence_page_num") or 0)
                )
            # FinanceBench records in this set point to one document; retain the
            # loop so a multi-document evidence record remains auditable.
            page_audits = [
                audit_gold_page(
                    filename=name,
                    gold_page_number=page,
                    benchmark_page_number=int(evidence.get("evidence_page_num") or 0) if evidence else None,
                    benchmark_evidence_text=str(
                        (evidence or {}).get("evidence_text_full_page")
                        or (evidence or {}).get("evidence_text")
                        or ""
                    ),
                    catalog=catalog,
                    pdfs=pdfs,
                )
                for name, pages in by_document.items() for page in sorted(set(pages))
                for evidence in [evidence_by_internal_gold.get((name, page))]
            ]
            category = _case_classification(page_audits)
            case = {
                "financebench_id": record.get("financebench_id"),
                "question": record.get("question"),
                "gold_document": ", ".join(sorted(by_document)),
                "classification": category,
                "classification_label": CLASS_LABELS[category],
                "gold_page_audits": page_audits,
            }
            cases.append(case)
            print(f"[{index:02d}/{len(focused)}] {case['financebench_id']}: {category} {CLASS_LABELS[category]}", flush=True)
    finally:
        pdfs.close()

    page_categories = Counter(
        audit["classification"] for case in cases for audit in case["gold_page_audits"]
    )
    case_categories = Counter(case["classification"] for case in cases)
    post_alignment_case_counts = Counter()
    for case in cases:
        statuses = [audit["post_alignment"]["status"] for audit in case["gold_page_audits"]]
        resolved = sum(status == "resolved_by_page_boundary" for status in statuses)
        if resolved == len(statuses):
            post_alignment_case_counts["fully_resolved"] += 1
        elif resolved:
            post_alignment_case_counts["partially_resolved"] += 1
        else:
            post_alignment_case_counts["not_resolved"] += 1
    payload = {
        "profile": "evidence_architecture_v5_table_association_audit",
        "scope": "fixed diagnostic14 selected from diagnostic30 missing-table records with adjacent tables",
        "classification_labels": CLASS_LABELS,
        "source_diagnostic": str(args.diagnostic),
        "summary": {
            "diagnostic_questions": len(diagnostic.get("records") or []),
            "all_missing_question_count": len(all_missing),
            "focused_case_count": len(cases),
            "gold_page_audit_count": sum(len(case["gold_page_audits"]) for case in cases),
            "case_classification_counts": dict(case_categories),
            "page_classification_counts": dict(page_categories),
            "benchmark_boundary_mismatch_pages": sum(
                audit["signals"]["benchmark_boundary_mismatch"]
                for case in cases for audit in case["gold_page_audits"]
            ),
            "intrinsic_identity_mismatch_pages": sum(
                audit["signals"]["intrinsic_identity_mismatch"]
                for case in cases for audit in case["gold_page_audits"]
            ),
            "raw_benchmark_page_has_table_pages": sum(
                audit["benchmark_page_table_count"] > 0
                for case in cases for audit in case["gold_page_audits"]
            ),
            "post_alignment_status_counts": dict(Counter(
                (
                    audit["post_alignment"]["status"]
                    if audit["post_alignment"]["status"] == "resolved_by_page_boundary"
                    else f"still_missing:{audit['post_alignment'].get('classification') or 'E'}"
                )
                for case in cases for audit in case["gold_page_audits"]
            )),
            "post_alignment_case_counts": {
                key: post_alignment_case_counts.get(key, 0)
                for key in ("fully_resolved", "partially_resolved", "not_resolved")
            },
        },
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {args.output_json}\nMarkdown: {args.output_markdown}", flush=True)


if __name__ == "__main__":
    main()
