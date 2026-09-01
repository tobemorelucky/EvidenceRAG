"""Offline Evidence Assembly A/B evaluation over frozen retrieval traces.

This script never runs retrieval, Jina, an answer model, or a Judge. It opens
the pages already selected in an existing retrieval diagnostic and compares:
A = page_text-only evidence, B = Evidence Assembly v1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from sqlalchemy import func


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from database import SessionLocal  # noqa: E402
from evidence_assembly_v1 import build_evidence_assembly_v1  # noqa: E402
from models import DocumentPage, DocumentTable  # noqa: E402


DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_FROZEN = ROOT / "reports" / "retrieval_document_local_diagnostic30.json"
DEFAULT_OUTPUT = ROOT / "reports" / "evidence_assembly_ab30.json"
QUESTION_TYPES = ("table_likely", "calculation", "comparison", "lookup")

_NUMBER_RE = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")
_PERIOD_RE = re.compile(r"\bFY\s*\d{2,4}\b|\b(?:19|20)\d{2}\b|\bQ[1-4]\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[a-z][a-z0-9_-]{2,}", re.IGNORECASE)
_COMPARISON_RE = re.compile(
    r"\b(compare|compared|difference|increase|decrease|change|higher|lower|most|least|largest|smallest|"
    r"grew|growth|decline|versus|vs\.?|between|which segment|which year)\b",
    re.IGNORECASE,
)
_CALCULATION_RE = re.compile(
    r"\b(calculate|ratio|margin|percentage|percent|round|divided|multiply|sum|average|per share|turnover|"
    r"working capital|cash flow ratio|return on)\b|[+*/]",
    re.IGNORECASE,
)
_TABLE_RE = re.compile(
    r"\b(statement|balance sheet|cash flow|assets|liabilities|revenue|sales|income|expense|margin|segment|"
    r"inventory|equity|debt|capital|fiscal|FY\d{4})\b",
    re.IGNORECASE,
)


def _filename(value: object) -> str:
    name = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return name if name.casefold().endswith(".pdf") else f"{name}.pdf"


def _page_key(item: dict) -> tuple[str, int]:
    return _filename(item.get("filename") or item.get("doc_name")).casefold(), int(item.get("page_number") or 0)


def _dedupe_page_keys(items: list[dict]) -> list[tuple[str, int]]:
    return list(dict.fromkeys(_page_key(item) for item in items or []))


def _parse_gold(row: dict) -> list[dict]:
    records = []
    for item in json.loads(row.get("evidence") or "[]"):
        records.append({
            "filename": _filename(item.get("doc_name")),
            "page_number": int(item.get("evidence_page_num") or 0),
            "evidence_text": str(item.get("evidence_text") or ""),
        })
    return records


def classify_question(row: dict) -> set[str]:
    question = str(row.get("question") or "")
    evidence = "\n".join(item["evidence_text"] for item in _parse_gold(row))
    kinds = set()
    if (
        _TABLE_RE.search(question)
        or len(_NUMBER_RE.findall(evidence)) >= 3
        and len([line for line in evidence.splitlines() if line.strip()]) >= 3
    ):
        kinds.add("table_likely")
    if str(row.get("question_reasoning") or "").casefold() == "numerical reasoning" or _CALCULATION_RE.search(question):
        kinds.add("calculation")
    if _COMPARISON_RE.search(question) or len(set(_PERIOD_RE.findall(question))) >= 2:
        kinds.add("comparison")
    if not ({"calculation", "comparison"} & kinds):
        kinds.add("lookup")
    return kinds


def select_rows(rows: list[dict], frozen_ids: list[str], *, limit: int = 30) -> list[dict]:
    by_id = {row["financebench_id"]: row for row in rows}
    available = [by_id[item] for item in frozen_ids if item in by_id]
    quota = max(1, math.ceil(min(limit, len(available)) / len(QUESTION_TYPES)))
    selected = []
    selected_ids = set()
    for kind in QUESTION_TYPES:
        for row in available:
            if len([item for item in selected if kind in classify_question(item)]) >= quota:
                break
            if row["financebench_id"] in selected_ids or kind not in classify_question(row):
                continue
            selected.append(row)
            selected_ids.add(row["financebench_id"])
    for row in available:
        if len(selected) >= limit:
            break
        if row["financebench_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["financebench_id"])
    return selected[:limit]


def _load_pages(keys: list[tuple[str, int]]) -> list[dict]:
    if not keys:
        return []
    filenames = {item[0] for item in keys}
    db = SessionLocal()
    try:
        rows = db.query(DocumentPage).filter(func.lower(DocumentPage.filename).in_(filenames)).all()
        by_key = {(row.filename.casefold(), row.page_number): row for row in rows}
        return [
            {
                "document_id": row.document_id,
                "page_id": row.page_id,
                "filename": row.filename,
                "page_number": row.page_number,
                "page_text": row.page_text,
            }
            for key in keys if (row := by_key.get(key)) is not None
        ]
    finally:
        db.close()


def _table_dict(row: DocumentTable) -> dict:
    return {
        "table_id": row.table_id,
        "document_id": row.document_id,
        "page_id": row.page_id,
        "filename": row.filename,
        "page_number": row.page_number,
        "start_page": row.start_page,
        "end_page": row.end_page,
        "table_index": row.table_index,
        "parser_backend": row.parser_backend,
        "quality_score": row.quality_score,
        "title": row.title,
        "caption": row.caption,
        "before_context": row.before_context,
        "after_context": row.after_context,
        "columns": list(row.columns or []),
        "rows": list(row.rows or []),
        "unit": row.unit,
        "scale": row.scale,
    }


def _load_tables(page_ids: list[str]) -> list[dict]:
    if not page_ids:
        return []
    db = SessionLocal()
    try:
        rows = db.query(DocumentTable).filter(DocumentTable.page_id.in_(page_ids)).all()
        return [_table_dict(row) for row in rows]
    finally:
        db.close()


def build_page_text_evidence(pages: list[dict], *, max_context_chars: int = 28000) -> str:
    if not pages:
        return ""
    slot = max(1, max_context_chars // len(pages))
    units = []
    for page in pages:
        value = (
            f"Source: {page.get('filename', '')}, internal page {page.get('page_number', 0)}\n"
            f"Page ID: {page.get('page_id', '')}\n[Page Text Evidence]\n{page.get('page_text', '')}"
        )
        units.append(value[:slot])
    return "\n\n".join(units)[:max_context_chars]


def _normalize_number(value: str) -> str:
    text = str(value or "").strip().replace("$", "").replace(",", "")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    return f"-{text}" if negative else text


def _numbers(value: str, *, exclude_years: bool = True) -> list[str]:
    out = []
    for match in _NUMBER_RE.finditer(str(value or "")):
        number = _normalize_number(match.group(0))
        plain = number.lstrip("-").rstrip("%")
        if exclude_years and plain.isdigit() and 1900 <= int(plain) <= 2099:
            continue
        if number not in out:
            out.append(number)
    return out


def _periods(value: str) -> list[str]:
    periods = []
    for match in _PERIOD_RE.finditer(str(value or "")):
        item = re.sub(r"\s+", "", match.group(0).upper())
        if item.startswith("FY"):
            year = item[2:]
            item = f"20{year}" if len(year) == 2 else year
        if item not in periods:
            periods.append(item)
    return periods


def _contains_all(values: list[str], evidence: str, extractor) -> bool | None:
    if not values:
        return None
    present = set(extractor(evidence))
    return all(value in present for value in values)


def _words(value: str) -> set[str]:
    stop = {"the", "and", "for", "with", "from", "that", "this", "were", "was", "are", "of", "in", "to"}
    return {item.casefold() for item in _WORD_RE.findall(str(value or "")) if item.casefold() not in stop}


def _required_numbers(row: dict) -> list[str]:
    """Return evidence operands/comparison values, excluding calculated outputs.

    FinanceBench justifications normally encode the exact operands or values used
    for the answer.  When a formula has a final equals sign, the right-hand side
    is a derived answer and therefore must not be required in source evidence.
    """
    justification = str(row.get("justification") or "").strip()
    source = justification
    if "=" in justification:
        left, right = justification.rsplit("=", 1)
        if _numbers(left) and _numbers(right):
            source = left
    values = _numbers(source)
    return values or _numbers(str(row.get("answer") or ""))


def gold_row_hit(gold: list[dict], evidence: str, *, required_numbers: list[str] | None = None) -> bool:
    evidence_words = _words(evidence)
    evidence_numbers = set(_numbers(evidence, exclude_years=False))
    if required_numbers and not set(required_numbers) <= set(_numbers(evidence)):
        return False
    for item in gold:
        text = item["evidence_text"]
        for line in text.splitlines():
            line_numbers = set(_numbers(line, exclude_years=False))
            line_words = _words(line)
            if not line_numbers or not line_words:
                continue
            word_overlap = len(line_words & evidence_words) / max(1, len(line_words))
            if line_numbers <= evidence_numbers and word_overlap >= 0.5:
                return True
    return False


def _trusted_table_evidence(assembly: str) -> str:
    """Keep only emitted page units that contain trusted table evidence."""
    units = re.split(r"(?=^Source: )", str(assembly or ""), flags=re.MULTILINE)
    return "\n".join(unit for unit in units if "[Trusted Table Evidence]" in unit)


def _hash_page_keys(keys: list[tuple[str, int]]) -> str:
    payload = json.dumps(keys, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _question_record(row: dict, trace: dict, variant_name: str) -> dict:
    variant = trace["variants"][variant_name]
    selected_keys = _dedupe_page_keys(variant.get("selected_pages") or [])
    candidate_trace = trace.get("page_candidate_traces", {}).get(variant_name, {})
    candidate_keys = _dedupe_page_keys(candidate_trace.get("expanded_pages") or [])
    pages = _load_pages(selected_keys)
    tables = _load_tables([page["page_id"] for page in pages if page.get("page_id")])
    baseline = build_page_text_evidence(pages)
    assembly, assembly_trace = build_evidence_assembly_v1(row["question"], pages, tables)
    gold = _parse_gold(row)
    required_numbers = _required_numbers(row)
    required_periods = _periods(row["question"])
    baseline_row_hit = gold_row_hit(gold, baseline, required_numbers=required_numbers)
    assembly_row_hit = gold_row_hit(
        gold,
        _trusted_table_evidence(assembly),
        required_numbers=required_numbers,
    )
    return {
        "financebench_id": row["financebench_id"],
        "question": row["question"],
        "question_types": sorted(classify_question(row)),
        "gold_evidence_pages": [
            {"filename": item["filename"], "page_number": item["page_number"]} for item in gold
        ],
        "frozen_retrieval": {
            "source_variant": variant_name,
            "candidate_pages": [{"filename": item[0], "page_number": item[1]} for item in candidate_keys],
            "candidate_pages_hash": _hash_page_keys(candidate_keys),
            "selected_pages": [{"filename": item[0], "page_number": item[1]} for item in selected_keys],
            "context_pages": [{"filename": item[0], "page_number": item[1]} for item in selected_keys],
            "selected_pages_hash": _hash_page_keys(selected_keys),
        },
        "baseline_evidence_chars": len(baseline),
        "assembly_evidence_chars": len(assembly),
        "trusted_tables": assembly_trace["trusted_table_ids"],
        "rejected_tables": assembly_trace["rejected_tables"],
        "page_text_fallback_count": assembly_trace["page_text_fallback_count"],
        "selected_page_count": len(pages),
        "gold_row_table_hit": {
            "baseline_page_text": baseline_row_hit,
            "assembly_v1": assembly_row_hit,
        },
        "required_numbers": required_numbers,
        "required_number_hit": {
            "baseline_page_text": _contains_all(required_numbers, baseline, _numbers),
            "assembly_v1": _contains_all(required_numbers, assembly, _numbers),
        },
        "required_periods": required_periods,
        "required_period_hit": {
            "baseline_page_text": _contains_all(required_periods, baseline, _periods),
            "assembly_v1": _contains_all(required_periods, assembly, _periods),
        },
    }


def _rate(values: list[bool | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return round(sum(bool(value) for value in usable) / len(usable), 4) if usable else None


def summarize(records: list[dict]) -> dict:
    selected_pages = sum(item["selected_page_count"] for item in records)
    fallback_pages = sum(item["page_text_fallback_count"] for item in records)

    def metrics(subset: list[dict]) -> dict:
        return {
            "questions": len(subset),
            "table_evidence_coverage": round(
                sum(bool(item["trusted_tables"]) for item in subset) / max(1, len(subset)), 4
            ),
            "page_fallback_ratio": round(
                sum(item["page_text_fallback_count"] for item in subset)
                / max(1, sum(item["selected_page_count"] for item in subset)), 4
            ),
            "average_baseline_evidence_chars": round(
                sum(item["baseline_evidence_chars"] for item in subset) / max(1, len(subset)), 2
            ),
            "average_assembly_evidence_chars": round(
                sum(item["assembly_evidence_chars"] for item in subset) / max(1, len(subset)), 2
            ),
            "baseline_gold_row_hit": _rate([item["gold_row_table_hit"]["baseline_page_text"] for item in subset]),
            "assembly_gold_row_table_hit": _rate([item["gold_row_table_hit"]["assembly_v1"] for item in subset]),
            "baseline_required_number_hit": _rate([item["required_number_hit"]["baseline_page_text"] for item in subset]),
            "assembly_required_number_hit": _rate([item["required_number_hit"]["assembly_v1"] for item in subset]),
            "baseline_required_period_hit": _rate([item["required_period_hit"]["baseline_page_text"] for item in subset]),
            "assembly_required_period_hit": _rate([item["required_period_hit"]["assembly_v1"] for item in subset]),
        }

    return {
        **metrics(records),
        "selected_pages": selected_pages,
        "fallback_pages": fallback_pages,
        "trusted_tables": sum(len(item["trusted_tables"]) for item in records),
        "rejected_tables": sum(len(item["rejected_tables"]) for item in records),
        "question_types": {
            kind: metrics([item for item in records if kind in item["question_types"]]) for kind in QUESTION_TYPES
        },
    }


def _percent(value) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Assembly v1 离线 A/B（冻结 Retrieval）",
        "",
        f"- Frozen report: `{payload['frozen_report']}`",
        f"- Frozen variant: `{payload['variant']}`",
        f"- Questions: {summary['questions']}",
        "- External calls: Jina=0, LLM=0, Judge=0, retrieval=0",
        "- Question types are multi-label; per-type N values therefore need not sum to the total.",
        "- Required numbers come from benchmark justification operands/comparison values; a derived value after the final `=` is excluded.",
        "",
        "## 汇总",
        "",
        f"- Table evidence coverage: {_percent(summary['table_evidence_coverage'])}",
        f"- Page fallback ratio: {_percent(summary['page_fallback_ratio'])}",
        f"- Trusted/rejected tables: {summary['trusted_tables']} / {summary['rejected_tables']}",
        f"- Average evidence chars A/B: {summary['average_baseline_evidence_chars']} / {summary['average_assembly_evidence_chars']}",
        f"- Gold row hit A/B: {_percent(summary['baseline_gold_row_hit'])} / {_percent(summary['assembly_gold_row_table_hit'])}",
        f"- Required number hit A/B: {_percent(summary['baseline_required_number_hit'])} / {_percent(summary['assembly_required_number_hit'])}",
        f"- Required period hit A/B: {_percent(summary['baseline_required_period_hit'])} / {_percent(summary['assembly_required_period_hit'])}",
        "",
        "## Question type",
        "",
        "| Type | N | Table coverage | Page fallback | Gold row A/B | Number A/B | Period A/B |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in QUESTION_TYPES:
        item = summary["question_types"][kind]
        lines.append(
            f"| {kind} | {item['questions']} | {_percent(item['table_evidence_coverage'])} | "
            f"{_percent(item['page_fallback_ratio'])} | {_percent(item['baseline_gold_row_hit'])} / "
            f"{_percent(item['assembly_gold_row_table_hit'])} | {_percent(item['baseline_required_number_hit'])} / "
            f"{_percent(item['assembly_required_number_hit'])} | {_percent(item['baseline_required_period_hit'])} / "
            f"{_percent(item['assembly_required_period_hit'])} |"
        )
    lines.extend(["", "## 逐题", ""])
    for index, item in enumerate(payload["records"], 1):
        selected = ", ".join(
            f"{page['filename']} p.{page['page_number']}" for page in item["frozen_retrieval"]["selected_pages"]
        )
        gold = ", ".join(f"{page['filename']} p.{page['page_number']}" for page in item["gold_evidence_pages"])
        rejected = ", ".join(
            f"{table.get('table_id', '(missing)')} ({table.get('reason', 'unknown')})"
            for table in item["rejected_tables"]
        )
        lines.extend([
            f"### {index}. {item['financebench_id']}",
            "",
            f"- Question: {item['question']}",
            f"- Types: {', '.join(item['question_types'])}",
            f"- Gold evidence page: {gold}",
            f"- Selected/context pages: {selected}",
            f"- Candidate pages: {len(item['frozen_retrieval']['candidate_pages'])}; hash `{item['frozen_retrieval']['candidate_pages_hash']}`",
            f"- Evidence chars A/B: {item['baseline_evidence_chars']} / {item['assembly_evidence_chars']}",
            f"- Trusted tables: {', '.join(item['trusted_tables']) or '(none)'}",
            f"- Rejected tables: {rejected or '(none)'}",
            f"- Gold row/table hit A/B: {item['gold_row_table_hit']['baseline_page_text']} / {item['gold_row_table_hit']['assembly_v1']}",
            f"- Required numbers: {', '.join(item['required_numbers']) or '(n/a)'}",
            f"- Required number hit A/B: {item['required_number_hit']['baseline_page_text']} / {item['required_number_hit']['assembly_v1']}",
            f"- Required periods: {', '.join(item['required_periods']) or '(n/a)'}",
            f"- Required period hit A/B: {item['required_period_hit']['baseline_page_text']} / {item['required_period_hit']['assembly_v1']}",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--frozen-report", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--variant", default="C_global_local_merge")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 30:
        raise SystemExit("--limit must be between 1 and 30")

    frozen = json.loads(args.frozen_report.read_text(encoding="utf-8"))
    traces = {item["financebench_id"]: item for item in frozen["records"]}
    with args.dataset.open(encoding="utf-8-sig", newline="") as handle:
        dataset_rows = list(csv.DictReader(handle))
    rows = select_rows(dataset_rows, list(traces), limit=args.limit)
    records = [_question_record(row, traces[row["financebench_id"]], args.variant) for row in rows]
    payload = {
        "evaluation": "evidence_assembly_v1_ab",
        "evaluation_scope": "frozen retrieval; evidence-only; no external calls",
        "frozen_report": str(args.frozen_report),
        "variant": args.variant,
        "context_page_source": "selected_pages (the frozen report did not persist context page keys separately)",
        "summary": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.output}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
