"""Audit why PostgreSQL tables fail Evidence Fact Store v1 extraction.

This is an offline diagnostic. FinanceBench gold pages are used only after
table classification to measure relevance; they never influence extraction or
failure labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from database import SessionLocal  # noqa: E402
from evidence_fact_store_v1 import facts_from_table  # noqa: E402
from models import DocumentTable  # noqa: E402
from table_store import TableStore  # noqa: E402


DEFAULT_FACT_STORE = ROOT / "reports" / "fact_store_v1.json"
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json"
DEFAULT_OUTPUT = ROOT / "reports" / "table_fact_extraction_failure_audit_v1.json"

_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_NUMBER_RE = re.compile(r"\(?\s*[-+]?\s*\$?\s*\d[\d,]*(?:\.\d+)?\s*%?\s*\)?")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_UNIT_RE = re.compile(r"\b(?:millions?|thousands?|USD)\b|%|\$", re.IGNORECASE)
_CATEGORY_NAMES = {
    "A": "header/year recovery failure",
    "B": "multi-level header",
    "C": "row alignment failure",
    "D": "unit/scale failure",
    "E": "parser empty/low quality",
    "F": "text-like table",
}


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _row_text(row: object) -> str:
    if isinstance(row, dict):
        return _clean(row.get("_raw_line")) or " ".join(
            _clean(value) for key, value in row.items()
            if not str(key).startswith("_") and _clean(value)
        )
    if isinstance(row, (list, tuple)):
        return " ".join(_clean(value) for value in row if _clean(value))
    return ""


def _table_text(table: dict) -> str:
    parts = [table.get("title"), table.get("caption"), table.get("before_context"), table.get("after_context")]
    parts.extend(table.get("columns") or [])
    parts.extend(_row_text(row) for row in table.get("rows") or [])
    parts.append(table.get("csv_text"))
    return "\n".join(_clean(part) for part in parts if _clean(part))


def _metric_row_count(table: dict) -> int:
    count = 0
    for row in table.get("rows") or []:
        line = _row_text(row)
        matches = list(_NUMBER_RE.finditer(line))
        if len(matches) < 2 or not _WORD_RE.search(line[: matches[0].start()]):
            continue
        tail = line[matches[0].start():]
        remainder = _NUMBER_RE.sub(" ", tail)
        # A table row normally becomes punctuation/whitespace after removing
        # its numeric cells. Narrative sentences still contain many words.
        if len(_WORD_RE.findall(remainder)) <= 4 or "..." in line:
            count += 1
    return count


def _header_like_year_row(value: str) -> bool:
    years = _YEAR_RE.findall(value)
    if len(years) < 2:
        return False
    return len(value) <= 140 and len(_WORD_RE.findall(value)) <= 8


def _is_text_like(table: dict, text: str) -> bool:
    columns = [_clean(item) for item in table.get("columns") or [] if _clean(item)]
    rows = [row for row in table.get("rows") or [] if isinstance(row, (dict, list, tuple))]
    words = len(_WORD_RE.findall(text))
    numbers = len(_NUMBER_RE.findall(text))
    weak_grid = len(columns) < 2 or len(rows) < 2
    row_word_counts = [len(_WORD_RE.findall(_row_text(row))) for row in rows]
    narrative_rows = bool(row_word_counts) and sum(row_word_counts) / len(row_word_counts) >= 8
    return weak_grid and words >= 12 and (numbers / max(1, words) < 0.25 or narrative_rows)


def classify_failed_table(table: dict, *, quality_threshold: float = 0.65) -> dict:
    facts, trace = facts_from_table(table, quality_threshold=quality_threshold)
    if facts:
        return {"passed": True, "fact_count": len(facts), "trace": trace}

    text = _table_text(table)
    columns = [_clean(item) for item in table.get("columns") or [] if _clean(item)]
    rows = list(table.get("rows") or [])
    years_anywhere = list(dict.fromkeys(_YEAR_RE.findall(text)))
    column_years = [year for column in columns for year in _YEAR_RE.findall(column)]
    row_year_sequences = [_YEAR_RE.findall(_row_text(row)) for row in rows[:6]]
    duplicate_year = bool(trace.get("ambiguous_period_header")) or any(
        len(sequence) != len(set(sequence)) for sequence in row_year_sequences if sequence
    )
    duplicate_columns = len([item.casefold() for item in columns]) != len(set(item.casefold() for item in columns))
    merged_header = any(len(_YEAR_RE.findall(column)) >= 2 for column in columns)
    year_header_rows = sum(
        _header_like_year_row(_row_text(row)) for row in rows[:6]
    )
    multi_level_header = duplicate_year or duplicate_columns or merged_header or year_header_rows >= 2
    metric_rows = _metric_row_count(table)
    unit_cues = list(dict.fromkeys(match.group(0) for match in _UNIT_RE.finditer(text)))
    unit_bound = bool(_clean(table.get("unit")) or _clean(table.get("scale")))
    text_like = _is_text_like(table, text)

    labels = []
    table_like_numeric = metric_rows > 0 or len(columns) >= 2
    if years_anywhere and not trace.get("periods") and table_like_numeric:
        labels.append("A")
    if multi_level_header:
        labels.append("B")
    if metric_rows and trace.get("reason") == "no_aligned_numeric_rows":
        labels.append("C")
    if unit_cues and not unit_bound:
        labels.append("D")
    if trace.get("reason") in {"empty_or_narrow_structure", "quality_below_threshold", "missing_identity"}:
        labels.append("E")
    if text_like:
        labels.append("F")

    # Root-cause precedence follows the extraction contract: ambiguous headers
    # before alignment, recoverable year-header failures before generic parser
    # quality, and narrative/table-shape failures before the E catch-all.
    if multi_level_header:
        primary = "B"
    elif text_like:
        primary = "F"
    elif years_anywhere and not trace.get("periods") and table_like_numeric:
        primary = "A"
    elif trace.get("reason") == "no_aligned_numeric_rows":
        primary = "C"
    elif trace.get("reason") in {"empty_or_narrow_structure", "quality_below_threshold", "missing_identity"}:
        primary = "E"
    else:
        primary = "E"
    if primary not in labels:
        labels.append(primary)

    return {
        "passed": False,
        "primary_category": primary,
        "primary_category_name": _CATEGORY_NAMES[primary],
        "diagnostic_labels": sorted(set(labels)),
        "fact_extraction_reason": trace.get("reason"),
        "features": {
            "years_anywhere": years_anywhere,
            "column_years": column_years,
            "periods_recovered": trace.get("periods") or [],
            "duplicate_year": duplicate_year,
            "duplicate_columns": duplicate_columns,
            "merged_header": merged_header,
            "year_header_rows": year_header_rows,
            "metric_rows": metric_rows,
            "unit_cues": unit_cues,
            "unit_bound": unit_bound,
            "text_like": text_like,
            "stored_quality_score": trace.get("stored_quality_score"),
            "structural_quality_score": trace.get("structural_quality_score"),
        },
        "trace": trace,
    }


def _preview(table: dict, *, rows: int = 3) -> dict:
    return {
        "table_id": table.get("table_id"),
        "document_id": table.get("document_id"),
        "page_id": table.get("page_id"),
        "filename": table.get("filename"),
        "page_number": int(table.get("page_number") or 0),
        "title": _clean(table.get("title") or table.get("caption"))[:500],
        "header": [_clean(item) for item in table.get("columns") or []],
        "rows_preview": [_row_text(row)[:500] for row in list(table.get("rows") or [])[:rows]],
        "unit": _clean(table.get("unit")),
        "scale": _clean(table.get("scale")),
    }


def _load_tables() -> list[dict]:
    db = SessionLocal()
    try:
        records = db.query(DocumentTable).order_by(
            DocumentTable.document_id.asc(), DocumentTable.page_number.asc(), DocumentTable.table_index.asc()
        ).all()
        return [TableStore._to_dict(record) for record in records]
    finally:
        db.close()


def _fixed30_ids(path: Path) -> tuple[list[str], dict[str, str]]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    ids = []
    groups = {}
    for group in ("candidate_miss10", "selection_loss10", "correct_regression10"):
        for item in fixture[group]:
            question_id = item["financebench_id"] if isinstance(item, dict) else item
            ids.append(question_id)
            groups[question_id] = group
    if len(ids) != 30 or len(set(ids)) != 30:
        raise ValueError("Expected a unique fixed 10+10+10 diagnostic set")
    return ids, groups


def _filename(value: object) -> str:
    name = str(value or "").replace("\\", "/").rsplit("/", 1)[-1]
    return name.casefold() if name.casefold().endswith(".pdf") else f"{name}.pdf".casefold()


def _gold_page_audit(
    tables: list[dict], analyses: dict[str, dict], fact_table_ids: set[str], dataset: Path, fixture: Path
) -> tuple[list[dict], dict]:
    ids, groups = _fixed30_ids(fixture)
    with dataset.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["financebench_id"]: row for row in csv.DictReader(handle) if row["financebench_id"] in set(ids)}
    by_page = defaultdict(list)
    for table in tables:
        by_page[(_filename(table.get("filename")), int(table.get("page_number") or 0))].append(table)

    records = []
    for question_id in ids:
        row = rows[question_id]
        gold_pages = []
        for evidence in json.loads(row.get("evidence") or "[]"):
            key = (_filename(evidence.get("doc_name")), int(evidence.get("evidence_page_num") or 0))
            if key not in gold_pages:
                gold_pages.append(key)
        page_records = []
        all_tables = []
        for filename, page_number in gold_pages:
            page_tables = by_page.get((filename, page_number), [])
            all_tables.extend(page_tables)
            page_records.append({
                "filename": filename,
                "page_number": page_number,
                "table_count": len(page_tables),
                "tables": [
                    {
                        **_preview(table),
                        "has_facts": table["table_id"] in fact_table_ids,
                        "failure_category": None if table["table_id"] in fact_table_ids else analyses[table["table_id"]]["primary_category"],
                        "failure_reason": None if table["table_id"] in fact_table_ids else analyses[table["table_id"]]["fact_extraction_reason"],
                    }
                    for table in page_tables
                ],
            })
        failed = [table for table in all_tables if table["table_id"] not in fact_table_ids]
        records.append({
            "question_id": question_id,
            "group": groups[question_id],
            "question": row.get("question"),
            "gold_pages": page_records,
            "gold_page_table_count": len(all_tables),
            "fact_covered_table_count": sum(table["table_id"] in fact_table_ids for table in all_tables),
            "fact_covered": any(table["table_id"] in fact_table_ids for table in all_tables),
            "no_table_on_gold_page": not all_tables,
            "failed_table_reasons": [
                {
                    "table_id": table["table_id"],
                    "category": analyses[table["table_id"]]["primary_category"],
                    "reason": analyses[table["table_id"]]["fact_extraction_reason"],
                }
                for table in failed
            ],
        })
    failed_counter = Counter(
        reason["category"] for record in records for reason in record["failed_table_reasons"]
    )
    return records, {
        "questions": len(records),
        "gold_pages": sum(len(record["gold_pages"]) for record in records),
        "gold_page_tables": sum(record["gold_page_table_count"] for record in records),
        "fact_covered_tables": sum(record["fact_covered_table_count"] for record in records),
        "failed_gold_tables": sum(len(record["failed_table_reasons"]) for record in records),
        "questions_with_any_gold_page_table": sum(not record["no_table_on_gold_page"] for record in records),
        "questions_with_fact_covered_gold_table": sum(record["fact_covered"] for record in records),
        "questions_with_recoverable_abc_gold_table": sum(
            any(reason["category"] in "ABC" for reason in record["failed_table_reasons"]) for record in records
        ),
        "questions_with_fact_or_recoverable_abc_gold_table": sum(
            record["fact_covered"] or any(reason["category"] in "ABC" for reason in record["failed_table_reasons"])
            for record in records
        ),
        "failed_gold_table_categories": dict(sorted(failed_counter.items())),
    }


def _percent(count: int, total: int) -> str:
    return f"{count / max(1, total):.2%}"


def _markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Table Fact Extraction Failure Audit v1", "",
        "- External calls: LLM=0, Jina=0, Judge=0, LangSmith=0",
        "- A–F primary categories are mutually exclusive; diagnostic labels may overlap.",
        "- FinanceBench gold pages are used only in the final fixed30 audit and never influence classification.",
        "- Unit/scale absence is not a Fact Store v1 rejection gate; D is therefore primarily a risk label rather than an assumed root cause.", "",
        "## 全库汇总", "",
        f"- PostgreSQL tables: {summary['tables_total']}",
        f"- Fact tables: {summary['tables_with_facts']}",
        f"- Failed tables: {summary['failed_tables']}", "",
        "| Category | Meaning | Count | Failed share | Diagnostic-label count |", "|---|---|---:|---:|---:|",
    ]
    for category in _CATEGORY_NAMES:
        item = summary["categories"][category]
        lines.append(
            f"| {category} | {_CATEGORY_NAMES[category]} | {item['primary_count']} | "
            f"{item['primary_share']:.2%} | {item['diagnostic_label_count']} |"
        )
    lines += [
        "", "## Fact Store v1 原始拒绝原因", "",
        "| Reason | Count | Failed share |", "|---|---:|---:|",
    ]
    for reason, count in summary["fact_extraction_reason_counts"].items():
        lines.append(f"| {reason} | {count} | {_percent(count, summary['failed_tables'])} |")
    lines += [
        "", "## 成本信号", "",
        f"- Potentially structure-recoverable A+B+C: {summary['potentially_recoverable_abc']} "
        f"({_percent(summary['potentially_recoverable_abc'], summary['failed_tables'])} of failures).",
        f"- Parser/text-like E+F: {summary['parser_or_text_like_ef']} "
        f"({_percent(summary['parser_or_text_like_ef'], summary['failed_tables'])} of failures).",
        f"- Unit/scale risk label D: {summary['unit_scale_risk_tables']} tables; this does not by itself explain extraction rejection.", "",
        "## 分类示例", "",
    ]
    for category in _CATEGORY_NAMES:
        lines += [f"### {category}. {_CATEGORY_NAMES[category]}", ""]
        examples = payload["examples"].get(category) or []
        if not examples:
            lines += ["(none)", ""]
            continue
        for example in examples:
            lines += [
                f"- `{example['table_id']}` — {example['filename']} p.{example['page_number']}",
                f"  - Title: {example['title'] or '(empty)'}",
                f"  - Header: `{example['header']}`",
                f"  - Rows: `{example['rows_preview']}`",
                f"  - Extraction reason: `{example['fact_extraction_reason']}`; labels: `{example['diagnostic_labels']}`",
            ]
        lines.append("")
    gold = payload["fixed30_gold_page_audit"]["summary"]
    lines += [
        "## 固定30题 Gold Page Table 审计", "",
        f"- Gold pages: {gold['gold_pages']}",
        f"- Gold-page tables: {gold['gold_page_tables']}",
        f"- Fact-covered tables: {gold['fact_covered_tables']}",
        f"- Failed gold-page tables: {gold['failed_gold_tables']}",
        f"- Questions with any gold-page table: {gold['questions_with_any_gold_page_table']}/30",
        f"- Questions with at least one fact-covered gold table: {gold['questions_with_fact_covered_gold_table']}/30",
        f"- Questions with an A/B/C failed gold table: {gold['questions_with_recoverable_abc_gold_table']}/30",
        f"- Optimistic fact-covered-or-A/B/C upper bound: {gold['questions_with_fact_or_recoverable_abc_gold_table']}/30",
        f"- Failed gold-table categories: `{gold['failed_gold_table_categories']}`", "",
        "| ID | Group | Gold tables | Fact tables | Covered | Failure categories |", "|---|---|---:|---:|---|---|",
    ]
    for record in payload["fixed30_gold_page_audit"]["records"]:
        categories = ", ".join(reason["category"] for reason in record["failed_table_reasons"]) or "-"
        lines.append(
            f"| {record['question_id']} | {record['group']} | {record['gold_page_table_count']} | "
            f"{record['fact_covered_table_count']} | {record['fact_covered']} | {categories} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fact-store", type=Path, default=DEFAULT_FACT_STORE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--examples-per-category", type=int, default=5)
    args = parser.parse_args()
    if args.examples_per_category < 1:
        parser.error("--examples-per-category must be positive")

    fact_store = json.loads(args.fact_store.read_text(encoding="utf-8"))
    if fact_store.get("schema") != "evidence_fact_store_v1":
        raise ValueError("Not an Evidence Fact Store v1 file")
    fact_table_ids = {fact["table_id"] for fact in fact_store["facts"]}
    quality_threshold = float(fact_store.get("stats", {}).get("quality_threshold", 0.65))
    tables = _load_tables()

    analyses = {}
    failed_records = []
    for table in tables:
        analysis = classify_failed_table(table, quality_threshold=quality_threshold)
        analyses[table["table_id"]] = analysis
        in_store = table["table_id"] in fact_table_ids
        if bool(analysis["passed"]) != in_store:
            raise ValueError(f"Fact store/table classifier drift for {table['table_id']}")
        if in_store:
            continue
        failed_records.append({**_preview(table), **analysis})

    primary_counts = Counter(record["primary_category"] for record in failed_records)
    label_counts = Counter(label for record in failed_records for label in record["diagnostic_labels"])
    total_failed = len(failed_records)
    categories = {
        category: {
            "name": _CATEGORY_NAMES[category],
            "primary_count": primary_counts[category],
            "primary_share": round(primary_counts[category] / max(1, total_failed), 6),
            "diagnostic_label_count": label_counts[category],
        }
        for category in _CATEGORY_NAMES
    }
    examples = {
        category: [record for record in failed_records if record["primary_category"] == category][: args.examples_per_category]
        for category in _CATEGORY_NAMES
    }
    gold_records, gold_summary = _gold_page_audit(
        tables, analyses, fact_table_ids, args.dataset, args.fixture
    )
    summary = {
        "tables_total": len(tables),
        "tables_with_facts": len(fact_table_ids),
        "failed_tables": total_failed,
        "categories": categories,
        "fact_extraction_reason_counts": dict(sorted(Counter(
            record["fact_extraction_reason"] for record in failed_records
        ).items())),
        "potentially_recoverable_abc": sum(primary_counts[item] for item in "ABC"),
        "parser_or_text_like_ef": sum(primary_counts[item] for item in "EF"),
        "unit_scale_risk_tables": label_counts["D"],
        "primary_category_total_check": sum(primary_counts.values()),
    }
    payload = {
        "schema": "table_fact_extraction_failure_audit_v1",
        "inputs": {"fact_store": str(args.fact_store), "table_store": "postgresql.document_tables", "fixed30_fixture": str(args.fixture)},
        "external_calls": {"llm": 0, "jina": 0, "judge": 0, "langsmith": 0},
        "category_definitions": _CATEGORY_NAMES,
        "summary": summary,
        "examples": examples,
        "failed_tables": failed_records,
        "fixed30_gold_page_audit": {"summary": gold_summary, "records": gold_records},
    }
    if summary["primary_category_total_check"] != total_failed:
        raise AssertionError("Primary categories must partition every failed table exactly once")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"summary": summary, "fixed30": gold_summary}, ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
