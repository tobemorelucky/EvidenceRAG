"""Offline A/B evaluation of Table Evidence Expansion v1 on 15 frozen failures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from sqlalchemy import func


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from database import SessionLocal  # noqa: E402
from models import DocumentPage, DocumentTable  # noqa: E402
from table_evidence_expander_v1 import expand_table_evidence_v1  # noqa: E402
from scripts.evaluate_evidence_assembly_ab import (  # noqa: E402
    _contains_all,
    _numbers,
    _parse_gold,
    _periods,
    _required_numbers,
    _rate,
    gold_row_hit,
)


CONFIG = ROOT / "configs/experiments/table_expansion_shadow_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "reports"


class FrozenTableStore:
    def __init__(self, tables: list[dict]):
        self.by_id = {table["table_id"]: table for table in tables}
        self.by_page: dict[str, list[dict]] = defaultdict(list)
        for table in tables:
            self.by_page[table["page_id"]].append(table)

    def get_tables_by_ids(self, table_ids: list[str]) -> list[dict]:
        return [self.by_id[table_id] for table_id in table_ids if table_id in self.by_id]

    def get_tables_by_page_ids(self, page_ids: list[str]) -> list[dict]:
        return [table for page_id in page_ids for table in self.by_page.get(page_id, [])]


def _filename(value: object) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _table_dict(row: DocumentTable) -> dict:
    return {
        "table_id": row.table_id,
        "document_id": row.document_id,
        "page_id": row.page_id,
        "filename": row.filename,
        "page_number": row.page_number,
        "title": row.title,
        "caption": row.caption,
        "columns": list(row.columns or []),
        "rows": list(row.rows or []),
        "unit": row.unit,
        "scale": row.scale,
    }


def load_identities_and_tables(records: list[dict]) -> tuple[dict[tuple[str, int], dict], list[dict]]:
    keys = {
        (_filename(chunk.get("filename")), int(chunk.get("page_number") or 0))
        for record in records for chunk in record.get("context_documents", [])
    }
    filenames = {filename for filename, _ in keys}
    db = SessionLocal()
    try:
        pages = db.query(DocumentPage).filter(func.lower(DocumentPage.filename).in_(filenames)).all()
        page_map = {
            (_filename(page.filename), int(page.page_number)): {
                "document_id": page.document_id,
                "page_id": page.page_id,
            }
            for page in pages
            if (_filename(page.filename), int(page.page_number)) in keys
        }
        page_ids = {value["page_id"] for value in page_map.values() if value["page_id"]}
        tables = db.query(DocumentTable).filter(DocumentTable.page_id.in_(page_ids)).all() if page_ids else []
        return page_map, [_table_dict(table) for table in tables]
    finally:
        db.close()


def enrich_chunks(chunks: list[dict], page_map: dict[tuple[str, int], dict]) -> list[dict]:
    values = []
    for rank, chunk in enumerate(chunks, 1):
        identity = page_map.get((_filename(chunk.get("filename")), int(chunk.get("page_number") or 0)), {})
        values.append({
            **chunk,
            "document_id": identity.get("document_id", ""),
            "page_id": identity.get("page_id", ""),
            "rank": rank,
        })
    return values


def build_record(source: dict, audit_item: dict, dataset_row: dict, page_map: dict, store, config: dict, mode: str) -> dict:
    chunks = enrich_chunks(source.get("context_documents", []), page_map)
    units, expanded_context, trace = expand_table_evidence_v1(
        chunks,
        table_store=store,
        mode=mode,
        max_tables=int(config["TABLE_EXPANSION_MAX_TABLES"]),
        max_table_chars=int(config["TABLE_EXPANSION_MAX_CHARS"]),
        max_context_chars=int(config["CONTEXT_MAX_CHARS"]),
        original_context=source["evidence"],
    )
    baseline = source["evidence"]
    gold = _parse_gold(dataset_row)
    required_numbers = _required_numbers(dataset_row)
    required_periods = _periods(dataset_row["question"])
    before_gold = gold_row_hit(gold, baseline, required_numbers=required_numbers)
    after_gold = gold_row_hit(gold, expanded_context, required_numbers=required_numbers)
    before_number = _contains_all(required_numbers, baseline, _numbers)
    after_number = _contains_all(required_numbers, expanded_context, _numbers)
    before_period = _contains_all(required_periods, baseline, _periods)
    after_period = _contains_all(required_periods, expanded_context, _periods)
    table_units = [unit for unit in units if unit["source_type"] == "table"]
    gold_pages = {(_filename(item["filename"]), int(item["page_number"])) for item in gold}
    added_gold_page_table = any(
        (_filename(unit["filename"]), int(unit["page_number"])) in gold_pages for unit in table_units
    )
    recovered = any((before is False and after is True) for before, after in (
        (before_gold, after_gold), (before_number, after_number), (before_period, after_period)
    ))
    return {
        "question_id": source["financebench_id"],
        "failure_type": audit_item["category"],
        "question": source["question"],
        "mode": mode,
        "before": {
            "context_chars": len(baseline),
            "selected_chunks": [
                {
                    "rank": chunk["rank"],
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": chunk.get("document_id"),
                    "page_id": chunk.get("page_id"),
                    "filename": chunk.get("filename"),
                    "page_number": chunk.get("page_number"),
                    "table_id": chunk.get("table_id") or "",
                }
                for chunk in chunks
            ],
            "evidence_coverage": before_gold,
            "required_number_hit": before_number,
            "required_period_hit": before_period,
        },
        "after": {
            "context_chars": len(expanded_context),
            "expanded_tables": [
                {
                    "table_id": unit["table_id"],
                    "document_id": unit["document_id"],
                    "page_id": unit["page_id"],
                    "filename": unit["filename"],
                    "page_number": unit["page_number"],
                    "title": unit["title"],
                    "header": unit["header"],
                    "unit": unit["unit"],
                    "rows_included": len(unit["rows"]),
                    "association_method": unit["association_method"],
                }
                for unit in table_units
            ],
            "evidence_coverage": after_gold,
            "gold_retention": not before_gold or after_gold,
            "required_number_hit": after_number,
            "required_period_hit": after_period,
        },
        "changes": {
            "new_gold_evidence": before_gold is False and after_gold is True,
            "new_required_number": before_number is False and after_number is True,
            "new_required_period": before_period is False and after_period is True,
            "recovered_key_evidence": recovered,
            "expanded_table_on_gold_page": added_gold_page_table,
        },
        "trace": trace,
    }


def summarize(records: list[dict]) -> dict:
    return {
        "questions": len(records),
        "questions_with_direct_table_id": sum(item["trace"]["direct_table_id_count"] > 0 for item in records),
        "questions_successfully_expanded": sum(item["trace"]["expanded_table_count"] > 0 for item in records),
        "expanded_tables": sum(item["trace"]["expanded_table_count"] for item in records),
        "association_mismatches": sum(item["trace"]["association_mismatch_count"] for item in records),
        "expanded_tables_on_gold_page": sum(item["changes"]["expanded_table_on_gold_page"] for item in records),
        "recovered_key_evidence_questions": sum(item["changes"]["recovered_key_evidence"] for item in records),
        "evidence_coverage": {
            "before": _rate([item["before"]["evidence_coverage"] for item in records]),
            "after": _rate([item["after"]["evidence_coverage"] for item in records]),
        },
        "gold_retention": _rate([item["after"]["gold_retention"] for item in records]),
        "required_number_hit": {
            "before": _rate([item["before"]["required_number_hit"] for item in records]),
            "after": _rate([item["after"]["required_number_hit"] for item in records]),
        },
        "required_period_hit": {
            "before": _rate([item["before"]["required_period_hit"] for item in records]),
            "after": _rate([item["after"]["required_period_hit"] for item in records]),
        },
        "average_context_chars": {
            "before": round(sum(item["before"]["context_chars"] for item in records) / max(1, len(records)), 2),
            "after": round(sum(item["after"]["context_chars"] for item in records) / max(1, len(records)), 2),
        },
        "removed_units": sum(len(item["trace"]["removed_units"]) for item in records),
        "acceptance": {
            "at_least_three_recovered": sum(item["changes"]["recovered_key_evidence"] for item in records) >= 3,
            "no_association_mismatches": not any(item["trace"]["association_mismatch_count"] for item in records),
        },
    }


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Table Evidence Expansion v1 Shadow",
        "",
        f"- Mode: `{payload['mode']}`",
        "- Frozen set: answer_failure_audit_v1, 15 questions",
        "- External calls: Retrieval=0, Jina=0, LLM=0, Judge=0, LangSmith=0",
        "- Gold evidence is used only for offline metrics, never for table selection.",
        "- Page fallback requires exact document_id + page_id; adjacent pages are never opened.",
        "",
        "## Summary",
        "",
        f"- Questions with direct table_id: {summary['questions_with_direct_table_id']}/{summary['questions']}",
        f"- Successfully expanded: {summary['questions_successfully_expanded']}/{summary['questions']}",
        f"- Expanded tables / association mismatches: {summary['expanded_tables']} / {summary['association_mismatches']}",
        f"- Expanded table on gold page: {summary['expanded_tables_on_gold_page']}",
        f"- Recovered key evidence: {summary['recovered_key_evidence_questions']}",
        f"- Evidence coverage A/B: {summary['evidence_coverage']['before']} / {summary['evidence_coverage']['after']}",
        f"- Gold retention: {summary['gold_retention']}",
        f"- Required number hit A/B: {summary['required_number_hit']['before']} / {summary['required_number_hit']['after']}",
        f"- Required period hit A/B: {summary['required_period_hit']['before']} / {summary['required_period_hit']['after']}",
        f"- Average context chars A/B: {summary['average_context_chars']['before']} / {summary['average_context_chars']['after']}",
        f"- Removed low-rank units: {summary['removed_units']}",
        f"- Acceptance: `{json.dumps(summary['acceptance'], ensure_ascii=False)}`",
        "",
        "## Per question",
        "",
    ]
    for index, item in enumerate(payload["records"], 1):
        before_chunks = "; ".join(
            f"#{chunk['rank']} {chunk['filename']} p.{chunk['page_number']} table_id={chunk['table_id'] or '(none)'}"
            for chunk in item["before"]["selected_chunks"]
        )
        after_tables = "; ".join(
            f"{table['table_id']} ({table['filename']} p.{table['page_number']}, {table['association_method']}, rows={table['rows_included']})"
            for table in item["after"]["expanded_tables"]
        ) or "(none)"
        lines.extend([
            f"### {index}. {item['question_id']} — `{item['failure_type']}`",
            "",
            f"- Question: {item['question']}",
            f"- Before selected chunks: {before_chunks}",
            f"- After expanded tables: {after_tables}",
            f"- Chars A/B: {item['before']['context_chars']} / {item['after']['context_chars']}",
            f"- Evidence coverage A/B: {item['before']['evidence_coverage']} / {item['after']['evidence_coverage']}",
            f"- Required number A/B: {item['before']['required_number_hit']} / {item['after']['required_number_hit']}",
            f"- Required period A/B: {item['before']['required_period_hit']} / {item['after']['required_period_hit']}",
            f"- New evidence: `{json.dumps(item['changes'], ensure_ascii=False)}`",
            f"- Removed units: `{json.dumps(item['trace']['removed_units'], ensure_ascii=False)}`",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct_table_id", "page_table_fallback"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    mode = args.mode or config["TABLE_EXPANSION_MODE"]
    state = json.loads((ROOT / config["source_state"]).read_text(encoding="utf-8"))
    audit = json.loads((ROOT / config["audit_json"]).read_text(encoding="utf-8"))
    with (ROOT / config["dataset"]).open(encoding="utf-8-sig", newline="") as handle:
        dataset = {row["financebench_id"]: row for row in csv.DictReader(handle)}
    audit_items = {item["question_id"]: item for item in audit["items"]}
    source = {record["financebench_id"]: record for record in state["records"]}
    records = [source[item["question_id"]] for item in audit["items"]]
    page_map, tables = load_identities_and_tables(records)
    store = FrozenTableStore(tables)
    evaluated = [
        build_record(record, audit_items[record["financebench_id"]], dataset[record["financebench_id"]], page_map, store, config, mode)
        for record in records
    ]
    payload = {
        "experiment": config["name"],
        "mode": mode,
        "production_enabled": config["ENABLE_TABLE_EXPANSION"],
        "external_calls": config["external_calls"],
        "summary": summarize(evaluated),
        "records": evaluated,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"table_expansion_shadow_v1_{mode}"
    json_path = args.output_dir / f"{stem}.json"
    markdown_path = args.output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
