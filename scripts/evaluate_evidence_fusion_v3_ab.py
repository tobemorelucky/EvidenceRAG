"""Offline Evidence Fusion v2 versus v3 evaluation on frozen selected pages."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evidence_fusion_v2 import build_evidence_fusion_v2  # noqa: E402
from evidence_fusion_v3 import build_evidence_fusion_v3  # noqa: E402
from scripts.evaluate_evidence_assembly_ab import (  # noqa: E402
    QUESTION_TYPES,
    _contains_all,
    _dedupe_page_keys,
    _hash_page_keys,
    _load_pages,
    _load_tables,
    _numbers,
    _parse_gold,
    _percent,
    _periods,
    _rate,
    _required_numbers,
    _trusted_table_evidence,
    classify_question,
    gold_row_hit,
    select_rows,
)


DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_FROZEN = ROOT / "reports" / "retrieval_document_local_diagnostic30.json"
DEFAULT_OUTPUT = ROOT / "reports" / "evidence_fusion_v3_ab30.json"


def _version_metrics(evidence: str, trace: dict, gold: list[dict], numbers: list[str], periods: list[str]) -> dict:
    table_evidence = _trusted_table_evidence(evidence)
    return {
        "evidence_chars": len(evidence),
        "gold_row_hit": gold_row_hit(gold, evidence, required_numbers=numbers),
        "required_number_hit": _contains_all(numbers, evidence, _numbers),
        "required_period_hit": _contains_all(periods, evidence, _periods),
        "table_contribution": {
            "trusted_table_count": trace["trusted_table_count"],
            "trusted_table_ids": trace["trusted_table_ids"],
            "rejected_table_count": trace["rejected_table_count"],
            "chars": trace["table_contribution_chars"],
            "ratio": trace["table_contribution_ratio"],
            "gold_row_hit": gold_row_hit(gold, table_evidence, required_numbers=numbers),
            "required_number_hit": _contains_all(numbers, table_evidence, _numbers),
        },
        "row_selection": trace.get("row_selection") or [],
    }


def build_record(row: dict, trace: dict, variant_name: str) -> dict:
    selected_keys = _dedupe_page_keys(trace["variants"][variant_name].get("selected_pages") or [])
    pages = _load_pages(selected_keys)
    tables = _load_tables([page["page_id"] for page in pages if page.get("page_id")])
    fusion_v2, trace_v2 = build_evidence_fusion_v2(row["question"], pages, tables)
    fusion_v3, trace_v3 = build_evidence_fusion_v3(row["question"], pages, tables)
    gold = _parse_gold(row)
    numbers = _required_numbers(row)
    periods = _periods(row["question"])
    trusted_ids = set(trace_v3["trusted_table_ids"])
    raw_trusted_tables = json.dumps(
        [table for table in tables if table.get("table_id") in trusted_ids],
        ensure_ascii=False,
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
            "selected_pages": [{"filename": key[0], "page_number": key[1]} for key in selected_keys],
            "context_pages": [{"filename": key[0], "page_number": key[1]} for key in selected_keys],
            "selected_pages_hash": _hash_page_keys(selected_keys),
        },
        "required_numbers": numbers,
        "required_periods": periods,
        "fusion_v2": _version_metrics(fusion_v2, trace_v2, gold, numbers, periods),
        "fusion_v3": _version_metrics(fusion_v3, trace_v3, gold, numbers, periods),
        "raw_trusted_table_ceiling": {
            "required_number_hit": _contains_all(numbers, raw_trusted_tables, _numbers),
            "gold_row_hit": gold_row_hit(gold, raw_trusted_tables, required_numbers=numbers),
        },
    }


def summarize(records: list[dict]) -> dict:
    def metrics(subset: list[dict]) -> dict:
        count = max(1, len(subset))

        def version(name: str) -> dict:
            return {
                "average_chars": round(sum(item[name]["evidence_chars"] for item in subset) / count, 2),
                "gold_row_hit": _rate([item[name]["gold_row_hit"] for item in subset]),
                "required_number_hit": _rate([item[name]["required_number_hit"] for item in subset]),
                "required_period_hit": _rate([item[name]["required_period_hit"] for item in subset]),
                "table_contribution_coverage": round(
                    sum(item[name]["table_contribution"]["trusted_table_count"] > 0 for item in subset) / count,
                    4,
                ),
                "average_table_chars": round(
                    sum(item[name]["table_contribution"]["chars"] for item in subset) / count,
                    2,
                ),
                "average_table_ratio": round(
                    sum(item[name]["table_contribution"]["ratio"] for item in subset) / count,
                    4,
                ),
                "table_gold_row_hit": _rate(
                    [item[name]["table_contribution"]["gold_row_hit"] for item in subset]
                ),
                "table_required_number_hit": _rate(
                    [item[name]["table_contribution"]["required_number_hit"] for item in subset]
                ),
            }

        return {
            "questions": len(subset),
            "fusion_v2": version("fusion_v2"),
            "fusion_v3": version("fusion_v3"),
            "gold_row_gains": sum(not item["fusion_v2"]["gold_row_hit"] and item["fusion_v3"]["gold_row_hit"] for item in subset),
            "gold_row_regressions": sum(item["fusion_v2"]["gold_row_hit"] and not item["fusion_v3"]["gold_row_hit"] for item in subset),
            "required_number_gains": sum(
                item["fusion_v2"]["required_number_hit"] is False
                and item["fusion_v3"]["required_number_hit"] is True
                for item in subset
            ),
            "required_number_regressions": sum(
                item["fusion_v2"]["required_number_hit"] is True
                and item["fusion_v3"]["required_number_hit"] is False
                for item in subset
            ),
            "raw_trusted_table_required_number_ceiling": _rate(
                [item["raw_trusted_table_ceiling"]["required_number_hit"] for item in subset]
            ),
            "raw_trusted_table_gold_row_ceiling": _rate(
                [item["raw_trusted_table_ceiling"]["gold_row_hit"] for item in subset]
            ),
        }

    return {
        **metrics(records),
        "selected_pages": sum(len(item["frozen_retrieval"]["selected_pages"]) for item in records),
        "row_selector_tables": sum(len(item["fusion_v3"]["row_selection"]) for item in records),
        "question_types": {
            kind: metrics([item for item in records if kind in item["question_types"]]) for kind in QUESTION_TYPES
        },
    }


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    v2 = summary["fusion_v2"]
    v3 = summary["fusion_v3"]
    lines = [
        "# Evidence Fusion v3 离线 A/B（冻结 Retrieval）",
        "",
        f"- Frozen report: `{payload['frozen_report']}`",
        f"- Frozen variant: `{payload['variant']}`",
        f"- Questions: {summary['questions']}",
        "- External calls: Retrieval=0, Jina=0, LLM=0, Judge=0",
        "- A: Fusion v2; B: Fusion v3 deterministic row relevance selector.",
        "- Question types are multi-label.",
        "",
        "## 汇总",
        "",
        f"- Average chars v2/v3: {v2['average_chars']} / {v3['average_chars']}",
        f"- Gold row hit v2/v3: {_percent(v2['gold_row_hit'])} / {_percent(v3['gold_row_hit'])}",
        f"- Required number hit v2/v3: {_percent(v2['required_number_hit'])} / {_percent(v3['required_number_hit'])}",
        f"- Required period hit v2/v3: {_percent(v2['required_period_hit'])} / {_percent(v3['required_period_hit'])}",
        f"- Table contribution coverage v2/v3: {_percent(v2['table_contribution_coverage'])} / {_percent(v3['table_contribution_coverage'])}",
        f"- Average table chars v2/v3: {v2['average_table_chars']} / {v3['average_table_chars']}",
        f"- Table-only gold row hit v2/v3: {_percent(v2['table_gold_row_hit'])} / {_percent(v3['table_gold_row_hit'])}",
        f"- Table-only required number hit v2/v3: {_percent(v2['table_required_number_hit'])} / {_percent(v3['table_required_number_hit'])}",
        f"- Gold row gains/regressions: {summary['gold_row_gains']} / {summary['gold_row_regressions']}",
        f"- Required number gains/regressions: {summary['required_number_gains']} / {summary['required_number_regressions']}",
        f"- Raw trusted-table number/row ceiling: {_percent(summary['raw_trusted_table_required_number_ceiling'])} / {_percent(summary['raw_trusted_table_gold_row_ceiling'])}",
        "",
        "## Question type",
        "",
        "| Type | N | Gold row v2/v3 | Number v2/v3 | Period v2/v3 | Table row v2/v3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for kind in QUESTION_TYPES:
        item = summary["question_types"][kind]
        a = item["fusion_v2"]
        b = item["fusion_v3"]
        lines.append(
            f"| {kind} | {item['questions']} | {_percent(a['gold_row_hit'])} / {_percent(b['gold_row_hit'])} | "
            f"{_percent(a['required_number_hit'])} / {_percent(b['required_number_hit'])} | "
            f"{_percent(a['required_period_hit'])} / {_percent(b['required_period_hit'])} | "
            f"{_percent(a['table_gold_row_hit'])} / {_percent(b['table_gold_row_hit'])} |"
        )
    lines.extend(["", "## 逐题", ""])
    for index, item in enumerate(payload["records"], 1):
        a = item["fusion_v2"]
        b = item["fusion_v3"]
        selected = ", ".join(
            f"{page['filename']} p.{page['page_number']}" for page in item["frozen_retrieval"]["selected_pages"]
        )
        lines.extend([
            f"### {index}. {item['financebench_id']}",
            "",
            f"- Question: {item['question']}",
            f"- Types: {', '.join(item['question_types'])}",
            f"- Frozen selected/context pages: {selected}",
            f"- Frozen page hash: `{item['frozen_retrieval']['selected_pages_hash']}`",
            f"- Average chars v2/v3: {a['evidence_chars']} / {b['evidence_chars']}",
            f"- Gold row hit v2/v3: {a['gold_row_hit']} / {b['gold_row_hit']}",
            f"- Required number hit v2/v3: {a['required_number_hit']} / {b['required_number_hit']}",
            f"- Required period hit v2/v3: {a['required_period_hit']} / {b['required_period_hit']}",
            f"- Table chars v2/v3: {a['table_contribution']['chars']} / {b['table_contribution']['chars']}",
            f"- Table gold row hit v2/v3: {a['table_contribution']['gold_row_hit']} / {b['table_contribution']['gold_row_hit']}",
            f"- v3 row-selected tables: {len(b['row_selection'])}",
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
    records = [build_record(row, traces[row["financebench_id"]], args.variant) for row in rows]
    payload = {
        "evaluation": "evidence_fusion_v3_ab",
        "evaluation_scope": "frozen selected pages; evidence-only; no external calls",
        "frozen_report": str(args.frozen_report),
        "variant": args.variant,
        "summary": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.output}")
    print(f"Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
