"""Offline page-text versus Evidence Fusion v2 evaluation on frozen pages."""

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
    build_page_text_evidence,
    classify_question,
    gold_row_hit,
    select_rows,
)


DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_FROZEN = ROOT / "reports" / "retrieval_document_local_diagnostic30.json"
DEFAULT_OUTPUT = ROOT / "reports" / "evidence_fusion_v2_ab30.json"


def build_record(row: dict, trace: dict, variant_name: str) -> dict:
    variant = trace["variants"][variant_name]
    selected_keys = _dedupe_page_keys(variant.get("selected_pages") or [])
    candidate_trace = trace.get("page_candidate_traces", {}).get(variant_name, {})
    candidate_keys = _dedupe_page_keys(candidate_trace.get("expanded_pages") or [])
    pages = _load_pages(selected_keys)
    tables = _load_tables([page["page_id"] for page in pages if page.get("page_id")])
    baseline = build_page_text_evidence(pages)
    fusion, fusion_trace = build_evidence_fusion_v2(row["question"], pages, tables)
    table_only = _trusted_table_evidence(fusion)
    gold = _parse_gold(row)
    required_numbers = _required_numbers(row)
    required_periods = _periods(row["question"])

    baseline_number_hit = _contains_all(required_numbers, baseline, _numbers)
    fusion_number_hit = _contains_all(required_numbers, fusion, _numbers)
    return {
        "financebench_id": row["financebench_id"],
        "question": row["question"],
        "question_types": sorted(classify_question(row)),
        "gold_evidence_pages": [
            {"filename": item["filename"], "page_number": item["page_number"]} for item in gold
        ],
        "frozen_retrieval": {
            "source_variant": variant_name,
            "candidate_page_count": len(candidate_keys),
            "candidate_pages_hash": _hash_page_keys(candidate_keys),
            "selected_pages": [{"filename": item[0], "page_number": item[1]} for item in selected_keys],
            "context_pages": [{"filename": item[0], "page_number": item[1]} for item in selected_keys],
            "selected_pages_hash": _hash_page_keys(selected_keys),
        },
        "average_inputs": {
            "selected_page_count": len(pages),
            "loaded_table_count": len(tables),
        },
        "evidence_chars": {
            "baseline_page_text": len(baseline),
            "fusion_v2": len(fusion),
        },
        "gold_row_hit": {
            "baseline_page_text": gold_row_hit(gold, baseline, required_numbers=required_numbers),
            "fusion_v2": gold_row_hit(gold, fusion, required_numbers=required_numbers),
        },
        "required_numbers": required_numbers,
        "required_number_hit": {
            "baseline_page_text": baseline_number_hit,
            "fusion_v2": fusion_number_hit,
        },
        "required_periods": required_periods,
        "required_period_hit": {
            "baseline_page_text": _contains_all(required_periods, baseline, _periods),
            "fusion_v2": _contains_all(required_periods, fusion, _periods),
        },
        "table_contribution": {
            "trusted_table_count": fusion_trace["trusted_table_count"],
            "trusted_table_ids": fusion_trace["trusted_table_ids"],
            "rejected_table_count": fusion_trace["rejected_table_count"],
            "chars": fusion_trace["table_contribution_chars"],
            "ratio": fusion_trace["table_contribution_ratio"],
            "gold_row_hit_in_table_layer": gold_row_hit(
                gold,
                table_only,
                required_numbers=required_numbers,
            ),
            "required_number_hit_in_table_layer": _contains_all(required_numbers, table_only, _numbers),
            "required_number_recovered_over_baseline": baseline_number_hit is False and fusion_number_hit is True,
        },
        "fusion_trace": fusion_trace,
    }


def summarize(records: list[dict]) -> dict:
    def metrics(subset: list[dict]) -> dict:
        count = max(1, len(subset))
        return {
            "questions": len(subset),
            "average_baseline_chars": round(
                sum(item["evidence_chars"]["baseline_page_text"] for item in subset) / count, 2
            ),
            "average_fusion_chars": round(
                sum(item["evidence_chars"]["fusion_v2"] for item in subset) / count, 2
            ),
            "baseline_gold_row_hit": _rate([item["gold_row_hit"]["baseline_page_text"] for item in subset]),
            "fusion_gold_row_hit": _rate([item["gold_row_hit"]["fusion_v2"] for item in subset]),
            "baseline_required_number_hit": _rate(
                [item["required_number_hit"]["baseline_page_text"] for item in subset]
            ),
            "fusion_required_number_hit": _rate([item["required_number_hit"]["fusion_v2"] for item in subset]),
            "baseline_required_period_hit": _rate(
                [item["required_period_hit"]["baseline_page_text"] for item in subset]
            ),
            "fusion_required_period_hit": _rate([item["required_period_hit"]["fusion_v2"] for item in subset]),
            "table_contribution_coverage": round(
                sum(item["table_contribution"]["trusted_table_count"] > 0 for item in subset) / count, 4
            ),
            "average_table_contribution_chars": round(
                sum(item["table_contribution"]["chars"] for item in subset) / count, 2
            ),
            "average_table_contribution_ratio": round(
                sum(item["table_contribution"]["ratio"] for item in subset) / count, 4
            ),
            "table_layer_gold_row_hit": _rate(
                [item["table_contribution"]["gold_row_hit_in_table_layer"] for item in subset]
            ),
            "required_numbers_recovered_over_baseline": sum(
                item["table_contribution"]["required_number_recovered_over_baseline"] for item in subset
            ),
        }

    return {
        **metrics(records),
        "selected_pages": sum(item["average_inputs"]["selected_page_count"] for item in records),
        "trusted_tables": sum(item["table_contribution"]["trusted_table_count"] for item in records),
        "rejected_tables": sum(item["table_contribution"]["rejected_table_count"] for item in records),
        "question_types": {
            kind: metrics([item for item in records if kind in item["question_types"]]) for kind in QUESTION_TYPES
        },
    }


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Fusion v2 离线 A/B（冻结 Retrieval）",
        "",
        f"- Frozen report: `{payload['frozen_report']}`",
        f"- Frozen variant: `{payload['variant']}`",
        f"- Questions: {summary['questions']}",
        "- External calls: Retrieval=0, Jina=0, LLM=0, Judge=0",
        "- A: page_text only; B: page_text + quality-gated trusted tables.",
        "- Question types are multi-label; per-type counts do not sum to 30.",
        "",
        "## 汇总",
        "",
        f"- Average chars A/B: {summary['average_baseline_chars']} / {summary['average_fusion_chars']}",
        f"- Gold row hit A/B: {_percent(summary['baseline_gold_row_hit'])} / {_percent(summary['fusion_gold_row_hit'])}",
        f"- Required number hit A/B: {_percent(summary['baseline_required_number_hit'])} / {_percent(summary['fusion_required_number_hit'])}",
        f"- Required period hit A/B: {_percent(summary['baseline_required_period_hit'])} / {_percent(summary['fusion_required_period_hit'])}",
        f"- Table contribution coverage: {_percent(summary['table_contribution_coverage'])}",
        f"- Average table contribution: {summary['average_table_contribution_chars']} chars / {_percent(summary['average_table_contribution_ratio'])}",
        f"- Table-only gold row hit: {_percent(summary['table_layer_gold_row_hit'])}",
        f"- Required-number recoveries over baseline: {summary['required_numbers_recovered_over_baseline']}",
        f"- Trusted/rejected tables: {summary['trusted_tables']} / {summary['rejected_tables']}",
        "",
        "## Question type",
        "",
        "| Type | N | Chars A/B | Gold row A/B | Number A/B | Period A/B | Table contribution |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in QUESTION_TYPES:
        item = summary["question_types"][kind]
        lines.append(
            f"| {kind} | {item['questions']} | {item['average_baseline_chars']} / {item['average_fusion_chars']} | "
            f"{_percent(item['baseline_gold_row_hit'])} / {_percent(item['fusion_gold_row_hit'])} | "
            f"{_percent(item['baseline_required_number_hit'])} / {_percent(item['fusion_required_number_hit'])} | "
            f"{_percent(item['baseline_required_period_hit'])} / {_percent(item['fusion_required_period_hit'])} | "
            f"{_percent(item['table_contribution_coverage'])} |"
        )
    lines.extend(["", "## 逐题", ""])
    for index, item in enumerate(payload["records"], 1):
        selected = ", ".join(
            f"{page['filename']} p.{page['page_number']}" for page in item["frozen_retrieval"]["selected_pages"]
        )
        gold = ", ".join(
            f"{page['filename']} p.{page['page_number']}" for page in item["gold_evidence_pages"]
        )
        contribution = item["table_contribution"]
        lines.extend([
            f"### {index}. {item['financebench_id']}",
            "",
            f"- Question: {item['question']}",
            f"- Types: {', '.join(item['question_types'])}",
            f"- Gold evidence page: {gold}",
            f"- Frozen selected/context pages: {selected}",
            f"- Frozen selected hash: `{item['frozen_retrieval']['selected_pages_hash']}`",
            f"- Evidence chars A/B: {item['evidence_chars']['baseline_page_text']} / {item['evidence_chars']['fusion_v2']}",
            f"- Gold row hit A/B: {item['gold_row_hit']['baseline_page_text']} / {item['gold_row_hit']['fusion_v2']}",
            f"- Required numbers: {', '.join(item['required_numbers']) or '(n/a)'}",
            f"- Required number hit A/B: {item['required_number_hit']['baseline_page_text']} / {item['required_number_hit']['fusion_v2']}",
            f"- Required periods: {', '.join(item['required_periods']) or '(n/a)'}",
            f"- Required period hit A/B: {item['required_period_hit']['baseline_page_text']} / {item['required_period_hit']['fusion_v2']}",
            f"- Table contribution: {contribution['trusted_table_count']} tables, {contribution['chars']} chars ({_percent(contribution['ratio'])})",
            f"- Table-only gold row/required number hit: {contribution['gold_row_hit_in_table_layer']} / {contribution['required_number_hit_in_table_layer']}",
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
        "evaluation": "evidence_fusion_v2_ab",
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
