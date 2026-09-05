"""Evaluate Evidence Package v1 on frozen Jina diagnostic30 without APIs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from statistics import fmean

from sqlalchemy import func


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
for path in (ROOT, BACKEND):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from database import SessionLocal  # noqa: E402
from evidence_package_v1 import build_evidence_packages_v1, render_evidence_packages_v1  # noqa: E402
from models import DocumentPage, DocumentTable  # noqa: E402
from table_store import TableStore  # noqa: E402
from scripts.evaluate_evidence_assembly_ab import (  # noqa: E402
    _contains_all,
    _numbers,
    _parse_gold,
    _periods,
    _rate,
    _required_numbers,
    gold_row_hit,
)
from scripts.evaluate_fact_store_shadow_v1 import _build_frozen_context  # noqa: E402
from scripts.evaluate_reranker_shadow_v1 import fixture_rows, validate_snapshot  # noqa: E402


DEFAULT_SNAPSHOT = ROOT / "reports" / "reranker_shadow_v1_rrf_top120.json"
DEFAULT_RERANK = ROOT / "reports" / "reranker_shadow_v1.json"
DEFAULT_OUTPUT = ROOT / "reports" / "evidence_package_shadow_v1.json"


def _filename(value: object) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _load_dataset() -> dict[str, dict]:
    path = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["financebench_id"]: row for row in csv.DictReader(handle)}


def _load_page_map_and_tables(records: list[dict], rankings: dict[str, list[dict]]) -> tuple[dict, dict]:
    keys = set()
    for record in records:
        for ranked in rankings[record["question_id"]][:12]:
            chunk = record["chunks"][int(ranked["index"])]
            keys.add((_filename(chunk.get("filename")), int(chunk.get("page_number") or 0)))
    filenames = {name for name, _ in keys}
    db = SessionLocal()
    try:
        pages = db.query(DocumentPage).filter(func.lower(DocumentPage.filename).in_(filenames)).all()
        page_map = {
            (_filename(page.filename), int(page.page_number)): {
                "document_id": page.document_id,
                "page_id": page.page_id,
            }
            for page in pages if (_filename(page.filename), int(page.page_number)) in keys
        }
        missing = sorted(keys - set(page_map))
        if missing:
            raise ValueError(f"Missing DocumentPage identities for {missing[:5]}")
        page_ids = {value["page_id"] for value in page_map.values()}
        table_rows = db.query(DocumentTable).filter(DocumentTable.page_id.in_(page_ids)).all() if page_ids else []
        tables_by_page = defaultdict(list)
        for table in table_rows:
            item = TableStore._to_dict(table)
            tables_by_page[item["page_id"]].append(item)
        return page_map, tables_by_page
    finally:
        db.close()


def _enrich_top12(record: dict, ranking: list[dict], page_map: dict) -> list[dict]:
    chunks = []
    for jina_rank, ranked in enumerate(ranking[:12], 1):
        source = record["chunks"][int(ranked["index"])]
        identity = page_map[(_filename(source.get("filename")), int(source.get("page_number") or 0))]
        chunks.append({**source, **identity, "jina_rank": jina_rank})
    return chunks


def _metrics(row: dict, text: str) -> dict:
    gold = _parse_gold(row)
    required_numbers = _required_numbers(row)
    required_periods = _periods(row["question"])
    return {
        "evidence_coverage": gold_row_hit(gold, text, required_numbers=required_numbers),
        "required_number_hit": _contains_all(required_numbers, text, _numbers),
        "required_period_hit": _contains_all(required_periods, text, _periods),
        "context_chars": len(text),
    }


def _summarize(records: list[dict]) -> dict:
    def group_metrics(items: list[dict]) -> dict:
        return {
            "questions": len(items),
            "evidence_coverage_a": _rate([item["metrics"]["a_jina_context"]["evidence_coverage"] for item in items]),
            "evidence_coverage_b": _rate([item["metrics"]["b_evidence_package"]["evidence_coverage"] for item in items]),
            "required_number_hit_a": _rate([item["metrics"]["a_jina_context"]["required_number_hit"] for item in items]),
            "required_number_hit_b": _rate([item["metrics"]["b_evidence_package"]["required_number_hit"] for item in items]),
            "required_period_hit_a": _rate([item["metrics"]["a_jina_context"]["required_period_hit"] for item in items]),
            "required_period_hit_b": _rate([item["metrics"]["b_evidence_package"]["required_period_hit"] for item in items]),
            "mean_chars_a": round(fmean(item["metrics"]["a_jina_context"]["context_chars"] for item in items), 2) if items else 0,
            "mean_chars_b": round(fmean(item["metrics"]["b_evidence_package"]["context_chars"] for item in items), 2) if items else 0,
        }
    return {
        **group_metrics(records),
        "groups": {group: group_metrics([item for item in records if item["group"] == group]) for group in sorted({item["group"] for item in records})},
        "questions_with_tables": sum(bool(item["table_contribution"]["included_table_ids"]) for item in records),
        "jina_top12_chunks": sum(len(item["frozen_jina_top12_chunk_ids"]) for item in records),
        "packages": sum(len(item["packages"]) for item in records),
        "multi_chunk_packages": sum(
            len(package["text_chunk_ids"]) > 1 for item in records for package in item["packages"]
        ),
        "included_tables": sum(len(item["table_contribution"]["included_table_ids"]) for item in records),
        "table_chars": sum(item["table_contribution"]["chars"] for item in records),
        "table_evidence_gains": [item["question_id"] for item in records if item["table_contribution"]["new_evidence_coverage"]],
        "table_number_gains": [item["question_id"] for item in records if item["table_contribution"]["new_required_number"]],
        "table_period_gains": [item["question_id"] for item in records if item["table_contribution"]["new_required_period"]],
        "evidence_gains_vs_a": [item["question_id"] for item in records if item["metrics"]["a_jina_context"]["evidence_coverage"] is False and item["metrics"]["b_evidence_package"]["evidence_coverage"] is True],
        "evidence_regressions_vs_a": [item["question_id"] for item in records if item["metrics"]["a_jina_context"]["evidence_coverage"] is True and item["metrics"]["b_evidence_package"]["evidence_coverage"] is False],
        "replaced_baseline_chunks": sum(len(item["replaced_chunks"]) for item in records),
        "budget_dropped_top12_chunks": sum(len(item["dropped_chunks"]) for item in records),
    }


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Package Shadow v1（固定 Jina 30题）", "",
        "- External calls: Retrieval=0, Jina=0, LLM=0, Judge=0, LangSmith=0",
        "- A uses frozen Jina Top8 under 28K, matching the existing shadow context contract.",
        "- B packages frozen Jina Top12 by exact document_id + page_id and renders under the same 28K budget.",
        "- Tables require exact document_id + page_id; adjacent pages are never used.",
        "- Gold evidence is used only for post-selection metrics.", "",
        "## 汇总", "", "| Metric | A Jina context | B Evidence Package |", "|---|---:|---:|",
        f"| Evidence coverage | {_pct(summary['evidence_coverage_a'])} | {_pct(summary['evidence_coverage_b'])} |",
        f"| Required number hit | {_pct(summary['required_number_hit_a'])} | {_pct(summary['required_number_hit_b'])} |",
        f"| Required period hit | {_pct(summary['required_period_hit_a'])} | {_pct(summary['required_period_hit_b'])} |",
        f"| Mean chars | {summary['mean_chars_a']} | {summary['mean_chars_b']} |", "",
        f"- Questions with included tables: {summary['questions_with_tables']}/30",
        f"- Jina Top12 chunks → packages → multi-chunk packages: {summary['jina_top12_chunks']} → {summary['packages']} → {summary['multi_chunk_packages']}",
        f"- Included tables / table chars: {summary['included_tables']} / {summary['table_chars']}",
        f"- Table-attributed evidence/number/period gains: `{summary['table_evidence_gains']}` / `{summary['table_number_gains']}` / `{summary['table_period_gains']}`",
        f"- Overall evidence gains/regressions vs A: `{summary['evidence_gains_vs_a']}` / `{summary['evidence_regressions_vs_a']}`",
        f"- Baseline Top8 chunks absent from B: {summary['replaced_baseline_chunks']}", "",
        f"- Top12 chunks dropped by B budget: {summary['budget_dropped_top12_chunks']}", "",
        "## 分组", "", "| Group | N | Evidence A/B | Number A/B | Period A/B | Chars A/B |", "|---|---:|---:|---:|---:|---:|",
    ]
    for group, item in summary["groups"].items():
        lines.append(
            f"| {group} | {item['questions']} | {_pct(item['evidence_coverage_a'])} / {_pct(item['evidence_coverage_b'])} | "
            f"{_pct(item['required_number_hit_a'])} / {_pct(item['required_number_hit_b'])} | "
            f"{_pct(item['required_period_hit_a'])} / {_pct(item['required_period_hit_b'])} | {item['mean_chars_a']} / {item['mean_chars_b']} |"
        )
    lines += ["", "## 逐题", "", "| ID | Group | Packages | Tables | New chars | Evidence A/B | Number A/B | Period A/B | Replaced chunks |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for item in payload["records"]:
        a = item["metrics"]["a_jina_context"]
        b = item["metrics"]["b_evidence_package"]
        lines.append(
            f"| {item['question_id']} | {item['group']} | {len(item['packages'])} | {len(item['table_contribution']['included_table_ids'])} | "
            f"{item['new_characters']} | {a['evidence_coverage']} / {b['evidence_coverage']} | "
            f"{a['required_number_hit']} / {b['required_number_hit']} | {a['required_period_hit']} / {b['required_period_hit']} | {len(item['replaced_chunks'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--reranker-result", type=Path, default=DEFAULT_RERANK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    validation_rows, groups = fixture_rows()
    frozen = validate_snapshot(json.loads(args.snapshot.read_text(encoding="utf-8")), validation_rows, groups)
    frozen_by_id = {record["question_id"]: record for record in frozen}
    reranker = json.loads(args.reranker_result.read_text(encoding="utf-8"))
    rerank_by_id = {record["question_id"]: record for record in reranker["records"]}
    rankings = {}
    for question_id, record in rerank_by_id.items():
        route = record["routes"].get("jina") or {}
        if route.get("status") != "ok" or len(route.get("ranked") or []) < 12:
            raise ValueError(f"Missing frozen Jina Top12 for {question_id}")
        if record.get("candidate_sha256") != frozen_by_id[question_id]["candidate_sha256"]:
            raise ValueError(f"Candidate drift for {question_id}")
        rankings[question_id] = route["ranked"]
    page_map, tables_by_page = _load_page_map_and_tables(frozen, rankings)
    dataset = _load_dataset()

    records = []
    for source in frozen:
        question_id = source["question_id"]
        route = rerank_by_id[question_id]["routes"]["jina"]
        baseline, baseline_chunks = _build_frozen_context(source["chunks"], route["ranked"])
        top12 = _enrich_top12(source, route["ranked"], page_map)
        page_ids = {chunk["page_id"] for chunk in top12}
        tables = [table for page_id in page_ids for table in tables_by_page.get(page_id, [])]
        packages = build_evidence_packages_v1(top12, tables)
        package_context, package_trace = render_evidence_packages_v1(packages, max_chars=28000)
        without_tables = deepcopy(packages)
        for package in without_tables:
            package["related_tables"] = []
        text_only_context, _ = render_evidence_packages_v1(without_tables, max_chars=28000)
        row = dataset[question_id]
        metrics_a = _metrics(row, baseline)
        metrics_b = _metrics(row, package_context)
        metrics_text_only = _metrics(row, text_only_context)
        baseline_ids = [chunk["chunk_id"] for chunk in baseline_chunks]
        included_ids = set(package_trace["included_chunk_ids"])
        records.append({
            "question_id": question_id,
            "group": source["group"],
            "question": source["question"],
            "frozen_jina_top12_chunk_ids": [chunk["chunk_id"] for chunk in top12],
            "packages": [
                {
                    "package_id": package["package_id"],
                    "anchor_chunk_id": package["anchor_chunk_id"],
                    "document_id": package["document_id"],
                    "page_id": package["page_id"],
                    "text_chunk_ids": [chunk["chunk_id"] for chunk in package["text_chunks"]],
                    "related_table_ids": [table["table_id"] for table in package["related_tables"]],
                    "metadata": package["metadata"],
                }
                for package in packages
            ],
            "metrics": {"a_jina_context": metrics_a, "b_evidence_package": metrics_b},
            "new_characters": metrics_b["context_chars"] - metrics_a["context_chars"],
            "table_contribution": {
                "included_table_ids": package_trace["included_table_ids"],
                "chars": package_trace["table_chars"],
                "new_evidence_coverage": metrics_text_only["evidence_coverage"] is False and metrics_b["evidence_coverage"] is True,
                "new_required_number": metrics_text_only["required_number_hit"] is False and metrics_b["required_number_hit"] is True,
                "new_required_period": metrics_text_only["required_period_hit"] is False and metrics_b["required_period_hit"] is True,
            },
            "replaced_chunks": [chunk_id for chunk_id in baseline_ids if chunk_id not in included_ids],
            "package_sources": package_trace["package_sources"],
            "dropped_chunks": package_trace["dropped_chunks"],
            "context_sha256": {
                "a": hashlib.sha256(baseline.encode("utf-8")).hexdigest(),
                "b": hashlib.sha256(package_context.encode("utf-8")).hexdigest(),
            },
        })

    payload = {
        "schema": "evidence_package_shadow_v1",
        "inputs": {"snapshot": str(args.snapshot), "reranker_result": str(args.reranker_result)},
        "configuration": {"jina_input_chunks": 12, "baseline_context_chunks": 8, "context_budget_chars": 28000},
        "external_calls": {"retrieval": 0, "jina": 0, "llm": 0, "judge": 0, "langsmith": 0},
        "summary": _summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
