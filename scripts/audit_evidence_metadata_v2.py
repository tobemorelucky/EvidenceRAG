"""Offline Evidence Metadata Audit v2 over frozen Ranking v1 results.

The audit does not rebuild Evidence Units or rerun retrieval/ranking. Complete
metadata is available only for selected units in the frozen JSON; unselected
rank-trace units are retained and explicitly marked unobservable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from database import SessionLocal  # noqa: E402
from models import DocumentPage  # noqa: E402
from scripts.audit_evidence_selection_failure_v1 import (  # noqa: E402
    _filename,
    _normalized,
    _periods,
    _words,
    evidence_coverage,
)


DEFAULT_RANKING = ROOT / "reports" / "evidence_ranking_v1_diagnostic30.json"
DEFAULT_SELECTION_AUDIT = ROOT / "reports" / "evidence_selection_failure_audit_v1.json"
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_JSON = ROOT / "reports" / "evidence_metadata_audit_v2.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "evidence_metadata_audit_v2.md"

AUDIT_GROUPS = ("selection_loss10", "correct_regression10", "random10")
_METRIC_STOPWORDS = {
    "annual", "company", "consolidated", "financial", "fiscal", "form",
    "report", "section", "statement", "table", "year", "years",
}


def _source_contract(source_type: str) -> dict[str, str]:
    if source_type == "table":
        return {
            "entity": "page.company > page.doc_name > table.filename",
            "period": "source_text year regex (header + row)",
            "metric": "first non-numeric table row value",
        }
    return {
        "entity": "chunk.company > page.company > page.doc_name > chunk.filename",
        "period": "source_text year regex",
        "metric": "chunk.section or chunk.section_title",
    }


def _page_map() -> dict[str, dict]:
    db = SessionLocal()
    try:
        rows = db.query(
            DocumentPage.page_id,
            DocumentPage.document_id,
            DocumentPage.filename,
            DocumentPage.doc_name,
            DocumentPage.page_number,
            DocumentPage.company,
        ).all()
    finally:
        db.close()
    return {
        str(page_id): {
            "document_id": document_id,
            "filename": filename,
            "doc_name": doc_name,
            "page_number": int(page_number),
            "company": company,
        }
        for page_id, document_id, filename, doc_name, page_number, company in rows
    }


def _dataset_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["financebench_id"]: row for row in csv.DictReader(handle)}


def _gold(row: dict) -> list[dict]:
    return [{
        "filename": _filename(item.get("doc_name")),
        "page_number": int(item.get("evidence_page_num") or 0),
        "evidence_text": str(item.get("evidence_text") or ""),
    } for item in json.loads(row.get("evidence") or "[]")]


def _gold_page_ids(gold: list[dict], pages: dict[str, dict]) -> set[str]:
    keys = {(item["filename"], item["page_number"]) for item in gold}
    return {
        page_id for page_id, page in pages.items()
        if (_filename(page.get("filename") or page.get("doc_name")), page["page_number"]) in keys
        or (_filename(page.get("doc_name")), page["page_number"]) in keys
    }


def entity_metadata(value: object, page: dict | None, *, observable: bool) -> dict:
    if not observable:
        return {
            "value": None, "source": "not serialized in unselected rank trace",
            "confidence": None, "status": "unobservable", "status_basis": "trace schema",
        }
    text = _normalized(value)
    if not text:
        return {"value": value, "source": "upstream fallback chain", "confidence": 0.0, "status": "missing"}
    page = page or {}
    company = _normalized(page.get("company"))
    doc_name = _normalized(page.get("doc_name"))
    filename = _normalized(page.get("filename"))
    if company and text == company:
        return {"value": value, "source": "page/company metadata", "confidence": 1.0, "status": "correct"}
    if doc_name and text == doc_name:
        return {"value": value, "source": "page.doc_name fallback", "confidence": 0.75, "status": "correct"}
    if filename and text == filename:
        return {"value": value, "source": "filename fallback", "confidence": 0.5, "status": "correct"}
    return {
        "value": value, "source": "unresolved upstream entity value",
        "confidence": 0.25, "status": "conflict", "status_basis": "does not match local page identity",
    }


def period_metadata(
    value: object,
    *,
    observable: bool,
    required_periods: list[str],
    period_match_feature: float | None,
) -> dict:
    if not observable:
        if required_periods and period_match_feature is not None:
            status = "correct" if period_match_feature > 0 else "conflict"
            return {
                "value": None,
                "source": "source_text regex; raw value not serialized",
                "confidence": float(period_match_feature),
                "status": status,
                "status_basis": "Ranking v1 period_match feature only",
                "observation": "feature_only",
            }
        return {
            "value": None, "source": "source_text regex; raw value not serialized",
            "confidence": None, "status": "unobservable", "status_basis": "trace schema",
        }
    periods = [str(item) for item in value or []]
    if not periods:
        return {"value": periods, "source": "source_text year regex", "confidence": 0.0, "status": "missing"}
    if not required_periods:
        return {
            "value": periods, "source": "source_text year regex", "confidence": 0.7,
            "status": "correct", "status_basis": "extraction observable; no question period to validate relevance",
        }
    matched = len(set(periods) & set(required_periods)) / len(set(required_periods))
    return {
        "value": periods,
        "source": "source_text year regex",
        "confidence": round(matched, 4),
        "status": "correct" if matched == 1.0 else "conflict",
        "status_basis": "compared with periods stated in the question",
    }


def metric_metadata(
    value: object,
    *,
    source_type: str,
    observable: bool,
    question: str,
    gold_text: str,
) -> dict:
    source = _source_contract(source_type)["metric"]
    if not observable:
        return {
            "value": None, "source": f"{source}; raw value not serialized",
            "confidence": None, "status": "unobservable", "status_basis": "trace schema",
        }
    metric = str(value or "").strip()
    if not metric:
        return {"value": None, "source": source, "confidence": 0.0, "status": "missing"}
    terms = _words(metric) - _METRIC_STOPWORDS
    question_overlap = len(terms & _words(question)) / max(1, len(terms))
    gold_overlap = len(terms & _words(gold_text)) / max(1, len(terms))
    if question_overlap > 0:
        confidence, status, basis = 0.9, "correct", "lexical overlap with question"
    elif gold_overlap > 0:
        confidence, status, basis = 0.75, "correct", "lexical overlap with annotated evidence"
    else:
        confidence, status, basis = 0.35, "conflict", "no lexical support in question or annotated evidence"
    return {"value": metric, "source": source, "confidence": confidence, "status": status, "status_basis": basis}


def _selected_by_rank(route: dict) -> dict[int, dict]:
    return {
        int(unit.get("ranking_v1_rank") or 0): unit
        for unit in route.get("selected_units") or []
        if int(unit.get("ranking_v1_rank") or 0) > 0
    }


def audit_unit(
    trace: dict,
    full_unit: dict | None,
    *,
    question: str,
    gold: list[dict],
    gold_page_ids: set[str],
    pages: dict[str, dict],
) -> dict:
    page_id = str(trace.get("page_id") or "")
    source_type = str(trace.get("source_type") or "")
    observable = full_unit is not None
    page = pages.get(page_id)
    page_gold = [item for item in gold if page and (
        (_filename(page.get("filename")), page["page_number"]) == (item["filename"], item["page_number"])
        or (_filename(page.get("doc_name")), page["page_number"]) == (item["filename"], item["page_number"])
    )]
    gold_text = "\n".join(item["evidence_text"] for item in page_gold)
    features = trace.get("ranking_v1_features") or {}
    entity = entity_metadata((full_unit or {}).get("entity"), page, observable=observable)
    period = period_metadata(
        (full_unit or {}).get("period"),
        observable=observable,
        required_periods=_periods(question),
        period_match_feature=(float(features["period_match"]) if "period_match" in features else None),
    )
    metric = metric_metadata(
        (full_unit or {}).get("metric"),
        source_type=source_type,
        observable=observable,
        question=question,
        gold_text=gold_text,
    )
    exact_coverage = evidence_coverage(page_gold, str((full_unit or {}).get("source_text") or "")) if page_gold else None
    return {
        "unit_id": f"rank:{int(trace.get('ranking_v1_rank') or 0)}",
        "rank": int(trace.get("ranking_v1_rank") or 0),
        "score": trace.get("ranking_v1_score"),
        "selected": bool(trace.get("selected")),
        "source_type": source_type,
        "document_id": (full_unit or {}).get("document_id") or (page or {}).get("document_id"),
        "page_id": page_id,
        "page_number": (page or {}).get("page_number"),
        "metadata_observable": observable,
        "metadata_source_contract": _source_contract(source_type),
        "entity": entity,
        "period": period,
        "metric": metric,
        "gold_page_associated": page_id in gold_page_ids,
        "exact_gold_evidence_coverage": exact_coverage,
        "ranking_features": features,
    }


def _status_counts(units: list[dict], field: str) -> dict[str, int]:
    return dict(Counter(str(unit[field]["status"]) for unit in units))


def audit_question(record: dict, row: dict, pages: dict[str, dict], selection_case: dict | None) -> dict:
    route = record["routes"]["evidence_ranking_v1"]
    selected = _selected_by_rank(route)
    gold = _gold(row)
    gold_page_ids = _gold_page_ids(gold, pages)
    units = [
        audit_unit(
            trace,
            selected.get(int(trace.get("ranking_v1_rank") or 0)),
            question=record["question"],
            gold=gold,
            gold_page_ids=gold_page_ids,
            pages=pages,
        )
        for trace in (route.get("trace") or {}).get("rank_trace") or []
    ]
    gold_units = [unit for unit in units if unit["gold_page_associated"]]
    observable_gold = [unit for unit in gold_units if unit["metadata_observable"]]
    exact_gold = [
        unit for unit in observable_gold
        if unit.get("exact_gold_evidence_coverage") and unit["exact_gold_evidence_coverage"]["matched_lines"] > 0
    ]
    return {
        "financebench_id": record["financebench_id"],
        "audit_group": (
            record["group"] if record["group"] in {"selection_loss10", "correct_regression10"} else "random10"
        ),
        "source_diagnostic_group": record["group"],
        "question": record["question"],
        "required_periods": _periods(record["question"]),
        "unit_count": len(units),
        "metadata_observable_unit_count": sum(unit["metadata_observable"] for unit in units),
        "gold_page_associated_unit_count": len(gold_units),
        "observable_gold_page_unit_count": len(observable_gold),
        "exact_gold_evidence_unit_count": len(exact_gold),
        "gold_metadata_status": {
            field: _status_counts(gold_units, field) for field in ("entity", "period", "metric")
        },
        "selection_failure_reason": (selection_case or {}).get("not_in_context_reason"),
        "selection_failure_classification": (selection_case or {}).get("classification"),
        "evidence_units": units,
    }


def _aggregate_status(records: list[dict], field: str) -> dict[str, int]:
    result: Counter = Counter()
    for record in records:
        result.update(record["gold_metadata_status"][field])
    return dict(result)


def _rate(counts: dict[str, int], key: str) -> float | None:
    comparable = sum(counts.get(name, 0) for name in ("correct", "missing", "conflict"))
    return round(counts.get(key, 0) / comparable, 4) if comparable else None


def _group_summary(records: list[dict]) -> dict:
    fields = {field: _aggregate_status(records, field) for field in ("entity", "period", "metric")}
    return {
        "questions": len(records),
        "evidence_units": sum(record["unit_count"] for record in records),
        "metadata_observable_units": sum(record["metadata_observable_unit_count"] for record in records),
        "gold_page_associated_units": sum(record["gold_page_associated_unit_count"] for record in records),
        "observable_gold_page_units": sum(record["observable_gold_page_unit_count"] for record in records),
        "exact_gold_evidence_units": sum(record["exact_gold_evidence_unit_count"] for record in records),
        "gold_metadata_status": fields,
        "gold_conflict_rates": {field: _rate(fields[field], "conflict") for field in fields},
        "gold_missing_rates": {field: _rate(fields[field], "missing") for field in fields},
        "packing_skip_questions": sum(
            record.get("selection_failure_reason") == "gold_unit_skipped_by_character_packing" for record in records
        ),
    }


def choose_next_direction(groups: dict[str, dict]) -> dict:
    selection = groups["selection_loss10"]
    regression = groups["correct_regression10"]
    selection_period = selection["gold_conflict_rates"]["period"]
    regression_period = regression["gold_conflict_rates"]["period"]
    period_gap = None if selection_period is None or regression_period is None else round(selection_period - regression_period, 4)
    packing_rate = round(selection["packing_skip_questions"] / max(1, selection["questions"]), 4)
    if period_gap is not None and period_gap >= 0.2:
        direction = "metadata"
        rationale = "Selection-loss has a materially higher gold-unit period conflict signal than correct-regression."
    elif packing_rate >= 0.5:
        direction = "packing"
        rationale = "At least half of selection-loss questions skip a gold-page candidate during character packing."
    else:
        direction = "evidence_planning"
        rationale = "Neither metadata conflict separation nor packing frequency is strong enough; test combination-aware selection next."
    return {
        "direction": direction,
        "rationale": rationale,
        "selection_vs_correct_period_conflict_rate_gap": period_gap,
        "selection_loss_packing_skip_rate": packing_rate,
        "confidence_limit": "Unselected entity/metric/raw-period values are absent from the frozen rank trace.",
    }


def summarize(records: list[dict]) -> dict:
    groups = {
        group: _group_summary([record for record in records if record["audit_group"] == group])
        for group in AUDIT_GROUPS
    }
    total_units = sum(record["unit_count"] for record in records)
    observable = sum(record["metadata_observable_unit_count"] for record in records)
    return {
        "questions": len(records),
        "evidence_units": total_units,
        "metadata_observable_units": observable,
        "metadata_observability_rate": round(observable / max(1, total_units), 4),
        "groups": groups,
        "next_direction": choose_next_direction(groups),
        "random10_definition": "the 10 remaining frozen diagnostic30 questions; source group is candidate_miss10",
        "limitations": [
            "Full metadata/source_text exists only for selected units; unselected units are feature-only in Ranking v1 JSON.",
            "Period status for unselected units is inferred from Ranking v1 period_match and is a conflict signal, not raw-value proof.",
            "Metric correctness uses generic lexical consistency only; semantic synonyms are not judged.",
            "Entity correctness compares against the local page identity contract; upstream fallback branch is inferred from value equality.",
        ],
        "external_calls": {"llm": 0, "jina": 0, "judge": 0, "langsmith": 0},
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    decision = summary["next_direction"]
    lines = [
        "# Evidence Metadata Audit v2",
        "",
        "> Frozen Evidence Ranking v1 diagnostic30 only. No retrieval/assembly/ranking rerun and no LLM, Jina, Judge, or LangSmith calls.",
        "",
        "## Audit boundary",
        "",
        "Ranking v1 serializes full `entity`, `period`, `metric`, and `source_text` only for selected units. Unselected units retain rank, page, score, and features. They remain in the JSON audit but unavailable raw metadata is marked `unobservable`; a zero `period_match` is reported only as a conflict signal.",
        "",
        f"The `random10` group is the ten remaining questions in the frozen diagnostic30 (their source group is `candidate_miss10`), so it is deterministic but not a statistically random population sample.",
        "",
        "## Summary",
        "",
        f"- Questions / Evidence Units: {summary['questions']} / {summary['evidence_units']}",
        f"- Full metadata observable: {summary['metadata_observable_units']} ({summary['metadata_observability_rate']:.2%})",
        f"- Recommended next focus: **{decision['direction']}**",
        f"- Reason: {decision['rationale']}",
        f"- Selection-vs-correct period conflict gap: {_pct(decision['selection_vs_correct_period_conflict_rate_gap'])}",
        f"- Selection-loss character-packing skip rate: {_pct(decision['selection_loss_packing_skip_rate'])}",
        "",
        "## Group comparison — gold-page-associated units",
        "",
        "| Group | Units | Observable | Gold-page units | Gold observable | Exact gold units | Entity conflict | Period conflict | Metric conflict | Packing skips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in AUDIT_GROUPS:
        item = summary["groups"][group]
        lines.append(
            f"| {group} | {item['evidence_units']} | {item['metadata_observable_units']} | "
            f"{item['gold_page_associated_units']} | {item['observable_gold_page_units']} | "
            f"{item['exact_gold_evidence_units']} | {_pct(item['gold_conflict_rates']['entity'])} | "
            f"{_pct(item['gold_conflict_rates']['period'])} | {_pct(item['gold_conflict_rates']['metric'])} | "
            f"{item['packing_skip_questions']} |"
        )
    lines.extend(["", "## Per-question gold metadata", ""])
    for index, record in enumerate(payload["records"], 1):
        lines.extend([
            f"### {index}. {record['financebench_id']} — {record['audit_group']}",
            "",
            f"- Question: {record['question']}",
            f"- Units / observable: `{record['unit_count']}` / `{record['metadata_observable_unit_count']}`",
            f"- Gold-page units / observable / exact-text: `{record['gold_page_associated_unit_count']}` / `{record['observable_gold_page_unit_count']}` / `{record['exact_gold_evidence_unit_count']}`",
            f"- Entity status: `{record['gold_metadata_status']['entity']}`",
            f"- Period status: `{record['gold_metadata_status']['period']}`",
            f"- Metric status: `{record['gold_metadata_status']['metric']}`",
        ])
        if record.get("selection_failure_reason"):
            lines.append(f"- Selection failure reason: `{record['selection_failure_reason']}`")
        gold_units = [unit for unit in record["evidence_units"] if unit["gold_page_associated"]]
        if gold_units:
            lines.extend([
                "",
                "| Rank | Selected | Score | Entity | Period | Metric | Metadata observable |",
                "|---:|---|---:|---|---|---|---|",
            ])
            for unit in gold_units:
                lines.append(
                    f"| {unit['rank']} | {unit['selected']} | {unit['score']} | "
                    f"{unit['entity']['status']} | {unit['period']['status']} | {unit['metric']['status']} | "
                    f"{unit['metadata_observable']} |"
                )
        lines.append("")
    lines.extend([
        "## Decision constraints",
        "",
        "- `metadata` is selected only when its conflict signal separates selection-loss from correct-regression by at least 20 percentage points.",
        "- Otherwise `packing` is selected when at least half of selection-loss questions skipped a gold-page candidate during character packing.",
        "- `Evidence Planning` is selected only when neither earlier signal is strong.",
        "- This audit does not authorize production changes; it identifies the next shadow experiment.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-json", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--selection-audit-json", type=Path, default=DEFAULT_SELECTION_AUDIT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    ranking = json.loads(args.ranking_json.read_text(encoding="utf-8"))
    selection = json.loads(args.selection_audit_json.read_text(encoding="utf-8"))
    selection_by_id = {item["financebench_id"]: item for item in selection.get("records") or []}
    rows = _dataset_rows(args.dataset)
    pages = _page_map()
    records = []
    for record in ranking.get("records") or []:
        row = rows.get(record["financebench_id"])
        if row is None:
            raise RuntimeError(f"Dataset row missing: {record['financebench_id']}")
        records.append(audit_question(record, row, pages, selection_by_id.get(record["financebench_id"])))

    payload = {
        "audit": "evidence_metadata_audit_v2",
        "source_ranking_json": str(args.ranking_json),
        "scope": "selection-loss10 + correct-regression10 + deterministic remaining10",
        "summary": summarize(records),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.output_json}")
    print(f"Markdown: {args.output_markdown}")


if __name__ == "__main__":
    main()
