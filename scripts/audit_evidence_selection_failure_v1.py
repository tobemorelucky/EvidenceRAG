"""Audit Evidence Ranking v1 failures on the frozen selection-loss group.

This is a read-only, deterministic audit.  It consumes the existing Ranking
v1 JSON, FinanceBench annotations, and the local document-page identity map.
It never reruns retrieval/ranking and never calls Jina, an LLM, or a Judge.
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
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from database import SessionLocal  # noqa: E402
from models import DocumentPage  # noqa: E402


DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_RANKING = ROOT / "reports" / "evidence_ranking_v1_diagnostic30.json"
DEFAULT_JSON = ROOT / "reports" / "evidence_selection_failure_audit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "evidence_selection_failure_audit_v1.md"

_NUMBER_RE = re.compile(r"\(?-?[$€£¥]?\s*\d[\d,]*(?:\.\d+)?%?\)?")
_PERIOD_RE = re.compile(r"\bFY\s*\d{2,4}\b|\b(?:19|20)\d{2}\b|\bQ[1-4]\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_SPACE_RE = re.compile(r"\s+")

CATEGORY_LABELS = {
    "A": "单 unit 排序失败",
    "B": "多 unit 组合缺失",
    "C": "period/entity 冲突",
    "D": "candidate 不足",
    "E": "其他",
}

DIRECTION_BY_CATEGORY = {
    "A": "Ranking 改进",
    "B": "Evidence Planning / 组合选择",
    "C": "period/entity 元数据对齐",
    "D": "候选构造或 Retrieval",
    "E": "Evidence Unit 表示或人工复核",
}


def _filename(value: object) -> str:
    name = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    if name and not name.casefold().endswith(".pdf"):
        name = f"{name}.pdf"
    return name.casefold()


def _normalized(value: object) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip().casefold()


def _words(value: object) -> set[str]:
    return {word.casefold() for word in _WORD_RE.findall(str(value or "")) if len(word) > 1}


def _numbers(value: object) -> list[str]:
    values = []
    for match in _NUMBER_RE.finditer(str(value or "")):
        normalized = match.group(0).replace(",", "").replace(" ", "").strip("()")
        normalized = normalized.lstrip("$€£¥")
        if normalized and normalized not in values:
            values.append(normalized)
    return values


def _periods(value: object) -> list[str]:
    values = []
    for match in _PERIOD_RE.finditer(str(value or "")):
        item = re.sub(r"\s+", "", match.group(0).upper())
        if item.startswith("FY"):
            year = item[2:]
            item = f"20{year}" if len(year) == 2 else year
        if item not in values:
            values.append(item)
    return values


def _required_numbers(row: dict) -> list[str]:
    justification = str(row.get("justification") or "").strip()
    source = justification
    if "=" in justification:
        left, right = justification.rsplit("=", 1)
        if _numbers(left) and _numbers(right):
            source = left
    return _numbers(source) or _numbers(row.get("answer"))


def evidence_coverage(gold: list[dict], evidence: str) -> dict:
    """Return deterministic line coverage using text, tokens, and numbers."""
    evidence_normalized = _normalized(evidence)
    evidence_words = _words(evidence)
    evidence_numbers = set(_numbers(evidence))
    lines: list[str] = []
    for item in gold:
        for line in str(item.get("evidence_text") or "").splitlines():
            normalized = _normalized(line)
            if len(normalized) >= 3 and normalized not in lines:
                lines.append(normalized)
    matched = 0
    for line in lines:
        if line in evidence_normalized:
            matched += 1
            continue
        line_words = _words(line)
        line_numbers = set(_numbers(line))
        overlap = len(line_words & evidence_words) / max(1, len(line_words))
        if overlap >= 0.7 and line_numbers <= evidence_numbers:
            matched += 1
    return {
        "matched_lines": matched,
        "total_lines": len(lines),
        "ratio": round(matched / len(lines), 4) if lines else None,
    }


def classify_selection_failure(
    *,
    gold_page_mapped: bool,
    gold_candidate_exists: bool,
    exact_coverage_ratio: float | None,
    required_periods: list[str],
    best_period_match: float | None,
    multiple_required_units: bool,
    selected_gold_page_unit_count: int,
) -> tuple[str, str]:
    """Classify one selection-loss case with an explicit priority order."""
    if not gold_page_mapped or not gold_candidate_exists:
        return "D", "No gold-page-associated Evidence Unit is visible in the frozen candidate trace."
    incomplete = exact_coverage_ratio is None or exact_coverage_ratio < 1.0
    if incomplete and required_periods and (best_period_match or 0.0) == 0.0:
        return "C", "Gold-page candidates exist, but none carries a matching required period in Ranking v1 features."
    if incomplete and multiple_required_units:
        return "B", "The question requires multiple evidence items/values and the selected package does not cover all of them."
    if incomplete and selected_gold_page_unit_count == 0:
        return "A", "A gold-page candidate exists but no unit from that page survives the ranking/budget selection."
    return "E", "A gold-page unit is selected, but its stored text does not fully cover the annotated evidence; inspect unit construction."


def non_entry_reason(
    *,
    gold_page_mapped: bool,
    gold_candidates: list[dict],
    selected_gold_page_unit_count: int,
    exact_coverage_ratio: float | None,
    selection_rank_frontier: int | None,
) -> str:
    if not gold_page_mapped:
        return "gold_page_not_mapped_to_local_page_id"
    if not gold_candidates:
        return "no_gold_page_associated_unit_in_top120_candidate_trace"
    if selected_gold_page_unit_count == 0:
        best_rank = min(int(item["ranking_v1_rank"]) for item in gold_candidates)
        if selection_rank_frontier is not None and best_rank > selection_rank_frontier:
            return "gold_unit_ranked_below_selected_frontier"
        return "gold_unit_skipped_by_character_packing"
    if exact_coverage_ratio is not None and exact_coverage_ratio < 1.0:
        return "gold_page_selected_but_exact_gold_text_incomplete"
    return "gold_evidence_present_in_context"


def _load_dataset_rows(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["financebench_id"]: row for row in csv.DictReader(handle)}


def _parse_gold(row: dict) -> list[dict]:
    items = json.loads(row.get("evidence") or "[]")
    return [{
        "filename": _filename(item.get("doc_name")),
        "page_number": int(item.get("evidence_page_num") or 0),
        "evidence_text": str(item.get("evidence_text") or "").strip(),
    } for item in items]


def _load_page_identity_map() -> dict[tuple[str, int], dict]:
    """Read the page identity contract without instantiating a mutating store."""
    db = SessionLocal()
    try:
        rows = db.query(
            DocumentPage.filename,
            DocumentPage.doc_name,
            DocumentPage.page_number,
            DocumentPage.page_id,
            DocumentPage.document_id,
            DocumentPage.company,
        ).all()
    finally:
        db.close()
    result: dict[tuple[str, int], dict] = {}
    for filename, doc_name, page_number, page_id, document_id, company in rows:
        value = {
            "page_id": page_id,
            "document_id": document_id,
            "company": company,
        }
        result[(_filename(filename), int(page_number))] = value
        if doc_name:
            result[(_filename(doc_name), int(page_number))] = value
    return result


def _gold_unit_records(gold: list[dict], page_map: dict[tuple[str, int], dict]) -> list[dict]:
    records = []
    for index, item in enumerate(gold, 1):
        identity = page_map.get((item["filename"], item["page_number"])) or {}
        records.append({
            "gold_unit_id": f"gold:{index}",
            **item,
            "page_id": identity.get("page_id"),
            "document_id": identity.get("document_id"),
            "mapping_status": "mapped" if identity.get("page_id") else "missing",
        })
    return records


def audit_record(record: dict, row: dict, page_map: dict[tuple[str, int], dict]) -> dict:
    route = record["routes"]["evidence_ranking_v1"]
    rank_trace = list((route.get("trace") or {}).get("rank_trace") or [])
    selected_units = list(route.get("selected_units") or [])
    gold_units = _gold_unit_records(_parse_gold(row), page_map)
    gold_page_ids = {item["page_id"] for item in gold_units if item.get("page_id")}
    gold_candidates = [item for item in rank_trace if item.get("page_id") in gold_page_ids]
    selected_gold_units = [item for item in selected_units if item.get("page_id") in gold_page_ids]
    selected_text = "\n\n".join(str(item.get("source_text") or "") for item in selected_gold_units)
    exact_coverage = evidence_coverage(gold_units, selected_text)
    selected_ranks = [int(item.get("ranking_v1_rank") or 0) for item in rank_trace if item.get("selected")]
    selection_frontier = max(selected_ranks) if selected_ranks else None
    required_periods = _periods(row.get("question"))
    required_numbers = _required_numbers(row)
    best_period_match = max(
        (float((item.get("ranking_v1_features") or {}).get("period_match") or 0.0) for item in gold_candidates),
        default=None,
    )
    multiple_required = (
        len(gold_units) > 1
        or len(gold_page_ids) > 1
        or len(required_numbers) > 1
    )
    category, rationale = classify_selection_failure(
        gold_page_mapped=len(gold_page_ids) == len({(x["filename"], x["page_number"]) for x in gold_units}),
        gold_candidate_exists=bool(gold_candidates),
        exact_coverage_ratio=exact_coverage["ratio"],
        required_periods=required_periods,
        best_period_match=best_period_match,
        multiple_required_units=multiple_required,
        selected_gold_page_unit_count=len(selected_gold_units),
    )
    reason = non_entry_reason(
        gold_page_mapped=bool(gold_page_ids),
        gold_candidates=gold_candidates,
        selected_gold_page_unit_count=len(selected_gold_units),
        exact_coverage_ratio=exact_coverage["ratio"],
        selection_rank_frontier=selection_frontier,
    )
    candidate_summaries = [{
        "source_type": item.get("source_type"),
        "page_id": item.get("page_id"),
        "rank": item.get("ranking_v1_rank"),
        "score": item.get("ranking_v1_score"),
        "features": item.get("ranking_v1_features") or {},
        "selected": bool(item.get("selected")),
    } for item in gold_candidates]
    best = min(candidate_summaries, key=lambda item: int(item["rank"])) if candidate_summaries else None
    return {
        "financebench_id": record["financebench_id"],
        "group": record["group"],
        "question": record["question"],
        "gold_evidence_units": gold_units,
        "candidate_presence_basis": "page_id association within Evidence Units derived from retrieval Top120",
        "candidate_exact_text_availability": "unknown_for_unselected_units_rank_trace_omits_source_text",
        "gold_unit_exists_top120": bool(gold_candidates),
        "gold_unit_best_rank": best["rank"] if best else None,
        "gold_unit_best_score": best["score"] if best else None,
        "gold_unit_best_features": best["features"] if best else None,
        "gold_page_candidate_units": candidate_summaries,
        "selected_gold_page_unit_count": len(selected_gold_units),
        "selected_gold_page_unit_ranks": sorted(int(item.get("ranking_v1_rank") or 0) for item in selected_gold_units),
        "selected_exact_gold_coverage": exact_coverage,
        "selection_rank_frontier": selection_frontier,
        "required_periods": required_periods,
        "required_numbers": required_numbers,
        "multiple_required_units": multiple_required,
        "period_conflict_signal": bool(required_periods and best_period_match == 0.0),
        "entity_conflict_signal": "not_determinable_from_existing_rank_trace",
        "not_in_context_reason": reason,
        "classification": {
            "code": category,
            "label": CATEGORY_LABELS[category],
            "rationale": rationale,
            "next_direction": DIRECTION_BY_CATEGORY[category],
        },
    }


def summarize(records: list[dict]) -> dict:
    category_counts = Counter(item["classification"]["code"] for item in records)
    reason_counts = Counter(item["not_in_context_reason"] for item in records)
    direction_counts = Counter(item["classification"]["next_direction"] for item in records)
    priority = direction_counts.most_common(1)[0][0] if direction_counts else None
    return {
        "questions": len(records),
        "gold_page_associated_candidate_rate": round(
            sum(item["gold_unit_exists_top120"] for item in records) / max(1, len(records)), 4
        ),
        "full_exact_gold_context_rate": round(
            sum(item["selected_exact_gold_coverage"]["ratio"] == 1.0 for item in records) / max(1, len(records)), 4
        ),
        "category_counts": {
            code: {
                "label": CATEGORY_LABELS[code],
                "count": category_counts.get(code, 0),
                "rate": round(category_counts.get(code, 0) / max(1, len(records)), 4),
            }
            for code in CATEGORY_LABELS
        },
        "not_in_context_reason_counts": dict(reason_counts),
        "next_direction_counts": dict(direction_counts),
        "primary_next_direction": priority,
        "method_limitations": [
            "Unselected rank-trace entries contain page_id/score/features but not source_text.",
            "Top120 presence therefore means a page-associated Evidence Unit exists, not that its exact text matches the gold snippet.",
            "Entity conflict cannot be derived reliably from the existing unselected rank trace and is reported as unknown.",
        ],
        "external_calls": {"llm": 0, "jina": 0, "judge": 0, "langsmith": 0},
    }


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Selection Failure Audit v1",
        "",
        "> Frozen Evidence Ranking v1 diagnostic30 JSON; selection-loss10 only. No retrieval/ranking rerun and no LLM, Jina, Judge, or LangSmith call.",
        "",
        "## Method boundary",
        "",
        "The frozen rank trace keeps `page_id`, score, generic features, rank, and selection status for every candidate, but omits `source_text` for unselected units. Therefore `gold_unit_exists_top120` is a **gold-page-associated unit proxy**. Exact gold-text coverage is evaluated only from selected units whose text is present in JSON.",
        "",
        "## Summary",
        "",
        f"- Questions: {summary['questions']}",
        f"- Gold-page-associated unit exists in Top120-derived candidates: {summary['gold_page_associated_candidate_rate']:.2%}",
        f"- Full exact annotated evidence in selected context: {summary['full_exact_gold_context_rate']:.2%}",
        f"- Primary next direction by deterministic classification: **{summary['primary_next_direction']}**",
        "- Category C is a trace-level metadata signal, not proof of a source-text conflict, because unselected candidate text is absent from the frozen JSON.",
        "",
        "| Category | Meaning | Count | Rate | Next direction |",
        "|---|---|---:|---:|---|",
    ]
    for code, item in summary["category_counts"].items():
        lines.append(
            f"| {code} | {item['label']} | {item['count']} | {item['rate']:.2%} | {DIRECTION_BY_CATEGORY[code]} |"
        )
    lines.extend(["", "## Per-question audit", ""])
    for index, record in enumerate(payload["records"], 1):
        classification = record["classification"]
        lines.extend([
            f"### {index}. {record['financebench_id']} — {classification['code']}: {classification['label']}",
            "",
            f"- Question: {record['question']}",
            f"- Gold unit exists in Top120-derived candidates: `{record['gold_unit_exists_top120']}`",
            f"- Best gold-page unit rank / score: `{record['gold_unit_best_rank']}` / `{record['gold_unit_best_score']}`",
            f"- Selected gold-page units: `{record['selected_gold_page_unit_count']}`; ranks: `{record['selected_gold_page_unit_ranks']}`",
            f"- Selected exact gold coverage: `{record['selected_exact_gold_coverage']['matched_lines']}/{record['selected_exact_gold_coverage']['total_lines']}` (`{record['selected_exact_gold_coverage']['ratio']}`)",
            f"- Required periods: `{record['required_periods']}`; required numbers: `{record['required_numbers']}`",
            f"- Not-in-context reason: `{record['not_in_context_reason']}`",
            f"- Classification rationale: {classification['rationale']}",
            f"- Next direction: **{classification['next_direction']}**",
            "",
            "Gold evidence units:",
            "",
        ])
        for unit in record["gold_evidence_units"]:
            preview = _normalized(unit["evidence_text"])
            if len(preview) > 260:
                preview = preview[:257] + "..."
            lines.append(
                f"- `{unit['gold_unit_id']}` — `{unit['filename']}`, page `{unit['page_number']}`, "
                f"page_id `{unit['page_id']}`: {preview}"
            )
        lines.extend(["", "Gold-page candidate trace:", ""])
        if record["gold_page_candidate_units"]:
            lines.extend([
                "| Rank | Score | Type | Selected | Retrieval | Lexical | Period | Numeric | Unit completeness |",
                "|---:|---:|---|---|---:|---:|---:|---:|---:|",
            ])
            for item in record["gold_page_candidate_units"]:
                features = item["features"]
                lines.append(
                    f"| {item['rank']} | {item['score']} | {item['source_type']} | {item['selected']} | "
                    f"{features.get('retrieval_score')} | {features.get('query_lexical_overlap')} | "
                    f"{features.get('period_match')} | {features.get('numeric_presence')} | "
                    f"{features.get('unit_completeness')} |"
                )
        else:
            lines.append("- No gold-page-associated candidate unit.")
        lines.append("")
    lines.extend([
        "## Decision",
        "",
        "The category distribution identifies the next experiment family; it does not authorize production changes. Category B supports combination-aware Evidence Planning, A supports a ranking experiment, C supports metadata alignment, and D points back to candidate construction/retrieval.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-json", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    ranking_payload = json.loads(args.ranking_json.read_text(encoding="utf-8"))
    dataset_rows = _load_dataset_rows(args.dataset)
    page_map = _load_page_identity_map()
    source_records = [item for item in ranking_payload.get("records") or [] if item.get("group") == "selection_loss10"]
    records = []
    for record in source_records:
        financebench_id = record["financebench_id"]
        row = dataset_rows.get(financebench_id)
        if row is None:
            raise RuntimeError(f"Dataset row not found: {financebench_id}")
        records.append(audit_record(record, row, page_map))

    payload = {
        "audit": "evidence_selection_failure_audit_v1",
        "source_ranking_json": str(args.ranking_json),
        "scope": "selection_loss10 from frozen diagnostic30",
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
