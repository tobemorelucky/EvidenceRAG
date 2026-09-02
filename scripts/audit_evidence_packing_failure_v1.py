"""Audit why frozen Evidence Units do not enter the 28K answer context."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evidence_assembly_v5 import EvidenceUnit, _render  # noqa: E402
from scripts.audit_evidence_selection_failure_v1 import _filename, _normalized, _numbers, evidence_coverage  # noqa: E402


DEFAULT_INPUT = ROOT / "reports" / "evidence_metadata_counterfactual_v1.json"
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_JSON = ROOT / "reports" / "evidence_packing_failure_audit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "evidence_packing_failure_audit_v1.md"
MAX_CONTEXT_CHARS = 28000
CATEGORIES = {
    "A": "unit rank低，被正常淘汰",
    "B": "unit rank足够高，但因为长度预算丢失",
    "C": "重复unit占用预算",
    "D": "page/group覆盖不足",
    "E": "其他",
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


def _unit_payload(item: dict) -> dict:
    return {
        name: item.get(name)
        for name in EvidenceUnit.__dataclass_fields__
    }


def replay_packing(candidates: list[dict], *, max_context_chars: int = MAX_CONTEXT_CHARS) -> tuple[list[dict], str]:
    """Replay the unchanged sequential first-fit packing with full trace."""
    ranked = sorted(candidates, key=lambda item: int((item.get("current_ranking") or {}).get("rank") or 0))
    traces = []
    rendered = []
    used = 0
    selected_count = 0
    for item in ranked:
        rank = int((item.get("current_ranking") or {}).get("rank") or 0)
        unit = EvidenceUnit(**_unit_payload(item))
        value = _render(unit, selected_count + 1)
        separator = 2 if rendered else 0
        required_chars = separator + len(value)
        remaining_before = max_context_chars - used
        if required_chars <= remaining_before:
            selected = True
            rejection_reason = None
            rendered.append(value)
            used += required_chars
            selected_count += 1
        else:
            selected = False
            rejection_reason = (
                "unit_exceeds_total_context_budget"
                if len(value) > max_context_chars else
                "exceeds_remaining_character_budget"
            )
        metadata = item.get("metadata") or {}
        traces.append({
            "unit_id": (
                metadata.get("chunk_id")
                or f"{metadata.get('table_id')}:{metadata.get('row_index')}"
                or f"rank:{rank}"
            ),
            "unit_score": (item.get("current_ranking") or {}).get("score"),
            "rank": rank,
            "source_text_chars": len(str(item.get("source_text") or "")),
            "rendered_char_length": len(value),
            "separator_chars": separator,
            "remaining_budget_before": remaining_before,
            "required_chars_at_attempt": required_chars,
            "source_page": {
                "filename": metadata.get("filename"),
                "page_number": int(metadata.get("page_number") or 0),
                "page_id": item.get("page_id"),
            },
            "source_type": item.get("source_type"),
            "selected": selected,
            "rejection_reason": rejection_reason,
            "source_text": str(item.get("source_text") or ""),
        })
    return traces, "\n\n".join(rendered)


def _similarity(left: str, right: str) -> float:
    left_tokens = re.findall(r"[a-z0-9%$.-]+", _normalized(left))
    right_tokens = re.findall(r"[a-z0-9%$.-]+", _normalized(right))
    if len(left_tokens) < 20 or len(right_tokens) < 20:
        return 0.0
    if set(_numbers(left)) != set(_numbers(right)):
        return 0.0
    length_ratio = min(len(left_tokens), len(right_tokens)) / max(len(left_tokens), len(right_tokens))
    if length_ratio < 0.9:
        return 0.0
    left_shingles = {tuple(left_tokens[index:index + 5]) for index in range(len(left_tokens) - 4)}
    right_shingles = {tuple(right_tokens[index:index + 5]) for index in range(len(right_tokens) - 4)}
    intersection = len(left_shingles & right_shingles)
    jaccard = intersection / max(1, len(left_shingles | right_shingles))
    return round(jaccard, 4)


def mark_selected_duplicates(traces: list[dict], *, threshold: float = 0.92) -> dict:
    """Mark exact or high-containment duplicates among selected units."""
    selected = [item for item in traces if item["selected"]]
    duplicate_chars = 0
    duplicate_units = 0
    for index, item in enumerate(selected):
        normalized = _normalized(item["source_text"])
        best_rank = None
        best_similarity = 0.0
        for earlier in selected[:index]:
            earlier_normalized = _normalized(earlier["source_text"])
            similarity = 1.0 if normalized and normalized == earlier_normalized else _similarity(
                item["source_text"], earlier["source_text"]
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_rank = earlier["rank"]
        is_duplicate = best_similarity >= threshold
        item["duplicate_of_selected_rank"] = best_rank if is_duplicate else None
        item["duplicate_similarity"] = best_similarity if is_duplicate else 0.0
        item["selected_duplicate"] = is_duplicate
        if is_duplicate:
            duplicate_units += 1
            duplicate_chars += item["rendered_char_length"] + item["separator_chars"]
    for item in traces:
        item.setdefault("duplicate_of_selected_rank", None)
        item.setdefault("duplicate_similarity", 0.0)
        item.setdefault("selected_duplicate", False)
    return {"selected_duplicate_units": duplicate_units, "selected_duplicate_chars": duplicate_chars}


def _gold_lines(gold: list[dict]) -> list[dict]:
    lines = []
    for item in gold:
        for line in str(item.get("evidence_text") or "").splitlines():
            if len(_normalized(line)) >= 3:
                lines.append({
                    "filename": item["filename"],
                    "page_number": item["page_number"],
                    "evidence_text": line,
                })
    return lines


def annotate_gold(traces: list[dict], gold: list[dict]) -> None:
    gold_pages = {(item["filename"], item["page_number"]) for item in gold}
    lines = _gold_lines(gold)
    for item in traces:
        page = item["source_page"]
        page_key = (_filename(page.get("filename")), int(page.get("page_number") or 0))
        matching_lines = [line for line in lines if (line["filename"], line["page_number"]) == page_key]
        coverage = evidence_coverage(matching_lines, item["source_text"]) if matching_lines else {
            "matched_lines": 0, "total_lines": 0, "ratio": None,
        }
        item["gold_page_associated"] = page_key in gold_pages
        item["gold_evidence_matched_lines"] = coverage["matched_lines"]
        item["contains_gold_evidence"] = coverage["matched_lines"] > 0


def classify_failure(
    *,
    candidate_coverage: float | None,
    selected_coverage: float | None,
    missing_gold_units: list[dict],
    selection_frontier: int | None,
    duplicate_reclaimable_chars: int,
) -> tuple[str, str]:
    if candidate_coverage is None or candidate_coverage < 1.0:
        return "D", "The complete candidate package does not contain every annotated evidence line."
    if selected_coverage == 1.0:
        return "E", "Annotated evidence is already present after packing; the historical failure is outside packing."
    if not missing_gold_units:
        return "E", "Candidate coverage is complete, but no single omitted unit is identified as the missing carrier."
    best = min(missing_gold_units, key=lambda item: item["rank"])
    deficit = max(0, best["required_chars_at_attempt"] - best["remaining_budget_before"])
    if duplicate_reclaimable_chars >= deficit > 0:
        return "C", "Earlier selected duplicate units consume enough characters to fit the best omitted gold unit."
    if selection_frontier is not None and best["rank"] <= selection_frontier:
        return "B", "A gold-bearing unit ranks inside the effective selected frontier but fails the remaining-budget check."
    return "A", "Gold-bearing units fall below the effective selected rank frontier."


def audit_record(record: dict, row: dict) -> dict:
    candidates = list(record.get("candidate_units") or [])
    traces, context = replay_packing(candidates)
    expected_selected = set(record["routes"]["current_ranking"].get("selected_unit_ranks") or [])
    actual_selected = {item["rank"] for item in traces if item["selected"]}
    if actual_selected != expected_selected:
        raise RuntimeError(
            f"Packing replay mismatch for {record['financebench_id']}: "
            f"expected={sorted(expected_selected)} actual={sorted(actual_selected)}"
        )
    duplicate_summary = mark_selected_duplicates(traces)
    gold = _gold(row)
    annotate_gold(traces, gold)
    candidate_text = "\n\n".join(item["source_text"] for item in traces)
    candidate_coverage = evidence_coverage(gold, candidate_text)
    selected_coverage = evidence_coverage(gold, context)
    selected_ranks = [item["rank"] for item in traces if item["selected"]]
    frontier = max(selected_ranks) if selected_ranks else None
    missing_gold = [item for item in traces if item["contains_gold_evidence"] and not item["selected"]]
    best_missing_rank = min((item["rank"] for item in missing_gold), default=None)
    duplicate_before = sum(
        item["rendered_char_length"] + item["separator_chars"]
        for item in traces
        if item["selected_duplicate"] and (best_missing_rank is None or item["rank"] < best_missing_rank)
    )
    category, rationale = classify_failure(
        candidate_coverage=candidate_coverage["ratio"],
        selected_coverage=selected_coverage["ratio"],
        missing_gold_units=missing_gold,
        selection_frontier=frontier,
        duplicate_reclaimable_chars=duplicate_before,
    )
    for item in traces:
        item.pop("source_text", None)
    return {
        "financebench_id": record["financebench_id"],
        "group": record["group"],
        "question": record["question"],
        "candidate_unit_count": len(traces),
        "selected_unit_count": len(actual_selected),
        "context_chars": len(context),
        "selection_frontier_rank": frontier,
        "candidate_gold_evidence_coverage": candidate_coverage,
        "selected_gold_evidence_coverage": selected_coverage,
        "gold_page_candidate_count": sum(item["gold_page_associated"] for item in traces),
        "gold_evidence_candidate_count": sum(item["contains_gold_evidence"] for item in traces),
        "missing_gold_unit_ranks": [item["rank"] for item in missing_gold],
        "duplicate_summary": {
            **duplicate_summary,
            "reclaimable_duplicate_chars_before_best_missing_gold": duplicate_before,
        },
        "classification": {"code": category, "label": CATEGORIES[category], "rationale": rationale},
        "candidate_units": traces,
    }


def _mean(values: list[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 4) if usable else None


def summarize(records: list[dict]) -> dict:
    selection = [item for item in records if item["group"] == "selection_loss10"]
    counts = Counter(item["classification"]["code"] for item in selection)
    direction_scores = {
        "packing优化": counts["B"] + counts["C"],
        "ranking优化": counts["A"],
        "evidence block重构": counts["D"] + counts["E"],
    }
    maximum = max(direction_scores.values(), default=0)
    leaders = [name for name, count in direction_scores.items() if count == maximum]
    next_direction = leaders[0] if len(leaders) == 1 else "mixed: " + " + ".join(leaders)
    return {
        "questions": len(records),
        "candidate_units": sum(item["candidate_unit_count"] for item in records),
        "selected_units": sum(item["selected_unit_count"] for item in records),
        "average_context_chars": _mean([item["context_chars"] for item in records]),
        "packing_replay_valid_questions": len(records),
        "selection_loss10": {
            "category_counts": {
                code: {"label": label, "count": counts[code], "rate": round(counts[code] / max(1, len(selection)), 4)}
                for code, label in CATEGORIES.items()
            },
            "direction_scores": direction_scores,
            "next_direction": next_direction,
            "average_candidate_coverage": _mean([
                item["candidate_gold_evidence_coverage"]["ratio"] for item in selection
            ]),
            "average_selected_coverage": _mean([
                item["selected_gold_evidence_coverage"]["ratio"] for item in selection
            ]),
            "duplicate_units": sum(item["duplicate_summary"]["selected_duplicate_units"] for item in selection),
            "duplicate_chars": sum(item["duplicate_summary"]["selected_duplicate_chars"] for item in selection),
        },
        "external_calls": {"retrieval": 0, "llm": 0, "jina": 0, "judge": 0, "langsmith": 0},
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    selection = summary["selection_loss10"]
    lines = [
        "# Evidence Packing Failure Audit v1",
        "",
        "> Existing frozen candidate units only. The unchanged 28K sequential first-fit packing is replayed exactly. Retrieval=0, LLM=0, Jina=0, Judge=0, LangSmith=0.",
        "",
        "## Replay summary",
        "",
        f"- Questions: {summary['questions']}",
        f"- Candidate / selected Evidence Units: {summary['candidate_units']} / {summary['selected_units']}",
        f"- Valid packing replays: {summary['packing_replay_valid_questions']}/30",
        f"- Average context chars: {summary['average_context_chars']}",
        "",
        "## Selection-loss classification",
        "",
        "| Category | Meaning | Count | Rate |",
        "|---|---|---:|---:|",
    ]
    for code, item in selection["category_counts"].items():
        lines.append(f"| {code} | {item['label']} | {item['count']} | {item['rate']:.2%} |")
    lines.extend([
        "",
        f"- Candidate evidence coverage: {_pct(selection['average_candidate_coverage'])}",
        f"- Selected evidence coverage: {_pct(selection['average_selected_coverage'])}",
        f"- Selected duplicate units/chars: {selection['duplicate_units']} / {selection['duplicate_chars']}",
        f"- Direction scores: `{selection['direction_scores']}`",
        f"- Recommended next shadow development: **{selection['next_direction']}**",
        "",
        "## Classification contract",
        "",
        "- D is assigned first when all candidates combined still lack annotated evidence; this is not blamed on packing.",
        "- C requires selected duplicates before the omitted gold unit to reclaim enough characters to fit it.",
        "- B requires a gold-bearing unit inside the effective selected rank frontier that fails the remaining-budget check.",
        "- A means the gold-bearing unit is below that frontier.",
        "- E means packing already retained the evidence or the remaining failure cannot be assigned safely.",
        "",
        "## Selection-loss details",
        "",
    ])
    selection_records = [item for item in payload["records"] if item["group"] == "selection_loss10"]
    for index, record in enumerate(selection_records, 1):
        classification = record["classification"]
        lines.extend([
            f"### {index}. {record['financebench_id']} — {classification['code']}: {classification['label']}",
            "",
            f"- Question: {record['question']}",
            f"- Candidate / selected units: `{record['candidate_unit_count']}` / `{record['selected_unit_count']}`",
            f"- Candidate / selected gold coverage: `{record['candidate_gold_evidence_coverage']['ratio']}` / `{record['selected_gold_evidence_coverage']['ratio']}`",
            f"- Gold page / gold-bearing candidates: `{record['gold_page_candidate_count']}` / `{record['gold_evidence_candidate_count']}`",
            f"- Selection frontier: `{record['selection_frontier_rank']}`; missing gold ranks: `{record['missing_gold_unit_ranks']}`",
            f"- Duplicate summary: `{record['duplicate_summary']}`",
            f"- Rationale: {classification['rationale']}",
            "",
            "| Rank | Score | Type | Page | Chars | Remaining before | Selected | Rejection | Gold | Duplicate |",
            "|---:|---:|---|---:|---:|---:|---|---|---|---|",
        ])
        for unit in record["candidate_units"]:
            if not (unit["contains_gold_evidence"] or unit["selected_duplicate"] or unit["rank"] <= 10):
                continue
            lines.append(
                f"| {unit['rank']} | {unit['unit_score']} | {unit['source_type']} | "
                f"{unit['source_page']['page_number']} | {unit['rendered_char_length']} | "
                f"{unit['remaining_budget_before']} | {unit['selected']} | "
                f"{unit['rejection_reason'] or 'selected'} | {unit['contains_gold_evidence']} | "
                f"{unit['selected_duplicate']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    frozen = json.loads(args.input_json.read_text(encoding="utf-8"))
    rows = _dataset_rows(args.dataset)
    records = []
    for index, record in enumerate(frozen.get("records") or [], 1):
        row = rows.get(record["financebench_id"])
        if row is None:
            raise RuntimeError(f"Dataset row missing: {record['financebench_id']}")
        result = audit_record(record, row)
        records.append(result)
        print(
            f"[{index:02d}/30] {result['financebench_id']} "
            f"coverage={result['candidate_gold_evidence_coverage']['ratio']}->"
            f"{result['selected_gold_evidence_coverage']['ratio']} "
            f"class={result['classification']['code']}",
            flush=True,
        )
    payload = {
        "audit": "evidence_packing_failure_audit_v1",
        "source": str(args.input_json),
        "scope": "frozen current-ranking candidate units and unchanged 28K packing",
        "summary": summarize(records),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {args.output_json}\nMarkdown: {args.output_markdown}", flush=True)


if __name__ == "__main__":
    main()
