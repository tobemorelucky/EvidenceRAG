"""Evidence Budget Audit v1 for frozen Top120 chunks and Assembly v5.

This is a local, evidence-only diagnostic.  It does not call Jina, an answer
model, a Judge, or LangSmith, and it does not mutate production components.
"""

from __future__ import annotations

import argparse
import json
import os
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

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from document_page_store import DocumentPageStore  # noqa: E402
from evidence_assembly_v5 import EvidenceUnit, _render, assemble_evidence_v5, build_evidence_units  # noqa: E402
from rag_core_v4 import retrieve_dense_primary  # noqa: E402
from runtime_profile import RETRIEVAL_DOCUMENT_LOCAL_PROFILE, apply_runtime_profile  # noqa: E402
from scripts.evaluate_evidence_assembly_ab import _numbers, _parse_gold, _periods, _required_numbers  # noqa: E402
from scripts.evaluate_oracle_evidence_block import answer_evidence_coverage  # noqa: E402
from scripts.evaluate_page_selector_v1 import GROUPS, _gold_pages, _load_rows, _page_key  # noqa: E402
from table_store import TableStore  # noqa: E402


DEFAULT_OUTPUT = ROOT / "reports" / "evidence_budget_audit_v1_diagnostic30.json"


def _unit_id(unit: EvidenceUnit | dict) -> str:
    value = unit.to_dict() if isinstance(unit, EvidenceUnit) else unit
    metadata = value.get("metadata") or {}
    if value.get("source_type") == "text":
        identity = metadata.get("chunk_id") or f"rank:{metadata.get('retrieval_rank', 0)}"
    elif value.get("source_type") == "table":
        identity = f"{metadata.get('table_id', '')}:row:{metadata.get('row_index', 0)}"
    else:
        identity = f"rank:{metadata.get('retrieval_rank', 0)}"
    return f"{value.get('source_type', 'unknown')}:{value.get('page_id', '')}:{identity}"


def _rank_score(unit: dict) -> tuple[float, str]:
    metadata = unit.get("metadata") or {}
    rank = max(1, int(metadata.get("retrieval_rank") or 1))
    reciprocal = 1.0 / rank
    if unit.get("source_type") == "table":
        overlap = int(metadata.get("query_overlap") or 0)
        return round(overlap + reciprocal, 8), "query_overlap + reciprocal_source_chunk_rank"
    return round(reciprocal, 8), "reciprocal_source_chunk_rank"


def _matched(required: list[str], source_text: str, extractor) -> list[str]:
    present = set(extractor(source_text))
    return [value for value in required if value in present]


def audit_selected_unit(
    unit: dict,
    *,
    index: int,
    gold: list[dict],
    required_numbers: list[str],
    required_periods: list[str],
) -> dict:
    source_text = str(unit.get("source_text") or "")
    coverage = answer_evidence_coverage(gold, source_text)
    matched_numbers = _matched(required_numbers, source_text, _numbers)
    matched_periods = _matched(required_periods, source_text, _periods)
    score, basis = _rank_score(unit)
    rendered_length = len(_render(EvidenceUnit(**unit), index))
    source_type = str(unit.get("source_type") or "")
    return {
        "unit_id": _unit_id(unit),
        "source_type": source_type,
        "page_id": unit.get("page_id") or "",
        "rank_score": score,
        "rank_score_basis": basis,
        "character_length": rendered_length,
        "source_text_length": len(source_text),
        "is_from_top_chunk": source_type == "text",
        "is_from_table": source_type == "table",
        "gold_evidence_covered": bool(coverage["matched_lines"]),
        "gold_evidence_coverage": coverage,
        "contains_required_number": bool(matched_numbers),
        "matched_required_numbers": matched_numbers,
        "contains_required_period": bool(matched_periods),
        "matched_required_periods": matched_periods,
        "retrieval_rank": int((unit.get("metadata") or {}).get("retrieval_rank") or 0),
        "filename": (unit.get("metadata") or {}).get("filename") or "",
        "page_number": int((unit.get("metadata") or {}).get("page_number") or 0),
    }


def _loss_reason(
    *,
    gold_pages: set[tuple[str, int]],
    candidate_pages: set[tuple[str, int]],
    selected_pages: set[tuple[str, int]],
    candidate_coverage: float | None,
    selected_coverage: float | None,
) -> str:
    candidate_ratio = float(candidate_coverage or 0.0)
    selected_ratio = float(selected_coverage or 0.0)
    if selected_ratio >= 1.0:
        return "not_lost"
    if not gold_pages & candidate_pages:
        return "gold_page_not_in_top120"
    if not gold_pages & selected_pages:
        return "gold_page_dropped_by_28k_budget"
    if candidate_ratio <= 0.0:
        return "gold_text_not_represented_in_candidate_units"
    if candidate_ratio > selected_ratio:
        return "gold_evidence_units_dropped_by_28k_budget"
    return "selected_gold_page_but_gold_evidence_partial"


def _mean(values: list[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 4) if usable else None


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _group_summary(records: list[dict]) -> dict:
    units = [unit for record in records for unit in record["selected_units"]]
    total_chars = sum(unit["character_length"] for unit in units)
    type_chars = Counter()
    for unit in units:
        type_chars[unit["source_type"]] += unit["character_length"]
    return {
        "questions": len(records),
        "average_context_chars": _mean([record["context_chars"] for record in records]),
        "average_budget_utilization": _mean([
            record["context_chars"] / max(1, record["assembly_trace"]["max_context_chars"])
            for record in records
        ]),
        "average_selected_units": _mean([len(record["selected_units"]) for record in records]),
        "text_character_ratio": round(type_chars["text"] / max(1, total_chars), 4),
        "table_character_ratio": round(type_chars["table"] / max(1, total_chars), 4),
        "mixed_character_ratio": round(type_chars["mixed"] / max(1, total_chars), 4),
        "non_gold_character_ratio": _mean([record["non_gold_character_ratio"] for record in records]),
        "conservative_budget_waste_ratio": _mean([record["conservative_budget_waste_ratio"] for record in records]),
        "selected_evidence_coverage": _mean([record["selected_evidence_coverage"] for record in records]),
        "candidate_evidence_coverage": _mean([record["candidate_evidence_coverage"] for record in records]),
        "candidate_to_selected_coverage_loss": _mean([
            max(0.0, float(record["candidate_evidence_coverage"] or 0.0) - float(record["selected_evidence_coverage"] or 0.0))
            for record in records
        ]),
        "average_non_gold_chars": _mean([
            record["unit_character_total_excluding_separators"] * record["non_gold_character_ratio"]
            for record in records
        ]),
        "average_conservative_waste_chars": _mean([
            record["unit_character_total_excluding_separators"] * record["conservative_budget_waste_ratio"]
            for record in records
        ]),
        "required_number_any_unit_hit": _rate([
            any(unit["contains_required_number"] for unit in record["selected_units"])
            for record in records if record["required_numbers"]
        ]),
        "required_period_any_unit_hit": _rate([
            any(unit["contains_required_period"] for unit in record["selected_units"])
            for record in records if record["required_periods"]
        ]),
        "gold_loss_reasons": dict(sorted(Counter(record["gold_evidence_loss_reason"] for record in records).items())),
        "budget_loss_questions": sum("28k_budget" in record["gold_evidence_loss_reason"] for record in records),
    }


def summarize(records: list[dict]) -> dict:
    summary = _group_summary(records)
    summary["groups"] = {
        group: _group_summary([record for record in records if record["group"] == group])
        for group in GROUPS
    }
    correct = summary["groups"]["correct_regression10"]
    selection = summary["groups"]["selection_loss10"]
    comparison_keys = (
            "text_character_ratio", "table_character_ratio", "non_gold_character_ratio",
            "conservative_budget_waste_ratio", "selected_evidence_coverage",
    )
    summary["correct_vs_selection_loss"] = {
        key: round(float(correct[key]) - float(selection[key]), 4)
        if correct.get(key) is not None and selection.get(key) is not None else None
        for key in comparison_keys
    }
    summary["external_calls"] = {"jina": 0, "answer_model": 0, "judge": 0, "langsmith": 0}
    return summary


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Budget Audit v1 — diagnostic30",
        "",
        "> Frozen Top120 retrieval and unchanged Evidence Assembly v5. No Jina, LLM, Judge, or LangSmith calls.",
        "",
        "## Metric definitions",
        "",
        "- `non_gold_character_ratio`: selected-unit characters in units that match no benchmark gold-evidence line.",
        "- `conservative_budget_waste_ratio`: characters in units that match neither gold evidence nor any required number/period.",
        "- Gold annotations are sparse, so non-gold characters are not automatically useless; the conservative ratio is the safer waste estimate.",
        "- `rank_score` reproduces v5 priority evidence: reciprocal source-chunk rank for text; query-token overlap plus reciprocal page-source rank for table rows.",
        "",
        "## Overall composition",
        "",
        f"- Questions: {summary['questions']}",
        f"- Average context chars/units: {summary['average_context_chars']} / {summary['average_selected_units']}",
        f"- Average 28k utilization: {_percent(summary['average_budget_utilization'])}",
        f"- Text/table/mixed character ratio: {_percent(summary['text_character_ratio'])} / {_percent(summary['table_character_ratio'])} / {_percent(summary['mixed_character_ratio'])}",
        f"- Candidate → selected evidence coverage: {_percent(summary['candidate_evidence_coverage'])} → {_percent(summary['selected_evidence_coverage'])}",
        f"- Candidate-to-selected coverage loss: {_percent(summary['candidate_to_selected_coverage_loss'])}",
        f"- Non-gold character ratio: {_percent(summary['non_gold_character_ratio'])}",
        f"- Conservative budget waste ratio: {_percent(summary['conservative_budget_waste_ratio'])}",
        f"- Average non-gold/conservative-waste chars: {summary['average_non_gold_chars']} / {summary['average_conservative_waste_chars']}",
        f"- Gold evidence loss reasons: `{summary['gold_loss_reasons']}`",
        f"- Questions losing gold evidence at the 28k allocation stage: {summary['budget_loss_questions']}/{summary['questions']}",
        "",
        "## Group comparison",
        "",
        "| Group | Text chars | Table chars | Non-gold chars | Conservative waste | Candidate→selected coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        item = summary["groups"][group]
        lines.append(
            f"| {group} | {_percent(item['text_character_ratio'])} | {_percent(item['table_character_ratio'])} | "
            f"{_percent(item['non_gold_character_ratio'])} | {_percent(item['conservative_budget_waste_ratio'])} | "
            f"{_percent(item['candidate_evidence_coverage'])} → {_percent(item['selected_evidence_coverage'])} |"
        )
    lines.extend([
        "",
        f"Correct-regression minus selection-loss: `{summary['correct_vs_selection_loss']}`",
        "",
        "## Findings",
        "",
        "1. The 28k package is almost fully occupied; the problem is allocation quality, not unused capacity.",
        "2. Candidate-unit gold coverage is high, but nearly half is lost before the final package. Most losses are pages or units excluded by the fixed budget.",
        "3. Text/table character ratios are nearly identical between correct-regression and selection-loss, so another global ratio change is unlikely to solve the gap.",
        "4. Conservative obvious waste is materially smaller than total coverage loss. Removing only clearly irrelevant units cannot recover all missing evidence; ranking must identify lower-ranked useful units.",
        "5. Mixed units are zero because unchanged Assembly v5 emits text and table units only; this audit does not invent a third assembly path.",
        "",
        "## Per question",
        "",
    ])
    for index, record in enumerate(payload["records"], 1):
        lines.extend([
            f"### {index}. {record['financebench_id']}",
            "",
            f"- Group: `{record['group']}`",
            f"- Question: {record['question']}",
            f"- Context chars; text/table/mixed: {record['context_chars']}; {record['text_chars']} / {record['table_chars']} / {record['mixed_chars']}",
            f"- Candidate → selected evidence coverage: {record['candidate_evidence_coverage']} → {record['selected_evidence_coverage']}",
            f"- Gold loss reason: `{record['gold_evidence_loss_reason']}`",
            f"- Non-gold/conservative waste: {_percent(record['non_gold_character_ratio'])} / {_percent(record['conservative_budget_waste_ratio'])}",
            "",
            "| Unit | Type | Page | Rank score | Chars | Top chunk | Table | Gold | Required number | Required period |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for unit in record["selected_units"]:
            lines.append(
                f"| `{unit['unit_id']}` | {unit['source_type']} | `{unit['page_id']}` | {unit['rank_score']} | "
                f"{unit['character_length']} | {unit['is_from_top_chunk']} | {unit['is_from_table']} | "
                f"{unit['gold_evidence_covered']} | {unit['contains_required_number']} | {unit['contains_required_period']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--retrieval-k", type=int, default=120)
    parser.add_argument("--max-context-chars", type=int, default=28000)
    parser.add_argument("--text-budget-ratio", type=float, default=0.78)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 30:
        raise SystemExit("--limit must be between 1 and 30")

    apply_runtime_profile(RETRIEVAL_DOCUMENT_LOCAL_PROFILE)
    rows = _load_rows(args.dataset, args.fixture)[:args.limit]
    page_store, table_store = DocumentPageStore(), TableStore()
    records = []
    print(f"[setup] questions={len(rows)} top_k={args.retrieval_k} jina=false llm=false judge=false", flush=True)
    for index, row in enumerate(rows, 1):
        retrieval = retrieve_dense_primary(row["question"], dense_k=args.retrieval_k, bm25_k=30)
        chunks = retrieval["merged"][:args.retrieval_k]
        page_keys = list(dict.fromkeys(
            (str(chunk.get("filename") or ""), int(chunk.get("page_number") or 0))
            for chunk in chunks if str(chunk.get("filename") or "").strip()
        ))
        pages = page_store.get_pages_by_keys(page_keys)
        tables = table_store.get_tables_by_page_keys(page_keys)
        candidate_units = build_evidence_units(row["question"], chunks, pages=pages, tables=tables)
        context, selected_units, trace = assemble_evidence_v5(
            row["question"], chunks, pages=pages, tables=tables,
            max_context_chars=args.max_context_chars, text_budget_ratio=args.text_budget_ratio,
        )
        gold = _parse_gold(row)
        required_numbers = _required_numbers(row)
        required_periods = _periods(row.get("question") or "")
        audited = [
            audit_selected_unit(
                unit, index=unit_index, gold=gold,
                required_numbers=required_numbers, required_periods=required_periods,
            )
            for unit_index, unit in enumerate(selected_units, 1)
        ]
        total_unit_chars = sum(unit["character_length"] for unit in audited)
        text_chars = sum(unit["character_length"] for unit in audited if unit["source_type"] == "text")
        table_chars = sum(unit["character_length"] for unit in audited if unit["source_type"] == "table")
        mixed_chars = sum(unit["character_length"] for unit in audited if unit["source_type"] == "mixed")
        non_gold_chars = sum(unit["character_length"] for unit in audited if not unit["gold_evidence_covered"])
        conservative_waste = sum(
            unit["character_length"] for unit in audited
            if not unit["gold_evidence_covered"]
            and not unit["contains_required_number"]
            and not unit["contains_required_period"]
        )
        candidate_source = "\n\n".join(unit.source_text for unit in candidate_units)
        candidate_coverage = answer_evidence_coverage(gold, candidate_source)["ratio"]
        selected_coverage = answer_evidence_coverage(gold, context)["ratio"]
        candidate_pages = {_page_key(chunk) for chunk in chunks}
        selected_pages = {(unit["filename"].casefold(), unit["page_number"]) for unit in audited if unit["filename"]}
        gold_pages = _gold_pages(row)
        loss_reason = _loss_reason(
            gold_pages=gold_pages, candidate_pages=candidate_pages, selected_pages=selected_pages,
            candidate_coverage=candidate_coverage, selected_coverage=selected_coverage,
        )
        records.append({
            "financebench_id": row["financebench_id"],
            "group": row["diagnostic_group"],
            "question": row["question"],
            "required_numbers": required_numbers,
            "required_periods": required_periods,
            "context_chars": len(context),
            "unit_character_total_excluding_separators": total_unit_chars,
            "separator_chars": max(0, len(context) - total_unit_chars),
            "text_chars": text_chars,
            "table_chars": table_chars,
            "mixed_chars": mixed_chars,
            "text_character_ratio": round(text_chars / max(1, total_unit_chars), 4),
            "table_character_ratio": round(table_chars / max(1, total_unit_chars), 4),
            "mixed_character_ratio": round(mixed_chars / max(1, total_unit_chars), 4),
            "non_gold_character_ratio": round(non_gold_chars / max(1, total_unit_chars), 4),
            "conservative_budget_waste_ratio": round(conservative_waste / max(1, total_unit_chars), 4),
            "candidate_evidence_coverage": candidate_coverage,
            "selected_evidence_coverage": selected_coverage,
            "gold_evidence_loss_reason": loss_reason,
            "candidate_unit_count": len(candidate_units),
            "selected_unit_count": len(audited),
            "assembly_trace": trace,
            "selected_units": audited,
        })
        print(
            f"[{index:02d}/{len(rows)}] {row['financebench_id']} units={len(audited)} "
            f"chars=text:{text_chars}/table:{table_chars} coverage={candidate_coverage}->{selected_coverage} "
            f"loss={loss_reason}",
            flush=True,
        )

    payload = {
        "audit": "evidence_budget_audit_v1_diagnostic30",
        "scope": "frozen Top120 plus unchanged Evidence Assembly v5; no external calls",
        "definitions": {
            "non_gold_character_ratio": "characters in selected units with no benchmark gold-evidence line match",
            "conservative_budget_waste_ratio": "characters in selected units with no gold-line, required-number, or required-period match",
        },
        "config": {
            "retrieval_k": args.retrieval_k,
            "max_context_chars": args.max_context_chars,
            "text_budget_ratio": args.text_budget_ratio,
        },
        "summary": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {args.output}\nMarkdown: {markdown}", flush=True)


if __name__ == "__main__":
    main()
