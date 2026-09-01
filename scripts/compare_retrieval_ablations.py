"""Combine frozen Core v3 and retrieval-only ablation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json"
DEFAULT_CORE = ROOT / "reports" / "evidencerag-rag-core-v3-skills-all100-final_answers.jsonl"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_filename(value: object) -> str:
    name = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return name.casefold() if name.casefold().endswith(".pdf") else f"{name}.pdf".casefold()


def _keys(items: list[dict]) -> set[tuple[str, int]]:
    result = set()
    for item in items or []:
        try:
            result.add((_normalize_filename(item.get("filename")), int(item.get("page_number") or 0)))
        except (TypeError, ValueError):
            continue
    return result


def _fixture(path: Path) -> tuple[list[str], dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids, groups = [], {}
    for group in ("candidate_miss10", "selection_loss10", "correct_regression10"):
        for item in payload[group]:
            financebench_id = item if isinstance(item, str) else item["financebench_id"]
            ids.append(financebench_id)
            groups[financebench_id] = group
    return ids, groups


def _core_metrics(dataset: Path, fixture: Path, answers: Path) -> tuple[dict, list[dict]]:
    ids, groups = _fixture(fixture)
    with dataset.open(encoding="utf-8-sig", newline="") as handle:
        gold = {
            row["financebench_id"]: {
                (_normalize_filename(item.get("doc_name")), int(item.get("evidence_page_num") or 0))
                for item in json.loads(row.get("evidence") or "[]")
            }
            for row in csv.DictReader(handle)
            if row["financebench_id"] in set(ids)
        }
    by_id = {item["financebench_id"]: item for item in _jsonl(answers)}
    records = []
    for financebench_id in ids:
        answer = by_id[financebench_id]
        trace = answer.get("rag_trace") or {}
        target = gold[financebench_id]
        records.append({
            "financebench_id": financebench_id,
            "group": groups[financebench_id],
            "candidate_hit": bool(target & _keys(trace.get("initial_retrieved_chunks") or [])),
            "selected_hit": bool(target & _keys(trace.get("selected_pages") or [])),
            "context_hit": bool(target & _keys(trace.get("answer_context_pages") or answer.get("citations") or [])),
            "jina_chars": int(trace.get("remote_rerank_input_chars") or 0),
            "latency_ms": float((answer.get("evaluation_latency") or {}).get("retrieval_ms") or 0),
        })
    metrics = _metrics(records)
    metrics["retrieval_calls"] = len(records)
    metrics["jina_calls"] = sum(bool(item["jina_chars"]) for item in records)
    metrics["jina_chars"] = sum(item["jina_chars"] for item in records)
    return metrics, records


def _metrics(records: list[dict]) -> dict:
    count = len(records)
    return {
        "questions": count,
        "candidate_hit": round(sum(item["candidate_hit"] for item in records) / max(1, count), 4),
        "selected_hit": round(sum(item["selected_hit"] for item in records) / max(1, count), 4),
        "context_hit": round(sum(item["context_hit"] for item in records) / max(1, count), 4),
        "average_latency_ms": round(sum(item.get("latency_ms", 0) for item in records) / max(1, count), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--core-answers", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--structural", type=Path, default=ROOT / "reports" / "retrieval_ablation_structural_diagnostic30.json")
    parser.add_argument("--field-aware", type=Path, default=ROOT / "reports" / "retrieval_ablation_field_aware_diagnostic30.json")
    parser.add_argument("--recall-k", type=Path, default=ROOT / "reports" / "retrieval_recall_k_ablation.json")
    parser.add_argument("--page-pool", type=Path, default=ROOT / "reports" / "page_level_candidate_pool_diagnostic30.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "retrieval_architecture_ablation.json")
    args = parser.parse_args()
    core_metrics, core_records = _core_metrics(args.dataset, args.fixture, args.core_answers)
    structural = json.loads(args.structural.read_text(encoding="utf-8"))
    field_aware = json.loads(args.field_aware.read_text(encoding="utf-8"))
    recall_k = json.loads(args.recall_k.read_text(encoding="utf-8"))
    page_pool = json.loads(args.page_pool.read_text(encoding="utf-8"))
    profiles = {
        "core_v3": core_metrics,
        "structural": structural["metrics"],
        "field_aware": field_aware["metrics"],
    }
    structural_by_id = {item["financebench_id"]: item for item in structural["records"]}
    field_by_id = {item["financebench_id"]: item for item in field_aware["records"]}
    migrations = []
    for item in core_records:
        financebench_id = item["financebench_id"]
        migrations.append({
            "financebench_id": financebench_id,
            "group": item["group"],
            "core_context_hit": item["context_hit"],
            "structural_context_hit": structural_by_id[financebench_id]["context_hit"],
            "field_aware_context_hit": field_by_id[financebench_id]["context_hit"],
        })
    group_context = {}
    for group in ("candidate_miss10", "selection_loss10", "correct_regression10"):
        subset = [item for item in migrations if item["group"] == group]
        group_context[group] = {
            "questions": len(subset),
            "core_v3": sum(item["core_context_hit"] for item in subset),
            "structural": sum(item["structural_context_hit"] for item in subset),
            "field_aware": sum(item["field_aware_context_hit"] for item in subset),
        }
    funnel = json.loads((ROOT / "reports" / "retrieval_funnel_audit.json").read_text(encoding="utf-8"))
    payload = {
        "profiles": profiles,
        "group_context_hits": group_context,
        "recall_k": recall_k["metrics"],
        "route_overlap": funnel.get("route_overlap") or {},
        "funnel_transitions": funnel.get("transitions") or {},
        "page_candidate_upper_bound": page_pool["metrics"],
        "migrations": migrations,
        "decision": {
            "keep_context_budget_v3_frozen": True,
            "field_aware_continue": False,
            "page_level_jina_continue": False,
            "reason": (
                "Structural page-first improves context hit, field-aware adds no aggregate hit and costs more calls; "
                "the page-level Jina prototype loses gold pages before Jina because Top100 chunks remain about 96 unique pages."
            ),
        },
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Retrieval Architecture Ablation", "",
        "> diagnostic30 retrieval-only; no answer model or Judge.", "",
        "## Profile comparison", "",
        "| Profile | Candidate | Selected | Context | Retrieval calls | Jina chars | Avg latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in profiles.items():
        lines.append(
            f"| {name} | {item['candidate_hit']:.2%} | {item['selected_hit']:.2%} | {item['context_hit']:.2%} | "
            f"{item.get('retrieval_calls', 0)} | {item.get('jina_chars', 0)} | {item['average_latency_ms']:.2f} ms |"
        )
    lines.extend([
        "", "## Context hit by diagnostic group", "",
        "| Group | Core v3 | Structural | Field-aware |",
        "|---|---:|---:|---:|",
    ])
    for group, item in group_context.items():
        lines.append(
            f"| {group} | {item['core_v3']}/{item['questions']} | {item['structural']}/{item['questions']} | "
            f"{item['field_aware']}/{item['questions']} |"
        )
    lines.extend([
        "", "## Recall@K", "",
        "| K | Dense | BM25 | RRF | Avg unique pages | Avg latency |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for k, item in recall_k["metrics"].items():
        lines.append(
            f"| {k} | {item['dense_gold_page_hit']:.2%} | {item['bm25_gold_page_hit']:.2%} | "
            f"{item['candidate_gold_page_hit']:.2%} | {item['average_unique_pages']:.2f} | "
            f"{item['average_retrieval_ms']:.2f} ms |"
        )
    lines.extend([
        "", "## Funnel findings", "",
        f"- Dense only / BM25 only / both / neither: {funnel['route_overlap']['dense_only']} / "
        f"{funnel['route_overlap']['bm25_only']} / {funnel['route_overlap']['both']} / {funnel['route_overlap']['neither']}.",
        f"- Dense hit@120 but RRF miss@120: {funnel['transitions']['dense_hit_at_120_but_rrf_miss_at_120']}.",
        f"- Frozen RRF hit outside Jina input: {funnel['transitions']['frozen_rrf_hit_but_outside_jina_input']}.",
        f"- Entered Jina but absent from Jina output: {funnel['transitions']['entered_jina_but_absent_from_jina_output']}.",
        f"- Page-ranked but not finally selected: {funnel['transitions']['page_ranked_but_not_selected']}.",
        "", "## Page-level Jina gate", "",
        f"- RRF Top-120 hit: {page_pool['metrics']['rrf_candidate_hit']:.2%}",
        f"- Top-30 page-candidate hit before Jina: {page_pool['metrics']['page_candidate_hit']:.2%}",
        "- Decision: stop the current page-level Jina prototype before a 30-call run.",
        "- Reason: Jina cannot recover gold pages removed by the page-candidate cutoff.",
        "", "## Decision", "",
        "- Keep Context Budget v3 and frozen Skills unchanged.",
        "- Do not continue field-aware retrieval: same aggregate hit, more calls and latency.",
        "- Preserve structural page-first as the strongest architecture signal, but diagnose its correct-regression losses.",
        "- Do not run the full 100-question answer experiment yet.", "",
    ])
    markdown = args.output.with_suffix(".md")
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.output}\nMarkdown: {markdown}")


if __name__ == "__main__":
    main()
