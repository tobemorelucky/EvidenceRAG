"""Offline Jina-context vs Jina-context-plus-facts evaluation on fixed30.

The script reads only frozen reranker output, the frozen RRF snapshot, the
FinanceBench CSV for post-selection metrics, and a prebuilt local fact store.
It never runs retrieval, Jina, an answer model, a Judge, or LangSmith.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from evidence_fact_store_v1 import fact_text  # noqa: E402
from scripts.evaluate_evidence_assembly_ab import (  # noqa: E402
    _contains_all,
    _numbers,
    _parse_gold,
    _periods,
    _rate,
    _required_numbers,
    gold_row_hit,
)
from scripts.evaluate_reranker_shadow_v1 import fixture_rows, validate_snapshot  # noqa: E402


DEFAULT_SNAPSHOT = ROOT / "reports" / "reranker_shadow_v1_rrf_top120.json"
DEFAULT_RERANK = ROOT / "reports" / "reranker_shadow_v1.json"
DEFAULT_FACTS = ROOT / "reports" / "fact_store_v1.json"
DEFAULT_OUTPUT = ROOT / "reports" / "evidence_fact_store_shadow_v1.json"
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]{2,}|(?:19|20)\d{2}", re.IGNORECASE)
_STOP = {
    "the", "and", "for", "from", "with", "that", "this", "what", "which", "were", "was",
    "are", "its", "into", "during", "according", "company", "year", "fiscal", "reported",
}


def _tokens(value: object) -> set[str]:
    return {item.casefold() for item in _TOKEN_RE.findall(str(value or "")) if item.casefold() not in _STOP}


def _filename(value: object) -> str:
    return str(value or "").replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _build_frozen_context(chunks: list[dict], ranking: list[dict], *, top_chunks: int = 8, budget: int = 28000) -> tuple[str, list[dict]]:
    ordered = [chunks[int(item["index"])] for item in ranking]
    parts: list[str] = []
    used: list[dict] = []
    remaining = budget
    for chunk in ordered[:top_chunks]:
        separator = 2 if parts else 0
        if remaining <= separator:
            break
        text = str(chunk.get("text") or "")[: remaining - separator]
        parts.append(text)
        used.append({**chunk, "context_text": text})
        remaining -= len(text) + separator
    return "\n\n".join(parts), used


def _fact_score(question: str, fact: dict, context_pages: set[tuple[str, int]]) -> tuple[float, dict]:
    query_tokens = _tokens(question)
    source = fact.get("source_table") or {}
    metric_tokens = _tokens(fact.get("metric"))
    title_tokens = _tokens(source.get("title"))
    entity_tokens = _tokens(fact.get("entity"))
    query_years = {token for token in query_tokens if token.isdigit() and len(token) == 4}
    lexical = len(query_tokens & (metric_tokens | title_tokens)) / max(1, len(query_tokens))
    metric_recall = len(query_tokens & metric_tokens) / max(1, len(metric_tokens))
    entity = len(query_tokens & entity_tokens) / max(1, len(entity_tokens)) if entity_tokens else 0.0
    period = 1.0 if fact.get("period") in query_years else (0.0 if query_years else 0.25)
    same_page = (_filename(source.get("filename")), int(source.get("page_number") or 0)) in context_pages
    score = 0.45 * lexical + 0.25 * metric_recall + 0.15 * entity + 0.10 * period + 0.05 * float(same_page)
    return round(score, 6), {
        "lexical": round(lexical, 6),
        "metric_recall": round(metric_recall, 6),
        "entity": round(entity, 6),
        "period": round(period, 6),
        "same_context_page": same_page,
    }


def retrieve_facts(
    question: str,
    facts_by_filename: dict[str, list[dict]],
    context_chunks: list[dict],
    *,
    max_facts: int = 20,
    max_chars: int = 4000,
) -> tuple[str, list[dict], dict]:
    context_filenames = {_filename(chunk.get("filename")) for chunk in context_chunks}
    context_pages = {
        (_filename(chunk.get("filename")), int(chunk.get("page_number") or 0)) for chunk in context_chunks
    }
    candidates = [fact for name in context_filenames for fact in facts_by_filename.get(name, [])]
    ranked = []
    for fact in candidates:
        score, components = _fact_score(question, fact, context_pages)
        if score <= 0:
            continue
        ranked.append((score, str(fact.get("fact_id") or ""), fact, components))
    ranked.sort(key=lambda item: (-item[0], item[1]))

    selected = []
    parts = []
    remaining = max_chars
    seen = set()
    for score, _, fact, components in ranked:
        identity = (fact.get("table_id"), fact.get("metric"), fact.get("period"), fact.get("value"))
        if identity in seen:
            continue
        rendered = fact_text(fact)
        separator = 2 if parts else 0
        if len(rendered) + separator > remaining:
            continue
        seen.add(identity)
        parts.append(rendered)
        remaining -= len(rendered) + separator
        selected.append({
            "fact_id": fact.get("fact_id"),
            "document_id": fact.get("document_id"),
            "page_id": fact.get("page_id"),
            "table_id": fact.get("table_id"),
            "entity": fact.get("entity"),
            "period": fact.get("period"),
            "metric": fact.get("metric"),
            "value": fact.get("value"),
            "unit": fact.get("unit"),
            "score": score,
            "score_components": components,
            "source_table": fact.get("source_table"),
        })
        if len(selected) >= max_facts:
            break
    return "\n\n".join(parts), selected, {
        "scope": "documents_present_in_frozen_jina_context",
        "candidate_facts": len(candidates),
        "positive_score_facts": len(ranked),
        "selected_facts": len(selected),
        "fact_chars": len("\n\n".join(parts)),
    }


def _load_rows() -> dict[str, dict]:
    path = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["financebench_id"]: row for row in csv.DictReader(handle)}


def _summary(records: list[dict]) -> dict:
    def metrics(items: list[dict]) -> dict:
        return {
            "questions": len(items),
            "baseline_required_number_hit": _rate([item["required_number_hit"]["jina_context"] for item in items]),
            "augmented_required_number_hit": _rate([item["required_number_hit"]["jina_context_plus_facts"] for item in items]),
            "baseline_required_period_hit": _rate([item["required_period_hit"]["jina_context"] for item in items]),
            "augmented_required_period_hit": _rate([item["required_period_hit"]["jina_context_plus_facts"] for item in items]),
            "fact_coverage": _rate([item["fact_coverage"] for item in items]),
            "candidate_fact_coverage": _rate([item["candidate_fact_coverage"] for item in items]),
            "candidate_fact_required_number_hit": _rate([item["candidate_fact_required_number_hit"] for item in items]),
            "questions_with_selected_facts": sum(bool(item["selected_facts"]) for item in items),
            "mean_baseline_context_chars": round(fmean(item["context_chars"]["jina_context"] for item in items), 2) if items else 0,
            "mean_augmented_context_chars": round(fmean(item["context_chars"]["jina_context_plus_facts"] for item in items), 2) if items else 0,
            "mean_fact_chars": round(fmean(item["fact_retrieval_trace"]["fact_chars"] for item in items), 2) if items else 0,
        }
    return {
        **metrics(records),
        "groups": {group: metrics([item for item in records if item["group"] == group]) for group in sorted({item["group"] for item in records})},
        "number_hit_gains": [item["question_id"] for item in records if item["required_number_hit"]["jina_context"] is False and item["required_number_hit"]["jina_context_plus_facts"] is True],
        "number_hit_regressions": [item["question_id"] for item in records if item["required_number_hit"]["jina_context"] is True and item["required_number_hit"]["jina_context_plus_facts"] is False],
        "period_hit_gains": [item["question_id"] for item in records if item["required_period_hit"]["jina_context"] is False and item["required_period_hit"]["jina_context_plus_facts"] is True],
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def _markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Evidence Fact Store v1 Shadow（固定 Jina 30题）", "",
        "- Retrieval/Jina/LLM/Judge/LangSmith calls: **0**",
        f"- Frozen snapshot: `{payload['inputs']['snapshot']}`",
        f"- Frozen Jina result: `{payload['inputs']['reranker_result']}`",
        f"- Fact store: `{payload['inputs']['fact_store']}`",
        "- Jina context严格复用缓存重排后的Top8 chunks、28K字符；fact selection仅在这些context所属文档内进行。",
        "- required number/period与gold fact coverage只用于选择完成后的离线评估，不参与fact检索与排序。", "",
        "## 汇总", "",
        "| 指标 | A: Jina context | B: Jina context + facts |", "|---|---:|---:|",
        f"| Required number hit | {_pct(summary['baseline_required_number_hit'])} | {_pct(summary['augmented_required_number_hit'])} |",
        f"| Required period hit | {_pct(summary['baseline_required_period_hit'])} | {_pct(summary['augmented_required_period_hit'])} |",
        f"| 平均context chars | {summary['mean_baseline_context_chars']} | {summary['mean_augmented_context_chars']} |",
        "", f"- Selected fact-only gold fact coverage: {_pct(summary['fact_coverage'])}",
        f"- Document-scoped candidate fact coverage（排序前上限）: {_pct(summary['candidate_fact_coverage'])}",
        f"- Candidate fact required-number hit（排序前上限）: {_pct(summary['candidate_fact_required_number_hit'])}",
        f"- 有fact入选的问题：{summary['questions_with_selected_facts']}/{summary['questions']}",
        f"- 平均新增fact chars：{summary['mean_fact_chars']}",
        f"- Required-number gains/regressions: `{summary['number_hit_gains']}` / `{summary['number_hit_regressions']}`",
        f"- Required-period gains: `{summary['period_hit_gains']}`", "",
        "## 分组", "", "| Group | N | Number A/B | Period A/B | Fact coverage | Avg chars A/B |", "|---|---:|---:|---:|---:|---:|",
    ]
    for group, item in summary["groups"].items():
        lines.append(
            f"| {group} | {item['questions']} | {_pct(item['baseline_required_number_hit'])} / {_pct(item['augmented_required_number_hit'])} | "
            f"{_pct(item['baseline_required_period_hit'])} / {_pct(item['augmented_required_period_hit'])} | {_pct(item['fact_coverage'])} | "
            f"{item['mean_baseline_context_chars']} / {item['mean_augmented_context_chars']} |"
        )
    lines += ["", "## 逐题", "", "| ID | Group | Facts | Number A/B | Period A/B | Fact coverage | Chars A/B |", "|---|---|---:|---:|---:|---:|---:|"]
    for item in payload["records"]:
        lines.append(
            f"| {item['question_id']} | {item['group']} | {len(item['selected_facts'])} | "
            f"{item['required_number_hit']['jina_context']} / {item['required_number_hit']['jina_context_plus_facts']} | "
            f"{item['required_period_hit']['jina_context']} / {item['required_period_hit']['jina_context_plus_facts']} | "
            f"{item['fact_coverage']} | {item['context_chars']['jina_context']} / {item['context_chars']['jina_context_plus_facts']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--reranker-result", type=Path, default=DEFAULT_RERANK)
    parser.add_argument("--fact-store", type=Path, default=DEFAULT_FACTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-facts", type=int, default=20)
    parser.add_argument("--max-fact-chars", type=int, default=4000)
    args = parser.parse_args()
    if args.max_facts < 1 or args.max_fact_chars < 1:
        parser.error("fact limits must be positive")

    rows_for_validation, groups = fixture_rows()
    snapshot_payload = json.loads(args.snapshot.read_text(encoding="utf-8"))
    frozen = validate_snapshot(snapshot_payload, rows_for_validation, groups)
    frozen_by_id = {item["question_id"]: item for item in frozen}
    rerank_payload = json.loads(args.reranker_result.read_text(encoding="utf-8"))
    rerank_by_id = {item["question_id"]: item for item in rerank_payload["records"]}
    if set(rerank_by_id) != set(frozen_by_id):
        raise ValueError("Jina result and frozen snapshot question sets differ")

    fact_payload = json.loads(args.fact_store.read_text(encoding="utf-8"))
    if fact_payload.get("schema") != "evidence_fact_store_v1":
        raise ValueError("Not an Evidence Fact Store v1 file")
    facts_by_filename: dict[str, list[dict]] = defaultdict(list)
    for fact in fact_payload["facts"]:
        facts_by_filename[_filename((fact.get("source_table") or {}).get("filename"))].append(fact)
    dataset_rows = _load_rows()

    records = []
    for question_id, frozen_record in frozen_by_id.items():
        route = rerank_by_id[question_id]["routes"].get("jina") or {}
        if route.get("status") != "ok":
            raise ValueError(f"Jina route is not complete for {question_id}")
        if rerank_by_id[question_id].get("candidate_sha256") != frozen_record["candidate_sha256"]:
            raise ValueError(f"Candidate hash drift for {question_id}")
        baseline, context_chunks = _build_frozen_context(frozen_record["chunks"], route["ranked"])
        expected = route["metrics"]
        actual_pages = list(dict.fromkeys(
            (_filename(chunk.get("filename")), int(chunk.get("page_number") or 0)) for chunk in context_chunks
        ))
        expected_pages = [(_filename(page[0]), int(page[1])) for page in expected["context_pages"]]
        if len(baseline) != int(expected["context_chars"]) or actual_pages != expected_pages:
            raise ValueError(f"Frozen Jina context reconstruction mismatch for {question_id}")

        fact_evidence, selected_facts, trace = retrieve_facts(
            frozen_record["question"], facts_by_filename, context_chunks,
            max_facts=args.max_facts, max_chars=args.max_fact_chars,
        )
        augmented = baseline + ("\n\n[Structured Evidence Facts]\n" + fact_evidence if fact_evidence else "")
        row = dataset_rows[question_id]
        gold = _parse_gold(row)
        required_numbers = _required_numbers(row)
        required_periods = _periods(row["question"])
        context_filenames = {_filename(chunk.get("filename")) for chunk in context_chunks}
        candidate_fact_evidence = "\n\n".join(
            fact_text(fact) for name in context_filenames for fact in facts_by_filename.get(name, [])
        )
        records.append({
            "question_id": question_id,
            "question": frozen_record["question"],
            "group": frozen_record["group"],
            "frozen_jina_context": {
                "chunk_ids": [chunk["chunk_id"] for chunk in context_chunks],
                "pages": [{"filename": chunk["filename"], "page_number": chunk["page_number"]} for chunk in context_chunks],
                "sha256": hashlib.sha256(baseline.encode("utf-8")).hexdigest(),
            },
            "required_numbers": required_numbers,
            "required_periods": required_periods,
            "required_number_hit": {
                "jina_context": _contains_all(required_numbers, baseline, _numbers),
                "jina_context_plus_facts": _contains_all(required_numbers, augmented, _numbers),
            },
            "required_period_hit": {
                "jina_context": _contains_all(required_periods, baseline, _periods),
                "jina_context_plus_facts": _contains_all(required_periods, augmented, _periods),
            },
            "fact_coverage": gold_row_hit(gold, fact_evidence, required_numbers=required_numbers) if fact_evidence else False,
            "candidate_fact_coverage": gold_row_hit(gold, candidate_fact_evidence, required_numbers=required_numbers) if candidate_fact_evidence else False,
            "candidate_fact_required_number_hit": _contains_all(required_numbers, candidate_fact_evidence, _numbers),
            "context_chars": {"jina_context": len(baseline), "jina_context_plus_facts": len(augmented)},
            "selected_facts": selected_facts,
            "fact_retrieval_trace": trace,
        })

    payload = {
        "schema": "evidence_fact_store_shadow_v1",
        "inputs": {
            "snapshot": str(args.snapshot),
            "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
            "reranker_result": str(args.reranker_result),
            "reranker_result_sha256": hashlib.sha256(args.reranker_result.read_bytes()).hexdigest(),
            "fact_store": str(args.fact_store),
            "fact_store_sha256": hashlib.sha256(args.fact_store.read_bytes()).hexdigest(),
        },
        "external_calls": {"retrieval": 0, "jina": 0, "llm": 0, "judge": 0, "langsmith": 0},
        "configuration": {"jina_context_top_chunks": 8, "jina_context_budget": 28000, "max_facts": args.max_facts, "max_fact_chars": args.max_fact_chars},
        "summary": _summary(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"Report: {args.output.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
