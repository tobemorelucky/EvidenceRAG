"""Run one isolated structural/field-aware retrieval profile on diagnostic30."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from runtime_profile import (  # noqa: E402
    RETRIEVAL_ABLATION_FIELD_AWARE_PROFILE,
    RETRIEVAL_ABLATION_STRUCTURAL_PROFILE,
    apply_runtime_profile,
)


PROFILES = {RETRIEVAL_ABLATION_STRUCTURAL_PROFILE, RETRIEVAL_ABLATION_FIELD_AWARE_PROFILE}
DEFAULT_DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json"


def _normalize_filename(value: object) -> str:
    name = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return name.casefold() if name.casefold().endswith(".pdf") else f"{name}.pdf".casefold()


def _gold(row: dict) -> set[tuple[str, int]]:
    return {
        (_normalize_filename(item.get("doc_name")), int(item.get("evidence_page_num") or 0))
        for item in json.loads(row.get("evidence") or "[]")
    }


def _keys(items: list[dict]) -> set[tuple[str, int]]:
    result = set()
    for item in items or []:
        try:
            result.add((_normalize_filename(item.get("filename")), int(item.get("page_number") or 0)))
        except (TypeError, ValueError):
            continue
    return result


def _diagnostic_rows(dataset: Path, fixture: Path) -> list[dict]:
    groups = json.loads(fixture.read_text(encoding="utf-8"))
    ids = []
    for group in ("candidate_miss10", "selection_loss10", "correct_regression10"):
        ids.extend(item if isinstance(item, str) else item["financebench_id"] for item in groups[group])
    wanted = set(ids)
    with dataset.open(encoding="utf-8-sig", newline="") as handle:
        by_id = {row["financebench_id"]: row for row in csv.DictReader(handle) if row["financebench_id"] in wanted}
    return [by_id[item] for item in ids if item in by_id]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    apply_runtime_profile(args.profile)
    # Import after the profile is applied because rag_utils snapshots feature
    # environment variables at module import time.
    from rag_utils import retrieve_documents

    rows = _diagnostic_rows(args.dataset, args.fixture)
    if args.limit:
        rows = rows[: args.limit]
    output = args.output or ROOT / "reports" / f"{args.profile}_diagnostic30.json"
    records = []
    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        result = retrieve_documents(row["question"], top_k=8, candidate_k=60, apply_page_merge=True)
        meta = result.get("meta") or {}
        gold = _gold(row)
        candidate = result.get("candidate_docs") or []
        selected = result.get("final_retrieved_docs") or []
        context = result.get("context_docs") or []
        record = {
            "financebench_id": row["financebench_id"],
            "candidate_hit": bool(gold & _keys(candidate)),
            "selected_hit": bool(gold & _keys(selected)),
            "context_hit": bool(gold & _keys(context)),
            "candidate_pages": sorted([list(item) for item in _keys(candidate)]),
            "selected_pages": sorted([list(item) for item in _keys(selected)]),
            "context_pages": sorted([list(item) for item in _keys(context)]),
            "retrieval_calls": len(meta.get("per_query_retrieval_counts") or []),
            "supplemental_attempted": bool(meta.get("supplemental_search_attempted")),
            "jina_chars": int(meta.get("remote_rerank_input_chars") or 0),
            "jina_calls": int(bool(meta.get("remote_success") or meta.get("rerank_cache_hit"))),
            "rerank_provider": meta.get("rerank_provider") or "none",
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        records.append(record)
        print(
            f"[{index:02d}/{len(rows)}] {row['financebench_id']} "
            f"candidate={record['candidate_hit']} selected={record['selected_hit']} context={record['context_hit']}",
            flush=True,
        )

    questions = len(records)
    metrics = {
        "questions": questions,
        "candidate_hit": round(sum(item["candidate_hit"] for item in records) / max(1, questions), 4),
        "selected_hit": round(sum(item["selected_hit"] for item in records) / max(1, questions), 4),
        "context_hit": round(sum(item["context_hit"] for item in records) / max(1, questions), 4),
        "retrieval_calls": sum(item["retrieval_calls"] for item in records),
        "supplemental_attempts": sum(item["supplemental_attempted"] for item in records),
        "jina_calls": sum(item["jina_calls"] for item in records),
        "jina_chars": sum(item["jina_chars"] for item in records),
        "average_latency_ms": round(sum(item["latency_ms"] for item in records) / max(1, questions), 2),
    }
    payload = {"profile": args.profile, "metrics": metrics, "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = output.with_suffix(".md")
    markdown.write_text(
        "\n".join([
            f"# {args.profile}", "",
            "> Retrieval-only diagnostic30; no answer model or Judge.", "",
            f"- Candidate hit: {metrics['candidate_hit']:.2%}",
            f"- Selected hit: {metrics['selected_hit']:.2%}",
            f"- Context hit: {metrics['context_hit']:.2%}",
            f"- Retrieval calls: {metrics['retrieval_calls']}",
            f"- Jina calls/chars: {metrics['jina_calls']} / {metrics['jina_chars']}",
            f"- Average latency: {metrics['average_latency_ms']:.2f} ms", "",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"JSON: {output}\nMarkdown: {markdown}")


if __name__ == "__main__":
    main()
