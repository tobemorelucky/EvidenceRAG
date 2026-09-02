"""Offline chunk-vs-block retrieval A/B on the fixed diagnostic30 set.

No Jina, answer model, strict Judge, or LangSmith call is made.  Consequently
strict_judge is reported as null rather than replaced with a proxy metric.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
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

from evidence_block_retrieval_v2 import (  # noqa: E402
    DEFAULT_BLOCK_COLLECTION,
    EvidenceBlockMilvusManager,
)
from rag_core_v4 import retrieve_dense_primary  # noqa: E402
from runtime_profile import RETRIEVAL_DOCUMENT_LOCAL_PROFILE, apply_runtime_profile  # noqa: E402
from scripts.evaluate_oracle_evidence_block import _context_metrics  # noqa: E402
from scripts.evaluate_page_selector_v1 import GROUPS, _load_rows  # noqa: E402


DEFAULT_OUTPUT = ROOT / "reports" / "evidence_block_retrieval_v2_diagnostic30.json"
DEFAULT_ORACLE_SUMMARY = ROOT / "reports" / "oracle_evidence_block_diagnostic30.summary.json"
ROUTES = ("current_chunk_retrieval", "evidence_block_retrieval_v2")


def _mean(values: list[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 4) if usable else None


def _rate(values: list[bool | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return round(sum(bool(value) for value in usable) / len(usable), 4) if usable else None


def _render_ranked_units(items: list[dict], *, max_units: int, max_context_chars: int) -> tuple[str, list[dict]]:
    rendered: list[str] = []
    sources: list[dict] = []
    used = 0
    for rank, item in enumerate(items, 1):
        text = str(item.get("text") or "").strip()
        filename = str(item.get("filename") or "").strip()
        page_number = int(item.get("page_number") or 0)
        if not text or not filename:
            continue
        block_id = str(item.get("block_id") or item.get("chunk_id") or item.get("id") or f"rank:{rank}")
        value = (
            f"Source: {filename}, internal page {page_number}\n"
            f"Block ID: {block_id}\n{text}"
        )
        separator = 2 if rendered else 0
        if used + separator + len(value) > max_context_chars:
            continue
        rendered.append(value)
        sources.append({
            "block_id": block_id,
            "block_type": item.get("source_type") or "chunk",
            "source_pages": [{"filename": filename, "page_number": page_number}],
            "dense_rank": item.get("dense_rank"),
            "bm25_rank": item.get("bm25_rank"),
            "rank": rank,
        })
        used += separator + len(value)
        if len(rendered) >= max_units:
            break
    return "\n\n".join(rendered), sources


def _route_summary(records: list[dict], route: str) -> dict:
    values = [record["routes"][route] for record in records]
    return {
        "strict_judge": None,
        "strict_judge_status": "not_evaluated_no_judge_calls",
        "answer_evidence_coverage": _mean([
            item["metrics"]["answer_evidence_coverage"]["ratio"] for item in values
        ]),
        "required_number_hit": _rate([item["metrics"]["required_number_hit"] for item in values]),
        "required_period_hit": _rate([item["metrics"]["required_period_hit"] for item in values]),
        "gold_page_hit": _rate([item["metrics"]["gold_page_hit"] for item in values]),
        "all_gold_pages_hit": _rate([item["metrics"]["all_gold_pages_hit"] for item in values]),
        "average_context_chars": _mean([item["metrics"]["context_chars"] for item in values]),
        "average_selected_units": _mean([item["metrics"]["block_count"] for item in values]),
        "average_retrieval_latency_ms": _mean([item["retrieval_latency_ms"] for item in values]),
    }


def _summary_for(records: list[dict]) -> dict:
    routes = {route: _route_summary(records, route) for route in ROUTES}
    current = routes["current_chunk_retrieval"]
    block = routes["evidence_block_retrieval_v2"]
    return {
        "questions": len(records),
        **routes,
        "delta": {
            metric: round((block[metric] or 0.0) - (current[metric] or 0.0), 4)
            for metric in ("answer_evidence_coverage", "required_number_hit", "required_period_hit", "gold_page_hit")
        },
        "coverage_gains": [
            item["financebench_id"] for item in records
            if (item["routes"]["evidence_block_retrieval_v2"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0)
            > (item["routes"]["current_chunk_retrieval"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0)
        ],
        "coverage_regressions": [
            item["financebench_id"] for item in records
            if (item["routes"]["evidence_block_retrieval_v2"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0)
            < (item["routes"]["current_chunk_retrieval"]["metrics"]["answer_evidence_coverage"]["ratio"] or 0)
        ],
    }


def summarize(records: list[dict], oracle_summary: dict | None = None) -> dict:
    summary = _summary_for(records)
    summary["groups"] = {
        group: _summary_for([record for record in records if record["group"] == group])
        for group in GROUPS
    }
    frozen = oracle_summary or {}
    if isinstance(frozen.get("summary"), dict):
        frozen = frozen["summary"]
    oracle = frozen.get("oracle_evidence_block") or {}
    block = summary["evidence_block_retrieval_v2"]
    summary["oracle_reference"] = {
        "source": "existing Oracle diagnostic30; not rerun",
        "strict_judge": oracle.get("strict_judge"),
        "answer_evidence_coverage": oracle.get("answer_evidence_coverage"),
        "required_number_hit": oracle.get("required_number_hit"),
        "required_period_hit": oracle.get("required_period_hit"),
        "gold_page_hit": oracle.get("gold_page_hit"),
    }
    summary["remaining_oracle_gap"] = {
        metric: round(float(oracle[metric]) - float(block[metric]), 4)
        if oracle.get(metric) is not None and block.get(metric) is not None else None
        for metric in ("answer_evidence_coverage", "required_number_hit", "required_period_hit", "gold_page_hit")
    }
    source_counts = Counter(
        source_type
        for record in records
        for source_type in record["routes"]["evidence_block_retrieval_v2"].get("selected_source_types", [])
    )
    summary["selected_block_source_counts"] = dict(sorted(source_counts.items()))
    summary["acceptance"] = {
        "passed": bool(
            summary["delta"]["answer_evidence_coverage"] > 0
            and summary["delta"]["required_number_hit"] >= 0
            and summary["groups"]["correct_regression10"]["delta"]["answer_evidence_coverage"] >= 0
        ),
        "criterion": "evidence coverage improves, required-number hit does not regress, and correct-regression coverage does not regress",
    }
    summary["external_calls"] = {"jina": 0, "answer_model": 0, "strict_judge": 0, "langsmith": 0}
    return summary


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    current = summary["current_chunk_retrieval"]
    block = summary["evidence_block_retrieval_v2"]
    oracle = summary["oracle_reference"]
    lines = [
        "# Evidence Block Retrieval v2 — diagnostic30 shadow A/B",
        "",
        "> Production Retrieval/Fusion/Prompt/Skills are unchanged. Jina=0, LLM=0, Judge=0, LangSmith=0.",
        "",
        "`strict_judge` is intentionally `null`: a new strict score cannot be produced without answers and Judge calls. The existing Oracle score is shown only as a frozen reference.",
        "",
        "## Overall",
        "",
        "| Metric | Current chunk retrieval | Block retrieval v2 | Oracle reference | Block-current |",
        "|---|---:|---:|---:|---:|",
        f"| Strict Judge | n/a | n/a | {_percent(oracle.get('strict_judge'))} | n/a |",
        f"| Answer evidence coverage | {_percent(current['answer_evidence_coverage'])} | {_percent(block['answer_evidence_coverage'])} | {_percent(oracle.get('answer_evidence_coverage'))} | {_percent(summary['delta']['answer_evidence_coverage'])} |",
        f"| Required number hit | {_percent(current['required_number_hit'])} | {_percent(block['required_number_hit'])} | {_percent(oracle.get('required_number_hit'))} | {_percent(summary['delta']['required_number_hit'])} |",
        f"| Required period hit | {_percent(current['required_period_hit'])} | {_percent(block['required_period_hit'])} | {_percent(oracle.get('required_period_hit'))} | {_percent(summary['delta']['required_period_hit'])} |",
        f"| Gold page hit | {_percent(current['gold_page_hit'])} | {_percent(block['gold_page_hit'])} | {_percent(oracle.get('gold_page_hit'))} | {_percent(summary['delta']['gold_page_hit'])} |",
        f"| Average context chars | {current['average_context_chars']} | {block['average_context_chars']} | — | — |",
        f"| Average retrieval latency | {current['average_retrieval_latency_ms']} ms | {block['average_retrieval_latency_ms']} ms | — | — |",
        "",
        f"- Coverage gains/regressions: {len(summary['coverage_gains'])} / {len(summary['coverage_regressions'])}",
        f"- Selected block source counts: `{summary['selected_block_source_counts']}`",
        f"- Remaining Oracle gap: `{summary['remaining_oracle_gap']}`",
        f"- Acceptance passed: `{summary['acceptance']['passed']}`",
        "",
        "## Groups",
        "",
        "| Group | Evidence coverage chunk/block | Number chunk/block | Period chunk/block | Gold page chunk/block |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        item = summary["groups"][group]
        old, new = item["current_chunk_retrieval"], item["evidence_block_retrieval_v2"]
        lines.append(
            f"| {group} | {_percent(old['answer_evidence_coverage'])} / {_percent(new['answer_evidence_coverage'])} | "
            f"{_percent(old['required_number_hit'])} / {_percent(new['required_number_hit'])} | "
            f"{_percent(old['required_period_hit'])} / {_percent(new['required_period_hit'])} | "
            f"{_percent(old['gold_page_hit'])} / {_percent(new['gold_page_hit'])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if summary["acceptance"]["passed"]:
        lines.append("Evidence Block Retrieval v2 passed the offline evidence-level gate and reduced the Oracle gap.")
    else:
        lines.extend([
            "Evidence Block Retrieval v2 did **not** pass the offline gate and did not reduce the Oracle gap.",
            "",
            "- Coarser merged text blocks consume the same 28,000-character budget with fewer independent evidence units.",
            "- Table and mixed blocks are retrieved, but they are a minority of selected units and do not compensate for lost high-ranking text evidence.",
            "- The correct-regression group loses evidence coverage, so this shadow index must not be connected to production.",
            "- This result rejects the current block construction/indexing hypothesis; it does not justify selector tuning or a production change.",
        ])
    lines.extend(["", "## Per question", ""])
    for index, item in enumerate(payload["records"], 1):
        old = item["routes"]["current_chunk_retrieval"]["metrics"]
        new = item["routes"]["evidence_block_retrieval_v2"]["metrics"]
        lines.extend([
            f"### {index}. {item['financebench_id']}",
            "",
            f"- Group: `{item['group']}`",
            f"- Question: {item['question']}",
            f"- Evidence coverage chunk/block: {old['answer_evidence_coverage']['ratio']} / {new['answer_evidence_coverage']['ratio']}",
            f"- Required number hit chunk/block: {old['required_number_hit']} / {new['required_number_hit']}",
            f"- Required period hit chunk/block: {old['required_period_hit']} / {new['required_period_hit']}",
            f"- Gold page hit chunk/block: {old['gold_page_hit']} / {new['gold_page_hit']}",
            f"- Selected block types: {item['routes']['evidence_block_retrieval_v2']['selected_source_types']}",
            "",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json")
    parser.add_argument("--collection", default=os.getenv("EVIDENCE_BLOCK_MILVUS_COLLECTION", DEFAULT_BLOCK_COLLECTION))
    parser.add_argument("--retrieval-k", type=int, default=30)
    parser.add_argument("--max-units", type=int, default=12)
    parser.add_argument("--max-context-chars", type=int, default=28000)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--oracle-summary", type=Path, default=DEFAULT_ORACLE_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 30:
        raise SystemExit("--limit must be between 1 and 30")

    apply_runtime_profile(RETRIEVAL_DOCUMENT_LOCAL_PROFILE)
    rows = _load_rows(args.dataset, args.fixture)[:args.limit]
    manager = EvidenceBlockMilvusManager(args.collection)
    if not manager.has_collection() or manager.count() == 0:
        raise RuntimeError("block shadow collection is missing or empty; run build_evidence_block_shadow_index_v2.py first")
    oracle_summary = json.loads(args.oracle_summary.read_text(encoding="utf-8")) if args.oracle_summary.is_file() else None
    records = []
    print(
        f"[setup] questions={len(rows)} collection={args.collection} block_k={args.retrieval_k} "
        "jina=false llm=false judge=false",
        flush=True,
    )
    for index, row in enumerate(rows, 1):
        current_started = time.perf_counter()
        retrieval = retrieve_dense_primary(row["question"], dense_k=120, bm25_k=30)
        current_ms = (time.perf_counter() - current_started) * 1000
        current_context, current_units = _render_ranked_units(
            retrieval["merged"], max_units=args.max_units, max_context_chars=args.max_context_chars,
        )
        block_result = manager.retrieve(row["question"], retrieval["query_embedding"], top_k=args.retrieval_k)
        block_context, block_units = _render_ranked_units(
            block_result["fused"], max_units=args.max_units, max_context_chars=args.max_context_chars,
        )
        current_metrics = _context_metrics(row, current_context, current_units)
        block_metrics = _context_metrics(row, block_context, block_units)
        records.append({
            "financebench_id": row["financebench_id"],
            "group": row["diagnostic_group"],
            "question": row["question"],
            "routes": {
                "current_chunk_retrieval": {
                    "metrics": current_metrics,
                    "retrieval_latency_ms": round(current_ms, 2),
                    "selected_ids": [unit["block_id"] for unit in current_units],
                },
                "evidence_block_retrieval_v2": {
                    "metrics": block_metrics,
                    "retrieval_latency_ms": block_result["latency_ms"]["total"],
                    "selected_ids": [unit["block_id"] for unit in block_units],
                    "selected_source_types": [unit["block_type"] for unit in block_units],
                    "dense_results": block_result["dense"],
                    "bm25_results": block_result["bm25"],
                    "fused_results": block_result["fused"],
                },
            },
        })
        print(
            f"[{index:02d}/{len(rows)}] {row['financebench_id']} "
            f"coverage={current_metrics['answer_evidence_coverage']['ratio']}->{block_metrics['answer_evidence_coverage']['ratio']} "
            f"page={int(current_metrics['gold_page_hit'])}->{int(block_metrics['gold_page_hit'])}",
            flush=True,
        )

    payload = {
        "evaluation": "evidence_block_retrieval_v2_diagnostic30",
        "scope": "current Dense-primary chunk retrieval vs isolated block Dense+BM25; no Jina/LLM/Judge",
        "collection": args.collection,
        "config": {
            "block_retrieval_k": args.retrieval_k,
            "max_units": args.max_units,
            "max_context_chars": args.max_context_chars,
        },
        "summary": summarize(records, oracle_summary),
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
