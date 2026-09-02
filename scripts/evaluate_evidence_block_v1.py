"""Offline Evidence Block v1 evaluation on the fixed diagnostic30 set.

Both routes start from the same retrieval merged[:120] chunks. The current
selector uses its existing page expansion/selection; the shadow route builds
and selects evidence blocks. No Jina, LLM, or Judge is called.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from document_page_store import DocumentPageStore  # noqa: E402
from evidence_block_v1 import select_evidence_blocks_v1  # noqa: E402
from rag_core_v3 import build_core_v3_evidence  # noqa: E402
from rag_core_v4 import expand_and_rank_pages, retrieve_dense_primary, select_document_first_pages  # noqa: E402
from runtime_profile import RETRIEVAL_DOCUMENT_LOCAL_PROFILE, apply_runtime_profile  # noqa: E402
from scripts.evaluate_evidence_assembly_ab import (  # noqa: E402
    _contains_all,
    _numbers,
    _parse_gold,
    _required_numbers,
    gold_row_hit,
)
from scripts.evaluate_page_selector_v1 import GROUPS, _gold_pages, _load_rows, _page_key  # noqa: E402
from table_store import TableStore  # noqa: E402


DEFAULT_OUTPUT = ROOT / "reports" / "evidence_block_v1_diagnostic30.json"


def _mean(values: list[int | float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 2) if usable else None


def _rate(values: list[bool | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return round(sum(bool(value) for value in usable) / len(usable), 4) if usable else None


def _source_page_keys(blocks: list[dict]) -> set[tuple[str, int]]:
    return {
        _page_key(source)
        for block in blocks
        for source in block.get("source_pages") or []
    }


def _current_selector_route(
    question: str,
    chunks: list[dict],
    query_embedding: list[float],
    gold_pages: set[tuple[str, int]],
    page_store: DocumentPageStore,
    table_store: TableStore,
) -> dict:
    candidate_pages, _ = expand_and_rank_pages(
        question, chunks, query_embedding, neighbor_window=1, page_store=page_store,
    )
    selected, selection_trace = select_document_first_pages(
        candidate_pages, final_page_k=8, global_escape_pages=1,
    )
    selected_keys = [(str(item.get("filename") or ""), int(item.get("page_number") or 0)) for item in selected]
    tables = table_store.get_tables_by_page_keys(selected_keys)
    evidence, context_meta = build_core_v3_evidence(question, selected, tables)
    selected_page_keys = {_page_key(item) for item in selected}
    context_page_keys = {_page_key(item) for item in context_meta.get("answer_context_pages") or []}
    return {
        "selected_hit": bool(gold_pages & selected_page_keys),
        "context_page_hit": bool(gold_pages & context_page_keys),
        "selected_pages": [{"filename": item[0], "page_number": item[1]} for item in sorted(selected_page_keys)],
        "context_pages": [{"filename": item[0], "page_number": item[1]} for item in sorted(context_page_keys)],
        "context": evidence,
        "context_chars": len(evidence),
        "selection_trace": selection_trace,
    }


def _route_metrics(route: dict, gold: list[dict], required_numbers: list[str]) -> dict:
    evidence = route["context"]
    return {
        "gold_row_hit": gold_row_hit(gold, evidence, required_numbers=required_numbers),
        "required_number_hit": _contains_all(required_numbers, evidence, _numbers),
    }


def _group_metrics(records: list[dict]) -> dict:
    current_context = _rate([item["current_selector"]["context_page_hit"] for item in records]) or 0.0
    block_context = _rate([item["evidence_block_v1"]["context_page_hit"] for item in records]) or 0.0
    return {
        "questions": len(records),
        "candidate_hit": _rate([item["candidate_hit"] for item in records]),
        "current_selector": {
            "selected_hit": _rate([item["current_selector"]["selected_hit"] for item in records]),
            "context_page_hit": current_context,
            "gold_row_hit": _rate([item["current_selector"]["gold_row_hit"] for item in records]),
            "required_number_hit": _rate([item["current_selector"]["required_number_hit"] for item in records]),
        },
        "evidence_block_v1": {
            "selected_evidence_block_hit": _rate([item["evidence_block_v1"]["selected_evidence_block_hit"] for item in records]),
            "context_page_hit": block_context,
            "gold_row_hit": _rate([item["evidence_block_v1"]["gold_row_hit"] for item in records]),
            "required_number_hit": _rate([item["evidence_block_v1"]["required_number_hit"] for item in records]),
            "average_selected_blocks": _mean([item["evidence_block_v1"]["selected_block_count"] for item in records]),
            "average_context_chars": _mean([item["evidence_block_v1"]["context_chars"] for item in records]),
        },
        "context_page_delta": round(block_context - current_context, 4),
        "recovered_vs_current": [
            item["financebench_id"] for item in records
            if item["evidence_block_v1"]["context_page_hit"] and not item["current_selector"]["context_page_hit"]
        ],
        "regressed_vs_current": [
            item["financebench_id"] for item in records
            if item["current_selector"]["context_page_hit"] and not item["evidence_block_v1"]["context_page_hit"]
        ],
    }


def summarize(records: list[dict]) -> dict:
    summary = _group_metrics(records)
    summary["groups"] = {
        group: _group_metrics([item for item in records if item["group"] == group])
        for group in GROUPS
    }
    selection = summary["groups"]["selection_loss10"]
    regression = summary["groups"]["correct_regression10"]
    summary["acceptance"] = {
        "selection_loss_context_improved": selection["context_page_delta"] > 0,
        "correct_regression_not_decreased": regression["context_page_delta"] >= 0,
        "passed": selection["context_page_delta"] > 0 and regression["context_page_delta"] >= 0,
    }
    summary["average_block_latency_ms"] = _mean([item["evidence_block_v1"]["latency_ms"] for item in records])
    summary["external_calls"] = {"jina": 0, "llm": 0, "judge": 0}
    return summary


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    current, block = summary["current_selector"], summary["evidence_block_v1"]
    lines = [
        "# Evidence Block v1 shadow — diagnostic30",
        "",
        "- Both routes start from the same retrieval merged[:120] chunks.",
        "- Current route: existing page expansion + document-first selector.",
        "- Shadow route: Table/Text/Merged Chunk evidence blocks.",
        "- External calls: Jina=0, LLM=0, Judge=0.",
        "- Production pipeline: unchanged.",
        "",
        "## Overall",
        "",
        f"- Candidate gold-page hit: {_percent(summary['candidate_hit'])}",
        f"- Selected evidence-block hit: {_percent(block['selected_evidence_block_hit'])}",
        f"- Context page hit current/block: {_percent(current['context_page_hit'])} / {_percent(block['context_page_hit'])}",
        f"- Gold row hit current/block: {_percent(current['gold_row_hit'])} / {_percent(block['gold_row_hit'])}",
        f"- Required number hit current/block: {_percent(current['required_number_hit'])} / {_percent(block['required_number_hit'])}",
        f"- Average selected blocks/context chars: {block['average_selected_blocks']} / {block['average_context_chars']}",
        f"- Average block latency: {summary['average_block_latency_ms']} ms/question",
        f"- Acceptance passed: `{summary['acceptance']['passed']}`",
        "",
        "## Groups",
        "",
        "| Group | Candidate | Context current/block | Row current/block | Number current/block | Recovered/regressed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        item = summary["groups"][group]
        old, new = item["current_selector"], item["evidence_block_v1"]
        lines.append(
            f"| {group} | {_percent(item['candidate_hit'])} | {_percent(old['context_page_hit'])} / {_percent(new['context_page_hit'])} | "
            f"{_percent(old['gold_row_hit'])} / {_percent(new['gold_row_hit'])} | "
            f"{_percent(old['required_number_hit'])} / {_percent(new['required_number_hit'])} | "
            f"{len(item['recovered_vs_current'])} / {len(item['regressed_vs_current'])} |"
        )
    lines.extend(["", "## Per question", ""])
    for index, item in enumerate(payload["records"], 1):
        old, new = item["current_selector"], item["evidence_block_v1"]
        lines.extend([
            f"### {index}. {item['financebench_id']}",
            "",
            f"- Group: `{item['group']}`",
            f"- Question: {item['question']}",
            f"- Candidate hit: {item['candidate_hit']}",
            f"- Context page hit current/block: {old['context_page_hit']} / {new['context_page_hit']}",
            f"- Gold row hit current/block: {old['gold_row_hit']} / {new['gold_row_hit']}",
            f"- Required number hit current/block: {old['required_number_hit']} / {new['required_number_hit']}",
            f"- Added block-source pages vs current: {new['difference_vs_current']['added_pages']}",
            f"- Removed pages vs current: {new['difference_vs_current']['removed_pages']}",
            f"- Selected block IDs: {new['selected_block_ids']}",
            "",
        ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv")
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests" / "fixtures" / "rag_core_v3_diagnostic_ids.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retrieval-k", type=int, default=120)
    parser.add_argument("--max-blocks", type=int, default=12)
    parser.add_argument("--max-context-chars", type=int, default=28000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    apply_runtime_profile(RETRIEVAL_DOCUMENT_LOCAL_PROFILE)
    rows = _load_rows(args.dataset, args.fixture)
    if args.limit:
        rows = rows[: args.limit]
    page_store, table_store = DocumentPageStore(), TableStore()
    records = []
    print(
        f"[setup] questions={len(rows)} retrieval_k={args.retrieval_k} max_blocks={args.max_blocks} "
        f"max_context_chars={args.max_context_chars} jina=false llm=false judge=false",
        flush=True,
    )
    for index, row in enumerate(rows, 1):
        retrieval = retrieve_dense_primary(row["question"], dense_k=args.retrieval_k, bm25_k=30)
        chunks = retrieval["merged"][: args.retrieval_k]
        chunk_page_keys = list(dict.fromkeys(
            (str(chunk.get("filename") or ""), int(chunk.get("page_number") or 0))
            for chunk in chunks if str(chunk.get("filename") or "").strip()
        ))
        pages = page_store.get_pages_by_keys(chunk_page_keys)
        tables = table_store.get_tables_by_page_keys(chunk_page_keys)
        gold_pages = _gold_pages(row)
        gold = _parse_gold(row)
        required_numbers = _required_numbers(row)
        candidate_hit = bool(gold_pages & {_page_key(chunk) for chunk in chunks})
        current = _current_selector_route(
            row["question"], chunks, retrieval["query_embedding"], gold_pages, page_store, table_store,
        )
        current.update(_route_metrics(current, gold, required_numbers))
        started = time.perf_counter()
        selected_blocks, block_context, block_trace = select_evidence_blocks_v1(
            row["question"], chunks, page_metadata=pages, table_metadata=tables,
            max_blocks=args.max_blocks, max_context_chars=args.max_context_chars,
        )
        block_latency = round((time.perf_counter() - started) * 1000, 2)
        block_pages = _source_page_keys(selected_blocks)
        block_route = {
            "selected_evidence_block_hit": bool(gold_pages & block_pages),
            "context_page_hit": bool(gold_pages & block_pages),
            "source_pages": [{"filename": item[0], "page_number": item[1]} for item in sorted(block_pages)],
            "context": block_context,
            "context_chars": len(block_context),
            "selected_block_count": len(selected_blocks),
            "selected_block_ids": [block["block_id"] for block in selected_blocks],
            "latency_ms": block_latency,
            "block_score_trace": block_trace["block_scores"],
        }
        block_route.update(_route_metrics(block_route, gold, required_numbers))
        current_pages = {_page_key(item) for item in current["context_pages"]}
        block_route["difference_vs_current"] = {
            "added_pages": [list(item) for item in sorted(block_pages - current_pages)],
            "removed_pages": [list(item) for item in sorted(current_pages - block_pages)],
            "context_page_hit_changed": current["context_page_hit"] != block_route["context_page_hit"],
            "gold_row_hit_changed": current["gold_row_hit"] != block_route["gold_row_hit"],
            "required_number_hit_changed": current["required_number_hit"] != block_route["required_number_hit"],
        }
        current.pop("context")
        block_route.pop("context")
        records.append({
            "financebench_id": row["financebench_id"],
            "group": row["diagnostic_group"],
            "question": row["question"],
            "gold_pages": [{"filename": item[0], "page_number": item[1]} for item in sorted(gold_pages)],
            "required_numbers": required_numbers,
            "candidate_hit": candidate_hit,
            "candidate_chunk_count": len(chunks),
            "current_selector": current,
            "evidence_block_v1": block_route,
        })
        print(
            f"[{index:02d}/{len(rows)}] {row['financebench_id']} group={row['diagnostic_group']} "
            f"candidate={int(candidate_hit)} context={int(current['context_page_hit'])}->{int(block_route['context_page_hit'])} "
            f"row={int(current['gold_row_hit'])}->{int(block_route['gold_row_hit'])}",
            flush=True,
        )
    payload = {
        "evaluation": "evidence_block_v1_diagnostic30",
        "scope": "same retrieval merged[:120]; current page selector vs shadow blocks; no Jina/LLM/Judge",
        "config": {
            "retrieval_k": args.retrieval_k,
            "max_blocks": args.max_blocks,
            "max_context_chars": args.max_context_chars,
        },
        "summary": summarize(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {args.output}\nMarkdown: {markdown}", flush=True)


if __name__ == "__main__":
    main()
