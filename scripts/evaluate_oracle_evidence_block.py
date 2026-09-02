"""Compare Oracle Evidence Blocks with Evidence Block v1 on diagnostic30.

Only the evidence string differs between routes. Both use the unchanged answer
generator and the same strict FinanceBench judge. LangSmith tracing is disabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
load_dotenv(ROOT / ".env", override=True)

from document_page_store import DocumentPageStore  # noqa: E402
from evidence_block_v1 import select_evidence_blocks_v1  # noqa: E402
from rag_core_v4 import retrieve_dense_primary  # noqa: E402
from runtime_profile import RETRIEVAL_DOCUMENT_LOCAL_PROFILE, apply_runtime_profile  # noqa: E402
from scripts.evaluate_evidence_assembly_ab import (  # noqa: E402
    _contains_all,
    _numbers,
    _parse_gold,
    _periods,
    _required_numbers,
)
from scripts.evaluate_page_selector_v1 import GROUPS, _gold_pages, _load_rows, _page_key  # noqa: E402
from table_store import TableStore  # noqa: E402


DEFAULT_OUTPUT = ROOT / "reports" / "oracle_evidence_block_diagnostic30.jsonl"
ROUTES = ("evidence_block_v1", "oracle_evidence_block")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")
_SPACE_RE = re.compile(r"\s+")


def _filename(value: object) -> str:
    name = str(value or "").strip().replace("\\", "/").rsplit("/", 1)[-1]
    return name if name.casefold().endswith(".pdf") else f"{name}.pdf"


def _parse_oracle_items(row: dict) -> list[dict]:
    try:
        items = json.loads(row.get("evidence") or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    records = []
    for item in items if isinstance(items, list) else [items]:
        if not isinstance(item, dict):
            continue
        try:
            page_number = int(item.get("evidence_page_num") or 0)
        except (TypeError, ValueError):
            continue
        records.append({
            "filename": _filename(item.get("doc_name")),
            "page_number": page_number,
            "evidence_text": str(item.get("evidence_text") or "").strip(),
            "evidence_text_full_page": str(item.get("evidence_text_full_page") or "").strip(),
        })
    return records


def build_oracle_evidence_blocks(row: dict, *, max_context_chars: int = 28000) -> tuple[str, list[dict]]:
    """Build source-preserving oracle blocks using the direct page contract."""
    grouped: dict[tuple[str, int], dict] = {}
    for item in _parse_oracle_items(row):
        key = (item["filename"], item["page_number"])
        group = grouped.setdefault(key, {"snippets": [], "full_page": ""})
        if item["evidence_text"] and item["evidence_text"] not in group["snippets"]:
            group["snippets"].append(item["evidence_text"])
        if len(item["evidence_text_full_page"]) > len(group["full_page"]):
            group["full_page"] = item["evidence_text_full_page"]
    if not grouped:
        return "", []
    separator_chars = 2 * max(0, len(grouped) - 1)
    slot = max(1, (max_context_chars - separator_chars) // len(grouped))
    units, blocks = [], []
    for index, ((filename, page_number), value) in enumerate(grouped.items(), 1):
        snippets = "\n".join(value["snippets"])
        header = (
            f"Source: {filename}, internal page {page_number}\n"
            f"Block ID: oracle:{index}\n[Oracle Evidence Block]\n"
        )
        snippet_section = f"Evidence text:\n{snippets}" if snippets else ""
        remaining = max(0, slot - len(header) - len(snippet_section) - 22)
        full_page = value["full_page"][:remaining]
        full_section = f"\nFull page context:\n{full_page}" if full_page else ""
        content = (header + snippet_section + full_section)[:slot]
        units.append(content)
        blocks.append({
            "block_id": f"oracle:{index}",
            "block_type": "oracle",
            "source_pages": [{"filename": filename, "page_number": page_number}],
            "evidence_text_chars": len(snippets),
            "full_page_chars_used": len(full_page),
            "content_chars": len(content),
        })
    return "\n\n".join(units)[:max_context_chars], blocks


def _normalized(value: str) -> str:
    return _SPACE_RE.sub(" ", str(value or "")).strip().casefold()


def _words(value: str) -> set[str]:
    return {word.casefold() for word in _WORD_RE.findall(str(value or "")) if len(word) > 1}


def answer_evidence_coverage(gold: list[dict], evidence: str) -> dict:
    """Measure how much benchmark evidence text is present in answer context."""
    evidence_normalized = _normalized(evidence)
    evidence_words = _words(evidence)
    evidence_numbers = set(_numbers(evidence, exclude_years=False))
    lines = []
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
        line_numbers = set(_numbers(line, exclude_years=False))
        word_overlap = len(line_words & evidence_words) / max(1, len(line_words))
        if word_overlap >= 0.7 and line_numbers <= evidence_numbers:
            matched += 1
    return {
        "matched_lines": matched,
        "total_lines": len(lines),
        "ratio": round(matched / len(lines), 4) if lines else None,
    }


def _source_pages(blocks: list[dict]) -> set[tuple[str, int]]:
    return {
        _page_key(source)
        for block in blocks
        for source in block.get("source_pages") or []
    }


def _context_metrics(row: dict, context: str, blocks: list[dict]) -> dict:
    gold = _parse_gold(row)
    gold_pages = _gold_pages(row)
    source_pages = _source_pages(blocks)
    required_numbers = _required_numbers(row)
    required_periods = _periods(row.get("question") or "")
    return {
        "answer_evidence_coverage": answer_evidence_coverage(gold, context),
        "required_numbers": required_numbers,
        "required_number_hit": _contains_all(required_numbers, context, _numbers),
        "required_periods": required_periods,
        "required_period_hit": _contains_all(required_periods, context, _periods),
        "gold_page_hit": bool(gold_pages & source_pages),
        "all_gold_pages_hit": bool(gold_pages) and gold_pages <= source_pages,
        "gold_page_coverage": round(len(gold_pages & source_pages) / len(gold_pages), 4) if gold_pages else None,
        "source_pages": [{"filename": item[0], "page_number": item[1]} for item in sorted(source_pages)],
        "context_chars": len(context),
        "block_count": len(blocks),
    }


def _input_tokens(usage: dict) -> int:
    return int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)


def _mean(values: list[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(statistics.fmean(usable), 4) if usable else None


def _rate(values: list[bool | None]) -> float | None:
    usable = [value for value in values if value is not None]
    return round(sum(bool(value) for value in usable) / len(usable), 4) if usable else None


def _route_summary(records: list[dict], route: str) -> dict:
    values = [record["routes"][route] for record in records]
    return {
        "strict_judge": _rate([item["judge_result"].get("score") == 1 for item in values if item.get("judge_result")]),
        "judged": sum(bool(item.get("judge_result")) for item in values),
        "answer_evidence_coverage": _mean([
            item["metrics"]["answer_evidence_coverage"]["ratio"] for item in values
        ]),
        "required_number_hit": _rate([item["metrics"]["required_number_hit"] for item in values]),
        "required_period_hit": _rate([item["metrics"]["required_period_hit"] for item in values]),
        "gold_page_hit": _rate([item["metrics"]["gold_page_hit"] for item in values]),
        "all_gold_pages_hit": _rate([item["metrics"]["all_gold_pages_hit"] for item in values]),
        "average_input_tokens": _mean([item["answer_input_tokens"] for item in values]),
        "average_latency_ms": _mean([item["latency_ms"] for item in values]),
        "errors": sum(bool(item.get("error")) for item in values),
    }


def _summary_for(records: list[dict]) -> dict:
    routes = {route: _route_summary(records, route) for route in ROUTES}
    block, oracle = routes["evidence_block_v1"], routes["oracle_evidence_block"]
    return {
        "questions": len(records),
        **routes,
        "strict_judge_delta": round((oracle["strict_judge"] or 0) - (block["strict_judge"] or 0), 4),
        "oracle_gains": [
            record["financebench_id"] for record in records
            if record["routes"]["oracle_evidence_block"]["judge_result"].get("score") == 1
            and record["routes"]["evidence_block_v1"]["judge_result"].get("score") != 1
        ],
        "oracle_regressions": [
            record["financebench_id"] for record in records
            if record["routes"]["evidence_block_v1"]["judge_result"].get("score") == 1
            and record["routes"]["oracle_evidence_block"]["judge_result"].get("score") != 1
        ],
    }


def summarize(records: list[dict]) -> dict:
    summary = _summary_for(records)
    summary["groups"] = {
        group: _summary_for([record for record in records if record["group"] == group])
        for group in GROUPS
    }
    summary["external_calls"] = {
        "answer_model": len(records) * 2,
        "strict_judge": len(records) * 2,
        "jina": 0,
        "langsmith": 0,
    }
    return summary


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.2f}%"


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    block, oracle = summary["evidence_block_v1"], summary["oracle_evidence_block"]
    lines = [
        "# Oracle Evidence Block vs Evidence Block v1 — diagnostic30",
        "",
        "- Same unchanged answer prompt/model settings for both routes.",
        "- Same DeepSeek-V4-Pro strict Judge for both routes.",
        "- LangSmith=0, Jina=0, production pipeline unchanged.",
        "",
        "## Overall",
        "",
        "| Metric | Evidence Block v1 | Oracle Evidence Block | Delta |",
        "|---|---:|---:|---:|",
        f"| Strict Judge | {_percent(block['strict_judge'])} | {_percent(oracle['strict_judge'])} | {_percent(summary['strict_judge_delta'])} |",
        f"| Answer evidence coverage | {_percent(block['answer_evidence_coverage'])} | {_percent(oracle['answer_evidence_coverage'])} | — |",
        f"| Required number hit | {_percent(block['required_number_hit'])} | {_percent(oracle['required_number_hit'])} | — |",
        f"| Required period hit | {_percent(block['required_period_hit'])} | {_percent(oracle['required_period_hit'])} | — |",
        f"| Gold page hit | {_percent(block['gold_page_hit'])} | {_percent(oracle['gold_page_hit'])} | — |",
        f"| Average answer input tokens | {block['average_input_tokens']} | {oracle['average_input_tokens']} | — |",
        "",
        f"- Oracle gains/regressions: {len(summary['oracle_gains'])} / {len(summary['oracle_regressions'])}",
        "",
        "## Groups",
        "",
        "| Group | Strict v1/oracle | Evidence coverage v1/oracle | Number v1/oracle | Page v1/oracle |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        item = summary["groups"][group]
        old, new = item["evidence_block_v1"], item["oracle_evidence_block"]
        lines.append(
            f"| {group} | {_percent(old['strict_judge'])} / {_percent(new['strict_judge'])} | "
            f"{_percent(old['answer_evidence_coverage'])} / {_percent(new['answer_evidence_coverage'])} | "
            f"{_percent(old['required_number_hit'])} / {_percent(new['required_number_hit'])} | "
            f"{_percent(old['gold_page_hit'])} / {_percent(new['gold_page_hit'])} |"
        )
    lines.extend(["", "## Per question", ""])
    for index, record in enumerate(payload["records"], 1):
        old = record["routes"]["evidence_block_v1"]
        new = record["routes"]["oracle_evidence_block"]
        lines.extend([
            f"### {index}. {record['financebench_id']}",
            "",
            f"- Group: `{record['group']}`",
            f"- Question: {record['question']}",
            f"- Reference: {record['reference_answer']}",
            f"- Strict Judge v1/oracle: {old['judge_result'].get('verdict')} / {new['judge_result'].get('verdict')}",
            f"- Evidence coverage v1/oracle: {old['metrics']['answer_evidence_coverage']['ratio']} / {new['metrics']['answer_evidence_coverage']['ratio']}",
            f"- Required number v1/oracle: {old['metrics']['required_number_hit']} / {new['metrics']['required_number_hit']}",
            f"- Required period v1/oracle: {old['metrics']['required_period_hit']} / {new['metrics']['required_period_hit']}",
            f"- Gold page v1/oracle: {old['metrics']['gold_page_hit']} / {new['metrics']['gold_page_hit']}",
            f"- Evidence Block answer: {old['answer']}",
            f"- Oracle answer: {new['answer']}",
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
    parser.add_argument("--request-interval-seconds", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.update({
        "LANGCHAIN_TRACING_V2": "false",
        "LANGSMITH_TRACING": "false",
        "ANSWER_THINKING_MODE": "disabled",
        "ANSWER_MAX_COMPLETION_TOKENS": "512",
        "ANSWER_TEMPERATURE": "0",
    })
    apply_runtime_profile(RETRIEVAL_DOCUMENT_LOCAL_PROFILE)
    from answer_generator import generate_answer
    from financebench_judge_common import JUDGE_PROMPT, _judge_model, _judge_with_retry

    judge_model = _judge_model()
    rows = _load_rows(args.dataset, args.fixture)
    if args.limit:
        rows = rows[: args.limit]
    completed = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                completed.add(json.loads(line)["financebench_id"])
            except (json.JSONDecodeError, KeyError):
                continue
        rows = [row for row in rows if row["financebench_id"] not in completed]
    page_store, table_store = DocumentPageStore(), TableStore()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[setup] questions={len(rows)} routes=2 retrieval_k={args.retrieval_k} "
        "answer=true strict_judge=true langsmith=false jina=false",
        flush=True,
    )
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            retrieval = retrieve_dense_primary(row["question"], dense_k=args.retrieval_k, bm25_k=30)
            chunks = retrieval["merged"][: args.retrieval_k]
            page_keys = list(dict.fromkeys(
                (str(chunk.get("filename") or ""), int(chunk.get("page_number") or 0))
                for chunk in chunks if str(chunk.get("filename") or "").strip()
            ))
            pages = page_store.get_pages_by_keys(page_keys)
            tables = table_store.get_tables_by_page_keys(page_keys)
            selected, block_context, block_trace = select_evidence_blocks_v1(
                row["question"], chunks, page_metadata=pages, table_metadata=tables,
                max_blocks=args.max_blocks, max_context_chars=args.max_context_chars,
            )
            oracle_context, oracle_blocks = build_oracle_evidence_blocks(
                row, max_context_chars=args.max_context_chars,
            )
            route_inputs = {
                "evidence_block_v1": (block_context, selected, block_trace),
                "oracle_evidence_block": (oracle_context, oracle_blocks, {}),
            }
            route_results = {}
            for route_index, route in enumerate(ROUTES):
                context, blocks, trace = route_inputs[route]
                started = time.perf_counter()
                answer, usage, error = "", {}, ""
                judge_result = {}
                try:
                    answer, usage = generate_answer(row["question"], context, [], "")
                    judge_result = _judge_with_retry(
                        judge_model,
                        JUDGE_PROMPT.format(
                            question=row["question"], reference=row.get("answer") or "", answer=answer,
                        ),
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    judge_result = {"score": 0, "verdict": "error", "reason": error}
                route_results[route] = {
                    "answer": answer,
                    "judge_result": judge_result,
                    "metrics": _context_metrics(row, context, blocks),
                    "answer_input_tokens": _input_tokens(usage),
                    "usage": usage,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "error": error,
                    "trace": trace,
                }
                if args.request_interval_seconds > 0 and (route_index < len(ROUTES) - 1 or index < len(rows)):
                    time.sleep(args.request_interval_seconds)
            record = {
                "financebench_id": row["financebench_id"],
                "group": row["diagnostic_group"],
                "question": row["question"],
                "reference_answer": row.get("answer") or "",
                "routes": route_results,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index:02d}/{len(rows)}] {row['financebench_id']} "
                f"strict={route_results['evidence_block_v1']['judge_result'].get('score', 0)}"
                f"->{route_results['oracle_evidence_block']['judge_result'].get('score', 0)} "
                f"coverage={route_results['evidence_block_v1']['metrics']['answer_evidence_coverage']['ratio']}"
                f"->{route_results['oracle_evidence_block']['metrics']['answer_evidence_coverage']['ratio']}",
                flush=True,
            )

    records = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = {
        "evaluation": "oracle_evidence_block_diagnostic30",
        "scope": "fixed diagnostic30; same answer prompt; strict judge; LangSmith/Jina disabled",
        "summary": summarize(records),
        "records": records,
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = args.output.with_suffix(".md")
    markdown.write_text(render_markdown(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"JSONL: {args.output}\nSummary: {summary_path}\nMarkdown: {markdown}", flush=True)


if __name__ == "__main__":
    main()
