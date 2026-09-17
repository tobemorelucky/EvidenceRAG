"""Shared runner for the pre-Jina Top120 answer-model comparison experiment.

This module is deliberately isolated from retrieval and reranking code. It only
reads the frozen final100 retrieval snapshot and sends every stored RRF chunk,
without truncation, to the answer model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence, TypeVar

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
SNAPSHOT = ROOT / "reports" / "final100" / "retrieval_snapshot.json"
PRO_MODEL = "deepseek-v4-pro-ga-260813"
FLASH_MODEL = "deepseek-v4-flash-ga-260731"
ANSWER_MODELS = {
    "pro": PRO_MODEL,
    "flash": FLASH_MODEL,
}
EXPECTED_CHUNKS = 120
T = TypeVar("T")

ANSWER_SYSTEM_PROMPT = """You are EvidenceRAG, a retrieval-augmented financial assistant.
Answer only from the Evidence supplied for the current question. Do not add factual claims from memory.
Cite important factual claims using [source: filename, page N]. Never invent a source or page.
Do not mix clearly different companies, reporting periods, currencies, units, or scopes.
If the Evidence is insufficient, state exactly what is missing instead of guessing.
Answer the question directly and concisely, following requested calculation and rounding requirements."""


def _clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            _clean(item.get("text")) if isinstance(item, dict) else _clean(item)
            for item in content
        )
    return _clean(content)


def normalize_usage(usage: Any) -> dict[str, int]:
    value = usage if isinstance(usage, dict) else {}
    input_tokens = int(value.get("input_tokens") or value.get("prompt_tokens") or 0)
    output_tokens = int(value.get("output_tokens") or value.get("completion_tokens") or 0)
    total_tokens = int(value.get("total_tokens") or input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_dataset(path: Path = DATASET) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [_clean(row.get("financebench_id")) for row in rows]
    if len(rows) != 100 or len(set(ids)) != 100 or any(not item for item in ids):
        raise ValueError("Expected exactly 100 unique FinanceBench rows")
    return rows


def load_snapshot(path: Path = SNAPSHOT) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or len(records) != 100:
        raise ValueError("Frozen retrieval snapshot must contain exactly 100 records")
    ids: set[str] = set()
    for record in records:
        question_id = _clean(record.get("question_id"))
        chunks = record.get("chunks")
        if not question_id or question_id in ids:
            raise ValueError("Frozen retrieval snapshot contains a missing or duplicate question ID")
        if record.get("status") != "ok" or not isinstance(chunks, list):
            raise ValueError(f"Frozen retrieval record is unavailable: {question_id}")
        if len(chunks) != EXPECTED_CHUNKS:
            raise ValueError(
                f"Expected {EXPECTED_CHUNKS} pre-Jina chunks for {question_id}, found {len(chunks)}"
            )
        ids.add(question_id)
    return records


def select_half(items: Sequence[T], split: str) -> list[T]:
    if len(items) != 100:
        raise ValueError("The first50/last50 split requires exactly 100 ordered items")
    if split == "first50":
        return list(items[:50])
    if split == "last50":
        return list(items[50:])
    raise ValueError(f"Unknown split: {split}")


def build_all_chunks_context(chunks: Sequence[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Serialize all frozen chunks in RRF order without truncating their text."""

    if len(chunks) != EXPECTED_CHUNKS:
        raise ValueError(f"Expected {EXPECTED_CHUNKS} chunks, found {len(chunks)}")
    sections: list[str] = []
    raw_chars = 0
    chunk_ids: list[str] = []
    for position, chunk in enumerate(chunks, 1):
        text = str(chunk.get("text") or "")
        raw_chars += len(text)
        chunk_id = _clean(chunk.get("chunk_id") or chunk.get("id"))
        chunk_ids.append(chunk_id)
        document = _clean(chunk.get("filename")) or "unknown"
        page = chunk.get("page_number")
        rrf_rank = chunk.get("rrf_rank") or position
        sections.append(
            f"[RRF Chunk {position}/{EXPECTED_CHUNKS}]\n"
            f"Source: {document}\n"
            f"Page: {page}\n"
            f"Chunk ID: {chunk_id}\n"
            f"RRF Rank: {rrf_rank}\n"
            f"{text}"
        )
    context = "\n\n---\n\n".join(sections)
    return context, {
        "chunk_count": len(chunks),
        "raw_fragment_chars": raw_chars,
        "assembled_context_chars": len(context),
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        "chunk_ids": chunk_ids,
        "truncated": False,
    }


def create_answer_model(model_name: str = PRO_MODEL) -> Any:
    api_key = os.getenv("ARK_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("ARK_API_KEY and BASE_URL are required")
    return init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        max_completion_tokens=1024,
        timeout=float(os.getenv("PRE_JINA_PRO_ANSWER_TIMEOUT_SECONDS", "180")),
        max_retries=int(os.getenv("PRE_JINA_PRO_ANSWER_MAX_RETRIES", "1")),
        stream_usage=True,
        extra_body={"thinking": {"type": "disabled"}},
    )


def invoke_answer(model: Any, question: str, context: str) -> tuple[str, dict[str, int]]:
    response = model.invoke(
        [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=f"Question:\n{question}\n\nEvidence:\n{context}"),
        ]
    )
    answer = _content_text(getattr(response, "content", response)).strip()
    if not answer:
        raise ValueError("Answer model returned empty content")
    return answer, normalize_usage(getattr(response, "usage_metadata", {}) or {})


def create_judge_model() -> Any:
    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("ARK_API_KEY")
    base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("JUDGE/ARK API key and base URL are required")
    return init_chat_model(
        model=PRO_MODEL,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        max_completion_tokens=512,
        timeout=float(os.getenv("PRE_JINA_PRO_JUDGE_TIMEOUT_SECONDS", "90")),
        max_retries=int(os.getenv("PRE_JINA_PRO_JUDGE_MAX_RETRIES", "1")),
        extra_body={"thinking": {"type": "disabled"}},
    )


def invoke_judge(model: Any, question: str, reference: str, answer: str) -> tuple[dict[str, Any], dict[str, int]]:
    from scripts.financebench_judge_common import JUDGE_PROMPT, _content_text as judge_text, _parse_verdict

    response = model.invoke(JUDGE_PROMPT.format(question=question, reference=reference, answer=answer))
    verdict = _parse_verdict(judge_text(getattr(response, "content", response)))
    usage = normalize_usage(getattr(response, "usage_metadata", {}) or {})
    if verdict.get("verdict") == "invalid_judge_output":
        retry = model.invoke(
            "Return exactly one JSON object with score, verdict, and reason.\n\n"
            + JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
        )
        verdict = _parse_verdict(judge_text(getattr(retry, "content", retry)))
        verdict["retried"] = True
        retry_usage = normalize_usage(getattr(retry, "usage_metadata", {}) or {})
        usage = {key: usage[key] + retry_usage[key] for key in usage}
    return verdict, usage


def prepare_output(output_dir: Path, resume: bool) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume:
        return None
    names = ("answers.jsonl", "judge_results.jsonl", "summary.json", "report.md", "state.json")
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if not existing:
        return None
    archive = output_dir / "archive" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    archive.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), str(archive / path.name))
    return archive


def _ordered(rows: Sequence[dict[str, str]], saved: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [saved[row["financebench_id"]] for row in rows if row["financebench_id"] in saved]


def write_state(output: Path, split: str, stage: str, records: Sequence[dict[str, Any]]) -> None:
    state_path = output / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state.update(
        {
            "schema": "pre_jina_all_chunks_pro_v1",
            "split": split,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    state[stage] = {
        "completed": len(records),
        "success": sum(record.get("status") == "ok" for record in records),
        "failed": sum(record.get("status") == "error" for record in records),
    }
    atomic_json(state_path, state)


def run_answers(
    rows: Sequence[dict[str, str]],
    snapshot_by_id: dict[str, dict[str, Any]],
    output: Path,
    split: str,
    max_new: int = 0,
    answer_model: str = PRO_MODEL,
) -> list[dict[str, Any]]:
    path = output / "answers.jsonl"
    saved = {record["financebench_id"]: record for record in load_jsonl(path)}
    pending = [row for row in rows if saved.get(row["financebench_id"], {}).get("status") != "ok"]
    pending = pending[:max_new] if max_new > 0 else pending
    pending_ids = {row["financebench_id"] for row in pending}
    model = create_answer_model(answer_model) if pending_ids else None
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        if saved.get(question_id, {}).get("status") == "ok":
            print(f"[answer {index:02d}/50] {question_id}: cached", flush=True)
            continue
        if question_id not in pending_ids:
            continue
        source = snapshot_by_id[question_id]
        context, context_trace = build_all_chunks_context(source["chunks"])
        started = time.perf_counter()
        try:
            answer, usage = invoke_answer(model, row["question"], context)
            record = {
                "financebench_id": question_id,
                "question": row["question"],
                "reference_answer": row["answer"],
                "answer": answer,
                "answer_model": answer_model,
                "usage": usage,
                "latency_seconds": round(time.perf_counter() - started, 4),
                "context_trace": context_trace,
                "status": "ok",
            }
        except Exception as exc:
            record = {
                "financebench_id": question_id,
                "question": row["question"],
                "reference_answer": row["answer"],
                "answer": "",
                "answer_model": answer_model,
                "usage": normalize_usage({}),
                "latency_seconds": round(time.perf_counter() - started, 4),
                "context_trace": context_trace,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        saved[question_id] = record
        records = _ordered(rows, saved)
        atomic_jsonl(path, records)
        if len(records) % 10 == 0 or record["status"] == "error":
            write_state(output, split, "answer", records)
        print(f"[answer {index:02d}/50] {question_id}: {record['status']}", flush=True)
    records = _ordered(rows, saved)
    write_state(output, split, "answer", records)
    return records


def run_judges(
    rows: Sequence[dict[str, str]], answers: Sequence[dict[str, Any]], output: Path, split: str
) -> list[dict[str, Any]]:
    path = output / "judge_results.jsonl"
    saved = {record["financebench_id"]: record for record in load_jsonl(path)}
    answer_by_id = {record["financebench_id"]: record for record in answers}
    pending = [
        row for row in rows
        if answer_by_id.get(row["financebench_id"], {}).get("status") == "ok"
        and saved.get(row["financebench_id"], {}).get("status") != "ok"
    ]
    model = create_judge_model() if pending else None
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        if saved.get(question_id, {}).get("status") == "ok":
            print(f"[judge  {index:02d}/50] {question_id}: cached", flush=True)
            continue
        answer_record = answer_by_id.get(question_id, {})
        if not answer_record:
            continue
        if answer_record.get("status") != "ok":
            record = {
                "financebench_id": question_id,
                "status": "skipped",
                "judge_model": PRO_MODEL,
                "judge": {"score": 0, "verdict": "not_judged", "reason": "answer_failed"},
                "usage": normalize_usage({}),
                "latency_seconds": 0.0,
            }
        else:
            started = time.perf_counter()
            try:
                verdict, usage = invoke_judge(
                    model, row["question"], row["answer"], answer_record["answer"]
                )
                record = {
                    "financebench_id": question_id,
                    "status": "ok",
                    "judge_model": PRO_MODEL,
                    "judge": verdict,
                    "usage": usage,
                    "latency_seconds": round(time.perf_counter() - started, 4),
                }
            except Exception as exc:
                record = {
                    "financebench_id": question_id,
                    "status": "error",
                    "judge_model": PRO_MODEL,
                    "judge": {"score": 0, "verdict": "not_judged", "reason": "judge_failed"},
                    "usage": normalize_usage({}),
                    "latency_seconds": round(time.perf_counter() - started, 4),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        saved[question_id] = record
        records = _ordered(rows, saved)
        atomic_jsonl(path, records)
        if len(records) % 10 == 0 or record["status"] == "error":
            write_state(output, split, "judge", records)
        print(f"[judge  {index:02d}/50] {question_id}: {record['judge']['verdict']}", flush=True)
    records = _ordered(rows, saved)
    write_state(output, split, "judge", records)
    return records


def build_summary(
    split: str,
    answers: Sequence[dict[str, Any]],
    judges: Sequence[dict[str, Any]],
    answer_model: str = PRO_MODEL,
) -> dict[str, Any]:
    successful_answers = [record for record in answers if record.get("status") == "ok"]
    successful_judges = [record for record in judges if record.get("status") == "ok"]
    correct = sum(int(record.get("judge", {}).get("score") or 0) for record in successful_judges)

    def usage_total(records: Sequence[dict[str, Any]]) -> dict[str, int]:
        return {
            key: sum(int(record.get("usage", {}).get(key) or 0) for record in records)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }

    return {
        "schema": "pre_jina_all_chunks_pro_v1",
        "split": split,
        "range": "1-50" if split == "first50" else "51-100",
        "total": len(answers),
        "answer_success": len(successful_answers),
        "judge_success": len(successful_judges),
        "strict_correct": correct,
        "strict_accuracy": round(correct / len(answers), 4) if answers else None,
        "judged_accuracy": round(correct / len(successful_judges), 4) if successful_judges else None,
        "answer_model": answer_model,
        "judge_model": PRO_MODEL,
        "pre_jina_chunks_per_question": EXPECTED_CHUNKS,
        "context_truncation": False,
        "average_context_chars": round(
            fmean(record["context_trace"]["assembled_context_chars"] for record in answers), 2
        ) if answers else 0,
        "answer_usage": usage_total(successful_answers),
        "judge_usage": usage_total(successful_judges),
        "average_answer_latency_seconds": round(
            fmean(float(record.get("latency_seconds") or 0) for record in successful_answers), 4
        ) if successful_answers else 0,
        "average_judge_latency_seconds": round(
            fmean(float(record.get("latency_seconds") or 0) for record in successful_judges), 4
        ) if successful_judges else 0,
        "upstream_calls": {"retrieval": 0, "query_rewrite": 0, "jina": 0},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_report(
    output: Path,
    summary: dict[str, Any],
    answers: Sequence[dict[str, Any]],
    judges: Sequence[dict[str, Any]],
) -> None:
    judge_by_id = {record["financebench_id"]: record for record in judges}
    lines = [
        f"# Pre-Jina Top120 → Answer Model Comparison ({summary['range']})",
        "",
        f"- Strict Judge: **{summary['strict_correct']}/{summary['total']} "
        f"({(summary['strict_accuracy'] or 0):.2%})**",
        f"- Answer model: `{summary['answer_model']}`",
        f"- Judge model: `{summary['judge_model']}`",
        f"- Evidence: all {EXPECTED_CHUNKS} frozen RRF chunks per question, no truncation",
        f"- Average context: {summary['average_context_chars']:,.0f} characters",
        "",
    ]
    for record in answers:
        judge = judge_by_id.get(record["financebench_id"], {})
        verdict = judge.get("judge", {})
        lines.extend(
            [
                f"## {record['financebench_id']}",
                "",
                "### Question",
                "",
                record["question"],
                "",
                "### Reference Answer",
                "",
                record["reference_answer"],
                "",
                "### Model Answer",
                "",
                record.get("answer") or f"ERROR: {record.get('error', 'unknown')}",
                "",
                "### Evaluation",
                "",
                f"- Verdict: {verdict.get('verdict', judge.get('status', 'missing'))}",
                f"- Reason: {verdict.get('reason', '')}",
                f"- Context chars: {record['context_trace']['assembled_context_chars']:,}",
                f"- Answer input/output tokens: {record.get('usage', {}).get('input_tokens', 0):,} / "
                f"{record.get('usage', {}).get('output_tokens', 0):,}",
                "",
            ]
        )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(
    split: str | None,
    default_output: Path | None,
    answer_model: str = PRO_MODEL,
) -> argparse.Namespace:
    default_answer_model = next(
        (alias for alias, model_id in ANSWER_MODELS.items() if model_id == answer_model),
        "pro",
    )
    split_label = split or "the selected split"
    parser = argparse.ArgumentParser(
        description=(
            f"Run {split_label} using all frozen pre-Jina RRF Top120 chunks. "
            "The answer model is selectable; the Judge always uses DeepSeek-V4-Pro."
        )
    )
    if split is None:
        parser.add_argument(
            "--split",
            choices=("first50", "last50"),
            required=True,
            help="Run FinanceBench questions 1-50 or 51-100.",
        )
    parser.add_argument(
        "--answer-model",
        choices=tuple(ANSWER_MODELS),
        default=default_answer_model,
        help="Answer model only: flash or pro. The Judge remains Pro.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume successful answer/Judge checkpoints. The default is a fresh run with archival.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. By default it is isolated by split and answer model.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "answer", "judge"),
        default="all",
        help="Run both stages, answer only, or Judge only. Judge-only never calls the answer model.",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=0,
        metavar="N",
        help="Run at most N new answer questions in this invocation; 0 runs all remaining questions.",
    )
    args = parser.parse_args()
    args.split = split or args.split
    args.answer_model_id = ANSWER_MODELS[args.answer_model]
    if args.output_dir is None:
        if default_output is not None and args.answer_model_id == answer_model:
            args.output_dir = default_output
        else:
            args.output_dir = (
                ROOT
                / "reports"
                / f"pre_jina_all_chunks_{args.answer_model}_{args.split}_v1"
            )
    if args.max_new < 0:
        parser.error("--max-new must be zero or a positive integer")
    return args


def run_split(
    split: str | None,
    default_output: Path | None,
    answer_model: str = PRO_MODEL,
) -> None:
    args = parse_args(split, default_output, answer_model=answer_model)
    split = args.split
    answer_model = args.answer_model_id
    load_dotenv(ROOT / ".env", override=False)
    os.environ.update(
        {
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    output = args.output_dir.resolve()
    resume = args.resume or args.stage == "judge"
    archived = prepare_output(output, resume)
    rows = select_half(load_dataset(), split)
    snapshot = load_snapshot()
    snapshot_by_id = {record["question_id"]: record for record in snapshot}
    missing = [row["financebench_id"] for row in rows if row["financebench_id"] not in snapshot_by_id]
    if missing:
        raise ValueError(f"Questions missing from frozen snapshot: {missing}")
    raw_sizes = [sum(len(str(chunk.get("text") or "")) for chunk in snapshot_by_id[row["financebench_id"]]["chunks"]) for row in rows]
    print(
        f"[setup] split={split} questions=50 chunks_per_question=120 answer={answer_model} "
        f"judge={PRO_MODEL} retrieval=false jina=false context_truncation=false "
        f"average_raw_chars={fmean(raw_sizes):.0f} cache_policy={'resume' if resume else 'fresh'} "
        f"max_new={args.max_new or 'all'}",
        flush=True,
    )
    if archived:
        print(f"[setup] previous result archived to: {archived}", flush=True)
    if args.stage in {"all", "answer"}:
        answers = run_answers(
            rows,
            snapshot_by_id,
            output,
            split,
            max_new=args.max_new,
            answer_model=answer_model,
        )
    else:
        answers = load_jsonl(output / "answers.jsonl")
        if not answers:
            raise RuntimeError("Judge-only stage requires an existing answers.jsonl checkpoint")
        print(
            f"[setup] judge-only: reusing {len(answers)} saved answers; answer model calls=0",
            flush=True,
        )
    if args.stage in {"all", "judge"}:
        judges = run_judges(rows, answers, output, split)
    else:
        judges = load_jsonl(output / "judge_results.jsonl")
    summary = build_summary(split, answers, judges, answer_model=answer_model)
    summary["cache_policy"] = "resume" if resume else "fresh"
    summary["archived_previous_results"] = str(archived) if archived else None
    atomic_json(output / "summary.json", summary)
    write_report(output, summary, answers, judges)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Output: {output}", flush=True)
