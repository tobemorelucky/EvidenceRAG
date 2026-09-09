"""Run the FinanceBench Answer-only Oracle Evidence Shadow v1 experiment.

This script deliberately has no dependency on retrieval, reranking, query
rewriting, or the production answer pipeline. It reads the frozen final100
artifacts only to validate the experiment population and recover the original
failure attribution. Answer evidence comes exclusively from the benchmark's
saved gold evidence records.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATASET = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
FINAL100 = ROOT / "reports" / "final100"
ATTRIBUTION = ROOT / "reports" / "answer_failure_attribution_v2" / "failure_cases.json"
OUTPUT = ROOT / "reports" / "financebench_answer_only_oracle_shadow_v1"
ANSWER_MODEL = "deepseek-v4-flash-ga-260731"
JUDGE_MODEL = "deepseek-v4-pro-ga-260813"
REQUIRED_FINAL100_FILES = (
    "retrieval_snapshot.json",
    "jina_results.json",
    "answers.jsonl",
    "judge_results.jsonl",
)

ORACLE_SYSTEM_PROMPT = """You are EvidenceRAG, a financial question-answering assistant.
Answer only from the gold Evidence supplied for the current question. Do not use external knowledge.
Use and cite the evidence numbers needed for the answer. Preserve company, period, sign, currency, unit, and scope.
Follow any output, calculation, and rounding requirements in the question.
Cite factual claims as [source: filename, page N]. If the supplied evidence genuinely does not support the answer, identify the exact missing fact instead of guessing."""

ORACLE_USER_TEMPLATE = """Question:
{question}

Evidence:
{evidence}

Instruction:
根据提供证据回答问题。
必须引用证据中的数字。
保持原回答格式要求。
不要使用外部知识。"""


def _clean(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path} at line {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object in {path} at line {line_number}")
        records.append(value)
    return records


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Frozen artifact has no records list: {path}")
    return [record for record in records if isinstance(record, dict)]


def _record_id(record: dict[str, Any]) -> str:
    return _clean(record.get("financebench_id") or record.get("question_id") or record.get("id"))


def validate_frozen_final100(final100_dir: Path = FINAL100) -> dict[str, Any]:
    """Validate, but never execute or import, any frozen upstream stage."""

    manifest: dict[str, Any] = {}
    expected_ids: set[str] | None = None
    for filename in REQUIRED_FINAL100_FILES:
        path = final100_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen final100 artifact: {path}")
        records = _artifact_records(path)
        ids = {_record_id(record) for record in records if _record_id(record)}
        if len(ids) != 100:
            raise ValueError(f"Expected 100 unique question IDs in {path}, found {len(ids)}")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"Question population mismatch in frozen artifact: {path}")
        manifest[filename] = {"records": len(records), "unique_question_ids": len(ids)}
    return {"question_ids": sorted(expected_ids or []), "artifacts": manifest}


def load_dataset(path: Path = DATASET) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = [_clean(row.get("financebench_id")) for row in rows]
    if len(rows) != 100 or len(set(ids)) != 100 or any(not item for item in ids):
        raise ValueError("Expected exactly 100 unique FinanceBench rows")
    return rows


def parse_gold_evidence(row: dict[str, str]) -> list[dict[str, Any]]:
    try:
        payload = json.loads(_clean(row.get("evidence")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid gold evidence JSON for {_clean(row.get('financebench_id'))}") from exc
    records = payload if isinstance(payload, list) else [payload]
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for raw in records:
        if not isinstance(raw, dict):
            continue
        document = _clean(raw.get("doc_name") or row.get("doc_name"))
        try:
            page = int(float(raw.get("evidence_page_num")))
        except (TypeError, ValueError):
            continue
        text = _clean(raw.get("evidence_text")) or _clean(raw.get("evidence_text_full_page"))
        if not document or not text:
            continue
        filename = document if document.lower().endswith(".pdf") else f"{document}.pdf"
        key = (filename, page, text)
        if key in seen:
            continue
        seen.add(key)
        evidence.append(
            {
                "document": filename,
                "page": page,
                "text": text,
                "source_kind": "evidence_text" if _clean(raw.get("evidence_text")) else "evidence_text_full_page",
            }
        )
    return evidence


def build_oracle_context(row: dict[str, str]) -> str:
    evidence = parse_gold_evidence(row)
    if not evidence:
        raise ValueError(f"Gold evidence is empty for {_clean(row.get('financebench_id'))}")
    return "\n\n---\n\n".join(
        f"[Gold Evidence {index}]\nSource: {item['document']}\nPage: {item['page']}\n{item['text']}"
        for index, item in enumerate(evidence, 1)
    )


def load_attribution(path: Path = ATTRIBUTION) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else payload.get("failure_cases", [])
    mapping = {
        "B": "metric",
        "E": "calculation",
        "F": "reasoning",
        "G": "refusal",
    }
    return {
        _record_id(record): mapping.get(_clean(record.get("category")).upper(), "other")
        for record in records
        if isinstance(record, dict) and _record_id(record)
    }


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
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def create_answer_model():
    api_key = os.getenv("ARK_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("ARK_API_KEY and BASE_URL are required")
    return init_chat_model(
        model=ANSWER_MODEL,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        max_completion_tokens=1024,
        timeout=float(os.getenv("ANSWER_TIMEOUT_SECONDS", "90")),
        max_retries=int(os.getenv("ANSWER_MAX_RETRIES", "2")),
        stream_usage=True,
        extra_body={"thinking": {"type": "disabled"}},
    )


def invoke_answer(model: Any, question: str, oracle_context: str) -> tuple[str, dict[str, int]]:
    response = model.invoke(
        [
            SystemMessage(content=ORACLE_SYSTEM_PROMPT),
            HumanMessage(content=ORACLE_USER_TEMPLATE.format(question=question, evidence=oracle_context)),
        ]
    )
    answer = _content_text(getattr(response, "content", response)).strip()
    if not answer:
        raise ValueError("Answer model returned empty content")
    return answer, normalize_usage(getattr(response, "usage_metadata", {}) or {})


def build_result_record(
    row: dict[str, str],
    oracle_context: str,
    model_answer: str,
    latency_seconds: float,
    token_usage: dict[str, int],
    attribution: str,
) -> dict[str, Any]:
    return {
        "financebench_id": _clean(row.get("financebench_id")),
        "question": _clean(row.get("question")),
        "gold_answer": _clean(row.get("answer")),
        "oracle_context": oracle_context,
        "model_answer": model_answer,
        "latency": round(float(latency_seconds), 4),
        "token_usage": normalize_usage(token_usage),
        "original_failure_attribution": attribution,
        "status": "ok",
    }


def _upsert_ordered(
    rows: list[dict[str, str]],
    saved: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [saved[row["financebench_id"]] for row in rows if row["financebench_id"] in saved]


def run_answers(
    rows: list[dict[str, str]],
    output_path: Path,
    attribution: dict[str, str],
    model_factory: Callable[[], Any] = create_answer_model,
) -> list[dict[str, Any]]:
    existing = load_jsonl(output_path)
    saved = {_record_id(record): record for record in existing if _record_id(record)}
    pending = [row for row in rows if saved.get(row["financebench_id"], {}).get("status") != "ok"]
    model = model_factory() if pending else None
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        if saved.get(question_id, {}).get("status") == "ok":
            print(f"[answer {index:03d}/100] {question_id}: cached", flush=True)
            continue
        started = time.perf_counter()
        context = build_oracle_context(row)
        try:
            answer, usage = invoke_answer(model, row["question"], context)
            record = build_result_record(
                row,
                context,
                answer,
                time.perf_counter() - started,
                usage,
                attribution.get(question_id, "not_applicable"),
            )
        except Exception as exc:
            record = {
                "financebench_id": question_id,
                "question": _clean(row.get("question")),
                "gold_answer": _clean(row.get("answer")),
                "oracle_context": context,
                "model_answer": "",
                "latency": round(time.perf_counter() - started, 4),
                "token_usage": normalize_usage({}),
                "original_failure_attribution": attribution.get(question_id, "not_applicable"),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        saved[question_id] = record
        atomic_jsonl(output_path, _upsert_ordered(rows, saved))
        print(f"[answer {index:03d}/100] {question_id}: {record['status']}", flush=True)
    return _upsert_ordered(rows, saved)


def run_judges(
    rows: list[dict[str, str]],
    answers: list[dict[str, Any]],
    output_path: Path,
) -> list[dict[str, Any]]:
    sys.path.insert(0, str(SCRIPTS))
    from financebench_judge_common import JUDGE_PROMPT, _judge_model, _judge_with_retry

    answer_by_id = {_record_id(record): record for record in answers}
    existing = load_jsonl(output_path)
    saved = {_record_id(record): record for record in existing if _record_id(record)}
    pending = [
        row for row in rows
        if answer_by_id.get(row["financebench_id"], {}).get("status") == "ok"
        and saved.get(row["financebench_id"], {}).get("status") != "ok"
    ]
    model = _judge_model() if pending else None
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        if saved.get(question_id, {}).get("status") == "ok":
            print(f"[judge  {index:03d}/100] {question_id}: cached", flush=True)
            continue
        answer_record = answer_by_id.get(question_id, {})
        started = time.perf_counter()
        if answer_record.get("status") != "ok":
            record = {
                "financebench_id": question_id,
                "status": "skipped",
                "judge": {"score": 0, "verdict": "not_judged", "reason": "answer_generation_failed"},
                "latency": 0.0,
            }
        else:
            try:
                verdict = _judge_with_retry(
                    model,
                    JUDGE_PROMPT.format(
                        question=row["question"],
                        reference=row["answer"],
                        answer=answer_record["model_answer"],
                    ),
                )
                record = {
                    "financebench_id": question_id,
                    "status": "ok",
                    "judge_model": JUDGE_MODEL,
                    "judge": verdict,
                    "latency": round(time.perf_counter() - started, 4),
                }
            except Exception as exc:
                record = {
                    "financebench_id": question_id,
                    "status": "error",
                    "judge": {"score": 0, "verdict": "not_judged", "reason": "judge_call_failed"},
                    "latency": round(time.perf_counter() - started, 4),
                    "error": f"{type(exc).__name__}: {exc}",
                }
        saved[question_id] = record
        atomic_jsonl(output_path, _upsert_ordered(rows, saved))
        verdict = (record.get("judge") or {}).get("verdict") or record["status"]
        print(f"[judge  {index:03d}/100] {question_id}: {verdict}", flush=True)
    return _upsert_ordered(rows, saved)


def build_summary(
    answers: list[dict[str, Any]],
    judges: list[dict[str, Any]],
    frozen_manifest: dict[str, Any],
) -> dict[str, Any]:
    judge_by_id = {_record_id(record): record for record in judges}
    successful_answers = [record for record in answers if record.get("status") == "ok"]
    judged = [record for record in judges if record.get("status") == "ok"]
    strict_correct = sum(int((record.get("judge") or {}).get("score") or 0) for record in judged)
    categories: dict[str, dict[str, int | float | None]] = {}
    for category in ("calculation", "reasoning", "refusal", "metric", "other"):
        members = [record for record in answers if record.get("original_failure_attribution") == category]
        correct = sum(
            int((judge_by_id.get(_record_id(record), {}).get("judge") or {}).get("score") or 0)
            for record in members
        )
        categories[category] = {
            "total": len(members),
            "strict_correct": correct,
            "accuracy": round(correct / len(members), 4) if members else None,
        }
    return {
        "schema": "financebench_answer_only_oracle_shadow_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(answers),
        "answer_success": len(successful_answers),
        "judge_success": len(judged),
        "strict_correct": strict_correct,
        "accuracy": round(strict_correct / len(judged), 4) if judged else None,
        "average_answer_tokens": round(
            fmean(record["token_usage"]["total_tokens"] for record in successful_answers), 2
        ) if successful_answers else 0.0,
        "average_answer_input_tokens": round(
            fmean(record["token_usage"]["input_tokens"] for record in successful_answers), 2
        ) if successful_answers else 0.0,
        "average_answer_output_tokens": round(
            fmean(record["token_usage"]["output_tokens"] for record in successful_answers), 2
        ) if successful_answers else 0.0,
        "average_latency": round(
            fmean(float(record.get("latency") or 0) for record in successful_answers), 4
        ) if successful_answers else 0.0,
        "answer_token_usage": {
            key: sum(int(record.get("token_usage", {}).get(key) or 0) for record in successful_answers)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        },
        "original_failure_attribution": categories,
        "original_failure_attribution_cases": sum(
            values["total"] for values in categories.values()
        ),
        "answer_errors": dict(Counter(record.get("status") or "unknown" for record in answers)),
        "judge_errors": dict(Counter(record.get("status") or "unknown" for record in judges)),
        "models": {"answer": ANSWER_MODEL, "judge": JUDGE_MODEL},
        "inference": {"temperature": 0.1, "thinking": "disabled", "max_tokens": 1024},
        "upstream_calls": {"retrieval": 0, "query_rewrite": 0, "jina": 0},
        "frozen_final100": frozen_manifest["artifacts"],
    }


def validate_results(rows: list[dict[str, str]], answers: list[dict[str, Any]]) -> None:
    ids = [_record_id(record) for record in answers]
    expected = {row["financebench_id"] for row in rows}
    if len(answers) != 100 or len(set(ids)) != 100 or set(ids) != expected:
        raise ValueError("Answer output does not contain exactly the 100 unique FinanceBench questions")
    empty = [record["financebench_id"] for record in answers if not _clean(record.get("oracle_context"))]
    if empty:
        raise ValueError(f"Oracle evidence is empty for: {', '.join(empty)}")
    required = {
        "financebench_id", "question", "gold_answer", "oracle_context",
        "model_answer", "latency", "token_usage",
    }
    incomplete = [record["financebench_id"] for record in answers if not required <= set(record)]
    if incomplete:
        raise ValueError(f"Incomplete answer records: {', '.join(incomplete)}")


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    os.environ.update(
        {
            "LANGSMITH_TRACING": "false",
            "LANGCHAIN_TRACING_V2": "false",
            "JUDGE_MODEL": JUDGE_MODEL,
            "JUDGE_THINKING_MODE": "disabled",
            "JUDGE_MAX_COMPLETION_TOKENS": "512",
        }
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = validate_frozen_final100()
    rows = load_dataset()
    if set(frozen["question_ids"]) != {row["financebench_id"] for row in rows}:
        raise ValueError("CSV question IDs do not match the frozen final100 experiment")
    attribution = load_attribution()
    print(
        f"[setup] questions=100 evidence=gold answer={ANSWER_MODEL} judge={JUDGE_MODEL} "
        "retrieval=false jina=false query_rewrite=false",
        flush=True,
    )
    answers = run_answers(rows, OUTPUT / "results.jsonl", attribution)
    validate_results(rows, answers)
    judges = run_judges(rows, answers, OUTPUT / "judge_results.jsonl")
    summary = build_summary(answers, judges, frozen)
    atomic_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Results: {OUTPUT / 'results.jsonl'}", flush=True)
    print(f"Judge: {OUTPUT / 'judge_results.jsonl'}", flush=True)
    print(f"Summary: {OUTPUT / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
