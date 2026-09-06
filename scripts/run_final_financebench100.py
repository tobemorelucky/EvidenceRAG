"""Staged final FinanceBench100 shadow experiment.

Pipeline: original + <=2 DeepSeek rewrites -> per-query Dense/BM25 -> RRF120
-> Jina input80 -> Top12/28k context -> DeepSeek Flash -> DeepSeek Pro Judge.
This script does not import or mutate the production retrieval pipeline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_query_rewrite_shadow_v1 import rrf_fuse  # noqa: E402
from evaluate_reranker_shadow_v1 import metrics  # noqa: E402
from jina_full_baseline_v1 import build_context, digest  # noqa: E402
from query_rewriter import create_query_rewrite_model, rewrite_queries  # noqa: E402
from run_jina_full_baseline_v1 import judge_answer  # noqa: E402
from shadow_rerankers_v1 import JinaReranker, validate_order  # noqa: E402


DATASET = ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"
DEFAULT_OUTPUT = ROOT / "reports/final100"
HISTORICAL = ROOT / "reports/jina_full_baseline_input120_all100"
LEAF_FILTER = '(evidence_type == "text_chunk" or evidence_type == "") and chunk_level == 3'
KNOWN_OUTPUTS = {
    "retrieval_snapshot.json", "jina_results.json", "answers.jsonl",
    "judge_results.jsonl", "state.json", "final_report.md", "final_summary.json",
}
CONFIG = {
    "query_rewrite": {"enabled": True, "max_rewrites": 2, "original_retained": True},
    "retrieval": {"dense_top_k": 240, "bm25_top_k": 240, "rrf_top_k": 120, "rrf_k": 60, "filter": LEAF_FILTER},
    "reranker": {"model": "jina-reranker-v3", "input_k": 80, "output_k": 12, "interval_seconds": 60},
    "context": {"top_k": 12, "max_chars": 28000},
    "answer": {"model": "deepseek-v4-flash-ga-260731", "profile": "clean_baseline", "prompt_mode": "baseline", "temperature": 0.1, "max_completion_tokens": 1024, "thinking": "disabled", "timeout_seconds": 60},
    "judge": {"model": "deepseek-v4-pro-ga-260813", "max_completion_tokens": 512, "timeout_seconds": 60},
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset() -> list[dict]:
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 100 or len({row["financebench_id"] for row in rows}) != 100:
        raise ValueError("Expected exactly 100 unique FinanceBench rows")
    return rows


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_output(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    unknown = {path.name for path in output.iterdir()} - KNOWN_OUTPUTS
    if unknown:
        raise ValueError(f"Output directory contains unknown files: {sorted(unknown)}")


def manifest() -> dict:
    return {
        "schema": "final_financebench100_v1", "dataset_sha256": sha(DATASET),
        "config": CONFIG,
        "code_sha256": {
            relative: sha(ROOT / relative) for relative in (
                "backend/query_rewriter.py", "backend/prompts.py", "backend/answer_generator.py",
                "scripts/evaluate_query_rewrite_shadow_v1.py", "scripts/jina_full_baseline_v1.py",
                "scripts/financebench_judge_common.py", "scripts/run_final_financebench100.py",
            )
        },
    }


def validate_manifest(payload: dict) -> None:
    expected = manifest()
    if payload.get("manifest") != expected:
        raise ValueError("Checkpoint configuration/code drift; use a new empty output directory")


def usage_sum(records: list[dict], getter) -> dict:
    usages = [getter(record) or {} for record in records if record.get("status") == "ok"]
    return {
        key: sum(int(usage.get(key) or 0) for usage in usages)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }


def write_state(output: Path, stage: str, records: list[dict], elapsed: float, force: bool = False) -> None:
    attempted = sum(record.get("status") in {"ok", "error"} for record in records)
    if not force and attempted % 10:
        return
    if stage == "recall":
        token_usage = usage_sum(records, lambda record: record.get("rewrite", {}).get("usage"))
    elif stage == "rerank":
        token_usage = usage_sum(records, lambda record: (record.get("trace", {}).get("usage") or {}))
    elif stage == "answer":
        token_usage = usage_sum(records, lambda record: record.get("usage"))
    elif stage == "judge":
        token_usage = usage_sum(records, lambda record: record.get("judge", {}).get("usage"))
    else:
        token_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    state_path = output / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"manifest": manifest(), "stages": {}}
    if state.get("manifest") != manifest():
        raise ValueError("State manifest drift")
    state["stages"][stage] = {
        "completed_count": attempted,
        "success_count": sum(record.get("status") == "ok" for record in records),
        "failed_count": sum(record.get("status") == "error" for record in records),
        "token_usage": token_usage,
        "elapsed_time_seconds": round(elapsed, 2),
    }
    atomic_json(state_path, state)


def _credential(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required credential: {name}")
    return value


def _configure_env() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    os.environ.update(
        LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
    )


def stage_recall(output: Path, rows: list[dict]) -> None:
    _credential("ARK_API_KEY")
    os.environ["MODEL"] = CONFIG["answer"]["model"]
    path = output / "retrieval_snapshot.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema": "final100_retrieval_snapshot_v1", "manifest": manifest(), "records": []}
    validate_manifest(payload)
    saved = {record["question_id"]: record for record in payload["records"]}
    from embedding import embedding_service
    from milvus_client import MilvusManager
    manager = MilvusManager()
    if not manager.uses_builtin_bm25:
        raise RuntimeError("Final experiment requires Milvus built-in BM25")
    model = create_query_rewrite_model()
    started = time.perf_counter()
    attempts = 0
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        if saved.get(question_id, {}).get("status") == "ok":
            continue
        attempts += 1
        question_started = time.perf_counter()
        try:
            rewrite = rewrite_queries(row["question"], model)
            if not 1 <= len(rewrite["queries"]) <= 3 or rewrite["queries"][0] != row["question"]:
                raise ValueError("Invalid original-first rewrite set")
            routes = []
            for query_index, query in enumerate(rewrite["queries"]):
                vector = embedding_service.get_embeddings([query])[0]
                dense = manager.dense_retrieve(vector, top_k=240, filter_expr=LEAF_FILTER)
                bm25 = manager.bm25_retrieve(query, top_k=240, filter_expr=LEAF_FILTER)
                routes.extend(((f"q{query_index}_dense", dense), (f"q{query_index}_bm25", bm25)))
            chunks = rrf_fuse(routes, rrf_k=60)[:120]
            if len(chunks) != 120:
                raise RuntimeError("Incomplete RRF Top120")
            record = {
                "question_id": question_id, "question": row["question"], "status": "ok",
                "rewrite": rewrite, "chunks": chunks, "candidate_sha256": digest(chunks),
                "retrieval_metrics": metrics(row, chunks),
                "latency_ms": (time.perf_counter() - question_started) * 1000,
            }
        except Exception as exc:
            record = {"question_id": question_id, "question": row["question"], "status": "error", "error_type": type(exc).__name__, "latency_ms": (time.perf_counter() - question_started) * 1000}
        saved[question_id] = record
        ordered = [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved]
        if attempts % 10 == 0:
            payload["records"] = ordered
            atomic_json(path, payload)
            write_state(output, "recall", ordered, time.perf_counter() - started)
        print(f"[recall {index}/100] {question_id} {record['status']}", flush=True)
    payload["records"] = [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved]
    atomic_json(path, payload)
    write_state(output, "recall", payload["records"], time.perf_counter() - started, force=True)


def stage_rerank(output: Path, rows: list[dict]) -> None:
    source_path = output / "retrieval_snapshot.json"
    if not source_path.exists():
        raise RuntimeError("Run --stage recall first")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_manifest(source)
    by_source = {record["question_id"]: record for record in source["records"]}
    path = output / "jina_results.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"schema": "final100_jina_results_v1", "manifest": manifest(), "records": []}
    validate_manifest(payload)
    saved = {record["question_id"]: record for record in payload["records"]}
    key = os.getenv("RERANK_API_KEY") or os.getenv("JINA_API_KEY")
    if not key:
        raise RuntimeError("Missing Jina credential")
    host = (os.getenv("RERANK_BINDING_HOST") or "https://api.jina.ai").rstrip("/")
    endpoint = host if host.endswith("/v1/rerank") else host + "/v1/rerank"
    reranker = JinaReranker(key, CONFIG["reranker"]["model"], endpoint, interval=60)
    started = time.perf_counter()
    attempts = 0
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        if saved.get(question_id, {}).get("status") == "ok":
            continue
        attempts += 1
        question_started = time.perf_counter()
        base = by_source.get(question_id, {})
        if base.get("status") != "ok":
            record = {"question_id": question_id, "status": "error", "error_type": "retrieval_unavailable"}
        else:
            chunks = base["chunks"][:80]
            try:
                ranked, trace = reranker.rank(row["question"], [chunk["text"] for chunk in chunks])
                ranked = validate_order(ranked, 80)
                ordered = [chunks[item["index"]] for item in ranked]
                record = {
                    "question_id": question_id, "status": "ok",
                    "candidate_sha256": base["candidate_sha256"], "input_k": 80, "output_k": 12,
                    "ranked": ranked, "trace": trace,
                    "rerank_metrics": metrics(row, ordered),
                    "latency_ms": (time.perf_counter() - question_started) * 1000,
                }
            except Exception as exc:
                message = str(exc) if isinstance(exc, RuntimeError) and str(exc).startswith("Jina ") else "Jina request failed"
                record = {"question_id": question_id, "status": "error", "error_type": type(exc).__name__, "reason": message, "latency_ms": (time.perf_counter() - question_started) * 1000}
        saved[question_id] = record
        ordered_records = [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved]
        payload["records"] = ordered_records
        atomic_json(path, payload)  # paid result checkpoint: prevents duplicate Jina calls
        if attempts % 10 == 0:
            write_state(output, "rerank", ordered_records, time.perf_counter() - started)
        print(f"[rerank {index}/100] {question_id} {record['status']}", flush=True)
    payload["records"] = [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved]
    atomic_json(path, payload)
    write_state(output, "rerank", payload["records"], time.perf_counter() - started, force=True)


def _configure_answer() -> None:
    from runtime_profile import apply_runtime_profile
    apply_runtime_profile("clean_baseline")
    os.environ.update(
        MODEL=CONFIG["answer"]["model"], ANSWER_PROMPT_MODE="baseline",
        ANSWER_TEMPERATURE="0.1", ANSWER_MAX_COMPLETION_TOKENS="1024",
        ANSWER_THINKING_MODE="disabled", ANSWER_TIMEOUT_SECONDS="60", ANSWER_MAX_RETRIES="0",
    )


def stage_answer(output: Path, rows: list[dict]) -> None:
    _credential("ARK_API_KEY")
    retrieval = json.loads((output / "retrieval_snapshot.json").read_text(encoding="utf-8"))
    rerank = json.loads((output / "jina_results.json").read_text(encoding="utf-8"))
    validate_manifest(retrieval)
    validate_manifest(rerank)
    sources = {record["question_id"]: record for record in retrieval["records"]}
    ranks = {record["question_id"]: record for record in rerank["records"]}
    path = output / "answers.jsonl"
    saved = {record["question_id"]: record for record in load_jsonl(path)}
    _configure_answer()
    from answer_generator import generate_answer
    started = time.perf_counter()
    attempts = 0
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        if saved.get(question_id, {}).get("status") == "ok":
            continue
        attempts += 1
        question_started = time.perf_counter()
        source, rank = sources.get(question_id, {}), ranks.get(question_id, {})
        if source.get("status") != "ok" or rank.get("status") != "ok":
            record = {"question_id": question_id, "status": "error", "error_type": "jina_unavailable"}
        else:
            try:
                input_chunks = source["chunks"][:80]
                ordered = [input_chunks[item["index"]] for item in validate_order(rank["ranked"], 80)]
                evidence, citations, documents = build_context(ordered, CONFIG["context"])
                answer, usage = generate_answer(row["question"], evidence, [], "", "clean_baseline", "baseline")
                if not str(answer).strip():
                    raise RuntimeError("Empty answer")
                record = {
                    "question_id": question_id, "question": row["question"], "status": "ok",
                    "evidence": evidence, "citations": citations, "context_documents": documents,
                    "context_metrics": metrics(row, documents), "answer": answer, "usage": usage,
                    "latency_ms": (time.perf_counter() - question_started) * 1000,
                }
            except Exception as exc:
                record = {"question_id": question_id, "question": row["question"], "status": "error", "error_type": type(exc).__name__, "latency_ms": (time.perf_counter() - question_started) * 1000}
        saved[question_id] = record
        ordered_records = [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved]
        atomic_jsonl(path, ordered_records)  # paid result checkpoint
        if attempts % 10 == 0:
            write_state(output, "answer", ordered_records, time.perf_counter() - started)
        print(f"[answer {index}/100] {question_id} {record['status']}", flush=True)
    records = [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved]
    atomic_jsonl(path, records)
    write_state(output, "answer", records, time.perf_counter() - started, force=True)


def stage_judge(output: Path, rows: list[dict]) -> None:
    _credential("ARK_API_KEY")
    answers = {record["question_id"]: record for record in load_jsonl(output / "answers.jsonl")}
    path = output / "judge_results.jsonl"
    saved = {record["question_id"]: record for record in load_jsonl(path)}
    config = CONFIG["judge"]
    holder = []
    started = time.perf_counter()
    attempts = 0
    for index, row in enumerate(rows, 1):
        question_id = row["financebench_id"]
        if saved.get(question_id, {}).get("status") == "ok":
            continue
        attempts += 1
        question_started = time.perf_counter()
        answer = answers.get(question_id, {})
        if answer.get("status") != "ok":
            record = {"question_id": question_id, "status": "error", "error_type": "answer_unavailable"}
        else:
            try:
                result = judge_answer(row["question"], row["answer"], answer["answer"], config, holder)
                record = {"question_id": question_id, "status": "ok", "judge": result, "latency_ms": (time.perf_counter() - question_started) * 1000}
            except Exception as exc:
                record = {"question_id": question_id, "status": "error", "error_type": type(exc).__name__, "latency_ms": (time.perf_counter() - question_started) * 1000}
        saved[question_id] = record
        ordered_records = [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved]
        atomic_jsonl(path, ordered_records)  # paid result checkpoint
        if attempts % 10 == 0:
            write_state(output, "judge", ordered_records, time.perf_counter() - started)
        print(f"[judge {index}/100] {question_id} {record['status']}", flush=True)
    records = [saved[item["financebench_id"]] for item in rows if item["financebench_id"] in saved]
    atomic_jsonl(path, records)
    write_state(output, "judge", records, time.perf_counter() - started, force=True)


def _historical_summary(rows: list[dict]) -> dict:
    summary = json.loads((HISTORICAL / "summary.json").read_text(encoding="utf-8"))
    state = json.loads((HISTORICAL / "state.json").read_text(encoding="utf-8"))
    records = state["records"]
    return {
        "candidate_hit": _rate([record.get("retrieval_metrics", {}).get("ranked", {}).get("candidate_gold_page_hit") for record in records]),
        "context_hit": _rate([record.get("retrieval_metrics", {}).get("actual_context", {}).get("candidate_gold_page_hit") for record in records]),
        "strict_judge": summary["strict_accuracy"], "strict_correct": summary["strict_correct"],
    }


def _rate(values) -> float | None:
    values = [value for value in values if value is not None]
    return fmean(values) if values else None


def stage_report(output: Path, rows: list[dict]) -> None:
    retrieval = {record["question_id"]: record for record in json.loads((output / "retrieval_snapshot.json").read_text(encoding="utf-8"))["records"]}
    rerank = {record["question_id"]: record for record in json.loads((output / "jina_results.json").read_text(encoding="utf-8"))["records"]}
    answers = {record["question_id"]: record for record in load_jsonl(output / "answers.jsonl")}
    judges = {record["question_id"]: record for record in load_jsonl(output / "judge_results.jsonl")}
    records = []
    for row in rows:
        question_id = row["financebench_id"]
        retrieval_record, rerank_record = retrieval.get(question_id, {}), rerank.get(question_id, {})
        answer_record, judge_record = answers.get(question_id, {}), judges.get(question_id, {})
        candidate_hit = retrieval_record.get("retrieval_metrics", {}).get("candidate_gold_page_hit") if retrieval_record.get("status") == "ok" else False
        jina_hit = rerank_record.get("rerank_metrics", {}).get("gold_page_rank") is not None and rerank_record["rerank_metrics"]["gold_page_rank"] <= 12 if rerank_record.get("status") == "ok" else False
        context_hit = answer_record.get("context_metrics", {}).get("candidate_gold_page_hit") if answer_record.get("status") == "ok" else False
        judge_correct = judge_record.get("judge", {}).get("score") == 1 if judge_record.get("status") == "ok" else False
        if not candidate_hit:
            failure = "retrieval_miss"
        elif not jina_hit:
            failure = "rerank_miss"
        elif not context_hit:
            failure = "context_loss"
        elif not judge_correct:
            failure = "answer_failure"
        else:
            failure = "correct"
        records.append({
            "id": question_id, "question": row["question"], "reference_answer": row["answer"],
            "rewrite_queries": retrieval_record.get("rewrite", {}).get("queries", []),
            "retrieval_status": retrieval_record.get("status", "missing"), "candidate_hit": candidate_hit,
            "jina_status": rerank_record.get("status", "missing"),
            "jina_gold_page_rank": rerank_record.get("rerank_metrics", {}).get("gold_page_rank"),
            "jina_top12_hit": jina_hit, "context_hit": context_hit,
            "context_summary": answer_record.get("evidence", "")[:1000].replace("\n", " "),
            "answer": answer_record.get("answer", ""), "judge_correct": judge_correct,
            "judge_reason": judge_record.get("judge", {}).get("reason", judge_record.get("error_type", "")),
            "failure_stage": failure,
        })
    correct = sum(record["judge_correct"] for record in records)
    funnel = {
        "rrf_candidate_hit": _rate([record["candidate_hit"] for record in records]),
        "jina_top12_hit": _rate([record["jina_top12_hit"] for record in records]),
        "context_hit": _rate([record["context_hit"] for record in records]),
        "retrieval_miss": sum(record["failure_stage"] == "retrieval_miss" for record in records),
        "rerank_miss": sum(record["failure_stage"] == "rerank_miss" for record in records),
        "context_loss": sum(record["failure_stage"] == "context_loss" for record in records),
        "answer_failure": sum(record["failure_stage"] == "answer_failure" for record in records),
    }
    retrieval_records, rerank_records = list(retrieval.values()), list(rerank.values())
    answer_records, judge_records = list(answers.values()), list(judges.values())
    cost = {
        "query_rewrite": usage_sum(retrieval_records, lambda record: record.get("rewrite", {}).get("usage")),
        "jina": usage_sum(rerank_records, lambda record: record.get("trace", {}).get("usage")),
        "flash": usage_sum(answer_records, lambda record: record.get("usage")),
        "judge": usage_sum(judge_records, lambda record: record.get("judge", {}).get("usage")),
        "latency_ms_per_question": {
            "retrieval": _rate([record.get("latency_ms") for record in retrieval_records if record.get("status") == "ok"]),
            "jina": _rate([record.get("latency_ms") for record in rerank_records if record.get("status") == "ok"]),
            "answer": _rate([record.get("latency_ms") for record in answer_records if record.get("status") == "ok"]),
            "judge": _rate([record.get("latency_ms") for record in judge_records if record.get("status") == "ok"]),
        },
    }
    for name in ("query_rewrite", "jina", "flash", "judge"):
        cost[name]["tokens_per_question"] = cost[name]["total_tokens"] / 100
    historical = _historical_summary(rows)
    summary = {
        "questions": 100, "strict_judge": {"correct": correct, "total": 100, "accuracy": correct / 100},
        "stage_failures": {
            "retrieval": sum(record.get("status") != "ok" for record in retrieval_records),
            "jina": sum(record.get("status") != "ok" for record in rerank_records),
            "answer": sum(record.get("status") != "ok" for record in answer_records),
            "judge": sum(record.get("status") != "ok" for record in judge_records),
        },
        "funnel": funnel, "cost": cost,
        "comparison": {
            "baseline_dense_bm25_rrf_jina120": historical,
            "final_query_rewrite_jina80": {"candidate_hit": funnel["rrf_candidate_hit"], "context_hit": funnel["context_hit"], "strict_judge": correct / 100, "strict_correct": correct},
        },
    }
    atomic_json(output / "final_summary.json", summary)
    priority = {"retrieval_miss": 0, "rerank_miss": 1, "context_loss": 2, "answer_failure": 3, "correct": 4}
    failures = sorted((record for record in records if record["failure_stage"] != "correct"), key=lambda record: (priority[record["failure_stage"]], record["id"]))[:20]
    lines = [
        "# Final FinanceBench100 Report", "",
        f"- Strict Judge: **{correct}/100 ({correct / 100:.2%})**", "",
        "## Retrieval funnel", "",
        f"- RRF candidate hit: {funnel['rrf_candidate_hit']:.2%}", f"- Jina Top12 hit: {funnel['jina_top12_hit']:.2%}", f"- Context hit: {funnel['context_hit']:.2%}",
        f"- Retrieval miss / rerank miss / context loss / answer failure: {funnel['retrieval_miss']} / {funnel['rerank_miss']} / {funnel['context_loss']} / {funnel['answer_failure']}", "",
        "## Cost", "",
        "| Stage | Input token | Output token | Total | Token/question | Latency/question |", "|---|---:|---:|---:|---:|---:|",
    ]
    latency_key = {"query_rewrite": "retrieval", "jina": "jina", "flash": "answer", "judge": "judge"}
    for name in ("query_rewrite", "jina", "flash", "judge"):
        item = cost[name]
        lines.append(f"| {name} | {item['input_tokens']:,} | {item['output_tokens']:,} | {item['total_tokens']:,} | {item['tokens_per_question']:.0f} | {cost['latency_ms_per_question'][latency_key[name]]:.0f} ms |")
    final = summary["comparison"]["final_query_rewrite_jina80"]
    lines.extend(["", "## Historical comparison", "", "| Experiment | Candidate hit | Context hit | Strict Judge |", "|---|---:|---:|---:|"])
    lines.append(f"| Dense+BM25+RRF+Jina120 | {historical['candidate_hit']:.2%} | {historical['context_hit']:.2%} | {historical['strict_correct']}/100 ({historical['strict_judge']:.2%}) |")
    lines.append(f"| Query Rewrite+Jina80 | {final['candidate_hit']:.2%} | {final['context_hit']:.2%} | {final['strict_correct']}/100 ({final['strict_judge']:.2%}) |")
    lines.extend(["", "## Top 20 failures", ""])
    for record in failures:
        lines.extend([
            f"### {record['id']} — {record['failure_stage']}", "", f"**Question:** {record['question']}", "",
            f"**Rewrite queries:** `{record['rewrite_queries']}`", "", f"**Retrieval/Jina:** {record['retrieval_status']} / {record['jina_status']}; gold page rank={record['jina_gold_page_rank']}", "",
            f"**Context摘要:** {record['context_summary']}", "", f"**Answer:** {record['answer']}", "", f"**Judge:** {record['judge_reason']}", "",
        ])
    (output / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Report: {output / 'final_report.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("recall", "rerank", "answer", "judge", "report"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    ensure_output(args.output_dir)
    _configure_env()
    rows = load_dataset()
    if args.stage == "recall":
        stage_recall(args.output_dir, rows)
    elif args.stage == "rerank":
        stage_rerank(args.output_dir, rows)
    elif args.stage == "answer":
        stage_answer(args.output_dir, rows)
    elif args.stage == "judge":
        stage_judge(args.output_dir, rows)
    else:
        stage_report(args.output_dir, rows)


if __name__ == "__main__":
    main()
