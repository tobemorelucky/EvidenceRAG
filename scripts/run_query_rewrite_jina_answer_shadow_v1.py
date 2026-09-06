"""Answer/Judge fixed30 using frozen Query Rewrite + Jina evidence only.

The original route is reused from the completed Jina Input120 all100 state.
The experiment performs no query rewrite, retrieval, or Jina requests.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import fmean


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_evidence_assembly_ab import (  # noqa: E402
    _contains_all, _numbers, _periods, _required_numbers,
)
from evaluate_reranker_shadow_v1 import fixture_rows, metrics  # noqa: E402
from jina_full_baseline_v1 import build_context, read_profile  # noqa: E402
from run_jina_full_baseline_v1 import configure, judge_answer, sha  # noqa: E402
from shadow_rerankers_v1 import validate_order  # noqa: E402


DATASET = ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"
BASELINE_STATE = ROOT / "reports/jina_full_baseline_input120_all100/state.json"
FROZEN_CANDIDATES = ROOT / "reports/query_rewrite_shadow_v1.json"
FROZEN_JINA = ROOT / "reports/query_rewrite_jina_shadow_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/query_rewrite_jina_answer_shadow_v1"
ROUTES = ("baseline", "rewrite_jina")
BASELINE_COMMIT = "0d0898ee04e7250b0010691a148450a40780277e"


def _frozen_answer_contract(expected_hashes: dict) -> dict[str, str]:
    """Load the exact clean-baseline prompt used by A without touching HEAD."""
    files = {}
    for relative in ("backend/prompts.py", "backend/answer_generator.py"):
        content = subprocess.check_output(["git", "show", f"{BASELINE_COMMIT}:{relative}"], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != expected_hashes.get(relative):
            raise ValueError(f"Frozen commit does not match A baseline manifest: {relative}")
        files[relative] = content
    names = {
        "CLEAN_BASELINE_PROMPT_VERSION", "CLEAN_BASELINE_ANSWER_SYSTEM_PROMPT",
        "CLEAN_BASELINE_ANSWER_USER_TEMPLATE",
    }
    values = {}
    tree = ast.parse(files["backend/prompts.py"].decode("utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                values[target.id] = ast.literal_eval(node.value)
    if set(values) != names:
        raise ValueError("Frozen clean-baseline prompt constants are incomplete")
    return values


def _dataset_rows() -> dict[str, dict]:
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["financebench_id"]: row for row in csv.DictReader(stream)}
    if len(rows) != 100:
        raise ValueError("Expected the fixed FinanceBench 100 dataset")
    return rows


def _usage_total(usage: dict | None) -> int:
    usage = usage or {}
    return int(usage.get("total_tokens") or (
        int(usage.get("input_tokens") or 0) + int(usage.get("output_tokens") or 0)
    ))


def _context_metrics(row: dict, evidence: str, documents: list[dict]) -> dict:
    scored = metrics(row, documents, context_chunks=max(1, len(documents)), budget=10**9)
    required_numbers = _required_numbers(row)
    required_periods = _periods(row["question"])
    return {
        "evidence_context_hit": scored["candidate_gold_page_hit"],
        "evidence_span_hit": scored["gold_chunk_rank"] is not None,
        "required_number_hit": _contains_all(required_numbers, evidence, _numbers),
        "required_period_hit": _contains_all(required_periods, evidence, _periods),
        "required_numbers": required_numbers,
        "required_periods": required_periods,
        "context_chars": len(evidence),
    }


def _validate_inputs() -> tuple[list[dict], dict[str, dict], dict[str, dict], dict, dict]:
    rows, groups = fixture_rows()
    dataset = _dataset_rows()
    baseline = json.loads(BASELINE_STATE.read_text(encoding="utf-8"))
    candidates = json.loads(FROZEN_CANDIDATES.read_text(encoding="utf-8"))
    jina = json.loads(FROZEN_JINA.read_text(encoding="utf-8"))
    ids = list(groups)

    if baseline.get("manifest", {}).get("profile", {}).get("name") != "jina_full_baseline_input120_v1":
        raise ValueError("A route is not the frozen Input120 baseline")
    profile = baseline["manifest"]["profile"]
    if (
        profile["answer"]["model"] != "deepseek-v4-flash-ga-260731"
        or profile["answer"]["profile"] != "clean_baseline"
        or profile["judge"]["model"] != "deepseek-v4-pro-ga-260813"
        or profile["reranker"]["input_k"] != 120
        or profile["reranker"]["output_k"] != 12
        or profile["context"] != {"top_k": 12, "max_chars": 28000, "strategy": "rerank_order_raw_chunks_with_sources"}
    ):
        raise ValueError("Frozen A route contract drift")
    expected_hashes = baseline["manifest"].get("code_sha256", {})
    answer_contract = _frozen_answer_contract(expected_hashes)
    for relative in ("backend/runtime_profile.py", "scripts/financebench_judge_common.py", "scripts/jina_full_baseline_v1.py", "scripts/run_jina_full_baseline_v1.py"):
        if expected_hashes.get(relative) != sha(ROOT / relative):
            raise ValueError(f"Current answer/Judge contract differs from A baseline: {relative}")

    baseline_by_id = {record["financebench_id"]: record for record in baseline.get("records", [])}
    candidate_by_id = {record["question_id"]: record for record in candidates.get("records", [])}
    jina_by_id = {record["question_id"]: record for record in jina.get("records", [])}
    if any(question_id not in baseline_by_id or question_id not in candidate_by_id or question_id not in jina_by_id for question_id in ids):
        raise ValueError("A/B inputs do not contain the same fixed30")

    frozen = []
    for question_id in ids:
        row = dataset[question_id]
        a = baseline_by_id[question_id]
        source = candidate_by_id[question_id]
        ranking = jina_by_id[question_id].get("routes", {}).get("query_rewrite", {})
        chunks = source.get("routes", {}).get("query_rewrite", {}).get("chunks") or []
        if row["question"] != rows[question_id]["question"] or a.get("question") != row["question"] or source.get("question") != row["question"]:
            raise ValueError(f"Question drift: {question_id}")
        if a.get("answer_status") != "ok" or a.get("judge_status") != "ok" or a.get("judge", {}).get("verdict") not in {"correct", "incorrect"}:
            raise ValueError(f"Incomplete A answer/Judge: {question_id}")
        if ranking.get("status") != "ok" or len(chunks) != 120:
            raise ValueError(f"Incomplete frozen B Jina route: {question_id}")
        if ranking.get("candidate_sha256") != source["routes"]["query_rewrite"].get("candidate_sha256"):
            raise ValueError(f"B candidate hash drift: {question_id}")
        ranked = validate_order(ranking.get("ranked") or [], 120)
        ordered = [chunks[item["index"]] for item in ranked]
        evidence, citations, documents = build_context(ordered, profile["context"])
        if not evidence or len(evidence) > 28000:
            raise ValueError(f"Invalid rebuilt B context: {question_id}")
        frozen.append({
            "question_id": question_id, "group": groups[question_id], "row": row,
            "baseline": a,
            "rewrite_jina": {"evidence": evidence, "citations": citations, "context_documents": documents},
        })
    return frozen, baseline_by_id, jina_by_id, profile, answer_contract


def _classification(record: dict) -> str:
    a_correct = record["baseline"]["judge"]["score"] == 1
    b_correct = record["rewrite_jina"]["judge"]["score"] == 1
    context_improved = (
        not record["baseline"]["evidence_metrics"]["evidence_context_hit"]
        and record["rewrite_jina"]["evidence_metrics"]["evidence_context_hit"]
    )
    if not a_correct and b_correct:
        return "retrieval_gain_converted"
    if context_improved and a_correct and not b_correct:
        return "retrieval_regression"
    if context_improved and not b_correct:
        return "retrieval_gain_unused"
    return "no_change"


def summarize(records: list[dict]) -> dict:
    complete = [record for record in records if record.get("rewrite_jina", {}).get("judge_status") == "ok"]
    def route_summary(route_name: str) -> dict:
        routes = [record[route_name] for record in complete]
        answer_latencies = [float(route.get("latency_ms", {}).get("answer") or 0) for route in routes]
        judge_latencies = [float(route.get("latency_ms", {}).get("judge") or 0) for route in routes]
        def rate(key: str):
            values = [route["evidence_metrics"][key] for route in routes if route["evidence_metrics"][key] is not None]
            return fmean(values) if values else None
        return {
            "judge_score": sum(route["judge"]["score"] for route in routes),
            "correct": sum(route["judge"]["score"] == 1 for route in routes),
            "accuracy": fmean(route["judge"]["score"] for route in routes) if routes else None,
            "evidence_context_hit": rate("evidence_context_hit"),
            "required_number_hit": rate("required_number_hit"),
            "required_period_hit": rate("required_period_hit"),
            "answer_token": sum(_usage_total(route.get("usage")) for route in routes),
            "judge_token": sum(_usage_total(route.get("judge", {}).get("usage")) for route in routes),
            "latency": {
                "answer_total_ms": sum(answer_latencies), "answer_mean_ms": fmean(answer_latencies) if answer_latencies else None,
                "judge_total_ms": sum(judge_latencies), "judge_mean_ms": fmean(judge_latencies) if judge_latencies else None,
            },
        }
    result = {
        "questions": len(records), "completed": len(complete),
        "baseline": route_summary("baseline"),
        "rewrite_jina": route_summary("rewrite_jina"),
        "classifications": dict(Counter(record.get("classification", "pending") for record in records)),
        "answer_calls": sum(record.get("rewrite_jina", {}).get("answer_status") == "ok" for record in records),
        "judge_calls": sum(record.get("rewrite_jina", {}).get("judge_status") == "ok" for record in records),
        "jina_calls": 0, "query_rewrite_calls": 0, "retrieval_calls": 0,
    }
    if complete:
        repaired = sum(record["baseline"]["judge"]["score"] == 0 and record["rewrite_jina"]["judge"]["score"] == 1 for record in complete)
        regressions = sum(record["baseline"]["judge"]["score"] == 1 and record["rewrite_jina"]["judge"]["score"] == 0 for record in complete)
        baseline_errors = sum(record["baseline"]["judge"]["score"] == 0 for record in complete)
        gain_opportunities = [record for record in complete if not record["baseline"]["evidence_metrics"]["evidence_context_hit"] and record["rewrite_jina"]["evidence_metrics"]["evidence_context_hit"] and record["baseline"]["judge"]["score"] == 0]
        converted_opportunities = sum(record["rewrite_jina"]["judge"]["score"] == 1 for record in gain_opportunities)
        result.update({
            "strict_judge_delta": result["rewrite_jina"]["accuracy"] - result["baseline"]["accuracy"],
            "repaired": repaired, "regressions": regressions,
            "repair_rate_among_baseline_errors": repaired / baseline_errors if baseline_errors else None,
            "retrieval_gain_opportunities": len(gain_opportunities),
            "retrieval_gain_converted": converted_opportunities,
            "retrieval_gain_conversion_rate": converted_opportunities / len(gain_opportunities) if gain_opportunities else None,
        })
    return result


def _write(output: Path, records: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary = output / "results.jsonl.tmp"
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    temporary.replace(output / "results.jsonl")
    lines = [
        "# Query Rewrite + Jina Full Answer Shadow v1", "",
        f"- Completed: {summary['completed']}/30",
        f"- Strict Judge A/B: {summary['baseline']['correct']}/30 → {summary['rewrite_jina']['correct']}/30",
        f"- Repaired / regressions: {summary.get('repaired', 0)} / {summary.get('regressions', 0)}", "",
    ]
    for record in records:
        a, b = record["baseline"], record["rewrite_jina"]
        lines.extend([
            f"## {record['question_id']}", "",
            f"**Question:** {record['question']}", "",
            f"**Baseline context摘要:** {a['evidence'][:800].replace(chr(10), ' ')}", "",
            f"**Rewrite context摘要:** {b['evidence'][:800].replace(chr(10), ' ')}", "",
            f"**Baseline answer:** {a['answer']}", "",
            f"**Rewrite answer:** {b.get('answer', 'PENDING')}", "",
            f"**Judge A/B:** {a['judge']['verdict']} / {b.get('judge', {}).get('verdict', 'PENDING')}", "",
            f"**是否修复:** {bool(b.get('judge_status') == 'ok' and a['judge']['score'] == 0 and b['judge']['score'] == 1)}",
            f"**分类:** {record.get('classification', 'pending')}", "",
        ])
    (output / "answers.md").write_text("\n".join(lines), encoding="utf-8")


def _create_frozen_answer_model(profile: dict):
    from langchain.chat_models import init_chat_model
    return init_chat_model(
        model=profile["answer"]["model"], model_provider="openai",
        api_key=os.getenv("ARK_API_KEY"), base_url=os.getenv("BASE_URL"),
        temperature=profile["answer"]["temperature"], stream_usage=True,
        timeout=profile["answer"]["timeout_seconds"], max_retries=0,
        max_completion_tokens=profile["answer"]["max_completion_tokens"],
        extra_body={"thinking": {"type": profile["answer"]["thinking"]}},
    )


def _generate_frozen_answer(model, contract: dict, question: str, evidence: str) -> tuple[str, dict]:
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content=f"Prompt-Version: {contract['CLEAN_BASELINE_PROMPT_VERSION']}\n\n{contract['CLEAN_BASELINE_ANSWER_SYSTEM_PROMPT']}"),
        HumanMessage(content=contract["CLEAN_BASELINE_ANSWER_USER_TEMPLATE"].format(question=question, evidence=evidence)),
    ]
    response = model.invoke(messages)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "".join(
            str(block.get("text") or "") if isinstance(block, dict) and block.get("type") == "text" else str(block)
            for block in content
        )
    usage = getattr(response, "usage_metadata", None) or getattr(response, "response_metadata", {}).get("token_usage") or {}
    return str(content or ""), dict(usage or {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    if not args.allow_paid:
        parser.error("Answer/Judge calls require explicit --allow-paid")

    frozen, _, _, frozen_profile, answer_contract = _validate_inputs()
    profile = read_profile(profile_name="jina_full_baseline_input120_v1")
    if profile != frozen_profile:
        raise ValueError("Current profile differs from frozen A baseline")
    configure(profile)
    os.environ.update(LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false")

    existing = {}
    results_path = args.output_dir / "results.jsonl"
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                existing[record["question_id"]] = record

    records = []
    for item in frozen:
        question_id, row = item["question_id"], item["row"]
        if question_id in existing:
            record = existing[question_id]
        else:
            a = item["baseline"]
            baseline = {
                "evidence": a["evidence"], "citations": a["citations"], "context_documents": a["context_documents"],
                "answer": a["answer"], "usage": a.get("usage") or {}, "answer_status": "ok",
                "judge": a["judge"], "judge_status": "ok", "latency_ms": a.get("latency_ms") or {},
            }
            baseline["evidence_metrics"] = _context_metrics(row, baseline["evidence"], baseline["context_documents"])
            rewrite = item["rewrite_jina"]
            rewrite["evidence_metrics"] = _context_metrics(row, rewrite["evidence"], rewrite["context_documents"])
            record = {
                "question_id": question_id, "group": item["group"], "question": row["question"],
                "reference_answer": row["answer"], "baseline": baseline, "rewrite_jina": rewrite,
            }
        records.append(record)
    _write(args.output_dir, records)

    answer_model = _create_frozen_answer_model(profile)
    for index, record in enumerate(records, 1):
        route = record["rewrite_jina"]
        if route.get("answer_status") == "ok":
            continue
        print(f"[answer {index}/30] {record['question_id']}", flush=True)
        started = time.perf_counter()
        answer, usage = _generate_frozen_answer(answer_model, answer_contract, record["question"], route["evidence"])
        if not str(answer).strip():
            raise RuntimeError("Empty answer; checkpoint preserved")
        route.update(answer=answer, usage=usage, answer_status="ok")
        route.setdefault("latency_ms", {})["answer"] = (time.perf_counter() - started) * 1000
        _write(args.output_dir, records)

    judge_holder = []
    for index, record in enumerate(records, 1):
        route = record["rewrite_jina"]
        if route.get("judge_status") == "ok":
            continue
        print(f"[judge {index}/30] {record['question_id']}", flush=True)
        started = time.perf_counter()
        route["judge"] = judge_answer(
            record["question"], record["reference_answer"], route["answer"], profile["judge"], judge_holder,
        )
        route["judge_status"] = "ok"
        route.setdefault("latency_ms", {})["judge"] = (time.perf_counter() - started) * 1000
        record["classification"] = _classification(record)
        _write(args.output_dir, records)

    _write(args.output_dir, records)
    print(json.dumps(summarize(records), ensure_ascii=False, indent=2))
    print(f"Report: {args.output_dir / 'answers.md'}")


if __name__ == "__main__":
    main()
