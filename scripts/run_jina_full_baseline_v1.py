"""Independent local Jina full baseline; check is offline, paid runs opt-in.

No production routing changes. Explicit recall freezes candidates separately;
run resumes Jina/answer/Judge checkpoints and refuses local rerank fallback.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from scripts.jina_full_baseline_v1 import (
    read_profile, digest, validate_candidates, rrf_merge, build_context, cached_jina, write_state, export_reports,
)
from scripts.evaluate_reranker_shadow_v1 import metrics
from scripts.shadow_rerankers_v1 import JinaReranker, validate_order

DATASET = ROOT / "data/financebench_top40_100_langsmith_with_evidence.csv"


def sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure(profile):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    from runtime_profile import apply_runtime_profile
    apply_runtime_profile("clean_baseline")
    os.environ.update({"LANGSMITH_TRACING": "false", "LANGSMITH_TRACING_V2": "false", "LANGCHAIN_TRACING_V2": "false",
        "LOCAL_RERANK_ENABLED": "false", "MODEL": profile["answer"]["model"],
        "ANSWER_TEMPERATURE": str(profile["answer"]["temperature"]),
        "ANSWER_MAX_COMPLETION_TOKENS": str(profile["answer"]["max_completion_tokens"]),
        "ANSWER_THINKING_MODE": profile["answer"]["thinking"], "ANSWER_TIMEOUT_SECONDS": str(profile["answer"]["timeout_seconds"]),
        "ANSWER_MAX_RETRIES": "0", "JUDGE_MODEL": profile["judge"]["model"],
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})


def load_rows(scope, limit):
    with DATASET.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 100 or len({r["financebench_id"] for r in rows}) != 100:
        raise ValueError("Expected the fixed 100-question dataset")
    if scope == "diagnostic30":
        fixture = json.loads((ROOT / "tests/fixtures/rag_core_v3_diagnostic_ids.json").read_text(encoding="utf-8"))
        ids = [v if isinstance(v, str) else v["financebench_id"] for group in ("candidate_miss10", "selection_loss10", "correct_regression10") for v in fixture[group]]
        by_id = {r["financebench_id"]: r for r in rows}
        rows = [by_id[i] for i in ids]
    return rows[:limit] if limit else rows


def recall(rows, path, profile):
    manifest = {"dataset_sha256": sha(DATASET), "retrieval": profile["retrieval"],
                "milvus_implementation": sha(ROOT / "backend/milvus_client.py"),
                "embedding_implementation": sha(ROOT / "backend/embedding.py"),
                "collection": os.getenv("MILVUS_COLLECTION", "embeddings_collection"),
                "embedding_config": {k: os.getenv(k) for k in ("EMBEDDING_MODEL", "EMBEDDING_DEVICE", "EMBEDDING_REVISION", "EMBEDDING_LOCAL_ONLY", "EMBEDDING_USE_FP16", "MILVUS_SEARCH_EF", "MILVUS_HOST", "MILVUS_PORT")},
                "question_ids": [r["financebench_id"] for r in rows]}
    payload = {"schema": "jina_full_baseline_recall_v1", "manifest": manifest,
               "retrieval": profile["retrieval"], "records": []}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("manifest") != manifest:
            raise ValueError("Recall checkpoint configuration drift")
    saved = {r["question_id"]: r for r in payload["records"]}
    manager = embedding = None
    for i, row in enumerate(rows, 1):
        key = row["financebench_id"]
        if key in saved:
            validate_candidates(saved[key], row["question"], profile["retrieval"]["rrf_top_k"])
            continue
        if manager is None:
            import psutil
            if psutil.virtual_memory().available < 4 * 1024**3:
                raise RuntimeError("At least 4 GiB available RAM required to initialize embedding; close unused apps manually")
            from milvus_client import MilvusManager
            from embedding import embedding_service
            manager, embedding = MilvusManager(), embedding_service
            if not manager.uses_builtin_bm25:
                raise RuntimeError("Native Milvus BM25 required")
        started = time.perf_counter()
        dense = embedding.get_embeddings([row["question"]])[0]
        dense_rows = manager.dense_retrieve(dense, top_k=profile["retrieval"]["dense_top_k"],
            filter_expr=profile["retrieval"]["filter"])
        bm25_rows = manager.bm25_retrieve(row["question"], top_k=profile["retrieval"]["bm25_top_k"],
            filter_expr=profile["retrieval"]["filter"])
        chunks = rrf_merge(dense_rows, bm25_rows, top_k=profile["retrieval"]["rrf_top_k"],
            rank_constant=profile["retrieval"]["rrf_rank_constant"])
        record = {"question_id": key, "question": row["question"], "chunks": chunks,
                  "candidate_sha256": digest(chunks), "retrieval_ms": (time.perf_counter() - started) * 1000}
        validate_candidates(record, row["question"], profile["retrieval"]["rrf_top_k"])
        payload["records"].append(record)
        write_state(path, payload)
        print(f"[recall {i}/{len(rows)}] frozen {len(chunks)} chunks", flush=True)


def run_question(record, source, profile, persist, jina, generate):
    """Injectable callbacks make no-network/full-flow contract tests possible."""
    if record["candidate_sha256"] != source["candidate_sha256"] or record["question"] != source["question"]:
        raise ValueError("Answer checkpoint candidate/question drift")
    if record.get("answer_status") == "ok":
        return
    input_chunks = source["chunks"][:profile["reranker"]["input_k"]]
    if record.get("jina", {}).get("status") != "ok":
        start = time.perf_counter()
        ranked, trace = jina(source["question"], [c["text"] for c in input_chunks])
        record["jina"] = {"status": "ok", "ranked": validate_order(ranked, len(input_chunks)), "trace": trace}
        record["latency_ms"]["jina"] = (time.perf_counter() - start) * 1000
        record["jina_cache_hit"] = False
        persist()
    ranked = validate_order(record["jina"]["ranked"], len(input_chunks))
    ordered = [input_chunks[r["index"]] for r in ranked]
    evidence, citations, included = build_context(ordered, profile["context"])
    if not evidence:
        raise RuntimeError("No context: not an answerable run")
    record.update(evidence=evidence, citations=citations, context_documents=included)
    persist()
    start = time.perf_counter()
    answer, usage = generate(source["question"], evidence)
    if not str(answer).strip():
        raise RuntimeError("Empty model answer; preserve Jina checkpoint and stop")
    record.update(answer=answer, usage=usage, answer_status="ok", evidence_status="retrieved_not_verified",
                  execution_mode="static", rag_trace={"profile": profile["name"], "rerank_provider": "jina",
                    "skills": [], "planner": False, "answer_context_chars": len(evidence)})
    record["latency_ms"]["answer"] = (time.perf_counter() - start) * 1000
    record.pop("last_error", None)
    persist()


def judge_answer(question, reference, answer, config, model_holder):
    from scripts.financebench_judge_common import JUDGE_PROMPT, _parse_verdict, _content_text
    if not model_holder:
        from langchain.chat_models import init_chat_model
        model_holder.append(init_chat_model(model=config["model"], model_provider="openai",
            api_key=os.getenv("JUDGE_API_KEY") or os.getenv("ARK_API_KEY"),
            base_url=os.getenv("JUDGE_BASE_URL") or os.getenv("BASE_URL"), temperature=0,
            max_completion_tokens=config["max_completion_tokens"], timeout=config["timeout_seconds"], max_retries=0,
            extra_body={"thinking": {"type": "disabled"}}))
    response = model_holder[0].invoke(JUDGE_PROMPT.format(question=question, reference=reference, answer=answer))
    result = _parse_verdict(_content_text(response.content))
    if result["verdict"] not in {"correct", "incorrect"} or result["score"] != int(result["verdict"] == "correct"):
        raise RuntimeError("Invalid Judge output; not counted as incorrect, retry explicitly")
    return {**result, "judge_model": config["model"], "usage": dict(getattr(response, "usage_metadata", {}) or {})}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["jina_full_baseline_v1", "jina_full_baseline_input120_v1", "jina_full_baseline_input80_v1"], default="jina_full_baseline_v1")
    parser.add_argument("--stage", choices=["check", "recall", "run", "report"], default="check")
    parser.add_argument("--scope", choices=["diagnostic30", "all100"], default="diagnostic30")
    parser.add_argument("--limit", type=int, default=2, help="0 = all questions in scope")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/jina_full_baseline_v1_smoke")
    parser.add_argument("--allow-retrieval", action="store_true")
    parser.add_argument("--allow-paid", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    profile = read_profile(profile_name=args.profile)
    configure(profile)
    if args.stage == "check":
        print(json.dumps({"profile": profile, "credentials_configured": {
            "jina": bool(os.getenv("RERANK_API_KEY") or os.getenv("JINA_API_KEY")),
            "answer": bool(os.getenv("ARK_API_KEY")), "judge": bool(os.getenv("JUDGE_API_KEY") or os.getenv("ARK_API_KEY"))},
            "answer_endpoint_configured": bool(os.getenv("BASE_URL")), "network_calls": 0}, ensure_ascii=True, indent=2))
        return
    rows = load_rows(args.scope, args.limit)
    references = {r["financebench_id"]: r for r in load_rows("all100", 0)}
    if args.stage == "report":
        state = json.loads((args.output_dir / "state.json").read_text(encoding="utf-8"))
        export_reports(args.output_dir, state, references)
        return
    snapshot = args.snapshot or args.output_dir / "recall.json"
    if args.stage == "recall":
        if not args.allow_retrieval:
            parser.error("Recall requires --allow-retrieval; no paid model calls")
        recall(rows, snapshot, profile)
        return
    if not args.allow_paid:
        parser.error("Run requires --allow-paid for Jina/answer/strict Judge; use --stage check first")
    if not snapshot.exists():
        parser.error("Missing snapshot; run recall first or pass an existing verified RRF snapshot")
    frozen = json.loads(snapshot.read_text(encoding="utf-8"))
    if frozen.get("schema") not in {"rrf_top120_shadow_v1", "jina_full_baseline_recall_v1"}:
        raise ValueError("Expected a verified RRF snapshot")
    frozen_retrieval = frozen.get("retrieval", {})
    legacy_snapshot = frozen_retrieval.get("method") == "MilvusManager.hybrid_retrieve"
    if legacy_snapshot:
        if (frozen_retrieval.get("top_k") != 120 or frozen_retrieval.get("rrf_k") != 60
                or profile["retrieval"]["rrf_top_k"] != 120
                or profile["retrieval"]["dense_top_k"] != frozen_retrieval.get("route_search_limit", 240)
                or profile["retrieval"]["bm25_top_k"] != frozen_retrieval.get("route_search_limit", 240)):
            raise ValueError("Legacy frozen Top120 retrieval parameters mismatch")
    elif frozen_retrieval != profile["retrieval"]:
        raise ValueError("Frozen retrieval parameters mismatch")
    if frozen.get("schema") == "jina_full_baseline_recall_v1" and frozen["manifest"]["dataset_sha256"] != sha(DATASET):
        raise ValueError("Recall dataset fingerprint mismatch")
    sources = {r["question_id"]: r for r in frozen["records"]}
    if len(sources) != len(frozen["records"]):
        raise ValueError("Duplicate snapshot question")
    for row in rows:
        validate_candidates(sources[row["financebench_id"]], row["question"], profile["retrieval"]["rrf_top_k"])
    manifest = {"profile": profile, "snapshot_sha256": sha(snapshot), "dataset_sha256": sha(DATASET),
        "question_ids": [r["financebench_id"] for r in rows],
        "code_sha256": {p: sha(ROOT / p) for p in ("backend/prompts.py", "backend/answer_generator.py", "backend/runtime_profile.py",
            "scripts/jina_full_baseline_v1.py", "scripts/run_jina_full_baseline_v1.py", "scripts/shadow_rerankers_v1.py", "scripts/financebench_judge_common.py")},
        "answer_base_url": os.getenv("BASE_URL"), "judge_base_url": os.getenv("JUDGE_BASE_URL") or os.getenv("BASE_URL")}
    state_path = args.output_dir / "state.json"
    state = {"manifest": manifest, "records": [{"financebench_id": row["financebench_id"], "question": row["question"],
             "candidate_sha256": sources[row["financebench_id"]]["candidate_sha256"], "latency_ms": {}} for row in rows]}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["manifest"] != manifest:
            raise ValueError("Run configuration/source drift; use a new output directory")
        if len(state.get("records", [])) != len(rows) or len({r.get("financebench_id") for r in state["records"]}) != len(rows):
            raise ValueError("Run checkpoint question set is incomplete or duplicated")
        expected = {r["financebench_id"]: r for r in rows}
        for item in state["records"]:
            key = item["financebench_id"]
            if key not in expected or item.get("question") != expected[key]["question"] or item.get("candidate_sha256") != sources[key]["candidate_sha256"]:
                raise ValueError("Run checkpoint question/candidate drift")
    cache_path = ROOT / "reports/reranker_shadow_v1.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    remote = None
    def jina(question, texts):
        nonlocal remote
        if remote is None:
            remote = JinaReranker(os.getenv("RERANK_API_KEY") or os.getenv("JINA_API_KEY"), profile["reranker"]["model"],
                                  profile["reranker"]["endpoint"], interval=profile["reranker"]["interval_seconds"])
        return remote.rank(question, texts)
    def generate(question, evidence):
        from answer_generator import generate_answer
        return generate_answer(question, evidence, [], "", profile["answer"]["profile"])
    def persist():
        write_state(state_path, state)
        export_reports(args.output_dir, state, references)
    persist()
    stage = "jina_or_answer"
    try:
        for i, record in enumerate(state["records"], 1):
            source = sources[record["financebench_id"]]
            if record.get("jina", {}).get("status") != "ok":
                cached = cached_jina(source, cache, profile["reranker"]["model"], profile["reranker"]["input_k"])
                if cached:
                    record["jina"] = {"status": "ok", "ranked": validate_order(cached["ranked"], profile["reranker"]["input_k"]), "trace": cached["trace"]}
                    record["jina_cache_hit"] = True
                    persist()
            print(f"[answer {i}/{len(rows)}] {record['financebench_id']} cached_jina={record.get('jina_cache_hit', False)}", flush=True)
            run_question(record, source, profile, persist, jina, generate)
            input_chunks = source["chunks"][:profile["reranker"]["input_k"]]
            ordered = [input_chunks[r["index"]] for r in record["jina"]["ranked"]]
            record["retrieval_metrics"] = {"ranked": metrics(references[record["financebench_id"]], ordered),
                "actual_context": metrics(references[record["financebench_id"]], record["context_documents"])}
            persist()
        # Judge only after every answer completed. Failed Judge never reruns answers.
        stage, judge_model = "judge", []
        for i, record in enumerate(state["records"], 1):
            if record.get("judge_status") == "ok":
                continue
            print(f"[judge {i}/{len(rows)}] {record['financebench_id']}", flush=True)
            start = time.perf_counter()
            reference = references[record["financebench_id"]]
            record["judge"] = judge_answer(record["question"], reference["answer"], record["answer"], profile["judge"], judge_model)
            record["judge_status"] = "ok"
            record["latency_ms"]["judge"] = (time.perf_counter() - start) * 1000
            record.pop("last_error", None)
            persist()
    except Exception as exc:
        # API exceptions can contain credentials/request bodies: don't persist them.
        error = {"stage": stage, "type": type(exc).__name__, "message": str(exc) if isinstance(exc, RuntimeError) and str(exc).startswith("Jina ") else "Stage failed; check service/configuration and explicitly resume"}
        record["last_error"] = error
        record.setdefault("error_history", []).append(error)
        persist()
        print(json.dumps(error), flush=True)
        raise SystemExit(2) from None
    state["complete"] = True
    persist()
    print(f"Completed {len(rows)} questions. Report: {args.output_dir / 'answers.md'}", flush=True)


if __name__ == "__main__":
    main()
