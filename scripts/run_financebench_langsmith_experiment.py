"""Run a controlled, traced FinanceBench static baseline in LangSmith."""

import argparse
import csv
import faulthandler
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.run_helpers import get_current_run_tree, traceable


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
DATA_PATH = ROOT / "data" / "financebench_top40_100_langsmith_with_evidence.csv"
load_dotenv(ROOT / ".env", override=True)


def _development_ids(rows: list[dict[str, str]], size: int = 20) -> set[str]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row.get("question_type") or "unknown", []).append(row)
    selected: list[dict[str, str]] = []
    while len(selected) < size:
        added = False
        for _, group in sorted(groups.items()):
            group.sort(key=lambda item: item.get("financebench_id") or "")
            if group:
                selected.append(group.pop(0))
                added = True
                if len(selected) == size:
                    break
        if not added:
            break
    return {row.get("financebench_id") or "" for row in selected}


def _configure_static_baseline(max_completion_tokens: int, thinking_mode: str, diagnose: bool) -> None:
    """Freeze the baseline before importing any backend module."""
    os.environ.update(
        {
            "RAG_QUERY_PLANNER_ENABLED": "false",
            "FINANCE_RAG_ENABLE_STEP_BACK": "false",
            "RAG_EXECUTION_MODE": "static",
            "RAG_PROFILE": "finance",
            "FINANCE_RAG_CANDIDATE_K": "40",
            "FINANCE_RAG_FINAL_TOP_K": "5",
            "RAG_PAGE_FIRST_ENABLED": "true",
            "RAG_PAGE_NEIGHBOR_WINDOW": "0",
            "RAG_ANCHOR_GUARD_ENABLED": "false",
            "RAG_COVER_FILTER_ENABLED": "false",
            "FINANCE_RAG_ENABLE_PAGE_MERGE": "false",
            "FINANCE_RAG_ADJACENT_PAGE_WINDOW": "0",
            "FINANCE_RAG_ADJACENT_CHUNK_WINDOW": "0",
            "AUTO_MERGE_ENABLED": "false",
            "TABLE_AWARE_RETRIEVAL": "off",
            "RAG_EVIDENCE_GROUPING_ENABLED": "false",
            "RERANK_MODEL": "",
            "RERANK_BINDING_HOST": "",
            "RERANK_API_KEY": "",
            "LOCAL_RERANK_ENABLED": "false",
            "ANSWER_MAX_COMPLETION_TOKENS": str(max_completion_tokens),
            "ANSWER_THINKING_MODE": thinking_mode,
            "ANSWER_TEMPERATURE": "0",
            "RAG_RETRIEVAL_DEBUG": "true" if diagnose else "false",
        }
    )


def _normalized_document_name(value: object) -> str:
    name = Path(str(value or "")).stem
    return name.strip().casefold()


def citation_document_hit(run, example) -> dict:
    outputs = getattr(run, "outputs", None) or {}
    metadata = getattr(example, "metadata", None) or {}
    cited = {
        _normalized_document_name(item.get("filename"))
        for item in (outputs.get("citations") or [])
        if isinstance(item, dict)
    }
    gold = {
        _normalized_document_name(item.get("doc_name"))
        for item in (metadata.get("gold_evidence") or [])
        if isinstance(item, dict)
    }
    matched = sorted(cited & gold)
    return {
        "key": "citation_document_hit",
        "score": bool(matched),
        "comment": f"matched documents: {', '.join(matched) if matched else 'none'}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a formal EvidenceRAG FinanceBench static experiment")
    parser.add_argument("--dataset-name", default="evidencerag_financebench_all100_v1")
    parser.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the whole selected split.")
    parser.add_argument("--experiment-prefix", default="evidencerag-finance-static")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--thinking", choices=("disabled", "auto", "enabled"), default="disabled")
    parser.add_argument("--diagnose", action="store_true", help="Print retrieval and generation stage timing.")
    parser.add_argument("--slow-question-seconds", type=int, default=90, help="Dump a stack trace after this many seconds per question (0 disables it).")
    parser.add_argument("--output", type=Path, help="Write each completed answer, citations, and trace to JSONL.")
    parser.add_argument("--resume", action="store_true", help="Skip completed FinanceBench IDs in --output.")
    args = parser.parse_args()

    _configure_static_baseline(args.max_completion_tokens, args.thinking, args.diagnose)
    with DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    id_by_question = {row.get("question", ""): row.get("financebench_id", "") for row in source_rows}
    dev_ids = _development_ids(source_rows)
    client = Client()
    try:
        examples = list(client.list_examples(dataset_name=args.dataset_name))
    except Exception as exc:
        raise SystemExit(
            "LangSmith is unavailable. Check access to api.smith.langchain.com:443 and your proxy settings."
        ) from exc
    if args.split != "all":
        use_dev = args.split == "dev"
        examples = [
            example
            for example in examples
            if ((getattr(example, "metadata", None) or {}).get("financebench_id") in dev_ids) == use_dev
        ]
    if args.limit > 0:
        examples = examples[: args.limit]
    if args.resume:
        if not args.output:
            raise SystemExit("--resume requires --output.")
        completed_ids = set()
        if args.output.exists():
            for line in args.output.read_text(encoding="utf-8").splitlines():
                try:
                    completed_ids.add(json.loads(line).get("financebench_id"))
                except json.JSONDecodeError:
                    continue
            examples = [
                example
                for example in examples
                if (getattr(example, "metadata", None) or {}).get("financebench_id") not in completed_ids
            ]
    if not examples:
        raise SystemExit("No LangSmith examples selected. Check --dataset-name and --split.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("a" if args.resume else "w", encoding="utf-8")
    else:
        output_handle = None

    # Do not load the local embedding model until the remote experiment service is reachable.
    sys.path.insert(0, str(BACKEND))
    from answer_generator import generate_answer
    from rag_orchestrator import prepare_rag_response

    @traceable(name="EvidenceRAG.retrieve", run_type="retriever")
    def retrieve(question: str) -> dict:
        return prepare_rag_response(question, profile="finance", mode="static")

    @traceable(name="EvidenceRAG.generate", run_type="llm")
    def generate(question: str, evidence: str) -> tuple[str, dict]:
        return generate_answer(question, evidence, [])

    @traceable(name="EvidenceRAG.financebench_static", run_type="chain")
    def target(inputs: dict) -> dict:
        question = str(inputs.get("question") or "")
        financebench_id = id_by_question.get(question, "")
        started = time.perf_counter()
        if args.diagnose:
            print(f"[question] {financebench_id} retrieve starting", flush=True)
        if args.slow_question_seconds > 0:
            faulthandler.dump_traceback_later(args.slow_question_seconds, repeat=False)
        try:
            prepared = retrieve(question)
        finally:
            if args.slow_question_seconds > 0:
                faulthandler.cancel_dump_traceback_later()
        retrieval_seconds = time.perf_counter() - started
        if args.diagnose:
            print(f"[question] {financebench_id} retrieve finished in {retrieval_seconds:.2f}s", flush=True)
        if prepared["evidence_status"] == "insufficient":
            answer, usage = "未检索到足够证据，无法基于当前知识库可靠回答。", {}
        else:
            if args.diagnose:
                print(f"[question] {financebench_id} generate starting", flush=True)
            started = time.perf_counter()
            if args.slow_question_seconds > 0:
                faulthandler.dump_traceback_later(args.slow_question_seconds, repeat=False)
            try:
                answer, usage = generate(question, prepared["evidence"])
            finally:
                if args.slow_question_seconds > 0:
                    faulthandler.cancel_dump_traceback_later()
            if args.diagnose:
                print(f"[question] {financebench_id} generate finished in {time.perf_counter() - started:.2f}s", flush=True)
        run_tree = get_current_run_tree()
        result = {
            "financebench_id": financebench_id,
            "question": question,
            "answer": answer,
            "citations": prepared["citations"],
            "evidence_status": prepared["evidence_status"],
            "execution_mode": prepared["execution_mode"],
            "route_reason": prepared["route_reason"],
            "rag_trace": prepared["rag_trace"],
            "usage": usage,
            "application_trace_id": prepared["trace_id"],
            "langsmith_trace_id": str(run_tree.id) if run_tree else "",
        }
        if output_handle:
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_handle.flush()
        return result

    metadata = {
        "profile": "finance",
        "execution_mode": "static",
        "query_planner": False,
        "step_back": False,
        "rerank": False,
        "thinking": args.thinking,
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": 0,
        "candidate_k": 40,
        "final_evidence_k": 5,
        "model": os.getenv("MODEL", ""),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    print(
        f"[setup] dataset={args.dataset_name} split={args.split} examples={len(examples)} "
        f"planner=false thinking={args.thinking}",
        flush=True,
    )
    try:
        results = evaluate(
            target,
            data=examples,
            evaluators=[citation_document_hit],
            experiment_prefix=args.experiment_prefix,
            description="Controlled EvidenceRAG static FinanceBench baseline.",
            metadata=metadata,
            max_concurrency=max(1, args.max_concurrency),
            client=client,
            blocking=True,
        )
        print(f"Experiment: {results.experiment_name}", flush=True)
    finally:
        if output_handle:
            output_handle.close()


if __name__ == "__main__":
    main()
