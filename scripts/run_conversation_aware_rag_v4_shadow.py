"""Run Conversation-aware RAG v4 end-to-end shadow on 30 financial dialogues.

This runner is intentionally isolated from the production chat route.  It uses
Conversation Understanding v3, a deterministic execution policy, local
Dense+BM25+RRF retrieval without any reranker, and the existing baseline answer
generator.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend.conversation_understanding import conversation_memory_enabled  # noqa: E402
from backend.conversation_understanding_v3 import (  # noqa: E402
    create_understanding_model_v3,
    understand_conversation_v3,
)
from backend.memory.conversation_policy import build_conversation_policy  # noqa: E402
from backend.memory.conversation_state import ConversationState  # noqa: E402


CASES = ROOT / "tests/fixtures/conversation_aware_rag_v4_cases.json"
FROZEN_FIRST_TURNS = ROOT / "reports/final100/answers.jsonl"
OUTPUT = ROOT / "reports/conversation_aware_rag_v4_shadow"
MAX_CONTEXT_CHARS = 28000
EXPECTED_CATEGORIES = {
    "follow_up", "period_modification", "entity_modification", "challenge",
    "citation_question", "calculation_explanation",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_cases(path: Path = CASES) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(case["category"] for case in cases)
    if len(cases) != 30 or len({case["id"] for case in cases}) != 30:
        raise ValueError("v4 requires exactly 30 unique scenarios")
    if set(counts) != EXPECTED_CATEGORIES or any(counts[name] != 5 for name in EXPECTED_CATEGORIES):
        raise ValueError("v4 requires five scenarios in each of the six categories")
    return cases


def load_frozen_first_turns(path: Path = FROZEN_FIRST_TURNS) -> dict[str, dict]:
    records = {str(row["question_id"]): row for row in load_jsonl(path)}
    if not records:
        raise FileNotFoundError(f"Frozen first-turn answers not found: {path}")
    return records


def configure_shadow_runtime() -> None:
    """Disable every reranker before rag_utils is imported."""
    os.environ.update({
        "ENABLE_CONVERSATION_MEMORY": "false",
        "RAG_QUERY_PLANNER_ENABLED": "false",
        "RAG_FIELD_AWARE_ENABLED": "false",
        "RAG_PAGE_FIRST_ENABLED": "false",
        "RAG_PAGE_LEVEL_FUSION_ENABLED": "false",
        "RAG_SUPPLEMENTAL_SEARCH_ENABLED": "false",
        "FINANCE_RAG_ENABLE_STEP_BACK": "false",
        "RERANK_MODEL": "",
        "RERANK_BINDING_HOST": "",
        "RERANK_API_KEY": "",
        "LOCAL_RERANK_ENABLED": "false",
        "FINANCE_RAG_CANDIDATE_K": "60",
        "MODEL": "deepseek-v4-flash-ga-260731",
        "ANSWER_PROMPT_MODE": "baseline",
    })


def retrieve_without_reranker(query: str, *, candidate_k: int = 60, context_k: int = 12) -> dict:
    """Run local candidate retrieval and stop before production finalization/rerank."""
    from backend.rag_utils import retrieve_candidate_documents

    started = time.perf_counter()
    result = retrieve_candidate_documents(query, candidate_k=candidate_k)
    candidates = list(result.get("docs") or [])
    selected = candidates[:context_k]
    blocks = []
    for index, doc in enumerate(selected, 1):
        text = str(doc.get("text") or doc.get("page_text") or "").strip()
        blocks.append(
            f"[{index}] Source: {doc.get('filename') or 'Unknown'} | "
            f"Page: {doc.get('page_number', 'N/A')}\n{text}"
        )
    evidence = "\n\n---\n\n".join(blocks)[:MAX_CONTEXT_CHARS]
    return {
        "evidence": evidence,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected": [
            {
                "chunk_id": doc.get("chunk_id"), "filename": doc.get("filename"),
                "page_number": doc.get("page_number"), "score": doc.get("score"),
            }
            for doc in selected
        ],
        "retrieval_mode": (result.get("meta") or {}).get("retrieval_mode"),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "reranker": "disabled",
    }


def build_shadow_evidence(previous: str, retrieved: str, *, reuse_previous: bool) -> tuple[str, dict]:
    previous = str(previous or "")
    retrieved = str(retrieved or "")
    if reuse_previous and retrieved:
        previous_budget = MAX_CONTEXT_CHARS // 2
        new_budget = MAX_CONTEXT_CHARS - previous_budget
        evidence = (
            "Previous-turn evidence:\n" + previous[:previous_budget]
            + "\n\n---\n\nFresh verification evidence:\n" + retrieved[:new_budget]
        )[:MAX_CONTEXT_CHARS]
        source = "previous_and_retrieved"
    elif reuse_previous:
        evidence, source = previous[:MAX_CONTEXT_CHARS], "previous"
    else:
        evidence, source = retrieved[:MAX_CONTEXT_CHARS], "retrieved"
    return evidence, {
        "source": source,
        "previous_evidence_available": bool(previous),
        "previous_evidence_reused": reuse_previous and bool(previous),
        "previous_chars": min(len(previous), MAX_CONTEXT_CHARS) if reuse_previous else 0,
        "retrieved_chars": min(len(retrieved), MAX_CONTEXT_CHARS),
        "final_chars": len(evidence),
    }


def _history_messages(state: ConversationState) -> list[Any]:
    from langchain_core.messages import AIMessage, HumanMessage

    mapping = {"user": HumanMessage, "assistant": AIMessage}
    return [mapping[turn.role](content=turn.content) for turn in state.recent(12)]


def answer_with_baseline(question: str, evidence: str, state: ConversationState) -> tuple[str, dict]:
    from backend.answer_generator import generate_answer

    return generate_answer(
        question,
        evidence,
        history=_history_messages(state),
        profile="rag_core_v3",
        prompt_mode="baseline",
    )


def continuity_checks(category: str, answer: str) -> dict:
    lowered = answer.casefold()
    if category == "citation_question":
        category_pass = any(token in lowered for token in ("page", ".pdf", "source", "页"))
    elif category == "calculation_explanation":
        category_pass = any(character.isdigit() for character in answer) and any(
            token in lowered for token in ("/", "=", "formula", "calculate", "计算")
        )
    else:
        category_pass = bool(answer.strip())
    return {"non_empty_answer": bool(answer.strip()), "category_continuity_proxy": category_pass}


def execute_case(
    case: dict,
    first_turn: dict,
    understanding_model: Any,
    *,
    retrieve_fn: Callable[..., dict] = retrieve_without_reranker,
    answer_fn: Callable[..., tuple[str, dict]] = answer_with_baseline,
) -> dict:
    state = ConversationState(case["id"])
    state.append("user", str(first_turn["question"]))
    state.append("assistant", str(first_turn.get("answer") or "No prior answer was available."))
    previous_evidence = str(first_turn.get("evidence") or "")

    understanding_started = time.perf_counter()
    decision, understanding_trace = understand_conversation_v3(case["user_input"], state, understanding_model)
    understanding_ms = round((time.perf_counter() - understanding_started) * 1000, 2)
    policy = build_conversation_policy(
        case["user_input"], decision, has_previous_evidence=bool(previous_evidence),
    )

    retrieval = {"called": False, "reranker": "disabled", "latency_ms": 0, "evidence": ""}
    if policy.retrieval:
        retrieval = {"called": True, **retrieve_fn(policy.retrieval_query)}
    evidence, memory_trace = build_shadow_evidence(
        previous_evidence,
        str(retrieval.get("evidence") or ""),
        reuse_previous=policy.reuse_previous_evidence,
    )

    answer_started = time.perf_counter()
    answer, answer_usage = answer_fn(case["user_input"], evidence, state)
    answer_ms = round((time.perf_counter() - answer_started) * 1000, 2)
    expected = case["expected"]
    policy_checks = {
        key: policy.to_dict()[key] == expected[key]
        for key in ("retrieval", "rewrite", "reuse_previous_evidence", "response_mode")
    }
    return {
        "id": case["id"], "category": case["category"], "base_question_id": case["base_question_id"],
        "base_question": first_turn["question"], "previous_answer": first_turn.get("answer"),
        "user_input": case["user_input"], "understanding": decision.to_dict(),
        "understanding_trace": understanding_trace,
        "policy": policy.to_dict(), "expected_policy": expected, "policy_checks": policy_checks,
        "query_trace": {
            "user_input": case["user_input"], "standalone_query": decision.standalone_query,
            "retrieval_query": policy.retrieval_query,
            "rewrite_applied": policy.rewrite,
            "query_changed": policy.retrieval_query.casefold() != case["user_input"].casefold(),
        },
        "retrieval": {key: value for key, value in retrieval.items() if key != "evidence"},
        "memory": memory_trace, "answer": answer, "answer_usage": answer_usage,
        "continuity": continuity_checks(case["category"], answer),
        "latency_ms": {
            "understanding": understanding_ms,
            "retrieval": retrieval.get("latency_ms", 0),
            "answer": answer_ms,
            "total": round(understanding_ms + float(retrieval.get("latency_ms") or 0) + answer_ms, 2),
        },
    }


def summarize(cases: list[dict], records: list[dict]) -> dict:
    successful = [record for record in records if record.get("status") == "ok"]
    by_category = {}
    for category in sorted(EXPECTED_CATEGORIES):
        subset = [record for record in successful if record["category"] == category]
        by_category[category] = {
            "completed": len(subset),
            "policy_accuracy": sum(all(record["policy_checks"].values()) for record in subset) / max(1, len(subset)),
            "retrieval_rate": sum(record["policy"]["retrieval"] for record in subset) / max(1, len(subset)),
            "memory_reuse_rate": sum(record["memory"]["previous_evidence_reused"] for record in subset) / max(1, len(subset)),
            "continuity_proxy": sum(record["continuity"]["category_continuity_proxy"] for record in subset) / max(1, len(subset)),
        }
    usages = [usage for record in successful for usage in (record.get("understanding_trace", {}).get("usage", {}), record.get("answer_usage", {}))]
    return {
        "cases": len(cases), "completed": len(successful), "failed": len(records) - len(successful),
        "policy_accuracy": sum(all(record["policy_checks"].values()) for record in successful) / max(1, len(successful)),
        "retrieval_calls": sum(record["retrieval"]["called"] for record in successful),
        "jina_calls": 0,
        "memory_reuse_cases": sum(record["memory"]["previous_evidence_reused"] for record in successful),
        "rewritten_queries": sum(record["query_trace"]["rewrite_applied"] for record in successful),
        "changed_queries": sum(record["query_trace"]["query_changed"] for record in successful),
        "answer_continuity_proxy": sum(record["continuity"]["category_continuity_proxy"] for record in successful) / max(1, len(successful)),
        "token_usage": {key: sum(int(usage.get(key) or 0) for usage in usages) for key in ("input_tokens", "output_tokens", "total_tokens")},
        "mean_latency_ms": sum(record["latency_ms"]["total"] for record in successful) / max(1, len(successful)),
        "by_category": by_category,
        "production_enabled": conversation_memory_enabled(),
    }


def write_report(output: Path, cases: list[dict], records: list[dict]) -> dict:
    summary = summarize(cases, records)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Conversation-aware RAG v4 End-to-End Shadow", "", "## Summary", "",
        f"- Completed: **{summary['completed']}/{summary['cases']}**; failures: **{summary['failed']}**",
        f"- Deterministic policy accuracy: **{summary['policy_accuracy']:.1%}**",
        f"- Answer continuity proxy: **{summary['answer_continuity_proxy']:.1%}**",
        f"- Retrieval calls: **{summary['retrieval_calls']}**; Jina calls: **0**",
        f"- Memory reuse: **{summary['memory_reuse_cases']}**; rewrite decisions: **{summary['rewritten_queries']}**",
        f"- Total model tokens: **{summary['token_usage']['total_tokens']:,}**",
        f"- Mean end-to-end latency: **{summary['mean_latency_ms']:.0f} ms/case**", "",
        "`ENABLE_CONVERSATION_MEMORY` remained false. This is a script-only shadow and does not alter the production chat route.", "",
        "## By category", "", "| Category | Done | Policy | Retrieval | Memory reuse | Continuity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["by_category"].items():
        lines.append(
            f"| {name} | {item['completed']} | {item['policy_accuracy']:.1%} | "
            f"{item['retrieval_rate']:.1%} | {item['memory_reuse_rate']:.1%} | {item['continuity_proxy']:.1%} |"
        )
    lines += ["", "## Per case", ""]
    for record in records:
        if record.get("status") != "ok":
            lines += [f"### {record['id']} — error", "", f"`{record.get('error')}`", ""]
            continue
        query = record["query_trace"]
        lines += [
            f"### {record['id']} — {record['category']}", "",
            f"- User turn: {record['user_input']}",
            f"- Standalone query: {query['standalone_query']}",
            f"- Retrieval query: {query['retrieval_query']}",
            f"- Decision: retrieval={record['policy']['retrieval']}, rewrite={record['policy']['rewrite']}, mode={record['policy']['response_mode']}",
            f"- Memory: {record['memory']['source']}; reused={record['memory']['previous_evidence_reused']}; context={record['memory']['final_chars']} chars",
            f"- Answer: {record['answer']}", "",
        ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def run(
    cases: list[dict], output: Path, *, limit: int = 0, retry_errors: bool = False,
    rerun_ids: set[str] | None = None,
) -> list[dict]:
    if conversation_memory_enabled():
        raise ValueError("ENABLE_CONVERSATION_MEMORY must remain false for this shadow")
    first_turns = load_frozen_first_turns()
    missing = sorted({case["base_question_id"] for case in cases} - set(first_turns))
    if missing:
        raise ValueError(f"Missing frozen first turns: {missing}")
    selected = [case for case in cases if case["id"] in rerun_ids] if rerun_ids else (cases[:limit] if limit else cases)
    if rerun_ids and len(selected) != len(rerun_ids):
        raise ValueError(f"Unknown --rerun-id values: {sorted(rerun_ids - {case['id'] for case in selected})}")
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.jsonl"
    saved = {record["id"]: record for record in load_jsonl(results_path)}
    understanding_model = create_understanding_model_v3()
    for index, case in enumerate(selected, 1):
        existing = saved.get(case["id"], {})
        if not rerun_ids and (existing.get("status") == "ok" or (existing and not retry_errors)):
            print(f"[{index}/{len(selected)}] {case['id']} cached", flush=True)
            continue
        started = time.perf_counter()
        try:
            record = execute_case(case, first_turns[case["base_question_id"]], understanding_model)
            record["status"] = "ok"
        except Exception as exc:
            record = {
                "id": case["id"], "category": case["category"], "status": "error",
                "error_type": type(exc).__name__, "error": str(exc)[:1000],
                "latency_ms": {"total": round((time.perf_counter() - started) * 1000, 2)},
            }
        saved[case["id"]] = record
        atomic_jsonl(results_path, [saved[item["id"]] for item in cases if item["id"] in saved])
        print(f"[{index}/{len(selected)}] {case['id']} {record['status']}", flush=True)
    return [saved[case["id"]] for case in cases if case["id"] in saved]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "report", "all"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--rerun-id", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    configure_shadow_runtime()
    cases = load_cases()
    records = load_jsonl(args.output_dir / "results.jsonl")
    if args.stage in {"run", "all"}:
        if not os.getenv("ARK_API_KEY") or not os.getenv("BASE_URL"):
            raise ValueError("ARK_API_KEY and BASE_URL are required")
        records = run(
            cases, args.output_dir, limit=args.limit, retry_errors=args.retry_errors,
            rerun_ids=set(args.rerun_id) or None,
        )
    if args.stage in {"report", "all"}:
        if args.limit and not args.rerun_id:
            print("[report] skipped because --limit creates a partial run", flush=True)
        else:
            summary = write_report(args.output_dir, cases, records)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
