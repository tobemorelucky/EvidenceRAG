"""Render one persisted online Conversation RAG trace without external calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from backend.database import SessionLocal  # noqa: E402
from backend.rag_trace_service import RagTraceRecord, RagTraceService  # noqa: E402


FINANCEBENCH_REFERENCE = {
    "profile": "finance",
    "execution_mode": "static",
    "query_rewrite": True,
    "dense_top_k": 240,
    "bm25_top_k": 240,
    "rrf_top_k": 120,
    "jina_input_k": 80,
    "jina_output_k": 12,
    "max_context_chars": 28000,
    "answer_prompt_mode": "baseline",
}


def _number(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def analyze_trace(trace: dict) -> dict:
    policy = dict(trace.get("policy_decision") or {})
    understanding = dict(trace.get("conversation_understanding") or {})
    rerank = dict(trace.get("rerank_parameters") or {})
    counts = dict(trace.get("retrieval_counts") or {})
    evidence = list(trace.get("evidence_items") or [])
    pages = []
    seen = set()
    for item in evidence:
        key = (item.get("filename"), item.get("page_number"))
        if key in seen:
            continue
        seen.add(key)
        pages.append({
            "filename": item.get("filename"),
            "page_number": item.get("page_number"),
            "score": item.get("score"),
            "text_preview": str(item.get("text") or "")[:400],
        })

    profile = policy.get("profile") or "unknown (legacy trace)"
    mode = policy.get("execution_mode") or policy.get("mode") or "unknown (legacy trace)"
    rewrite = policy.get("query_rewrite_executed")
    if rewrite is None:
        rewrite = bool(policy.get("rewrite"))
    provider = str(rerank.get("provider") or "").lower()
    jina_called = bool(rerank.get("applied") and ("jina" in str(rerank.get("model") or "").lower() or provider == "remote"))
    result = {
        "trace_id": trace.get("trace_id"),
        "created_at": trace.get("created_at"),
        "profile": profile,
        "mode": mode,
        "conversation_id": trace.get("conversation_id"),
        "user_query": trace.get("user_query"),
        "conversation_understanding": understanding,
        "standalone_query": trace.get("standalone_query") or understanding.get("standalone_query"),
        "need_retrieval": understanding.get("need_retrieval"),
        "retrieval_executed": bool(trace.get("retrieval_executed")),
        "query_rewrite_executed": bool(rewrite),
        "profile_config": policy.get("profile_config") or rerank.get("profile_config"),
        "query_rewrite_enabled": policy.get("query_rewrite_enabled"),
        "retrieval_counts": counts,
        "retrieval_depth": {
            "dense_top_k": _number(rerank.get("dense_top_k")),
            "bm25_top_k": _number(rerank.get("bm25_top_k")),
            "rrf_top_k": _number(rerank.get("rrf_top_k")),
        },
        "jina": {
            "called": jina_called,
            "enabled": bool(rerank.get("enabled")),
            "applied": bool(rerank.get("applied")),
            "provider": rerank.get("provider"),
            "model": rerank.get("model"),
            "input_k": _number(rerank.get("input_k") or rerank.get("candidate_k")),
            "output_k": _number(rerank.get("output_k") or rerank.get("final_top_k")),
        },
        "context_budget": _number(policy.get("context_budget") or rerank.get("context_budget")),
        "final_evidence_pages": pages,
        "answer_model": trace.get("answer_model"),
        "answer_prompt_mode": policy.get("answer_prompt_mode") or "unknown (legacy trace)",
        "token_usage": trace.get("token_usage") or {},
        "latency_ms": trace.get("latency_ms") or {},
        "financebench_reference": FINANCEBENCH_REFERENCE,
    }
    differences = []
    if profile != "finance":
        differences.append("线上 trace 未能确认 finance profile。")
    if result["profile_config"] != "finance_online_v1":
        differences.append(
            f"profile_config={result['profile_config'] or 'unknown'}，目标为 finance_online_v1。"
        )
    if not result["query_rewrite_executed"]:
        differences.append("线上未执行 Query Rewrite；最终 FinanceBench 链路会保留原查询并生成 rewrite。")
    if not result["retrieval_executed"]:
        differences.append("本轮未执行新检索，而是由会话策略复用历史证据。")
    for name, expected in (("dense_top_k", 240), ("bm25_top_k", 240), ("rrf_top_k", 120)):
        actual = result["retrieval_depth"][name]
        if actual != expected:
            differences.append(f"{name}={actual if actual is not None else 'unknown'}，FinanceBench 为 {expected}。")
    if not jina_called:
        differences.append("线上 trace 未确认执行 Jina rerank。")
    elif result["jina"]["input_k"] != 80 or result["jina"]["output_k"] != 12:
        differences.append(
            f"线上 Jina 深度为 {result['jina']['input_k']}→{result['jina']['output_k']}，FinanceBench 为 80→12。"
        )
    if result["context_budget"] != 28000:
        differences.append(
            f"context_budget={result['context_budget'] if result['context_budget'] is not None else 'unknown'}，目标为 28000。"
        )
    result["pipeline_differences"] = differences
    result["matches_financebench_pipeline"] = not differences
    return result


def render_markdown(result: dict) -> str:
    pages = result["final_evidence_pages"]
    lines = [
        "# Online RAG Trace Diagnosis",
        "",
        f"- Trace ID: `{result['trace_id']}`",
        f"- Conversation ID: `{result['conversation_id']}`",
        f"- Time: {result.get('created_at') or 'unknown'}",
        f"- Query: {result.get('user_query') or ''}",
        "",
        "## Runtime",
        "",
        f"- Profile: `{result['profile']}`",
        f"- Mode: `{result['mode']}`",
        f"- Standalone query: {result.get('standalone_query') or ''}",
        f"- need_retrieval: `{result.get('need_retrieval')}`",
        f"- Retrieval executed: `{result['retrieval_executed']}`",
        f"- Query Rewrite executed: `{result['query_rewrite_executed']}`",
        f"- Profile config: `{result.get('profile_config') or 'unknown'}`",
        f"- Retrieval counts: `{json.dumps(result['retrieval_counts'], ensure_ascii=False)}`",
        f"- Retrieval TopK: `{json.dumps(result['retrieval_depth'], ensure_ascii=False)}`",
        f"- Jina: `{json.dumps(result['jina'], ensure_ascii=False)}`",
        f"- Context budget: `{result.get('context_budget')}` chars",
        f"- Answer model: `{result.get('answer_model') or 'unknown'}`",
        f"- Answer prompt mode: `{result['answer_prompt_mode']}`",
        f"- Token: `{json.dumps(result['token_usage'], ensure_ascii=False)}`",
        f"- Latency ms: `{json.dumps(result['latency_ms'], ensure_ascii=False)}`",
        "",
        "## Conversation Understanding",
        "",
        "```json",
        json.dumps(result["conversation_understanding"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Final Evidence Pages",
        "",
    ]
    if pages:
        for page in pages:
            lines.extend([
                f"- `{page.get('filename')}` p.{page.get('page_number')} (score={page.get('score')})",
                f"  - {page.get('text_preview') or '(no preview)'}",
            ])
    else:
        lines.append("- No evidence pages recorded.")
    lines.extend(["", "## FinanceBench Pipeline Comparison", ""])
    if result["pipeline_differences"]:
        lines.extend(f"- {item}" for item in result["pipeline_differences"])
    else:
        lines.append("- The persisted settings match the final FinanceBench reference pipeline.")
    lines.extend([
        "",
        "> `unknown (legacy trace)` means the historical schema did not persist that field; it is not inferred from the current environment.",
        "",
    ])
    return "\n".join(lines)


def load_trace(trace_id: str | None, latest_query: str | None) -> dict:
    db = SessionLocal()
    try:
        query = db.query(RagTraceRecord)
        if trace_id:
            row = query.filter(RagTraceRecord.trace_id == trace_id).first()
        elif latest_query:
            row = query.filter(RagTraceRecord.user_query.ilike(f"%{latest_query}%")).order_by(RagTraceRecord.id.desc()).first()
        else:
            row = query.order_by(RagTraceRecord.id.desc()).first()
        if row is None:
            raise SystemExit("No matching rag trace found")
        return RagTraceService._serialize(row)
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_id", nargs="?", help="Persisted rag trace id; defaults to latest")
    parser.add_argument("--latest-query", help="Use the newest trace whose user query contains this text")
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "online_rag_trace"))
    args = parser.parse_args()
    result = analyze_trace(load_trace(args.trace_id, args.latest_query))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(result["trace_id"] or "latest")
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
