"""Re-run selected historical questions through finance_online_v1 retrieval only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
load_dotenv(ROOT / ".env", override=False)

from backend.rag_orchestrator import prepare_rag_response  # noqa: E402
from backend.rag_trace_service import build_rag_trace_payload  # noqa: E402
from scripts.analyze_online_rag_trace import analyze_trace, load_trace, render_markdown  # noqa: E402


DEFAULT_TRACE_IDS = (
    "7758c420-a9b1-4e30-8a70-a5e8146395d2",  # Block
    "ac36a801-d5e3-4436-ae24-b30fbe8cc81d",  # Adobe
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_ids", nargs="*", default=list(DEFAULT_TRACE_IDS))
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "online_rag_trace" / "finance_alignment_v1_after"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = []
    for trace_id in args.trace_ids:
        before = load_trace(trace_id, None)
        result = prepare_rag_response(str(before.get("user_query") or ""), profile="finance", mode="auto")
        rag_trace = dict(result.get("rag_trace") or {})
        payload = build_rag_trace_payload(
            conversation_id=str(before.get("conversation_id") or "diagnostic"),
            user_query=str(before.get("user_query") or ""),
            decision=dict(before.get("conversation_understanding") or {}),
            policy={"retrieval": True, "rewrite": False},
            rag_result=result,
            citations=list(result.get("citations") or []),
            answer_model=os.getenv("MODEL", ""),
            understanding_usage={},
            answer_usage={},
            latency_ms=dict(rag_trace.get("latency_breakdown") or {}),
            profile="finance",
            execution_mode="auto",
            answer_prompt_mode="baseline",
        )
        payload.update({"trace_id": f"alignment-{trace_id}", "created_at": "diagnostic-replay"})
        analysis = analyze_trace(payload)
        (output_dir / f"{trace_id}.json").write_text(
            json.dumps({
                "source_trace_id": trace_id,
                "analysis": analysis,
                "rag_trace": rag_trace,
                "evidence": result.get("evidence") or "",
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / f"{trace_id}.md").write_text(render_markdown(analysis), encoding="utf-8")
        combined.append(analysis)
        print(f"[{trace_id}] match={analysis['matches_financebench_pipeline']} pages={len(analysis['final_evidence_pages'])}", flush=True)
    (output_dir / "summary.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
