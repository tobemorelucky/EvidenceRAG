import json
from pathlib import Path

from scripts.evaluate_query_rewrite_jina_shadow_v1 import (
    candidate_signature,
    route_metrics,
    summarize,
    validate_frozen_input,
)


def _chunk(index, page=None, text=None):
    return {
        "chunk_id": f"c{index}", "filename": "report.pdf",
        "page_number": index if page is None else page,
        "text": text or f"chunk {index}", "rrf_rank": index + 1,
    }


def test_candidate_signature_includes_jina_text_and_source_identity():
    left = candidate_signature([_chunk(0)])
    right = candidate_signature([{**_chunk(0), "text": "changed"}])
    assert left != right


def test_route_metrics_uses_build_context_budget():
    row = {"evidence": json.dumps([{
        "doc_name": "report.pdf", "evidence_page_num": 2,
        "evidence_text": "This is a sufficiently long literal gold evidence sentence for matching.",
    }])}
    chunks = [_chunk(index, page=index, text="x" * 5000) for index in range(12)]
    chunks[2]["text"] = "This is a sufficiently long literal gold evidence sentence for matching."
    ranked = [{"index": index, "score": 12 - index} for index in range(12)]
    result = route_metrics(row, chunks, ranked)
    assert result["jina_top12_gold_page_hit"] is True
    assert result["context_hit"] is True
    assert result["context_chars"] <= 28000


def test_context_gate_requires_at_least_five_percentage_points():
    records = []
    for index in range(30):
        a = index < 15
        b = index < 17
        def route(value):
            return {"status": "ok", "cache_hit": False, "metrics": {
                "candidate_hit": True, "jina_top12_gold_chunk_hit": value,
                "jina_top12_gold_page_hit": value, "context_hit": value,
                "gold_chunk_rank": 1 if value else None, "gold_page_rank": 1 if value else None,
            }}
        records.append({"question_id": str(index), "routes": {"baseline": route(a), "query_rewrite": route(b)}})
    result = summarize(records)
    assert result["context_hit_delta"] == 0.066667
    assert result["context_gate_passed"] is True


def test_evaluator_contains_no_answer_judge_or_langsmith_call():
    source = (Path(__file__).resolve().parents[1] / "scripts/evaluate_query_rewrite_jina_shadow_v1.py").read_text(encoding="utf-8").casefold()
    assert "answer_generator" not in source
    assert "judge_financebench" not in source
    assert "langsmith.client" not in source
