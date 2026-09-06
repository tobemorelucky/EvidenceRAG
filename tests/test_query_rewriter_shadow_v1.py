from pathlib import Path

from backend.query_rewriter import normalize_retrieval_queries, rewrite_queries
from scripts.evaluate_query_rewrite_shadow_v1 import rrf_fuse, summarize


class _Response:
    content = '{"queries": ["FY2023 revenue", "2023 net sales"]}'
    usage_metadata = {"input_tokens": 10, "output_tokens": 8}


class _Model:
    def invoke(self, messages):
        assert "Original question:" in messages[-1].content
        return _Response()


def test_rewrite_retains_original_and_caps_total_at_three():
    result = rewrite_queries("What was FY2023 revenue?", _Model())
    assert result["queries"] == [
        "What was FY2023 revenue?", "FY2023 revenue", "2023 net sales",
    ]
    assert result["usage"]["input_tokens"] == 10


def test_normalize_deduplicates_original_case_insensitively():
    assert normalize_retrieval_queries("Revenue 2023", ["revenue 2023", "net sales 2023"]) == [
        "Revenue 2023", "net sales 2023",
    ]


def test_multi_route_rrf_keeps_all_route_ranks():
    a = {"chunk_id": "a", "filename": "a.pdf", "page_number": 1, "chunk_idx": 0}
    b = {"chunk_id": "b", "filename": "a.pdf", "page_number": 2, "chunk_idx": 1}
    ranked = rrf_fuse([("q0_dense", [a, b]), ("q0_bm25", [b, a])])
    assert {item["chunk_id"] for item in ranked} == {"a", "b"}
    assert ranked[0]["rrf_rank"] == 1
    assert set(ranked[0]["route_ranks"]) == {"q0_dense", "q0_bm25"}


def test_three_percentage_point_gate():
    records = []
    for index in range(30):
        a_hit = index < 20
        b_hit = index < 21
        metric_a = {"candidate_gold_page_hit": a_hit, "gold_chunk_rank": 1 if a_hit else None, "gold_page_rank": 1 if a_hit else None}
        metric_b = {"candidate_gold_page_hit": b_hit, "gold_chunk_rank": 1 if b_hit else None, "gold_page_rank": 1 if b_hit else None}
        records.append({
            "question_id": str(index), "rewrite": {"status": "ok", "queries": ["q", "r"]},
            "routes": {"baseline": {"metrics": metric_a}, "query_rewrite": {"metrics": metric_b}},
        })
    result = summarize(records)
    assert result["candidate_hit_delta"] == 0.033333
    assert result["candidate_gate_passed"] is True


def test_shadow_script_does_not_import_jina_answer_or_judge():
    source = (Path(__file__).resolve().parents[1] / "scripts/evaluate_query_rewrite_shadow_v1.py").read_text(encoding="utf-8").casefold()
    assert "jinareranker" not in source
    assert "answer_generator" not in source
    assert "judge_financebench" not in source
