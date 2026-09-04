import copy
import json
from unittest.mock import Mock, patch

import pytest

from scripts.shadow_rerankers_v1 import IdentityReranker, JinaReranker, validate_order
from scripts.evaluate_reranker_shadow_v1 import metrics, validate_snapshot, digest, summarize, scoring_manifest, finalize_report


def chunk(i=0, text="A sufficiently long literal evidence statement about production output."):
    return {"chunk_id": str(i), "filename": "sample.pdf", "page_number": i,
            "text": text, "rrf_rank": i + 1}


def row(page=0):
    return {"evidence": json.dumps([{"doc_name": "sample", "evidence_page_num": page,
                                    "evidence_text": chunk()["text"]}])}


def test_identity_is_permutation_without_mutation():
    texts = [str(i) for i in range(120)]
    saved = texts[:]
    ranked, trace = IdentityReranker().rank("Query", texts)
    assert [r["index"] for r in validate_order(ranked, 120)] == list(range(120))
    assert texts == saved and trace["requests"] == 0


@pytest.mark.parametrize("items", [[{"index": 0, "score": 1}] * 2,
    [{"index": 0, "score": float("nan")}, {"index": 1, "score": 1}],
    [{"index": True, "score": 1}, {"index": 0, "score": 2}]])
def test_invalid_backend_output_fails(items):
    with pytest.raises(ValueError):
        validate_order(items, 2)


def test_unique_page_rank_not_chunk_rank_and_zero_based():
    ordered = [chunk(4), chunk(4), chunk(0)]
    result = metrics(row(0), ordered)
    assert result["gold_chunk_rank"] == 3
    assert result["gold_page_rank"] == 2
    assert result["page_hit_at_5"]
    assert result["context_hit"]


def test_missing_gold_chunk_text_is_not_page_match():
    result = metrics(row(), [chunk(text="Unrelated text.")])
    assert result["gold_chunk_rank"] is None
    assert result["gold_page_rank"] == 1
    assert not result["context_evidence_span_hit"]


def test_context_budget_truncates_actual_text_not_only_metadata():
    result = metrics(row(), [chunk()], budget=10)
    assert result["context_chars"] == 10
    assert result["context_hit"]
    assert not result["context_evidence_span_hit"]
    assert not metrics(row(), [chunk(1), chunk()], context_chunks=1)["context_hit"]


def test_jina_one_call_full120_no_gold_or_fallback():
    response = Mock(status_code=200)
    response.json.return_value = {"results": [{"index": i, "relevance_score": i} for i in range(120)]}
    with patch("requests.post", return_value=response) as post:
        ranked, _ = JinaReranker("secret", interval=0).rank("Question only", ["text"] * 120)
    assert ranked[0]["index"] == 119
    assert post.call_count == 1
    payload = post.call_args.kwargs["json"]
    assert payload["top_n"] == 120 and len(payload["documents"]) == 120
    assert set(payload) == {"model", "query", "documents", "top_n", "return_documents"}


def test_jina_errors_redacted_and_no_retry():
    with patch("requests.post", return_value=Mock(status_code=429, text="secret")) as post:
        with pytest.raises(RuntimeError, match="HTTP 429") as exc:
            JinaReranker("secret", interval=0).rank("Q", ["T"])
    assert "secret" not in str(exc.value) and post.call_count == 1


def test_jina_rejects_missing_indices_and_foreign_endpoint():
    response = Mock(status_code=200)
    response.json.return_value = {"results": [{"index": 0, "relevance_score": 1}]}
    with patch("requests.post", return_value=response), pytest.raises(RuntimeError, match="incomplete"):
        JinaReranker("s").rank("Q", ["a", "b"])
    with pytest.raises(ValueError):
        JinaReranker("s", endpoint="https://example.com/v1/rerank")


def test_snapshot_rejects_dense_and_content_drift():
    records = [{"question_id": str(i), "question": "Q", "group": "g",
                "chunks": [chunk(j) for j in range(120)],
                "candidate_sha256": digest([chunk(j) for j in range(120)])} for i in range(30)]
    payload = {"schema": "rrf_top120_shadow_v1", "retrieval": {"method": "MilvusManager.hybrid_retrieve"}, "records": records}
    rows = {str(i): {"question": "Q"} for i in range(30)}
    groups = dict.fromkeys(rows, "g")
    assert len(validate_snapshot(payload, rows, groups)) == 30
    saved = copy.deepcopy(payload)
    payload["retrieval"]["method"] = "dense_primary"
    with pytest.raises(ValueError):
        validate_snapshot(payload, rows, groups)
    saved["records"][0]["chunks"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="drift"):
        validate_snapshot(saved, rows, groups)


def test_failed_backend_metrics_are_unavailable_not_zero():
    records = [{"question_id": "q", "group": "selection_loss10", "routes": {"jina": {"status": "error"}}}]
    result = summarize(records, ["jina"])["jina"]
    assert result["completed"] == 0 and result["context_hit"] is None


def test_candidate_hit_invariant_under_reranking():
    chunks = [chunk(2), chunk(0), chunk(1)]
    original = metrics(row(), chunks, context_chunks=1)
    ranked = metrics(row(), list(reversed(chunks)), context_chunks=1)
    assert original["candidate_gold_page_hit"] == ranked["candidate_gold_page_hit"] == True
    records = [{"question_id": "q", "group": "selection_loss10", "routes": {
        "identity": {"status": "ok", "metrics": original}, "bge": {"status": "ok", "metrics": ranked}}}]
    result = summarize(records, ["identity", "bge"])
    assert result["identity"]["candidate_gold_page_hit"] == result["bge"]["candidate_gold_page_hit"] == 1


def test_resume_allows_pacing_only_not_scoring_changes():
    first = {"input_sha256": "frozen", "backends": {"jina": {"model": "jina", "interval_seconds": 8}}}
    paced = copy.deepcopy(first)
    paced["backends"]["jina"]["interval_seconds"] = 35
    assert scoring_manifest(first) == scoring_manifest(paced)
    assert first["backends"]["jina"]["interval_seconds"] == 8
    paced["backends"]["jina"]["model"] = "other"
    assert scoring_manifest(first) != scoring_manifest(paced)


def test_common_subset_and_cached_metric_validation():
    chunks = [chunk()]
    frozen = [{"question_id": "q", "chunks": chunks, "candidate_sha256": digest(chunks)}]
    ok = {"status": "ok", "ranked": [{"index": 0, "score": 1}], "metrics": metrics(row(), chunks)}
    record = {"question_id": "q", "group": "selection_loss10", "candidate_sha256": digest(chunks),
              "routes": {"identity": ok, "jina": {"status": "error"}}}
    payload = {"manifest": {"backends": {"identity": {}, "jina": {}}}, "records": [record]}
    result = finalize_report(payload, frozen, {"q": row()})
    assert result["common_subset"]["question_ids"] == []
    assert result["verification"]["completed_routes_recomputed"] == 1
    ok["metrics"]["context_hit"] = False
    with pytest.raises(ValueError, match="mismatch"):
        finalize_report(payload, frozen, {"q": row()})
