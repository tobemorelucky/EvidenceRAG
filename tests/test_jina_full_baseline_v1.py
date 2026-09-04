import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.jina_full_baseline_v1 import (
    read_profile, digest, validate_candidates, rrf_merge, build_context, cached_jina, export_reports,
)
from scripts.run_jina_full_baseline_v1 import run_question


def chunks(n=120):
    return [{"chunk_id": str(i), "text": f"Evidence {i} amount {i}.00", "filename": "sample.pdf",
             "page_number": i, "rrf_rank": i + 1} for i in range(n)]


def source():
    values = chunks()
    return {"question_id": "q", "question": "What amount?", "chunks": values,
            "candidate_sha256": digest(values)}


def ranked(n=120):
    return [{"index": i, "score": float(n - i)} for i in range(n)]


def test_profile_is_static_pure_jina_and_pinned_models():
    profile = read_profile({})
    assert profile["name"] == "jina_full_baseline_v1"
    assert profile["skills"] == [] and not profile["planner"] and not profile["langsmith"]
    assert profile["retrieval"] == {"method": "experimental_independent_dense_bm25_rrf", "dense_top_k": 240,
        "bm25_top_k": 240, "rrf_top_k": 120, "rrf_rank_constant": 60,
        "filter": '(evidence_type == "text_chunk" or evidence_type == "") and chunk_level == 3'}
    assert profile["reranker"]["input_k"] == 120 and profile["reranker"]["output_k"] == 8
    assert profile["reranker"]["model"] == "jina-reranker-v3"
    assert profile["reranker"]["fallback"] is False
    assert profile["answer"]["model"] == "deepseek-v4-flash-ga-260731"
    assert profile["judge"]["model"] == "deepseek-v4-pro-ga-260813"


def test_all_depth_parameters_resolve_from_environment_and_validate_contract():
    profile = read_profile({"DENSE_TOP_K": "90", "BM25_TOP_K": "30", "RRF_TOP_K": "60",
                            "JINA_INPUT_K": "40", "JINA_OUTPUT_K": "6"})
    assert profile["retrieval"]["dense_top_k"] == 90
    assert profile["retrieval"]["bm25_top_k"] == 30
    assert profile["retrieval"]["rrf_top_k"] == 60
    assert profile["reranker"]["input_k"] == 40
    assert profile["reranker"]["output_k"] == profile["context"]["top_k"] == 6
    with pytest.raises(ValueError, match="JINA_INPUT_K"):
        read_profile({"RRF_TOP_K": "40", "JINA_INPUT_K": "41"})
    with pytest.raises(ValueError, match="positive integer"):
        read_profile({"DENSE_TOP_K": "0"})


def test_converged_quality_and_budget_profiles_are_distinct_and_overridable():
    quality = read_profile({}, profile_name="jina_full_baseline_input120_v1")
    budget = read_profile({}, profile_name="jina_full_baseline_input80_v1")
    assert (quality["reranker"]["input_k"], quality["reranker"]["output_k"]) == (120, 12)
    assert (budget["reranker"]["input_k"], budget["reranker"]["output_k"]) == (80, 10)
    assert quality["context"]["max_chars"] == budget["context"]["max_chars"] == 28000
    assert read_profile({"JINA_OUTPUT_K": "8"}, profile_name="jina_full_baseline_input120_v1")["context"]["top_k"] == 8


def test_experimental_rrf_uses_independent_route_depths_and_stable_order():
    dense = [{"chunk_id": "a", "text": "a"}, {"chunk_id": "b", "text": "b"}]
    bm25 = [{"chunk_id": "b", "text": "b"}, {"chunk_id": "c", "text": "c"}]
    result = rrf_merge(dense, bm25, top_k=3, rank_constant=60)
    assert [item["chunk_id"] for item in result] == ["b", "a", "c"]
    assert result[0]["dense_rank"] == 2 and result[0]["bm25_rank"] == 1
    assert [item["rrf_rank"] for item in result] == [1, 2, 3]


def test_candidates_are_exact_top120_and_tamper_detected():
    item = source()
    validate_candidates(item, item["question"])
    bad = copy.deepcopy(item)
    bad["chunks"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_candidates(bad, bad["question"])
    with pytest.raises(ValueError):
        validate_candidates({**item, "chunks": item["chunks"][:-1]}, item["question"])


def test_context_uses_rerank_order_raw_text_and_budget():
    values = chunks(4)
    values[3]["text"] = "X" * 100
    context, citations, docs = build_context(list(reversed(values)), {"top_k": 3, "max_chars": 120})
    assert context.startswith("Source: sample.pdf | Page: 3\n")
    assert [c["page_number"] for c in citations] == [3]
    assert citations[0]["truncated"] and len(context) == 120
    assert docs[0]["text"] in context


def test_cached_jina_requires_model_question_candidate_and_endpoint():
    item = source()
    route = {"status": "ok", "ranked": ranked(), "trace": {"usage": {"total_tokens": 1}}}
    cache = {"manifest": {"backends": {"jina": {"model": "jina-reranker-v3", "endpoint": "https://api.jina.ai/v1/rerank"}}},
             "records": [{"question": item["question"], "candidate_sha256": item["candidate_sha256"], "routes": {"jina": route}}]}
    assert cached_jina(item, cache, "jina-reranker-v3") == route
    assert cached_jina({**item, "candidate_sha256": "other"}, cache, "jina-reranker-v3") is None
    assert cached_jina(item, cache, "other") is None
    cache["manifest"]["backends"]["jina"]["endpoint"] = "https://example.com/v1/rerank"
    assert cached_jina(item, cache, "jina-reranker-v3") is None


def test_question_checkpoint_persists_jina_before_answer_and_resume_does_not_rebill():
    profile = read_profile({})
    item = source()
    record = {"financebench_id": "q", "question": item["question"],
              "candidate_sha256": item["candidate_sha256"], "latency_ms": {}}
    events = []
    jina = Mock(return_value=(ranked(), {"requests": 1, "usage": {"total_tokens": 100}}))
    generate = Mock(side_effect=RuntimeError("answer down"))
    with pytest.raises(RuntimeError, match="answer down"):
        run_question(record, item, profile, lambda: events.append(copy.deepcopy(record)), jina, generate)
    assert jina.call_count == 1 and events[0]["jina"]["status"] == "ok"
    generate.side_effect = None
    generate.return_value = ("Answer [source: sample.pdf, page 0]", {"total_tokens": 10})
    run_question(record, item, profile, lambda: events.append(copy.deepcopy(record)), jina, generate)
    assert jina.call_count == 1 and generate.call_count == 2
    assert record["answer_status"] == "ok" and record["citations"]
    run_question(record, item, profile, lambda: None, jina, generate)
    assert jina.call_count == 1 and generate.call_count == 2


def test_jina_failure_never_falls_back_or_generates():
    profile, item = read_profile({}), source()
    record = {"financebench_id": "q", "question": item["question"],
              "candidate_sha256": item["candidate_sha256"], "latency_ms": {}}
    generate = Mock()
    with pytest.raises(RuntimeError, match="Jina HTTP 429"):
        run_question(record, item, profile, lambda: None,
                     Mock(side_effect=RuntimeError("Jina HTTP 429; no local fallback")), generate)
    generate.assert_not_called()
    assert "answer_status" not in record and "jina" not in record


def test_report_has_reference_answer_model_answer_citations_judge_and_cost(tmp_path):
    record = {"financebench_id": "q", "question": "Question", "answer": "Model answer",
        "answer_status": "ok", "citations": [{"filename": "a.pdf", "page_number": 0}],
        "judge_status": "ok", "judge": {"score": 1, "verdict": "correct", "reason": "match", "total_tokens": 5},
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}, "jina_cache_hit": False,
        "jina": {"status": "ok", "trace": {"usage": {"total_tokens": 99}}}, "latency_ms": {}}
    export_reports(tmp_path, {"records": [record]}, {"q": {"question": "Question", "answer": "Reference"}})
    text = (tmp_path / "answers.md").read_text(encoding="utf-8")
    assert all(value in text for value in ("问题", "参考答案", "模型答案", "引用", "Judge", "Reference", "Model answer", "a.pdf"))
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["strict_accuracy"] == 1 and summary["jina_new_reported_tokens"] == 99


def test_runner_contains_no_production_router_or_local_fallback():
    import ast
    tree = ast.parse(Path("scripts/run_jina_full_baseline_v1.py").read_text(encoding="utf-8"))
    names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "LocalReranker" not in names
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert "rag_orchestrator" not in imports
