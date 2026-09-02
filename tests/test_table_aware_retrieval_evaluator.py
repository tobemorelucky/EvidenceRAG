import json

from backend.evidence_identity import build_page_id
from scripts.evaluate_table_aware_retrieval import _gold_pages, summarize


def _record():
    return {
        "financebench_id": "generic_1",
        "gold_table_ids": ["t1"],
        "gold_table_diagnostics": {
            "quality_scores": [0.8], "quality_at_least_065": 1,
            "rows_nonempty": 1, "columns_nonempty": 1, "adjacent_table_count": 0,
        },
        "text": {"candidate_hit": False, "gold_page_rank": None, "candidate_count": 10},
        "table": {
            "candidate_count": 5,
            "dense_gold_table_rank": 4,
            "bm25_gold_table_rank": None,
            "fused_gold_table_rank": 3,
        },
        "text_plus_table": {"candidate_hit": True, "gold_page_rank": 3, "candidate_count": 11},
        "latency_ms": {
            "frozen_text": 100,
            "table_query_embedding": 10,
            "table_dense": 5,
            "table_bm25": 4,
            "table_total_excluding_embedding": 10,
            "shadow_incremental_total": 20,
        },
    }


def test_summary_reports_candidate_page_table_recall_cost_and_recovery():
    summary = summarize([_record()])

    assert summary["candidate_hit"] == {"text": 0.0, "text_plus_table": 1.0}
    assert summary["gold_page_hit_at_k"]["5"]["text_plus_table"] == 1.0
    assert summary["gold_table_hit_at_30"] == 1.0
    assert summary["table_recall_at_k"]["dense"]["5"] == 1.0
    assert summary["table_recall_at_k"]["bm25"]["30"] == 0.0
    assert summary["average_candidate_count"]["text_plus_table"] == 11.0
    assert summary["recovered_candidate_ids"] == ["generic_1"]


def test_table_recall_uses_only_questions_with_gold_tables():
    without_table = _record()
    without_table["financebench_id"] = "generic_2"
    without_table["gold_table_ids"] = []
    without_table["gold_table_diagnostics"] = {
        "quality_scores": [], "quality_at_least_065": 0,
        "rows_nonempty": 0, "columns_nonempty": 0, "adjacent_table_count": 1,
    }
    without_table["table"]["fused_gold_table_rank"] = None

    summary = summarize([_record(), without_table])

    assert summary["gold_table_eligible_questions"] == 1
    assert summary["gold_table_hit_at_30"] == 1.0
    assert summary["table_extraction"]["gold_page_table_coverage"] == 0.5
    assert summary["table_extraction"]["adjacent_table_present_when_gold_page_missing"] == 1


def test_financebench_page_number_is_the_internal_page_without_offset():
    row = {
        "evidence": json.dumps([
            {"doc_name": "GENERIC_2024_10K", "evidence_page_num": 59},
        ])
    }

    gold_pages = _gold_pages(row)

    assert gold_pages == {("generic_2024_10k.pdf", 59)}
    _, page_number = next(iter(gold_pages))
    assert build_page_id("doc_contract", page_number) == "doc_contract:page:000059"
