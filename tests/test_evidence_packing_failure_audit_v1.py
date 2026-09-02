from scripts.audit_evidence_packing_failure_v1 import (
    classify_failure,
    mark_selected_duplicates,
    replay_packing,
)


def _candidate(rank: int, text: str):
    return {
        "document_id": "doc",
        "page_id": f"page-{rank}",
        "source_type": "text",
        "entity": "Example Corp",
        "period": [],
        "metric": None,
        "value": None,
        "unit": None,
        "source_text": text,
        "metadata": {"filename": "example.pdf", "page_number": rank, "chunk_id": f"chunk-{rank}"},
        "current_ranking": {"rank": rank, "score": 1 / rank},
    }


def test_packing_records_remaining_budget_and_rejection():
    traces, context = replay_packing([
        _candidate(1, "short fact"),
        _candidate(2, "long fact " * 100),
    ], max_context_chars=300)

    assert traces[0]["selected"] is True
    assert traces[1]["selected"] is False
    assert traces[1]["rejection_reason"] == "unit_exceeds_total_context_budget"
    assert len(context) <= 300


def test_duplicate_detection_counts_reclaimable_selected_chars():
    text = " ".join(f"token{i}" for i in range(30))
    traces, _ = replay_packing([_candidate(1, text), _candidate(2, text)], max_context_chars=5000)

    result = mark_selected_duplicates(traces)

    assert result["selected_duplicate_units"] == 1
    assert result["selected_duplicate_chars"] > 0


def test_same_table_header_with_different_numbers_is_not_duplicate():
    shared = "header revenue operating income assets liabilities " * 8
    traces, _ = replay_packing([
        _candidate(1, shared + " row revenue 120"),
        _candidate(2, shared + " row income 30"),
    ], max_context_chars=10000)

    result = mark_selected_duplicates(traces)

    assert result["selected_duplicate_units"] == 0


def test_classification_separates_upstream_coverage_from_packing():
    assert classify_failure(
        candidate_coverage=0.5,
        selected_coverage=0.2,
        missing_gold_units=[],
        selection_frontier=10,
        duplicate_reclaimable_chars=0,
    )[0] == "D"
    missing = [{"rank": 5, "required_chars_at_attempt": 500, "remaining_budget_before": 100}]
    assert classify_failure(
        candidate_coverage=1.0,
        selected_coverage=0.5,
        missing_gold_units=missing,
        selection_frontier=10,
        duplicate_reclaimable_chars=0,
    )[0] == "B"
    assert classify_failure(
        candidate_coverage=1.0,
        selected_coverage=0.5,
        missing_gold_units=missing,
        selection_frontier=10,
        duplicate_reclaimable_chars=500,
    )[0] == "C"
