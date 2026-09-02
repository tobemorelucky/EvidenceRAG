from scripts.evaluate_evidence_block_v1 import summarize


def _record(group: str, current: bool, block: bool) -> dict:
    return {
        "financebench_id": f"{group}-{current}-{block}",
        "group": group,
        "candidate_hit": True,
        "current_selector": {
            "selected_hit": current, "context_page_hit": current,
            "gold_row_hit": current, "required_number_hit": current,
        },
        "evidence_block_v1": {
            "selected_evidence_block_hit": block, "context_page_hit": block,
            "gold_row_hit": block, "required_number_hit": block,
            "selected_block_count": 2, "context_chars": 100, "latency_ms": 1.0,
        },
    }


def test_acceptance_compares_selection_loss_and_correct_regression_to_current_selector():
    records = [
        _record("candidate_miss10", False, False),
        _record("selection_loss10", False, True),
        _record("correct_regression10", True, True),
    ]

    summary = summarize(records)

    assert summary["groups"]["selection_loss10"]["context_page_delta"] == 1.0
    assert summary["groups"]["correct_regression10"]["context_page_delta"] == 0.0
    assert summary["acceptance"]["passed"] is True
