from scripts.evaluate_page_selector_v1 import summarize


def _record(group: str, *, old: bool, new: bool) -> dict:
    return {
        "financebench_id": f"{group}-{old}-{new}",
        "group": group,
        "candidate_hit": True,
        "selector_latency_ms": 1.0,
        "baseline": {"selected_hit": old, "context_hit": old, "gold_page_rank": 12},
        "page_selector_v1": {"selected_hit": new, "context_hit": new, "gold_page_rank": 4},
    }


def test_summary_applies_selection_and_regression_acceptance_contract():
    records = [
        _record("candidate_miss10", old=False, new=False),
        _record("selection_loss10", old=False, new=True),
        _record("correct_regression10", old=True, new=True),
    ]

    summary = summarize(records)

    assert summary["groups"]["selection_loss10"]["context_hit_delta"] == 1.0
    assert summary["groups"]["correct_regression10"]["context_hit_delta"] == 0.0
    assert summary["acceptance"]["passed"] is True
