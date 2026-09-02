from scripts.evaluate_page_selector_v2 import summarize


def _record(group: str, baseline: bool, v1: bool, v2: bool) -> dict:
    route = lambda value: {"selected_hit": value, "context_hit": value, "gold_page_rank": 1 if value else None}
    return {
        "financebench_id": f"{group}-{baseline}-{v1}-{v2}",
        "group": group,
        "candidate_hit": True,
        "baseline": route(baseline),
        "page_selector_v1": route(v1),
        "page_selector_v2": {**route(v2), "gold_group_hit": v2, "latency_ms": 2.0},
    }


def test_v2_acceptance_compares_against_production_not_failed_v1():
    records = [
        _record("candidate_miss10", False, False, False),
        _record("selection_loss10", False, False, True),
        _record("correct_regression10", True, False, True),
    ]

    summary = summarize(records)

    assert summary["groups"]["selection_loss10"]["v2_context_delta_vs_baseline"] == 1.0
    assert summary["groups"]["correct_regression10"]["v2_context_delta_vs_baseline"] == 0.0
    assert summary["acceptance"]["passed"] is True
