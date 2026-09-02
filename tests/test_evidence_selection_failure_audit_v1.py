from scripts.audit_evidence_selection_failure_v1 import (
    classify_selection_failure,
    evidence_coverage,
    non_entry_reason,
)


def _classify(**overrides):
    values = {
        "gold_page_mapped": True,
        "gold_candidate_exists": True,
        "exact_coverage_ratio": 0.0,
        "required_periods": [],
        "best_period_match": None,
        "multiple_required_units": False,
        "selected_gold_page_unit_count": 0,
    }
    values.update(overrides)
    return classify_selection_failure(**values)[0]


def test_failure_classification_priority_is_deterministic():
    assert _classify(gold_candidate_exists=False) == "D"
    assert _classify(required_periods=["2024"], best_period_match=0.0) == "C"
    assert _classify(multiple_required_units=True) == "B"
    assert _classify() == "A"
    assert _classify(selected_gold_page_unit_count=1) == "E"


def test_non_entry_reason_distinguishes_rank_and_packing():
    candidate = [{"ranking_v1_rank": 30}]

    assert non_entry_reason(
        gold_page_mapped=True,
        gold_candidates=candidate,
        selected_gold_page_unit_count=0,
        exact_coverage_ratio=0.0,
        selection_rank_frontier=20,
    ) == "gold_unit_ranked_below_selected_frontier"
    assert non_entry_reason(
        gold_page_mapped=True,
        gold_candidates=candidate,
        selected_gold_page_unit_count=0,
        exact_coverage_ratio=0.0,
        selection_rank_frontier=40,
    ) == "gold_unit_skipped_by_character_packing"


def test_evidence_coverage_requires_words_and_numbers_for_fuzzy_match():
    gold = [{"evidence_text": "Net sales were $120 million in 2024."}]

    assert evidence_coverage(gold, "Net sales were $120 million in 2024.")["ratio"] == 1.0
    assert evidence_coverage(gold, "Net sales were $90 million in 2024.")["ratio"] == 0.0
