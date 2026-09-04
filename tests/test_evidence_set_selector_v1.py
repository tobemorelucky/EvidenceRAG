import copy

import pytest

from backend.evidence_packing_v1 import _render_selection
from backend.evidence_set_selector_v1 import FIELDS, marginal_gain, metadata_features, select_evidence_set_v1


def unit(rank, score=1.0, **changes):
    return dict(document_id="doc", page_id="doc:p0", source_type="text", entity="Entity",
                period=["2024"], metric="quantity", value=["100"], unit="items",
                source_text="Reported production quantity was 100 items in 2024.",
                metadata={"chunk_id": str(rank), "page_number": 0, "filename": "file.pdf"},
                current_ranking={"rank": rank, "score": score}) | changes


def test_dynamic_choice_recomputes_after_first_unit():
    units = [unit(1, 1), unit(2, 0.9), unit(3, 0.8, page_id="doc:p1", value=["200"], source_text="A separate measured volume is 200.")]
    _, _, trace = select_evidence_set_v1(units)
    assert trace["selection_order"] == [1, 3, 2]
    assert trace["steps"][1]["novelty_by_field"]["numeric_value"]["new_values"] == ["200"]
    assert trace["steps"][2]["marginal_information_gain"] == 0
    assert trace["dynamic_utility_evaluations"] == 6


def test_gain_is_bounded_and_missing_fields_get_no_reward():
    covered = {name: set() for name in FIELDS}
    empty = metadata_features(unit(1, entity=None, period=[], metric=None, value=[], unit=None))
    assert marginal_gain(empty, covered)[0] == 0
    full = metadata_features(unit(1))
    assert marginal_gain(full, covered)[0] == 1
    assert marginal_gain(full, full)[0] == 0
    many = metadata_features(unit(1, value=list(range(100, 300))))
    assert marginal_gain(many, covered)[0] == 1


def test_decimal_normalization_preserves_sign_percent_currency_and_excludes_period():
    features = metadata_features(unit(1, value=["1,000.00", "1000", "(100)", "100", "100%", "$100", "2024", "2024,", "unknown"]))
    assert features["numeric_value"] == {"1000", "-100", "100", "100%", "$100"}


def test_page_text_and_numeric_redundancy_are_visible_soft_penalties():
    units = [unit(1), unit(2, 0.9)]
    _, selected, trace = select_evidence_set_v1(units)
    assert len(selected) == 2
    second = trace["steps"][1]
    assert second["same_page_penalty"] == 0.5
    assert second["max_text_similarity"] == 1
    assert second["max_numeric_similarity"] == 1
    assert second["redundancy_denominator"] == 3.5


def test_different_documents_do_not_share_page_penalty():
    _, _, trace = select_evidence_set_v1([unit(1), unit(2, 0.9, document_id="other", page_id="doc:p0")])
    assert trace["steps"][1]["same_page_penalty"] == 0


def test_empty_numeric_sets_do_not_count_as_numeric_duplicates():
    _, _, trace = select_evidence_set_v1([unit(1, value=[]), unit(2, value=[])])
    assert trace["steps"][1]["max_numeric_similarity"] == 0


def test_exact_budget_and_multidigit_ordinals():
    units = [unit(rank, source_text=f"Fact {rank}.") for rank in range(1, 14)]
    expected_chars = len(_render_selection(units)[0])
    context, selected, trace = select_evidence_set_v1(units[::-1], max_context_chars=expected_chars)
    assert len(selected) == 13
    assert len(context) == expected_chars == trace["context_chars"]
    context, selected, trace = select_evidence_set_v1(units, max_context_chars=expected_chars - 1)
    assert len(context) <= expected_chars - 1
    assert len(selected) < 13
    assert all(item["reason"] == "budget" for item in trace["unselected"])


def test_oversized_unit_is_skipped_without_truncation():
    units = [unit(1, source_text="Long fact. " * 1000), unit(2, source_text="Short fact.")]
    context, selected, _ = select_evidence_set_v1(units, max_context_chars=500)
    assert selected == [units[1]]
    assert len(context) <= 500


def test_candidate_fields_and_ranking_preserved_gold_fields_ignored():
    units = [unit(1), unit(2, 0.8)]
    original = copy.deepcopy(units)
    expected = select_evidence_set_v1(units)
    annotated = copy.deepcopy(units)
    for value in annotated:
        value.update(financebench_id="sentinel", gold_evidence="Do not use", reference_answer="999")
    actual = select_evidence_set_v1(annotated)
    assert expected[0] == actual[0]
    assert expected[2] == actual[2]
    assert units == original


def test_tie_breaking_is_deterministic_and_zero_budget_is_empty():
    units = [unit(2), unit(1)]
    assert select_evidence_set_v1(units)[2]["selection_order"][0] == 1
    assert select_evidence_set_v1(units, max_context_chars=0)[:2] == ("", [])
    assert select_evidence_set_v1([])[:2] == ("", [])


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1])
def test_invalid_scores_rejected(score):
    with pytest.raises(ValueError):
        select_evidence_set_v1([unit(1, score)])


def test_duplicate_ranks_and_negative_budget_rejected():
    with pytest.raises(ValueError):
        select_evidence_set_v1([unit(1), unit(1)])
    with pytest.raises(ValueError):
        select_evidence_set_v1([unit(1)], max_context_chars=-1)


def test_empty_source_text_is_not_selected():
    _, selected, trace = select_evidence_set_v1([unit(1, source_text=""), unit(2)])
    assert len(selected) == 1
    assert trace["unselected"][0]["reason"] == "empty_source_text"


def test_zero_gain_zero_score_stops():
    empty = unit(1, 0, entity=None, period=[], metric=None, value=[], unit=None)
    _, selected, trace = select_evidence_set_v1([empty])
    assert selected == []
    assert trace["unselected"][0]["reason"] == "nonpositive_utility"
