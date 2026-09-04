import copy

import pytest

from backend.evidence_answerability_ranker import WEIGHTS, answer_type, numbers, rank_answerability, score_answerability


def unit(rank=1, text="Production volume reached 170 in 2024.", **kwargs):
    return dict(document_id="doc", page_id="doc:p0", source_type="text", entity="Example Entity",
                period=["2024"], metric=None, value=None, unit=None, source_text=text,
                metadata={"chunk_id": str(rank), "page_number": 0, "filename": "file.pdf"},
                current_ranking={"rank": rank, "score": 1 / rank}) | kwargs


@pytest.mark.parametrize("query,expected", [
    ("How much was production volume?", "numeric"), ("Compare production across 2023 and 2024", "comparison"),
    ("Which location had the largest volume?", "selection"), ("Why did operations stop?", "explanation"),
    ("What did the note disclose?", "lookup"),
])
def test_generic_answer_types(query, expected):
    assert answer_type(query) == expected


def test_numeric_presence_excludes_years_and_saturates():
    assert not numbers("Year 2024")
    a = score_answerability("How much production volume?", unit(text="Production volume 170."))
    b = score_answerability("How much production volume?", unit(text="Production volume 170 180 190."))
    assert a["features"]["numeric_presence"] == b["features"]["numeric_presence"] == 1


def test_local_support_beats_disconnected_numbers():
    query = "How much production volume in 2024?"
    linked = score_answerability(query, unit())
    disconnected = score_answerability(query, unit(text="Production volume discussed.\nUnrelated costs were 170 in 2024."))
    assert linked["local_numeric_support"] > disconnected["local_numeric_support"]
    assert linked["score"] > disconnected["score"]


def test_unknown_period_and_entity_are_not_conflicts():
    item = score_answerability("Production volume in 2024?", unit(text="Production volume disclosed.", period=[], entity=""))
    assert item["period_status"] == "unknown"
    assert item["entity_status"] == "unspecified_or_unresolved"
    assert item["features"]["period_consistency"] == 0.5


def test_explicit_period_mismatch_and_metadata_only_are_distinguished():
    mismatch = score_answerability("Production volume 2024?", unit(text="Production volume 2023 was 100."))
    metadata = score_answerability("Production volume 2024?", unit(text="Production volume was 100."))
    assert mismatch["period_status"] == "text_mismatch"
    assert mismatch["features"]["period_consistency"] == 0
    assert metadata["period_status"] == "metadata_only_match"
    assert metadata["features"]["period_consistency"] == 0.5


def test_ranker_is_gold_free_and_preserves_input():
    candidates = [unit(1), unit(2, text="Unrelated narrative.")]
    original = copy.deepcopy(candidates)
    expected = rank_answerability("How much production volume in 2024?", candidates)
    polluted = copy.deepcopy(candidates)
    for value in polluted:
        value.update(financebench_id="sentinel", gold_evidence="999", reference_answer="special")
    assert rank_answerability("How much production volume in 2024?", polluted) == expected
    assert candidates == original


def test_scores_bounded_and_zero_empty_evidence():
    assert sum(WEIGHTS.values()) == pytest.approx(1)
    for text in ("", "Short", "Production volume was 170 in 2024."):
        score = score_answerability("How much production volume?", unit(text=text))
        assert 0 <= score["score"] <= 1
        assert all(0 <= value <= 1 for value in score["features"].values())
    assert score_answerability("Question", unit(text=""))["score"] == 0


def test_ties_use_original_rank_and_duplicate_rank_rejected():
    assert [item["original_rank"] for item in rank_answerability("volume", [unit(2), unit(1)])] == [1, 2]
    with pytest.raises(ValueError):
        rank_answerability("volume", [unit(1), unit(1)])


def test_adapter_changes_only_shadow_ranking_and_packing_respects_budget():
    from scripts.evaluate_evidence_answerability_v1 import shadow_inputs
    from backend.evidence_packing_v1 import select_evidence_packing_v1

    candidates = [unit(1), unit(2, text="Unrelated narrative discussion.")]
    original = copy.deepcopy(candidates)
    ranked = shadow_inputs(candidates, rank_answerability("How much production volume?", candidates))
    assert candidates == original
    for old, new in zip(candidates, ranked):
        assert {k: v for k, v in old.items() if k != "current_ranking"} == {k: v for k, v in new.items() if k != "current_ranking"}
    context, _, trace = select_evidence_packing_v1("How much production volume?", ranked, max_context_chars=300)
    assert len(context) <= 300
    assert trace["replacement_threshold"] == 1.0
    assert trace["max_replacements"] is None
    assert trace["protected_anchor_count"] == 0


def test_table_structure_completeness_requires_header_and_row():
    query = "How much production volume in 2024?"
    complete = score_answerability(query, unit(source_type="table", text="Header: 2024 | 2023\nRow: Production volume | 170 | 160"))
    missing_header = score_answerability(query, unit(source_type="table", text="Row: Production volume | 170 | 160"))
    assert complete["table_header_present"] and complete["table_row_present"]
    assert complete["features"]["evidence_completeness"] > missing_header["features"]["evidence_completeness"]


def test_entity_name_alone_does_not_count_as_target_relevance():
    result = score_answerability("What was Example Entity production volume?", unit(text="Example Entity has offices."))
    assert result["features"]["query_lexical_relevance"] == 0
    assert result["entity_status"] == "query_metadata_text_match"


@pytest.mark.parametrize("period", ["2024", "FY2024", "FY 2024", "fy2024", "FY-2024"])
def test_period_format_does_not_change_year_or_lexical_support(period):
    result = score_answerability(f"How much production volume in {period}?", unit())
    plain = score_answerability("How much production volume in 2024?", unit())
    assert result["query_periods"] == ["2024"]
    assert result["period_status"] == "text_match"
    assert result["features"] == plain["features"]
