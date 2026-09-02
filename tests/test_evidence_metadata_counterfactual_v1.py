from scripts.evaluate_evidence_metadata_counterfactual_v1 import (
    _feature_distance,
    _raw_retrieval_rank,
    _selected_unit_match,
    extract_counterfactual_metadata,
)


def test_raw_retrieval_rank_is_recovered_without_retrieval():
    trace = {"ranking_v1_features": {"retrieval_score": 0.05}}

    assert _raw_retrieval_rank(trace) == 20


def test_period_source_priority_uses_local_before_document_metadata():
    unit = {
        "source_type": "text",
        "entity": "Example Corp",
        "period": [],
        "metric": None,
        "source_text": "Revenue increased during 2023.",
        "metadata": {},
    }
    corrected, trace = extract_counterfactual_metadata(
        "What was revenue in 2023?",
        unit,
        {"company": "Example Corp", "report_year": 2024, "page_text": "Annual report 2024"},
        nearby_text="Comparison for 2022",
    )

    assert corrected["period"] == ["2023"]
    assert trace["period"]["source"] == "local_text"


def test_document_period_is_only_a_fallback():
    unit = {
        "source_type": "text", "entity": "", "period": [], "metric": None,
        "source_text": "Revenue increased.", "metadata": {},
    }
    corrected, trace = extract_counterfactual_metadata(
        "What was revenue?", unit,
        {"company": "Example Corp", "report_year": 2024, "page_text": "Annual report"},
    )

    assert corrected["period"] == ["2024"]
    assert trace["period"]["source"] == "document_metadata"
    assert corrected["entity"] == "Example Corp"


def test_feature_distance_ignores_unchanged_retrieval_component():
    expected = {
        "retrieval_score": 1.0, "query_lexical_overlap": 0.5,
        "numeric_presence": 1.0, "period_match": 1.0, "unit_completeness": 1.0,
    }
    actual = {**expected, "retrieval_score": 0.1}

    assert _feature_distance(expected, actual) == 0.0


def test_selected_replay_validation_requires_identity_and_text():
    metadata = {"chunk_id": "chunk-1", "table_id": None, "row_index": None}
    frozen = [{"source_type": "text", "page_id": "page-1", "source_text": "fact", "metadata": metadata}]

    assert _selected_unit_match(frozen, frozen)["selected_units_valid"] is True
    changed = [{**frozen[0], "source_text": "different"}]
    assert _selected_unit_match(changed, frozen)["selected_units_valid"] is False
