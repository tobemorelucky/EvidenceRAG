from scripts.audit_evidence_metadata_v2 import (
    choose_next_direction,
    entity_metadata,
    metric_metadata,
    period_metadata,
)


def test_unselected_metadata_is_unobservable_not_missing():
    result = entity_metadata(None, {"company": "Example Corp"}, observable=False)

    assert result["status"] == "unobservable"
    assert result["confidence"] is None


def test_entity_uses_page_identity_without_company_specific_rules():
    page = {"company": "Example Corp", "doc_name": "EXAMPLE_2024_10K", "filename": "example.pdf"}

    assert entity_metadata("Example Corp", page, observable=True)["status"] == "correct"
    assert entity_metadata("Different Corp", page, observable=True)["status"] == "conflict"


def test_period_feature_is_only_a_feature_level_conflict_signal():
    result = period_metadata(
        None, observable=False, required_periods=["2024"], period_match_feature=0.0,
    )

    assert result["status"] == "conflict"
    assert result["observation"] == "feature_only"
    assert result["value"] is None


def test_metric_confidence_uses_generic_lexical_support():
    result = metric_metadata(
        "Operating income",
        source_type="table",
        observable=True,
        question="What was operating income?",
        gold_text="",
    )

    assert result["status"] == "correct"
    assert result["confidence"] == 0.9


def test_direction_prefers_metadata_only_with_group_separation():
    groups = {
        "selection_loss10": {
            "questions": 10,
            "gold_conflict_rates": {"period": 0.8},
            "packing_skip_questions": 5,
        },
        "correct_regression10": {
            "gold_conflict_rates": {"period": 0.4},
        },
    }

    assert choose_next_direction(groups)["direction"] == "metadata"
