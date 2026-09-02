from backend.evidence_assembly_v5 import EvidenceUnit
from scripts.audit_evidence_budget_v1 import (
    _loss_reason,
    audit_selected_unit,
    summarize,
)


def _unit():
    return EvidenceUnit(
        document_id="doc_generic",
        page_id="doc_generic:page:000005",
        source_type="table",
        entity="Generic Corp",
        period=["2024", "2023"],
        metric="Revenue",
        value=["120", "100"],
        unit="USD millions",
        source_text="Header: Metric | 2024 | 2023\nRow: Revenue | 120 | 100",
        metadata={
            "filename": "generic.pdf",
            "page_number": 5,
            "table_id": "table_1",
            "row_index": 0,
            "retrieval_rank": 4,
            "query_overlap": 2,
        },
    )


def test_unit_audit_reports_required_fields_and_matches():
    audited = audit_selected_unit(
        _unit().to_dict(), index=1,
        gold=[{"evidence_text": "Revenue | 120 | 100"}],
        required_numbers=["120", "100"], required_periods=["2024", "2023"],
    )

    assert audited["unit_id"].startswith("table:doc_generic:page:000005:table_1")
    assert audited["source_type"] == "table"
    assert audited["rank_score"] == 2.25
    assert audited["is_from_top_chunk"] is False
    assert audited["is_from_table"] is True
    assert audited["gold_evidence_covered"] is True
    assert audited["matched_required_numbers"] == ["120", "100"]
    assert audited["matched_required_periods"] == ["2024", "2023"]
    assert audited["character_length"] > audited["source_text_length"]


def test_loss_reason_distinguishes_retrieval_budget_and_unit_representation():
    gold = {("generic.pdf", 5)}

    assert _loss_reason(
        gold_pages=gold, candidate_pages=set(), selected_pages=set(),
        candidate_coverage=0, selected_coverage=0,
    ) == "gold_page_not_in_top120"
    assert _loss_reason(
        gold_pages=gold, candidate_pages=gold, selected_pages=set(),
        candidate_coverage=0.5, selected_coverage=0,
    ) == "gold_page_dropped_by_28k_budget"
    assert _loss_reason(
        gold_pages=gold, candidate_pages=gold, selected_pages=gold,
        candidate_coverage=0, selected_coverage=0,
    ) == "gold_text_not_represented_in_candidate_units"
    assert _loss_reason(
        gold_pages=gold, candidate_pages=gold, selected_pages=gold,
        candidate_coverage=0.8, selected_coverage=0.4,
    ) == "gold_evidence_units_dropped_by_28k_budget"


def _record(group: str, *, waste: float, coverage: float) -> dict:
    return {
        "financebench_id": group,
        "group": group,
        "context_chars": 100,
        "unit_character_total_excluding_separators": 100,
        "assembly_trace": {"max_context_chars": 100},
        "non_gold_character_ratio": waste,
        "conservative_budget_waste_ratio": waste,
        "selected_evidence_coverage": coverage,
        "candidate_evidence_coverage": 1.0,
        "required_numbers": ["120"],
        "required_periods": ["2024"],
        "gold_evidence_loss_reason": "gold_evidence_units_dropped_by_28k_budget",
        "selected_units": [{
            "source_type": "text",
            "character_length": 100,
            "contains_required_number": True,
            "contains_required_period": True,
        }],
    }


def test_summary_compares_correct_and_selection_loss_groups():
    records = [
        _record("candidate_miss10", waste=0.8, coverage=0.2),
        _record("selection_loss10", waste=0.6, coverage=0.4),
        _record("correct_regression10", waste=0.2, coverage=0.8),
    ]

    result = summarize(records)

    assert result["questions"] == 3
    assert result["groups"]["selection_loss10"]["conservative_budget_waste_ratio"] == 0.6
    assert result["correct_vs_selection_loss"]["conservative_budget_waste_ratio"] == -0.4
    assert result["correct_vs_selection_loss"]["selected_evidence_coverage"] == 0.4
    assert result["external_calls"] == {"jina": 0, "answer_model": 0, "judge": 0, "langsmith": 0}
