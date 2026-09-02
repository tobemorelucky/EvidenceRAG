import json

from backend.evidence_assembly_v5 import EvidenceUnit
from backend.evidence_ranking_v1 import (
    rank_evidence_units_v1,
    score_evidence_unit,
    select_ranked_evidence_v1,
)
from scripts.evaluate_evidence_ranking_v1 import gold_evidence_retention, oracle_budget_curve


def _unit(text: str, *, rank: int, source_type: str = "text", period=None, unit=None):
    return EvidenceUnit(
        document_id="doc_generic",
        page_id=f"doc_generic:page:{rank:06d}",
        source_type=source_type,
        entity="Generic Corp",
        period=period or [],
        metric="Revenue" if source_type == "table" else None,
        value=["120"] if source_type == "table" else None,
        unit=unit,
        source_text=text,
        metadata={"filename": "generic.pdf", "page_number": rank, "retrieval_rank": rank},
    )


def test_scoring_uses_only_declared_generic_features():
    unit = _unit("Revenue was $120 million in 2024.", rank=2, period=["2024"])

    features = score_evidence_unit("What was revenue in 2024?", unit)

    assert set(features) == {
        "retrieval_score", "query_lexical_overlap", "numeric_presence",
        "period_match", "unit_completeness",
    }
    assert features["retrieval_score"] == 0.5
    assert features["query_lexical_overlap"] > 0
    assert features["numeric_presence"] == 1.0
    assert features["period_match"] == 1.0


def test_ranking_can_promote_query_matching_lower_retrieval_unit():
    high_rank_irrelevant = _unit("General corporate information without figures.", rank=1)
    lower_rank_relevant = _unit("Revenue was 120 in 2024.", rank=8, period=["2024"])

    ranked = rank_evidence_units_v1("What was revenue in 2024?", [high_rank_irrelevant, lower_rank_relevant])

    assert ranked[0]["source_text"] == "Revenue was 120 in 2024."


def test_ranked_selector_respects_same_character_budget():
    units = [
        _unit("Revenue was 120 in 2024." * 5, rank=3, period=["2024"]),
        _unit("Unrelated narrative." * 20, rank=1),
    ]

    context, selected, trace = select_ranked_evidence_v1(
        "What was revenue in 2024?", units, max_context_chars=500,
    )

    assert len(context) <= 500
    assert selected
    assert selected[0]["source_text"].startswith("Revenue")
    assert trace["max_context_chars"] == 500
    assert trace["selected_unit_count"] == len(selected)


def test_gold_retention_is_relative_to_candidate_matched_lines():
    gold = [{"evidence_text": "Revenue was 120.\nIncome was 30."}]

    retention = gold_evidence_retention(
        gold, "Revenue was 120. Income was 30.", "Revenue was 120.",
    )

    assert retention == {
        "candidate_matched_lines": 2,
        "selected_matched_lines": 1,
        "ratio": 0.5,
    }


def test_oracle_budget_curve_is_monotonic_and_uses_direct_page_number():
    row = {
        "evidence": json.dumps([{
            "doc_name": "GENERIC_2024_10K",
            "evidence_page_num": 59,
            "evidence_text": "Revenue was 120 in 2024.",
            "evidence_text_full_page": "Revenue was 120 in 2024. " + "context " * 2000,
        }]),
    }

    curve = oracle_budget_curve(row, budgets=(100, 500, 1000))

    values = [curve[str(budget)]["strict_evidence_coverage"] for budget in (100, 500, 1000)]
    assert values == sorted(values)
    assert all(curve[str(budget)]["context_chars"] <= budget for budget in (100, 500, 1000))
