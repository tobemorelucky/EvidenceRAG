from backend.evidence_packing_v1 import (
    coverage_features,
    near_duplicate_similarity,
    select_evidence_packing_v1,
)


def _unit(rank: int, text: str, *, score: float = 1.0, page: int | None = None):
    return {
        "document_id": "doc",
        "page_id": f"page-{page if page is not None else rank}",
        "source_type": "text",
        "entity": "Example Corp",
        "period": [],
        "metric": None,
        "value": None,
        "unit": None,
        "source_text": text,
        "metadata": {
            "filename": "example.pdf", "page_number": page if page is not None else rank,
            "chunk_id": f"chunk-{rank}",
        },
        "current_ranking": {"rank": rank, "score": score},
    }


def test_coverage_features_are_gold_free_and_generic():
    features = coverage_features(
        "What was revenue in 2024?",
        _unit(1, "Revenue was $120 in 2024."),
    )

    assert "query:revenue" in features
    assert "period:2024" in features
    assert "number:120" in features


def test_packing_respects_budget_and_preserves_candidate_objects():
    candidates = [
        _unit(1, "Revenue was 120 in 2024. " * 30),
        _unit(2, "Income was 30 in 2024. " * 20),
        _unit(3, "Assets were 400 in 2024. " * 20),
    ]
    context, selected, trace = select_evidence_packing_v1(
        "Compare revenue and income in 2024", candidates, max_context_chars=1800,
    )

    assert len(context) <= 1800
    assert trace["candidate_unit_count"] == len(candidates)
    assert all(unit in candidates for unit in selected)
    assert trace["gold_used_for_selection"] is False


def test_near_duplicate_requires_same_numeric_content():
    shared = "header revenue operating income assets liabilities " * 8

    assert near_duplicate_similarity(shared + " 120", shared + " 30") == 0.0
    assert near_duplicate_similarity(shared + " 120", shared + " 120") > 0.92


def test_page_repetition_is_penalty_not_hard_limit():
    candidates = [
        _unit(1, "Revenue 100 in 2024", page=1),
        _unit(2, "Income 20 in 2024", page=1),
        _unit(3, "Assets 300 in 2024", page=1),
    ]
    _, selected, _ = select_evidence_packing_v1(
        "Revenue income assets 2024", candidates, max_context_chars=5000,
    )

    assert len(selected) == 3
