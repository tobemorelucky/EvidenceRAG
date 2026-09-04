import pytest

from backend.evidence_packing_guard_v1 import (
    select_anchor_keys,
    select_evidence_packing_guard_v1,
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
            "filename": "example.pdf",
            "page_number": page if page is not None else rank,
            "chunk_id": f"chunk-{rank}",
        },
        "current_ranking": {"rank": rank, "score": score},
    }


def test_anchor_protection_requires_both_top_rank_and_relevance():
    units = [
        _unit(1, "Revenue in 2024 was 120."),
        _unit(2, "Unrelated governance discussion."),
        _unit(3, "Revenue in 2024 was 100."),
    ]

    anchors = select_anchor_keys(
        "What was revenue in 2024?",
        units,
        anchor_top_n=2,
        anchor_min_query_relevance=0.3,
    )

    assert len(anchors) == 1
    assert next(iter(anchors))[2] == "chunk-1"


@pytest.mark.parametrize("threshold", [1.05, 1.10])
@pytest.mark.parametrize("replacement_limit", [None, 5, 10])
def test_guard_respects_budget_and_configuration(threshold, replacement_limit):
    units = [
        _unit(index, f"Revenue income assets in 2024 value {index}. " * (5 + index), score=1 / index)
        for index in range(1, 16)
    ]
    context, _, trace = select_evidence_packing_guard_v1(
        "Compare revenue income and assets in 2024",
        units,
        max_context_chars=2500,
        replacement_threshold=threshold,
        max_replacements=replacement_limit,
    )

    assert len(context) <= 2500
    assert trace["replacement_threshold"] == threshold
    assert trace["max_replacements"] == replacement_limit
    assert trace["gold_used_for_selection"] is False
    if replacement_limit is not None:
        assert trace["replacement_count"] <= replacement_limit


def test_invalid_guard_values_fail_fast():
    units = [_unit(1, "Revenue in 2024 was 120.")]

    with pytest.raises(ValueError):
        select_evidence_packing_guard_v1(
            "Revenue 2024", units, replacement_threshold=0.99,
        )
    with pytest.raises(ValueError):
        select_evidence_packing_guard_v1(
            "Revenue 2024", units, max_replacements=-1,
        )
