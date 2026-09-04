import copy

import pytest

from backend.evidence_bundle_v1 import build_evidence_bundles, pack_evidence_bundles, unit_id
from backend.evidence_packing_v1 import _render_selection, select_evidence_packing_v1


def unit(rank, page=0, doc="document", period=None, entity="Entity", text=None):
    return {
        "document_id": doc, "page_id": f"{doc}:page:{page}", "source_type": "text",
        "entity": entity, "period": ["2024"] if period is None else period,
        "metric": None, "value": None, "unit": None,
        "source_text": text or f"Revenue income assets increased by {rank} in 2024.",
        "metadata": {"chunk_id": f"{doc}:chunk:{rank}", "page_number": page, "filename": "same.pdf"},
        "current_ranking": {"rank": rank, "score": 1 / rank},
    }


def test_same_page_partition_and_weighted_score():
    units = [unit(1), unit(2)]
    original = copy.deepcopy(units)
    bundles = build_evidence_bundles("Revenue 2024", units)
    assert len(bundles) == 1
    bundle = bundles[0]
    assert bundle["bundle_type"] == "same_page"
    assert set(bundle["unit_ids"]) == {unit_id(u) for u in units}
    lengths = [len(_render_selection([u])[0]) for u in units]
    assert bundle["score"] == pytest.approx(sum(u["current_ranking"]["score"] * n for u, n in zip(units, lengths)) / sum(lengths))
    assert units == original


@pytest.mark.parametrize("change", [
    {"doc": "other"}, {"entity": "Other entity"}, {"period": ["2023"]},
])
def test_same_page_never_crosses_document_or_known_metadata_conflict(change):
    assert len(build_evidence_bundles("Revenue", [unit(1), unit(2, **change)])) == 2


def test_unknown_period_does_not_bridge_conflicting_periods():
    units = [unit(1, period=[]), unit(2, period=["2023"]), unit(3, period=["2024"])]
    bundles = build_evidence_bundles("Revenue", units)
    assert len(bundles) == 2
    assert not any({unit_id(units[1]), unit_id(units[2])} <= set(b["unit_ids"]) for b in bundles)


def test_adjacent_pairs_need_distance_overlap_and_lexical_similarity():
    units = [unit(1), unit(2, page=2)]
    assert build_evidence_bundles("Revenue", units)[0]["bundle_type"] == "adjacent_page"
    for other in [unit(2, page=3), unit(2, page=2, period=[]), unit(2, page=2, text="Unrelated governance committee discussion")]:
        assert len(build_evidence_bundles("Revenue", [unit(1), other])) == 2


def test_no_transitive_page_chain_or_duplicate_membership():
    units = [unit(rank, page=rank - 1) for rank in range(1, 5)]
    bundles = build_evidence_bundles("Revenue", units)
    assert all(len(b["pages"]) <= 2 for b in bundles)
    ids = [uid for b in bundles for uid in b["unit_ids"]]
    assert len(ids) == len(set(ids)) == len(units)


def test_builder_ignores_benchmark_annotations_and_is_order_invariant():
    units = [unit(1), unit(2), unit(3, page=1)]
    expected = build_evidence_bundles("Revenue 2024", units)
    annotated = copy.deepcopy(units)
    for u in annotated:
        u.update(gold_evidence="DO NOT READ", reference_answer="999", financebench_id="sentinel")
    assert build_evidence_bundles("Revenue 2024", annotated[::-1]) == expected


def test_atomic_packing_does_not_split_oversized_bundle():
    units = [unit(1), unit(2)]
    bundles = build_evidence_bundles("Revenue", units)
    context, selected, trace = pack_evidence_bundles(units, bundles, budget=bundles[0]["length"] - 1)
    assert context == "" and selected == []
    assert trace["trace"][0]["reason"] == "oversize_atomic_bundle"


def test_singletons_match_v1_and_budget_with_no_guard():
    units = [unit(rank, page=rank * 10, text=f"Revenue was {rank} in 2024. " * 15) for rank in range(1, 10)]
    question = "Revenue 2024"
    bundles = build_evidence_bundles(question, units)
    assert all(b["bundle_type"] == "singleton" for b in bundles)
    old, old_selected, _ = select_evidence_packing_v1(question, units, max_context_chars=2000)
    new, new_selected, trace = pack_evidence_bundles(units, bundles, budget=2000)
    assert old == new
    assert old_selected == new_selected
    assert len(new) <= 2000
    for event in trace["replacement_events"]:
        assert event["new_total_utility"] > event["old_total_utility"]


def test_invalid_partition_and_duplicate_candidates_rejected():
    units = [unit(1)]
    with pytest.raises(ValueError):
        build_evidence_bundles("Revenue", units * 2)
    bundles = build_evidence_bundles("Revenue", units)
    with pytest.raises(ValueError):
        pack_evidence_bundles(units, bundles * 2)


def test_empty_inputs():
    assert build_evidence_bundles("Revenue", []) == []
    context, selected, _ = pack_evidence_bundles([], [])
    assert context == "" and selected == []


def test_baseline_event_replay_and_gold_only_offline_audit():
    from scripts.evaluate_evidence_bundle_v1 import replay_baseline_events, audit_events

    units = [unit(rank, page=rank * 10) for rank in range(1, 10)]
    question = "Revenue 2024"
    _, selected, trace = select_evidence_packing_v1(question, units, max_context_chars=1000)
    prior = {"packing_trace": trace, "selected_unit_ranks": [u["current_ranking"]["rank"] for u in selected]}
    events = replay_baseline_events(question, units, prior)
    assert len(events) == trace["replacement_count"]
    audit = audit_events([{"evidence_text": units[0]["source_text"]}], units, events)
    assert audit["count"] == len(events)


def test_offline_event_audit_detects_actual_evidence_loss():
    from scripts.evaluate_evidence_bundle_v1 import audit_events

    units = [unit(1, text="Revenue increased to 913 in 2024."), unit(2, text="Committee governance discussion.")]
    events = [{"before_unit_ids": [unit_id(units[0])], "after_unit_ids": [unit_id(units[1])]}]
    audit = audit_events([{"evidence_text": units[0]["source_text"]}], units, events)
    assert audit["coverage_decreasing_count"] == 1


def test_selected_multimember_bundles_are_complete():
    units = [unit(rank, page=(rank // 3) * 10) for rank in range(1, 16)]
    bundles = build_evidence_bundles("Revenue 2024", units)
    context, selected, trace = pack_evidence_bundles(units, bundles, budget=2000)
    selected_ids = {unit_id(u) for u in selected}
    for bundle in bundles:
        members = set(bundle["unit_ids"])
        assert not members.intersection(selected_ids) or members <= selected_ids
    assert len(context) <= 2000
    for event in trace["replacement_events"]:
        assert event["new_total_utility"] > event["old_total_utility"]
