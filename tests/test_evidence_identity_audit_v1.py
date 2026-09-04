import copy
import inspect
import json

import pytest

from scripts.audit_evidence_identity_v1 import (
    build_registry, check_period, facts_from_reference, inspect_unit,
    legacy_hit, literal_hit, profile, question_requirement,
)


def unit(text="Output was 100 in 2024.", **kwargs):
    return dict(document_id="doc", page_id="doc:p0", entity="Example", period=["2024"],
                metric="Output", value=["100"], unit=None, source_type="text", source_text=text,
                metadata={"filename": "example.pdf", "page_number": 0, "chunk_id": "u"},
                current_ranking={"rank": 1, "score": 1}) | kwargs


def req(**kwargs):
    return dict(entity=["example"], entity_reliable=True, period=["2024"], scope=[],
                metric={"question_terms": ["output"]}, required_number=["100"]) | kwargs


def fact(**kwargs):
    return dict(fact_id=0, text="Output was 100 in 2024.", document_id="doc", page_id="doc:p0") | kwargs


def test_foreign_entity_is_A_and_never_F_despite_same_numbers():
    result = inspect_unit(unit(entity="Other", document_id="other"), req(), [fact()])
    assert result["confirmed_categories"] == ["A"]
    assert result["qualified_fact_ids"] == []
    assert result["values"]["required_present"] == ["100"]


@pytest.mark.parametrize("entity", ["", "unknown", "None", None])
def test_unknown_entity_is_not_mismatch_or_correct(entity):
    result = inspect_unit(unit(entity=entity), req(), [fact()])
    assert result["entity_status"] == "unknown"
    assert result["confirmed_categories"] == []
    assert result["unverified"]


def test_period_conflict_partial_and_metadata_only_differ():
    assert check_period(["2024"], ["2023"], ["2024"])[0] == "visible_period_conflict"
    assert check_period(["2023", "2024"], ["2023"], [])[0] == "partial"
    assert check_period(["2024", "q1"], ["2024", "q2"], [])[0] == "visible_period_conflict"
    assert check_period(["2024"], [], ["2024"])[0] == "metadata_only"
    assert check_period([], [], [])[0] == "not_requested"


def test_partial_period_support_is_not_B_for_comparison_operand():
    result = inspect_unit(unit(), req(period=["2023", "2024"]), [fact()])
    assert "B" not in result["suspected_categories"]
    assert "F" in result["confirmed_categories"]


def test_missing_metric_not_forced_to_C():
    result = inspect_unit(unit(text="Further details.", metric=None), req(), [])
    assert "C" not in result["suspected_categories"]
    assert result["metric"]["status"] == "unverified"


def test_metric_lexical_mismatch_only_suspected():
    result = inspect_unit(unit(text="Volume was 100 in 2024.", metric="Volume"), req(), [])
    assert "C" in result["suspected_categories"]
    assert "C" not in result["confirmed_categories"]


def test_scope_mismatch_requires_explicit_metadata_not_absence():
    requested = req(scope=["income_statement"])
    unknown = inspect_unit(unit(), requested, [fact()])
    assert unknown["scope"]["status"] == "unknown"
    assert "D" not in unknown["suspected_categories"]
    conflict = inspect_unit(unit(metadata={"scope": "balance sheet"}), requested, [fact()])
    assert "D" in conflict["suspected_categories"]
    assert "F" not in conflict["confirmed_categories"]


def test_numeric_absence_not_a_proven_false_fact():
    result = inspect_unit(unit(text="Output may vary in 2024."), req(), [])
    assert "E" in result["suspected_categories"] and not result["confirmed_categories"]


def test_F_requires_stable_document_and_page_and_literal_span():
    assert inspect_unit(unit(), req(), [fact()])["confirmed_categories"] == ["F"]
    assert inspect_unit(unit(page_id="doc:p1"), req(), [fact()])["qualified_fact_ids"] == []
    assert inspect_unit(unit(), req(), [fact(page_id=None)])["qualified_fact_ids"] == []
    assert inspect_unit(unit(text="Output was 1000 in 2024."), req(), [fact()])["qualified_fact_ids"] == []


def test_legacy_collision_vs_literal_reference():
    assert legacy_hit("output 100", profile("Output elsewhere. 100 other things."))
    assert not literal_hit("output 100", "Output elsewhere. 100 other things.")
    assert not literal_hit("output 100", "output 1000")
    assert literal_hit("output 100", "\nOutput   100\n")


def test_bare_values_are_not_guessed_into_facts():
    requested = req() | {"parsed_requirement": {"requires_numeric_evidence": True},
        "reference_documents": [{"document_id": "doc", "page_id": "doc:p0", "page_number": 0,
                                  "evidence_text": "100\nOutput\nOutput was 100 in 2024.\nUnrelated expenses were 10."}]}
    facts, excluded = facts_from_reference(requested)
    assert len(facts) == 1
    assert facts[0]["text"] == "Output was 100 in 2024."
    assert excluded["bare_value_or_short_header"] == 2


def test_reference_filename_bridge_detects_ambiguity_and_preserves_zero_page():
    row = {"question": "How much output in 2024?", "answer": "100", "justification": "",
           "evidence": '[{"doc_name":"example", "evidence_page_num":0, "evidence_text":"Output was 100 in 2024."}]'}
    good = build_registry([{"candidate_units": [unit()]}])
    r = question_requirement(row, good)
    assert r["reference_documents"][0]["page_id"] == "doc:p0"
    assert r["entity_source"].startswith("offline gold")
    ambiguous = build_registry([{"candidate_units": [unit(), unit(document_id="doc2")]}])
    r = question_requirement(row, ambiguous)
    assert not r["entity_reliable"]
    assert r["unresolved_reference_mappings"]


def test_inconsistent_document_entities_do_not_establish_identity():
    row = {"question": "Output?", "evidence": '[{"doc_name":"example", "evidence_page_num":0}]'}
    registry = build_registry([{"candidate_units": [unit(), unit(entity="Other")]}])
    assert not question_requirement(row, registry)["entity_reliable"]


def test_inspection_does_not_mutate_input_or_select():
    u, r, f = unit(), req(), [fact()]
    saved = copy.deepcopy((u, r, f))
    assert inspect_unit(u, r, f) == inspect_unit(u, r, f)
    assert (u, r, f) == saved
    import scripts.audit_evidence_identity_v1 as module
    text = inspect.getsource(module)
    assert "select_evidence_packing_v1" not in text
    assert "matching_inputs(" not in text
    assert "financebench_id_0" not in text


def test_complete_audit_detects_foreign_coverage_and_available_fact_dropped():
    from scripts.audit_evidence_identity_v1 import audit_record, digest
    from scripts.evaluate_evidence_metadata_counterfactual_v1 import _context_metrics
    from backend.evidence_packing_v1 import _render_selection
    good = unit()
    foreign = unit(entity="Other", document_id="other", page_id="other:p0",
                   metadata={"filename": "other.pdf", "page_number": 0, "chunk_id": "v"},
                   current_ranking={"rank": 2, "score": .5})
    values = [good, foreign]
    row = {"question": "How much output in 2024?", "answer": "100", "justification": "",
           "evidence": json.dumps([{"doc_name": "example", "evidence_page_num": 0,
                                    "evidence_text": "Output was 100 in 2024."}])}
    source = {"financebench_id": "case", "group": "selection_loss10", "question": row["question"],
              "candidate_units": values, "_packing_trace": [{"rank": 1, "selection_reason": "budget"}]}
    prior = {"candidate_sha256": digest(values), "routes": {"packing_v1": {
        "selected_ranks": [2], "metrics": _context_metrics(row, _render_selection([foreign])[0], [foreign])}}}
    saved = copy.deepcopy(source)
    record = audit_record(source, prior, row, build_registry([source]))
    assert record["selected_ranks"] == [2]
    assert record["coverage"]["legacy"] == 1
    assert record["coverage"]["entity_bound"] == 0
    assert record["foreign_entity_only_line_hits"] == 1
    assert record["identity_conflicting_line_hits"] == 1
    assert record["coverage"]["candidate_qualified_facts"] == 1
    assert record["coverage"]["selected_qualified_facts"] == 0
    assert record["selection_loss_diagnosis"] == "qualified_candidate_fact_lost_in_frozen_selection"
    assert record["source_flow"]["stage"] == "reference_page_candidates_not_selected"
    assert record["source_flow"]["candidate_reference_page_ranks"] == [1]
    assert record["source_flow"]["selected_reference_page_ranks"] == []
    assert record["candidate_fact_loss"][0]["packing_decisions"][0]["selection_reason"] == "budget"
    assert record["coverage_but_fact_wrong_count"] is None
    assert source == saved
