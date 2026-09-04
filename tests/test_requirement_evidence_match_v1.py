import copy
import inspect

import pytest

from backend.evidence_packing_v1 import _render_selection, select_evidence_packing_v1
from backend.query_requirement_v1 import parse_query_requirement
from backend.requirement_evidence_match_v1 import entity_match, match_requirement_evidence, matching_inputs


def unit(rank=1, text="Example output was 100 in 2024.", **kwargs):
    return dict(document_id="doc", page_id=f"doc:{rank}", source_type="text", entity="Example",
                period=["2024"], metric=None, value=None, unit=None, source_text=text,
                metadata={"chunk_id": str(rank), "page_number": rank, "filename": "file.pdf"},
                current_ranking={"rank": rank, "score": 1 / rank}) | kwargs


def match(question, value):
    return match_requirement_evidence(question, parse_query_requirement(question), value)


@pytest.mark.parametrize("query,entity,score,status", [
    ("What did Example Works produce?", "EXAMPLEWORKS", 1, "query_entity_surface_match"),
    ("What did Example Works produce?", "Example Works Corporation", 1, "query_entity_surface_match"),
    ("What did 7Z produce?", "7Z", 1, "query_entity_surface_match"),
    ("What did exampleworksother produce?", "Example Works", 0, "target_unresolved_or_not_mentioned"),
    ("What did EW produce?", "Example Works", 0, "target_unresolved_or_not_mentioned"),
    ("What was output?", "", 0, "unknown_metadata"),
])
def test_generic_entity_surface_matching(query, entity, score, status):
    assert entity_match(query, entity)[:2] == (score, status)


def test_numeric_support_requires_local_target_overlap():
    q = "How much output did Example produce in 2024?"
    local = match(q, unit(text="Example output produced was 100 in 2024."))
    unrelated = match(q, unit(text="Example output produced is discussed.\nOther items were 100 in 2024."))
    assert local["numeric_availability"] == 1
    assert unrelated["numeric_availability"] == 0
    assert local["compatibility_score"] > unrelated["compatibility_score"]


def test_period_local_vs_metadata_only_and_partial():
    q = "Compare Example output between 2023 and 2024."
    a = match(q, unit(text="Example output 100 in 2023 and 200 in 2024."))
    b = match(q, unit(text="Example output 100 and 200.", period=["2023", "2024"]))
    c = match(q, unit(text="Example output 100 in 2024.", period=[]))
    assert a["period_match"] == 1 and a["period_status"] == "local_text_full"
    assert b["period_match"] == .5 and b["period_status"] == "nonlocal_or_metadata_only"
    assert c["period_match"] == .5 and c["period_status"] == "local_text_partial"


def test_calculation_is_support_only_and_requires_nonperiod_numbers():
    q = "Calculate Example output for FY22 Q4."
    a = match(q, unit(text="Example output FY22 Q4.", period=[]))
    b = match(q, unit(text="Example output 100 FY22 Q4.", period=[]))
    c = match(q, unit(text="Example output 100 200 FY22 Q4.", period=[]))
    assert a["calculation_support"] == 0
    assert b["calculation_support"] == .5
    assert c["calculation_support"] == 1
    assert "result" not in c and "formula" not in c


def test_existing_table_header_row_together_but_no_new_table_parsing():
    q = "Calculate Example output for 2024."
    value = unit(text="Table title: Output\nHeader: 2024 | 2023\nRow: Output | 100 | 200", source_type="table")
    result = match(q, value)
    assert result["period_match"] == 1
    assert result["numeric_availability"] == 1
    assert result["calculation_support"] == 1
    assert result["best_fragment_index"] == 0


def test_entity_year_only_do_not_establish_metric_support():
    q = "How much output did Example produce in 2024?"
    value = unit(text="Example 2024. Other details 100 200.")
    assert match(q, value)["compatibility_score"] == 0


@pytest.mark.parametrize("name", ["7Z", "Example", "Example Works"])
@pytest.mark.parametrize("possessive", ["'s", "’s", "'"])
def test_entity_surface_tokens_do_not_leak_into_metric_terms(name, possessive):
    q = f"How much was {name}{possessive} output in 2024?"
    value = unit(entity=name.replace(" ", ""), text=f"{name} reported in 2024.")
    result = match(q, value)
    assert result["entity_match"] == 1
    assert result["target_terms"] == ["output"]
    assert result["metric_relevance"] == 0


def test_metadata_metric_cannot_create_numeric_support():
    q = "How much output?"
    result = match(q, unit(text="Unrelated 100 200.", metric="output"))
    assert result["metadata_metric_relevance"] == 1
    assert result["metric_relevance"] == .5
    assert result["numeric_availability"] == 0


def test_missing_or_incompatible_period_is_not_hard_rejection():
    unknown = match("How much Example output in 2024?", unit(text="Output was 100.", period=[], entity=""))
    mismatch = match("How much Example output in 2024?", unit(text="Example output was 100 in 2023.", period=[]))
    assert unknown["period_status"] == "unknown"
    assert mismatch["period_status"] == "requested_period_not_visible"
    assert 0 <= mismatch["compatibility_score"] <= 1


def test_whitelisted_adapter_is_gold_free_and_nonmutating():
    candidates = [unit(), unit(2)]
    original = copy.deepcopy(candidates)
    q = "How much output did Example produce in 2024?"
    req = parse_query_requirement(q)
    adapted, trace = matching_inputs(q, req, candidates)
    contaminated = copy.deepcopy(candidates)
    for u in contaminated:
        u.update(financebench_id="sentinel", gold_evidence="1000", reference_answer="999")
    assert matching_inputs(q, req, contaminated) == (adapted, trace)
    assert candidates == original
    for before, after in zip(candidates, adapted):
        assert before["current_ranking"]["rank"] == after["current_ranking"]["rank"]
        assert {k: v for k, v in before.items() if k != "current_ranking"} == {k: v for k, v in after.items() if k != "current_ranking"}
        assert before["current_ranking"]["score"] <= after["current_ranking"]["score"] <= 2 * before["current_ranking"]["score"]


def test_no_external_services_or_classification_calls_in_matcher():
    import backend.requirement_evidence_match_v1 as module
    source = inspect.getsource(module)
    for banned in ("financebench_id", "parse_query_requirement", "requests.", "langsmith", "jina", "openai"):
        assert banned not in source


def test_duplicate_rank_empty_and_deterministic_budget():
    q = "How much Example output in 2024?"
    req = parse_query_requirement(q)
    assert matching_inputs(q, req, [])[0] == []
    with pytest.raises(ValueError):
        matching_inputs(q, req, [unit(), unit()])
    candidates = [unit(i, text=f"Example output {i * 100} in 2024. " * 15) for i in range(1, 9)]
    adapted, _ = matching_inputs(q, req, candidates)
    result = select_evidence_packing_v1(q, adapted, max_context_chars=2000)
    assert result == select_evidence_packing_v1(q, adapted, max_context_chars=2000)
    context, selected, trace = result
    ranks = {u["current_ranking"]["rank"] for u in selected}
    assert len(context) <= 2000
    assert context == _render_selection([u for u in candidates if u["current_ranking"]["rank"] in ranks])[0]
    assert trace["replacement_threshold"] == 1 and trace["max_replacements"] is None


def test_loss_audit_coverage_can_be_distributed_not_single_unit():
    from scripts.evaluate_requirement_evidence_match_v1 import covered_lines
    gold = [{"evidence_text": "Output 100\nVolume 200"}]
    assert covered_lines(gold, "Output 100\nVolume 200") == {"output 100", "volume 200"}
    assert covered_lines(gold, "Volume 200") == {"volume 200"}
