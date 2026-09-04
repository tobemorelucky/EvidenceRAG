import copy
import json
from pathlib import Path

import pytest

from backend.evidence_packing_v1 import _render_selection, select_evidence_packing_v1
from backend.query_requirement_v1 import parse_query_requirement, periods, requirement_guided_inputs, requirement_support


def unit(rank=1, text="Example produced 100 units in 2024.", **kwargs):
    return dict(document_id="doc", page_id=f"doc:{rank}", source_type="text", entity="Example",
                period=["2024"], metric=None, value=None, unit=None, source_text=text,
                metadata={"chunk_id": str(rank), "page_number": rank, "filename": "example.pdf"},
                current_ranking={"rank": rank, "score": 1 / rank}) | kwargs


@pytest.mark.parametrize("question,kind,numeric,comparison,calculation", [
    ("How much did Example produce in FY2024?", "numeric", True, False, False),
    ("Compare output between 2023 and 2024.", "numeric", True, True, False),
    ("Which location produced the most units?", "selection", True, True, False),
    ("Why did operations stop?", "explanation", False, False, False),
    ("What caused the increase in output?", "explanation", False, False, False),
    ("Did output increase?", "judgment", True, True, False),
    ("Calculate the average of the disclosed values.", "numeric", True, False, True),
    ("What percentage of total output came from this site?", "numeric", True, False, True),
    ("Which products are manufactured?", "lookup", False, False, False),
    ("What is described in the note?", "lookup", False, False, False),
])
def test_general_requirements(question, kind, numeric, comparison, calculation):
    result = parse_query_requirement(question)
    assert (result.answer_type, result.requires_numeric_evidence, result.requires_comparison, result.requires_calculation) == (kind, numeric, comparison, calculation)


def test_primary_task_not_overridden_by_conditional_explanation():
    q = "How much output was produced? Calculate the average; if it is not meaningful explain why."
    assert parse_query_requirement(q).answer_type == "numeric"
    assert parse_query_requirement(q).requires_calculation
    assert parse_query_requirement("If capacity doubles, calculate expected output.").requires_calculation


def test_period_is_not_numeric_evidence_and_fy_quarters_normalize():
    assert periods("FY22 FY 2023 Q4 FY99 2024") == {"2022", "2023", "q4", "1999", "2024"}
    r = parse_query_requirement("What products existed in FY22?")
    assert r.requires_period and not r.requires_numeric_evidence
    r = parse_query_requirement("How much in FY22 Q4?")
    s = requirement_support("How much in FY22 Q4?", r, unit(text="FY22 Q4 only", period=[]))
    assert s["active_support"]["numeric"] == 0


def test_entity_is_not_guessed_from_missing_company_context():
    assert not parse_query_requirement("How much output in 2024?").requires_entity
    assert parse_query_requirement("How much output did Example produce?").requires_entity
    assert parse_query_requirement("Which location produced the most?").requires_entity


def test_no_financial_metric_formulas_are_inferred():
    assert not parse_query_requirement("What is the undisclosed efficiency metric?").requires_calculation
    assert not parse_query_requirement("What is the quick ratio?").requires_calculation


def test_missing_metadata_is_soft_and_bonus_bounded():
    q = "Calculate Example output for 2024."
    r = parse_query_requirement(q)
    s = requirement_support(q, r, unit(text="Unrelated narrative", entity="", period=[]))
    assert s["multiplier"] == 1
    for text in ("", "Example output 100 in 2024", "Example output 100 200 300 400 500 2024"):
        assert 1 <= requirement_support(q, r, unit(text=text))["multiplier"] <= 2


def test_year_alone_and_more_numbers_do_not_create_unlimited_bonus():
    q = "Calculate Example output for 2024."
    r = parse_query_requirement(q)
    a = requirement_support(q, r, unit(text="Example output 100 200 in 2024"))
    b = requirement_support(q, r, unit(text="Example output 100 200 300 400 in 2024"))
    assert a == b


def test_empty_and_invalid_inputs():
    assert not any(v for k, v in parse_query_requirement("").to_dict().items() if k != "answer_type")
    with pytest.raises(TypeError):
        parse_query_requirement(None)
    assert requirement_guided_inputs("", [])[0] == []
    with pytest.raises(ValueError):
        requirement_guided_inputs("query", [unit(), unit()])


def test_only_shadow_score_changes_and_labels_ignored():
    candidates = [unit(), unit(2, text="Example output 200 in 2024")]
    original = copy.deepcopy(candidates)
    a, trace = requirement_guided_inputs("How much output did Example produce in 2024?", candidates)
    for before, after in zip(candidates, a):
        assert before["current_ranking"]["rank"] == after["current_ranking"]["rank"]
        assert {k: v for k, v in before.items() if k != "current_ranking"} == {k: v for k, v in after.items() if k != "current_ranking"}
    contaminated = copy.deepcopy(candidates)
    for item in contaminated:
        item.update(financebench_id="fake", reference_answer="gold", gold_evidence="123")
    assert requirement_guided_inputs("How much output did Example produce in 2024?", contaminated) == (a, trace)
    assert candidates == original


def test_packer_unchanged_rendering_budget_and_determinism():
    candidates = [unit(i, text=f"Example output {i * 100} in 2024. " * 8) for i in range(1, 8)]
    adapted, _ = requirement_guided_inputs("How much output did Example produce in 2024?", candidates)
    result = select_evidence_packing_v1("How much output did Example produce in 2024?", adapted, max_context_chars=1600)
    assert result == select_evidence_packing_v1("How much output did Example produce in 2024?", adapted, max_context_chars=1600)
    context, selected, trace = result
    assert len(context) <= 1600
    ranks = {u["current_ranking"]["rank"] for u in selected}
    assert context == _render_selection([u for u in candidates if u["current_ranking"]["rank"] in ranks])[0]
    assert trace["replacement_threshold"] == 1 and trace["max_replacements"] is None


def test_no_active_requirement_exactly_preserves_score():
    q = "describe this."
    adapted, _ = requirement_guided_inputs(q, [unit()])
    assert adapted == [unit()]


def test_silver_accuracy_denominators_and_nulls():
    from scripts.evaluate_query_requirement_v1 import requirement_accuracy
    labels = {"provenance": "test", "fields": ["answer_type", "requires_calculation"],
              "labels": {"a": ["lookup", None], "b": ["numeric", True]}}
    records = [{"financebench_id": "a", "requirement_trace": {"requirement": {"answer_type": "lookup", "requires_calculation": False}}},
               {"financebench_id": "b", "requirement_trace": {"requirement": {"answer_type": "lookup", "requires_calculation": True}}}]
    stats = requirement_accuracy(records, labels)
    assert stats["scored_fields"] == 3
    assert stats["micro_accuracy"] == pytest.approx(2 / 3)
    assert stats["full_spec_exact"] == {"correct": 0, "total": 1}
    assert stats["fields"]["requires_calculation"]["unscored"] == 1


def test_labels_have_expected_contract_and_are_evaluation_only():
    path = Path(__file__).parent / "fixtures/query_requirement_v1_labels.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["labels"]) == 30 and len(data["fields"]) == 6
    assert all(len(values) == 6 for values in data["labels"].values())
    import backend.query_requirement_v1 as module
    import inspect
    source = inspect.getsource(module)
    assert "fixtures" not in source and "financebench_id" not in source
