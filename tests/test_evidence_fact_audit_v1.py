import copy
import inspect

import pytest

from scripts.audit_evidence_fact_v1 import (
    compare_claim, inspect_fact_unit, numeric_signature, period_binding, signature_values,
)


def unit(**kwargs):
    return dict(document_id="doc", page_id="p0", source_type="text", entity="Example",
                metric="Output", period=["2024"], value=["100"], unit=None,
                source_text="Output was 100 in 2024.", metadata={},
                current_ranking={"rank": 1, "score": .5}) | kwargs


def req(**kwargs):
    return dict(question="What was Example output in 2024?", entity=["example"],
                entity_reliable=True, period=["2024"], scope=[]) | kwargs


def fact(**kwargs):
    return dict(fact_id=0, text="Output was 100 in 2024.", document_id="doc", page_id="p0") | kwargs


def test_exact_source_A_and_other_source_B():
    assert compare_claim(fact(), fact()["text"], unit(), req())["category"] == "A"
    assert compare_claim(fact(), fact()["text"], unit(document_id="doc2", page_id="p3"), req())["category"] == "B"


def test_matching_number_foreign_entity_C():
    result = inspect_fact_unit(unit(entity="Elsewhere"), req(), [fact()])
    assert result["categories"] == ["C"]
    assert not result["compatible_reference_fact_ids"]


@pytest.mark.parametrize("entity", [None, "unknown", ""])
def test_unknown_entity_not_C_or_A(entity):
    assert compare_claim(fact(), fact()["text"], unit(entity=entity), req())["category"] == "G"


def test_no_period_binding_from_distant_year_or_metadata():
    f = fact(text="Output was 100.")
    result = inspect_fact_unit(unit(source_text="Report for 2024.\nOutput was 100."), req(), [f])
    assert result["categories"] == ["G"]
    assert result["claim_comparisons"][0]["reason"] == "unknown_local_binding"


def test_local_period_D_but_reference_question_disagreement_G():
    assert compare_claim(fact(), "Output was 100 in 2023.", unit(), req())["category"] == "D"
    result = compare_claim(fact(text="Output was 100 in 2023."), fact()["text"], unit(), req())
    assert result["category"] == "G"
    assert result["reason"] == "question_reference_period_disagreement"


def test_partial_comparison_period_is_valid_not_conflict():
    assert period_binding(fact()["text"], fact()["text"], ["2023", "2024"]) == "explicit_local_match"
    assert period_binding("Output 100 2023 2024", "Output 100 2023 2024", ["2023", "2024"]) == "multiple_period_value_binding_unknown"
    assert period_binding("Output 100 in 2023", "Output 100 in 2024", []) == "reference_period_conflict"


def test_same_claim_numeric_contradiction_F_including_sign():
    for text in ("Output was 101 in 2024.", "Output was (100) in 2024."):
        assert compare_claim(fact(), text, unit(), req())["category"] == "F"


def test_missing_numbers_or_different_metric_not_F_or_E():
    assert compare_claim(fact(), "Output was unavailable in 2024.", unit(), req()) is None
    assert compare_claim(fact(), "Volume was 100 in 2024.", unit(), req()) is None
    assert compare_claim(fact(), fact()["text"], unit(metric="Volume"), req())["category"] == "G"


def test_numeric_formatting_sign_currency_percent_preserved():
    assert signature_values("1,000.00") == signature_values("1000")
    assert signature_values("(100)") == signature_values("-100")
    assert signature_values("10%") != signature_values("10")
    assert signature_values("$10") != signature_values("€10")
    assert numeric_signature("Result 100.")[0]["value"] == "1E+2"


def test_normalized_same_fact_does_not_change_source_class():
    f = fact(text="Output was 1,000.00 in 2024.")
    assert compare_claim(f, "Output was 1000 in 2024.", unit(), req())["category"] == "A"


def test_currency_scale_and_percent_not_automatically_equivalent():
    assert compare_claim(fact(text="Output was $100 in 2024."), "Output was €100 in 2024.", unit(), req())["category"] == "G"
    assert compare_claim(fact(text="Output was 100 million in 2024."), "Output was 100 billion in 2024.", unit(), req()) is None
    assert compare_claim(fact(text="Output was 10% in 2024."), "Output was 10 in 2024.", unit(), req())["category"] == "G"


def test_scope_unknown_not_promoted_and_prefix_numbers_do_not_match():
    assert compare_claim(fact(), fact()["text"], unit(), req(scope=["consolidated"]))["category"] == "G"
    assert compare_claim(fact(), "Output was 1000 in 2024.", unit(), req())["category"] == "F"


def test_no_metric_or_no_gold_does_not_invent_fact_truth():
    assert inspect_fact_unit(unit(metric=None), req(), [fact()])["categories"] == ["G"]
    assert inspect_fact_unit(unit(), req(), [])["categories"] == ["G"]


def test_audit_does_not_mutate_or_invoke_selection_or_api():
    u, r, f = unit(), req(), [fact()]
    saved = copy.deepcopy((u, r, f))
    assert inspect_fact_unit(u, r, f) == inspect_fact_unit(u, r, f)
    assert saved == (u, r, f)
    import scripts.audit_evidence_fact_v1 as module
    source = inspect.getsource(module)
    assert "select_evidence_packing_v1(" not in source
    assert "financebench_id_0" not in source
    assert "create_connection\", denied" in source


def test_unrequested_period_change_not_mislabelled_numeric_error():
    result = compare_claim(fact(), "Output was 100 in 2023.", unit(), req(period=[]))
    assert result["category"] == "D"


def test_complete_frozen_audit_reproduces_selection_and_retains_unknown():
    from scripts.audit_evidence_fact_v1 import audit
    from scripts.audit_evidence_identity_v1 import build_registry, digest
    from scripts.evaluate_evidence_metadata_counterfactual_v1 import _context_metrics
    from backend.evidence_packing_v1 import _render_selection
    import json
    u = unit(metadata={"filename": "example.pdf", "page_number": 0, "chunk_id": "u"})
    row = {"question": req()["question"], "answer": "100", "justification": "",
           "evidence": json.dumps([{"doc_name": "example", "evidence_page_num": 0, "evidence_text": fact()["text"]}])}
    source = {"financebench_id": "opaque-test-id", "question": row["question"],
              "group": "selection_loss10", "candidate_units": [u]}
    previous = {"candidate_sha256": digest([u]), "routes": {"packing_v1": {
        "selected_ranks": [1], "metrics": _context_metrics(row, _render_selection([u])[0], [u])}}}
    saved = copy.deepcopy((source, previous, row))
    result = audit(source, previous, row, build_registry([source]))
    assert result["frozen_selection_verified"]
    assert result["selected_ranks"] == [1]
    assert result["selected_category_counts"] == {"A": 1}
    assert result["coverage"]["fact_compatible_partial_support_lower_bound"] == 1
    assert result["actual_answer_effectiveness"] is None
    assert (source, previous, row) == saved
