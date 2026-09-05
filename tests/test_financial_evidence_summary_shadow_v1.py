from copy import deepcopy

from backend.financial_evidence_summary_shadow_v1 import (
    detect_metric_substitution_v1,
    extract_financial_evidence_summary_v1,
)


def test_extracts_metric_operands_period_values_and_exact_source_span():
    evidence = """Source: ACME_2022_10K.pdf | Page: 10
In millions 2022 2021
Net sales $1,000 $900
Cost of sales $600 $570
"""
    result = extract_financial_evidence_summary_v1(
        "Did Acme have an improving gross margin profile as of FY2022?", evidence,
        [{"filename": "ACME_2022_10K.pdf", "page_number": 10, "company": "Acme", "report_year": 2022}],
    )
    metrics = {fact["metric"] for fact in result["facts"]}
    assert result["extraction_status"] == "operand_supported"
    assert {"revenue", "cost of sales"} <= metrics
    assert {fact["period"] for fact in result["facts"]} >= {"2022", "2021"}
    assert all(fact["source_span"]["document"] == "ACME_2022_10K.pdf" for fact in result["facts"])
    assert any(fact["source_span"]["text"] == "Net sales $1,000 $900" for fact in result["facts"])


def test_uses_only_frozen_evidence_text_not_metadata_text():
    evidence = "Source: ACME_2022_10K.pdf | Page: 10\nUnrelated narrative."
    metadata = [{
        "filename": "ACME_2022_10K.pdf", "page_number": 10, "company": "Acme", "report_year": 2022,
        "text": "Net sales $1,000 and cost of sales $600",
    }]
    result = extract_financial_evidence_summary_v1("What was Acme's gross margin in FY2022?", evidence, metadata)
    assert result["facts"] == []
    assert result["source_contract"] == "exact_frozen_answer_context_only"


def test_preserves_qualitative_fact_and_marks_ambiguity():
    evidence = """Source: ACME_2023_10K.pdf | Page: 4
The increase in merchandise inventories reflected new stores and product launches.
"""
    result = extract_financial_evidence_summary_v1(
        "What drove the increase in Acme's merchandise inventories balance at end of FY2023?", evidence
    )
    fact = result["facts"][0]
    assert fact["value"] is None
    assert "qualitative_value" in fact["ambiguity_flags"]
    assert fact["source_span"]["text"].startswith("The increase")


def test_input_metadata_is_not_mutated():
    evidence = "Source: ACME_2022_10K.pdf | Page: 2\nTotal current assets $20"
    metadata = [{"filename": "ACME_2022_10K.pdf", "page_number": 2, "company": "Acme", "report_year": 2022}]
    original = deepcopy(metadata)
    extract_financial_evidence_summary_v1("Does Acme have positive working capital based on FY2022 data?", evidence, metadata)
    assert metadata == original


def test_metric_substitution_detector_is_target_scoped():
    answer = "SG&A decreased, but store payroll and benefits deleveraged."
    assert "sg&a" in detect_metric_substitution_v1("wages expense as percent of sales", answer)
    assert detect_metric_substitution_v1("inventory turnover", answer) == []


def test_quarter_is_recovered_from_document_identifier():
    evidence = "Source: ACME_2023Q2_10Q.pdf | Page: 4\nThe company announced a business separation."
    result = extract_financial_evidence_summary_v1("As of Q2 2023, is Acme spinning off a business?", evidence)
    assert result["facts"][0]["period"] == "2023Q2"


def test_rejects_matching_metric_from_a_different_entity_document():
    evidence = """Source: OTHER_2022_10K.pdf | Page: 3
Net sales $900
Source: ACME_2022_10K.pdf | Page: 4
Cost of sales $600
"""
    metadata = [
        {"filename": "OTHER_2022_10K.pdf", "page_number": 3, "company": "Other", "report_year": 2022},
        {"filename": "ACME_2022_10K.pdf", "page_number": 4, "company": "Acme", "report_year": 2022},
    ]
    result = extract_financial_evidence_summary_v1("What was Acme's gross margin in FY2022?", evidence, metadata)
    assert {fact["source_span"]["document"] for fact in result["facts"]} == {"ACME_2022_10K.pdf"}
    assert {fact["metric"] for fact in result["facts"]} == {"cost of sales"}
