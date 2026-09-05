from backend.evidence_frame_shadow_v1 import build_evidence_frame_shadow_v1, explain_answer_failure


def test_extracts_explicit_formula_period_operands_and_provenance():
    question = "What is the FY2022 cash ratio, defined as: cash / total current liabilities?"
    evidence = """Source: SAMPLE_2022_10K.pdf | Page: 10
Consolidated Balance Sheets
FY2022
Cash $50 million
Total current liabilities $100 million
"""
    metadata = [{"filename": "SAMPLE_2022_10K.pdf", "page_number": 10, "company": "SAMPLE"}]
    frame = build_evidence_frame_shadow_v1(question, evidence, metadata)
    assert frame["question_type"] == "calculation"
    assert frame["formula_candidates"][0]["operands"] == ["cash", "total current liabilities"]
    assert frame["diagnostics"]["requested_period_found"] is True
    assert frame["diagnostics"]["required_operands_found"] is True
    assert {item["page"] for item in frame["operand_candidates"]} == {10}


def test_never_reads_page_metadata_text_outside_frozen_context():
    frame = build_evidence_frame_shadow_v1(
        "What is the FY2022 cash ratio, defined as: cash / liabilities?",
        "Source: SAMPLE.pdf | Page: 1\nCash was $5 in FY2022.",
        [{"filename": "SAMPLE.pdf", "page_number": 1, "company": "SAMPLE", "text": "Liabilities were $10."}],
    )
    assert frame["diagnostics"]["required_operands_status"] == "partial"
    assert all("Liabilities were $10" not in span["text"] for span in frame["evidence_spans"])


def test_unnecessary_refusal_is_explained_only_when_operands_are_complete():
    frame = build_evidence_frame_shadow_v1(
        "What is the FY2022 ratio using cash as the numerator and liabilities as the denominator?",
        "Source: SAMPLE.pdf | Page: 2\nFY2022 cash $5\nFY2022 liabilities $10",
        [],
    )
    explanation = explain_answer_failure(frame, "I cannot calculate this ratio.")
    assert "unnecessary_refusal_candidate" in explanation["signals"]
    assert "refusal_failure" in explanation["explained_failure_types"]


def test_implicit_financial_formula_is_not_invented():
    frame = build_evidence_frame_shadow_v1(
        "What was inventory turnover in FY2022?",
        "Source: SAMPLE.pdf | Page: 3\nFY2022 inventory was $20 and cost was $100.",
        [],
    )
    assert frame["formula_candidates"] == []
    assert frame["diagnostics"]["required_operands_status"] == "formula_not_specified"
    assert frame["diagnostics"]["required_operands_found"] is None


def test_short_fiscal_year_is_normalized_without_matching_arbitrary_two_digit_numbers():
    frame = build_evidence_frame_shadow_v1(
        "Did cash increase in FY22?",
        "Source: SAMPLE.pdf | Page: 4\nCash was $47 in FY2022.",
        [],
    )
    assert frame["diagnostics"]["requested_periods"] == ["2022"]
    assert frame["diagnostics"]["requested_period_found"] is True
