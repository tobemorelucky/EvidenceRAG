from query_parser import assess_required_field_coverage, build_required_field_query, parse_query


def test_operating_cash_flow_ratio_has_required_fields():
    parsed = parse_query("What is Adobe's FY2017 operating cash flow ratio? Cash from operations / total current liabilities.")
    assert parsed["task_type"] == "calculation"
    assert parsed["required_fields"] == ["cash_from_operations", "current_liabilities"]


def test_coverage_reports_partial_without_supplemental_search():
    parsed = parse_query("What is the operating cash flow ratio?")
    coverage = assess_required_field_coverage(
        parsed,
        [{"text": "Net cash provided by operating activities was $100."}],
    )
    assert coverage["status"] == "partial"
    assert coverage["missing_fields"] == ["current_liabilities"]
    assert coverage["supplemental_search_attempted"] is False


def test_ebitda_less_capex_contract_is_complete_when_operands_are_present():
    parsed = parse_query("What is unadjusted EBITDA less capex? EBITDA is operating income plus depreciation and amortization.")
    coverage = assess_required_field_coverage(
        parsed,
        [{"text": "Operating income, depreciation and amortization, and capital expenditures are shown below."}],
    )
    assert coverage["status"] == "complete"
    assert coverage["formula"] == "operating_income + depreciation_amortization - capital_expenditures"


def test_required_field_query_keeps_question_and_adds_statement_labels():
    question = "What is Adobe's FY2017 operating cash flow ratio?"
    rewritten = build_required_field_query(question)
    assert question in rewritten
    assert "net cash provided by operating activities" in rewritten
    assert "total current liabilities" in rewritten
