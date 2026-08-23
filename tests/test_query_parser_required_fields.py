from query_parser import (
    assess_required_field_coverage,
    build_answer_directives,
    build_required_field_query,
    build_supplemental_field_query,
    infer_page_statement_types,
    match_required_fields_in_text,
    parse_query,
)


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
        [{"text": "Operating income 120; depreciation and amortization 30; capital expenditures 20."}],
    )
    assert coverage["status"] == "complete"
    assert coverage["formula"] == "operating_income + depreciation_amortization - capital_expenditures"


def test_required_field_query_keeps_question_and_adds_statement_labels():
    question = "What is Adobe's FY2017 operating cash flow ratio?"
    rewritten = build_required_field_query(question)
    assert question in rewritten
    assert "net cash provided by operating activities" in rewritten
    assert "total current liabilities" in rewritten


def test_coverage_requires_numeric_evidence_near_each_field():
    parsed = parse_query("What is the operating cash flow ratio?")
    coverage = assess_required_field_coverage(
        parsed,
        [{"text": "The report discusses net cash provided by operating activities and current liabilities."}],
    )

    assert coverage["status"] == "insufficient"
    assert coverage["missing_fields"] == ["cash_from_operations", "current_liabilities"]


def test_coverage_rejects_fields_from_a_different_company():
    parsed = parse_query("What is Adobe's FY2022 operating margin?")
    coverage = assess_required_field_coverage(
        parsed,
        [{"filename": "AES_2022_10K.pdf", "text": "Operating income 100; total revenue 500."}],
    )

    assert coverage["status"] == "insufficient"
    assert coverage["scope_status"] == "company_mismatch"


def test_comparison_coverage_requires_both_reporting_periods():
    parsed = parse_query("Did Pfizer grow its PPNE between FY2020 and FY2021?")
    coverage = assess_required_field_coverage(
        parsed,
        [{"filename": "PFIZER_2021_10K.pdf", "text": "Property, plant and equipment, net was $10 in 2021."}],
    )

    assert parsed["required_fields"] == ["ppe"]
    assert coverage["status"] == "partial"
    assert coverage["missing_periods"] == ["2020"]


def test_operating_margin_targets_income_statement():
    parsed = parse_query("What was Adobe's operating margin in FY2022?")

    assert parsed["statement_types"] == ["income_statement"]


def test_quick_ratio_targets_balance_sheet():
    parsed = parse_query("Calculate the FY2022 quick ratio.")

    assert parsed["statement_types"] == ["balance_sheet"]


def test_page_statement_classifier_uses_filing_headings():
    statement_types = infer_page_statement_types(
        "CONSOLIDATED STATEMENTS OF CASH FLOWS\nNet cash provided by operating activities 1,234"
    )

    assert "cash_flow" in statement_types


def test_supplemental_query_contains_only_missing_field_anchors():
    query = build_supplemental_field_query(
        "Calculate AMD's FY2022 quick ratio.",
        {
            "missing_fields": ["accounts_receivable"],
            "missing_periods": [],
        },
    )

    assert "accounts receivable" in query
    assert "balance sheet" in query
    assert "short-term investments" not in query


def test_required_field_match_requires_number_near_alias():
    matched = match_required_fields_in_text(
        ["accounts_receivable", "current_liabilities"],
        "Accounts receivable, net 4,126; current liabilities are discussed below.",
    )

    assert matched == {"accounts_receivable": "accounts receivable"}


def test_supplemental_query_does_not_run_for_period_only_gap():
    query = build_supplemental_field_query(
        "Compare revenue between FY2021 and FY2022.",
        {"missing_fields": [], "missing_periods": ["2021"]},
    )

    assert query == ""


def test_working_capital_uses_difference_formula():
    parsed = parse_query("Does Corning have positive working capital based on FY2022 data?")

    assert parsed["task_type"] == "calculation"
    assert parsed["required_fields"] == [
        "accounts_receivable",
        "inventory",
        "other_current_assets",
        "accounts_payable",
        "other_accrued_liabilities",
    ]
    assert parsed["formula"] == "accounts_receivable + inventory + other_current_assets - accounts_payable - other_accrued_liabilities"
    assert parsed["company"] == "corning"


def test_selection_question_requires_metric_field():
    parsed = parse_query("Which of JPM's business segments had the lowest net revenue in 2021 Q1?")

    assert parsed["task_type"] == "selection"
    assert parsed["required_fields"] == ["revenue"]


def test_fixed_asset_turnover_prefers_primary_statements():
    parsed = parse_query("What is Activision Blizzard's FY2019 fixed asset turnover ratio?")

    assert parsed["company"] == "activision_blizzard"
    assert parsed["statement_types"] == ["income_statement", "balance_sheet"]


def test_company_store_count_directive_requires_total_row():
    question = "What was the change in Best Buy's total store count from FY2021 to FY2022?"
    directives = build_answer_directives(question, parse_query(question))

    assert any("Total row" in directive for directive in directives)
    assert any("branded 'Best Buy' subrow" in directive for directive in directives)


def test_quick_ratio_directive_requires_direct_conclusion():
    question = "Calculate the FY2022 quick ratio and say whether it is healthy."
    directives = build_answer_directives(question, parse_query(question))

    assert any("give the requested healthy/not-healthy conclusion directly" in directive for directive in directives)
    assert any("generic business-model or cash-flow caveat" in directive for directive in directives)


def test_capital_intensity_directive_requires_yes_no_conclusion():
    question = "Based on capital spending, net PP&E and revenue, is the company capital-intensive?"
    directives = build_answer_directives(question, parse_query(question))

    assert any("yes/no capital-intensity conclusion" in directive for directive in directives)
    assert any("not capital-intensive" in directive for directive in directives)
