from query_parser import (
    assess_required_field_coverage,
    build_answer_directives,
    build_finance_query_rewrite,
    build_required_field_query,
    build_supplemental_field_query,
    infer_page_statement_types,
    match_required_fields_in_text,
    parse_query,
)


def test_requested_years_preserve_question_order():
    parsed = parse_query(
        "What is the FY2019 ratio using the average balance between FY2018 and FY2019?"
    )

    assert parsed["required_periods"] == ["2019", "2018"]


def test_ratio_directive_preserves_decimal_units_when_percent_not_requested():
    question = "What is FY2022 return on assets? Round the ratio to two decimals."
    parsed = {
        "task_type": "calculation",
        "formula": "net_income / average(total_assets)",
        "required_periods": ["2022"],
    }

    directives = build_answer_directives(question, parsed)

    assert any("decimal without a percent sign" in directive for directive in directives)


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


def test_required_field_query_returns_concise_statement_anchors():
    question = "What is Adobe's FY2017 operating cash flow ratio?"
    rewritten = build_required_field_query(question)
    assert question not in rewritten
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


def test_plain_working_capital_uses_standard_difference_formula():
    parsed = parse_query("Does Corning have positive working capital based on FY2022 data?")

    assert parsed["task_type"] == "calculation"
    assert parsed["required_fields"] == ["current_assets", "current_liabilities"]
    assert parsed["formula"] == "current_assets - current_liabilities"
    assert parsed["calculation_basis"] == "standard_working_capital"
    assert parsed["company"] == "corning"


def test_explicit_operating_working_capital_uses_operating_formula():
    parsed = parse_query("Calculate operating working capital for FY2022.")

    assert parsed["required_fields"] == [
        "accounts_receivable",
        "inventory",
        "other_current_assets",
        "accounts_payable",
        "other_accrued_liabilities",
    ]
    assert parsed["formula"] == "accounts_receivable + inventory + other_current_assets - accounts_payable - other_accrued_liabilities"
    assert parsed["calculation_basis"] == "operating_working_capital"


def test_selection_question_requires_metric_field():
    parsed = parse_query("Which of JPM's business segments had the lowest net revenue in 2021 Q1?")

    assert parsed["task_type"] == "selection"
    assert parsed["required_fields"] == ["revenue"]


def test_fixed_asset_turnover_prefers_primary_statements():
    parsed = parse_query("What is Activision Blizzard's FY2019 fixed asset turnover ratio?")

    assert parsed["company"] == "activision_blizzard"
    assert parsed["statement_types"] == ["income_statement", "balance_sheet"]


def test_capex_lookup_requires_capital_expenditures_field():
    parsed = parse_query("How much did the company spend in capex in FY2018?")

    assert parsed["task_type"] == "lookup"
    assert parsed["required_fields"] == ["capital_expenditures"]
    assert parsed["statement_types"] == ["cash_flow"]


def test_company_store_count_directive_requires_total_row():
    question = "What was the change in the company's total store count from FY2021 to FY2022?"
    directives = build_answer_directives(question, parse_query(question))

    assert any("Total row" in directive for directive in directives)
    assert any("brand, segment, geography" in directive for directive in directives)
    assert all("Best Buy" not in directive for directive in directives)


def test_quick_ratio_directive_requires_direct_conclusion():
    question = "Calculate the FY2022 quick ratio and say whether it is healthy."
    directives = build_answer_directives(question, parse_query(question))

    assert any("give the requested healthy/not-healthy conclusion directly" in directive for directive in directives)
    assert any("generic business-model or cash-flow caveat" in directive for directive in directives)


def test_comparison_directive_requires_one_conclusion_after_comparison():
    question = "Did operating margin increase between FY2021 and FY2022?"
    directives = build_answer_directives(question, parse_query(question))

    assert any("before stating the directional conclusion" in directive for directive in directives)
    assert any("without an initial guess or self-correction" in directive for directive in directives)


def test_capital_intensity_directive_requires_yes_no_conclusion():
    question = "Based on capital spending, net PP&E and revenue, is the company capital-intensive?"
    directives = build_answer_directives(question, parse_query(question))

    assert any("capital-intensity conclusion" in directive for directive in directives)
    assert any("universal threshold" in directive for directive in directives)
    assert all("not capital-intensive" not in directive for directive in directives)


def test_acquisition_directive_requires_transaction_evidence():
    directives = build_answer_directives(
        "What are the main companies acquired during the year?",
        parse_query("What are the main companies acquired during the year?"),
    )

    assert any("transaction statements" in directive for directive in directives)
    assert any("glossary" in directive for directive in directives)


def test_best_performance_directive_uses_growth_not_mix():
    question = "Which category performed best on the top line?"
    directives = build_answer_directives(question, parse_query(question))

    assert any("change or growth measure" in directive for directive in directives)
    assert any("revenue mix" in directive for directive in directives)


def test_forecast_and_driver_directives_preserve_task_semantics():
    forecast = "What production changes are expected next year?"
    drivers = "What drove the change in operating margin?"

    assert any("future action and direction" in item for item in build_answer_directives(forecast, parse_query(forecast)))
    assert any("explicit MD&A attribution" in item for item in build_answer_directives(drivers, parse_query(drivers)))


def test_domestic_scope_directive_excludes_international_table():
    question = "Which category performed best in the domestic USA market?"

    assert any("Do not substitute an International" in item for item in build_answer_directives(question, parse_query(question)))


def test_parse_query_extracts_explicit_rounding_precision():
    parsed = parse_query("Calculate ROA and round your answer to two decimal places.")

    assert parsed["rounding_decimal_places"] == 2


def test_margin_uses_percent_unit_but_roa_formula_remains_decimal():
    margin_question = "What is FY2015 depreciation and amortization % margin?"
    margin = parse_query(margin_question)
    roa = parse_query("What is ROA? Round your answer to two decimal places.")

    assert margin["result_unit"] == "percent"
    assert roa["result_unit"] == "decimal"
    assert any(
        "multiplying the validated decimal ratio by 100" in item
        for item in build_answer_directives(margin_question, margin)
    )


def test_driver_rewrite_keeps_mda_intent():
    rewrite = build_finance_query_rewrite("What drove operating margin change in FY2022?")

    assert "drivers" in rewrite
    assert "management discussion and analysis" in rewrite


def test_exchange_does_not_trigger_change_comparison():
    parsed = parse_query("Which securities trade on a national securities exchange in 2022?")

    assert parsed["task_type"] == "lookup"


def test_finance_rewrite_is_concise_and_keeps_financial_anchors():
    question = "According to the statements, what is AMD FY2015 depreciation and amortization margin?"
    rewrite = build_finance_query_rewrite(question)

    assert question not in rewrite
    assert "amd" in rewrite
    assert "depreciation and amortization" in rewrite
    assert "net revenue" in rewrite
    assert "2015" in rewrite


def test_revenue_field_match_rejects_percentage_context():
    text = (
        "International sales as a percentage of net revenue were 75% in 2015.\n"
        "The increase in net revenue from domestic products reflected demand.\n"
        "Liquidity was $785 million."
    )

    assert match_required_fields_in_text(["revenue"], text) == {}


def test_exchange_registered_securities_uses_cover_page_field():
    question = "Which debt securities are registered to trade on a national securities exchange?"
    parsed = parse_query(question)
    directives = build_answer_directives(question, parsed)

    assert parsed["task_type"] == "lookup"
    assert parsed["required_fields"] == ["exchange_registered_securities"]
    assert any("Section 12(b)" in directive for directive in directives)
