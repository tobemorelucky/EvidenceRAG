from query_parser import (
    assess_answer_facets,
    build_answer_directives,
    assess_required_field_coverage,
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


def test_forecast_and_driver_directives_preserve_task_semantics():
    forecast = "What production changes are expected next year?"
    drivers = "What drove the change in operating margin?"

    assert any("future action and direction" in item for item in build_answer_directives(forecast, parse_query(forecast)))
    assert any("explicit MD&A attribution" in item for item in build_answer_directives(drivers, parse_query(drivers)))


def test_parse_query_extracts_explicit_rounding_precision():
    parsed = parse_query("Calculate ROA and round your answer to two decimal places.")

    assert parsed["rounding_decimal_places"] == 2


def test_parse_query_exposes_generic_frame_alignment_fields():
    parsed = parse_query("Which reporting segment had the highest net revenue in FY2022?")

    assert parsed["target_measure"] == "net revenue"
    assert "net revenue" in parsed["required_concepts"]
    assert parsed["required_periods"] == ["2022"]
    assert parsed["candidate_dimension"] == "reporting segment"
    assert parsed["scope"] == "segment"
    assert parsed["operation"] == "argmax"
    assert parsed["selection_direction"] == "max"


def test_parse_query_builds_generic_target_measure_without_company_rules():
    parsed = parse_query("What was Example Holdings' days payable outstanding for FY2021?")

    assert "days payable outstanding" in parsed["target_measure"]
    assert parsed["required_concepts"] == [parsed["target_measure"]]
    assert parsed["required_periods"] == ["2021"]


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

    assert parsed["task_type"] == "lookup"
    assert parsed["required_fields"] == ["exchange_registered_securities"]


def test_generic_selection_is_inferred_from_requested_ranking_operation():
    parsed = parse_query("Which activity brought in the most cash during the period?")

    assert parsed["task_type"] == "selection"


def test_generic_comparison_is_inferred_without_registered_metric_name():
    parsed = parse_query("Did the disclosed expense rate improve year-over-year?")

    assert parsed["task_type"] == "comparison"


def test_generic_calculation_is_inferred_without_registered_formula():
    parsed = parse_query("Calculate the disclosed book value per share for the period.")

    assert parsed["task_type"] == "calculation"
    assert parsed["formula"] == ""
    assert parsed["required_fields"] == []


def test_generic_hypothetical_is_inferred_as_judgment():
    parsed = parse_query("If all assets were liquidated, how much could investors receive?")

    assert parsed["task_type"] == "judgment"


def test_calculation_intent_precedes_secondary_metric_relevance_caveat():
    parsed = parse_query(
        "How many times was inventory sold? If that measure is not meaningful, explain why."
    )

    assert parsed["task_type"] == "calculation"


def test_company_name_containing_best_does_not_trigger_selection():
    parsed = parse_query("Was there any change in the number of Best Holdings stores between FY2024 and FY2023?")

    assert parsed["task_type"] == "comparison"


def test_percentage_change_operation_is_independent_from_comparison_task_type():
    parsed = parse_query(
        "What is the year-over-year percentage change in operating income from FY2015 to FY2016?"
    )

    assert parsed["task_type"] == "comparison"
    assert parsed["operation"] == "percentage_change"
    assert parsed["baseline_period"] == "2015"
    assert parsed["target_period"] == "2016"
    assert parsed["period_order"] == ["2015", "2016"]
    assert parsed["result_unit"] == "percent"


def test_yoy_change_with_percent_output_unit_requests_percentage_change():
    parsed = parse_query(
        "What is the year-over-year change in unadjusted operating income from FY2015 to FY2016 "
        "(in units of percents and round to one decimal place)?"
    )

    assert parsed["operation"] == "percentage_change"
    assert parsed["result_unit"] == "percent"
    assert parsed["rounding_decimal_places"] == 1


def test_directional_and_absolute_change_choose_different_existing_operations():
    direction = parse_query("Did revenue increase from FY2022 to FY2023?")
    difference = parse_query("By how many dollars did revenue change from FY2022 to FY2023?")

    assert direction["operation"] == "compare"
    assert difference["operation"] == "subtract"
    assert direction["baseline_period"] == difference["baseline_period"] == "2022"
    assert direction["target_period"] == difference["target_period"] == "2023"


def test_between_periods_preserves_list_order_but_does_not_invent_direction():
    parsed = parse_query("Was revenue higher between FY2023 and FY2024?")

    assert parsed["period_order"] == ["2023", "2024"]
    assert parsed["baseline_period"] == ""
    assert parsed["target_period"] == ""
    assert parsed["period_semantics_confidence"] < 0.8


def test_ambiguous_best_does_not_request_authoritative_argmax():
    parsed = parse_query("Which segment performed best in FY2022?")

    assert parsed["task_type"] == "selection"
    assert parsed["operation"] == "select"
    assert parsed["operation_confidence"] < 0.8


def test_explicit_selection_measure_can_request_argmax():
    parsed = parse_query("Which segment had the highest revenue in FY2022?")

    assert parsed["operation"] == "argmax"
    assert parsed["candidate_dimension"] == "segment"
    assert parsed["operation_confidence"] >= 0.8


def test_multi_part_answer_contract_comes_only_from_question_semantics(monkeypatch):
    monkeypatch.setenv("ANSWER_REQUIRED_FACETS_ENABLED", "true")
    question = "Which customer accounted for what percentage of FY2023 revenue?"
    parsed = parse_query(question)

    assert parsed["answer_type"] == "multi_part"
    assert parsed["required_facets"] == ["entity", "numeric_value", "percentage"]
    directives = build_answer_directives(question, parsed)
    assert any("entity, numeric_value, percentage" in directive for directive in directives)


def test_percentage_change_contract_requires_final_percentage_not_all_operands():
    parsed = parse_query(
        "What was the percentage change in revenue from FY2022 to FY2023, rounded to one decimal place?"
    )

    assert "percentage" in parsed["required_facets"]
    assert "baseline_value" not in parsed["required_facets"]
    assert "target_value" not in parsed["required_facets"]


def test_answer_facet_check_records_omission_without_filling_it():
    task = {
        "answer_type": "multi_part",
        "required_facets": ["entity", "numeric_value", "percentage"],
    }

    trace = assess_answer_facets("The U.S. government was the primary customer.", task)

    assert trace["facet_checks"]["entity"] is True
    assert trace["missing_facets"] == ["numeric_value", "percentage"]
    assert trace["complete"] is False


def test_explicit_formula_contract_extracts_existing_fields_acronym_and_period_bindings(monkeypatch):
    monkeypatch.setenv("EXPLICIT_FORMULA_ADVISORY_ENABLED", "true")
    parsed = parse_query(
        "What is Example's FY2024 efficiency index? The index is defined as: "
        "365 * (average accounts payable between FY2023 and FY2024) / "
        "(FY2024 COGS + change in inventory between FY2023 and FY2024). "
        "Round your answer to two decimal places."
    )

    operands = {item["key"]: item for item in parsed["explicit_formula_operands"]}
    assert parsed["explicit_formula_present"] is True
    assert parsed["explicit_formula_source"] == "question_explicit_definition"
    assert parsed["explicit_formula_confidence"] == 1.0
    assert parsed["task_type_original"] == "comparison"
    assert parsed["task_type"] == "calculation"
    assert parsed["task_type_resolution_method"] == "explicit_formula_direct_value"
    assert parsed["answer_type"] == "numeric_value"
    assert list(operands) == ["accounts_payable", "cogs", "inventory"]
    assert operands["accounts_payable"]["periods"] == ["2023", "2024"]
    assert operands["accounts_payable"]["transform"] == "average"
    assert operands["cogs"]["field"] == ""
    assert operands["cogs"]["periods"] == ["2024"]
    assert operands["inventory"]["transform"] == "change"
    # Advisory extraction must not alter the existing retrieval rewrite fields.
    assert parsed["required_fields"] == ["inventory"]
    expression = parsed["explicit_formula_expression"]
    assert expression["kind"] == "question_defined_expression"
    assert expression["operand_keys"] == ["accounts_payable", "cogs", "inventory"]
    assert expression["constants"] == ["365"]
    assert expression["execution_allowed"] is False


def test_explicit_formula_does_not_override_real_comparison_intent(monkeypatch):
    monkeypatch.setenv("EXPLICIT_FORMULA_ADVISORY_ENABLED", "true")
    parsed = parse_query(
        "The efficiency index is defined as revenue / assets. "
        "Did the efficiency index increase from FY2023 to FY2024?"
    )

    assert parsed["task_type_original"] == "comparison"
    assert parsed["task_type"] == "comparison"
    assert parsed["task_type_resolution_method"] == "explicit_formula_outer_comparison"


def test_explicit_formula_does_not_override_real_selection_intent(monkeypatch):
    monkeypatch.setenv("EXPLICIT_FORMULA_ADVISORY_ENABLED", "true")
    parsed = parse_query(
        "The efficiency index is defined as revenue / assets. "
        "Which segment had the highest efficiency index in FY2024?"
    )

    assert parsed["task_type"] == "selection"
    assert parsed["task_type_resolution_method"] == "explicit_formula_outer_selection"


def test_explicit_formula_task_type_override_is_fully_reversible(monkeypatch):
    monkeypatch.setenv("EXPLICIT_FORMULA_ADVISORY_ENABLED", "false")
    parsed = parse_query(
        "What is the efficiency index? The index is defined as revenue / assets."
    )

    assert parsed["explicit_formula_present"] is True
    assert parsed["task_type"] == parsed["task_type_original"]
    assert parsed["task_type_resolution_method"] == "explicit_formula_advisory_disabled"


def test_formula_advisory_does_not_infer_formula_without_explicit_definition():
    parsed = parse_query("How efficiently did the company manage payables and inventory in FY2024?")

    assert parsed["explicit_formula_present"] is False
    assert parsed["explicit_formula_operands"] == []
    assert parsed["explicit_formula_confidence"] == 0.0


def test_explicit_formula_reuses_existing_metric_alias_mapping():
    parsed = parse_query("Free cash flow is defined as (cash from operations - capex).")

    assert [item["key"] for item in parsed["explicit_formula_operands"]] == [
        "cash_from_operations", "capital_expenditures",
    ]
    assert parsed["explicit_formula_confidence"] == 1.0
