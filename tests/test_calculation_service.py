from calculation_service import (
    build_calculation_result,
    match_evidence_frames_detailed,
    validate_execution_contract,
)


def _evidence_frame(evidence_id, row_label, value, *, period="2024"):
    return {
        "evidence_id": evidence_id,
        "source_type": "table_cell",
        "company": "Example Co",
        "document": "report.pdf",
        "page_number": 8,
        "statement_type": "income_statement",
        "table_id": "table-1",
        "row_label": row_label,
        "row_path": ["Financial results", row_label],
        "column_path": [period] if period else [],
        "section": "Consolidated Statements of Income",
        "period": period,
        "normalized_value": str(value),
        "currency": "USD",
        "scale": "millions",
        "scope": "consolidated",
        "descriptor": f"{row_label} | Financial results | Consolidated Statements of Income | income statement | consolidated",
        "citation": "[source: report.pdf, page 8]",
    }


def test_frame_matching_uses_descriptor_context_after_alias_matching(monkeypatch):
    monkeypatch.setenv("FRAME_ALIGNMENT_ENABLED", "true")
    frames = [
        _evidence_frame("ef_receivables", "Trade receivables, net", "25"),
        _evidence_frame("ef_noise", "Allowance", "2"),
    ]

    matches, trace = match_evidence_frames_detailed(
        "accounts_receivable",
        frames,
        concepts=["accounts receivable"],
        statement_types=["balance_sheet"],
        scope="consolidated",
    )

    assert [item["evidence_id"] for item in matches] == ["ef_receivables"]
    assert trace["candidates"][0]["match_method"] in {"canonical_alias", "phrase_overlap"}
    assert trace["candidates"][0]["match_score"] > 0.7


def test_frame_matching_exact_layer_suppresses_partial_rows_and_deduplicates(monkeypatch):
    monkeypatch.setenv("FRAME_ALIGNMENT_ENABLED", "true")
    exact = _evidence_frame("ef_exact", "Total current assets", "100")
    partial = _evidence_frame("ef_partial", "Other current assets", "20")

    matches, _ = match_evidence_frames_detailed(
        "current_assets",
        [exact, dict(exact), partial],
        concepts=["total current assets", "current assets"],
    )

    assert [item["evidence_id"] for item in matches] == ["ef_exact"]


def test_comparison_executor_builds_auditable_period_pair(monkeypatch):
    monkeypatch.setenv("FRAME_ALIGNMENT_ENABLED", "true")
    monkeypatch.setenv("STRUCTURED_TASK_EXECUTOR_ENABLED", "true")
    current = _evidence_frame("ef_current", "Revenue", "120", period="2024")
    prior = _evidence_frame("ef_prior", "Revenue", "100", period="2023")
    task = {
        "task_type": "comparison",
        "company": "Example Co",
        "target_measure": "revenue",
        "required_fields": ["revenue"],
        "required_periods": ["2024", "2023"],
        "operation": "compare",
    }

    result = build_calculation_result(task, {"status": "complete"}, [], evidence_frames=[current, prior])

    assert result["executor"] == "evidence_frame"
    assert result["operation"] == "compare"
    assert result["comparison_direction"] == "increased"
    assert result["operand_evidence_ids"] == ["ef_current", "ef_prior"]
    assert result["candidate_matrix"][0]["period"] == "2024"


def test_percentage_change_uses_target_then_baseline_but_preserves_requested_period_order(monkeypatch):
    monkeypatch.setenv("FRAME_ALIGNMENT_ENABLED", "true")
    monkeypatch.setenv("STRUCTURED_TASK_EXECUTOR_ENABLED", "true")
    baseline = _evidence_frame("ef_2015", "Operating income", "903095", period="2015")
    target = _evidence_frame("ef_2016", "Operating income", "1493602", period="2016")
    task = {
        "task_type": "comparison",
        "company": "Example Co",
        "target_measure": "operating income",
        "required_fields": ["operating_income"],
        "required_periods": ["2015", "2016"],
        "baseline_period": "2015",
        "target_period": "2016",
        "period_order": ["2015", "2016"],
        "operation": "percentage_change",
        "operation_confidence": 1.0,
        "result_unit": "percent",
        "rounding_decimal_places": 1,
    }

    result = build_calculation_result(task, {"status": "complete"}, [], evidence_frames=[target, baseline])

    assert result["operation"] == "percentage_change"
    assert result["display_result"] == "65.4"
    assert result["requested_period_order"] == ["2015", "2016"]
    assert result["resolved_period_order"] == ["2015", "2016"]
    assert result["baseline_value"] == "903095"
    assert result["target_value"] == "1493602"
    assert result["authoritative"] is True
    assert result["execution_contract"]["passed"] is True


def test_wrong_operation_result_never_becomes_authoritative():
    task = {
        "task_type": "comparison",
        "operation": "percentage_change",
        "operation_confidence": 1.0,
        "required_periods": ["2015", "2016"],
        "baseline_period": "2015",
        "target_period": "2016",
        "period_order": ["2015", "2016"],
        "target_measure": "operating income",
        "required_concepts": ["operating income"],
        "result_unit": "percent",
    }
    frames = [
        _evidence_frame("ef_2015", "Operating income", "90", period="2015"),
        _evidence_frame("ef_2016", "Operating income", "150", period="2016"),
    ]
    result = {
        "operation": "compare",
        "unit": "percent",
        "rounding": None,
        "candidate_matrix": frames,
        "requested_period_order": ["2015", "2016"],
        "resolved_period_order": ["2015", "2016"],
    }

    contract = validate_execution_contract(task, result, frames)

    assert contract["passed"] is False
    assert contract["operation_match"] is False
    assert "operation_mismatch" in contract["failure_reasons"]


def test_unknown_period_result_never_becomes_authoritative():
    task = {
        "task_type": "comparison",
        "operation": "compare",
        "operation_confidence": 1.0,
        "required_periods": ["2023", "2024"],
        "baseline_period": "2023",
        "target_period": "2024",
        "period_order": ["2023", "2024"],
        "target_measure": "revenue",
        "required_concepts": ["revenue"],
    }
    frames = [
        _evidence_frame("ef_a", "Revenue", "100", period=None),
        _evidence_frame("ef_b", "Revenue", "120", period=None),
    ]
    result = {"operation": "compare", "candidate_matrix": frames}

    contract = validate_execution_contract(task, result, frames)

    assert contract["passed"] is False
    assert contract["period_match"] is False
    assert "required_period_missing" in contract["failure_reasons"]


def test_selection_executor_requires_complete_same_table_candidate_matrix(monkeypatch):
    monkeypatch.setenv("FRAME_ALIGNMENT_ENABLED", "true")
    monkeypatch.setenv("STRUCTURED_TASK_EXECUTOR_ENABLED", "true")
    north = _evidence_frame("ef_north", "North region revenue", "80", period="2024")
    south = _evidence_frame("ef_south", "South region revenue", "120", period="2024")
    task = {
        "task_type": "selection",
        "company": "Example Co",
        "target_measure": "revenue",
        "candidate_dimension": "region",
        "required_periods": ["2024"],
        "selection_direction": "max",
        "operation": "argmax",
    }

    result = build_calculation_result(task, {"status": "complete"}, [], evidence_frames=[north, south])

    assert result["operation"] == "argmax"
    assert result["selected_entity"] == "South region revenue"
    assert result["selected_evidence_id"] == "ef_south"


def test_selection_executor_rejects_ambiguous_candidate_groups(monkeypatch):
    monkeypatch.setenv("FRAME_ALIGNMENT_ENABLED", "true")
    monkeypatch.setenv("STRUCTURED_TASK_EXECUTOR_ENABLED", "true")
    first = _evidence_frame("ef_a", "North revenue", "80")
    second = _evidence_frame("ef_b", "South revenue", "120")
    duplicate_group_a = {**first, "evidence_id": "ef_c", "table_id": "table-2"}
    duplicate_group_b = {**second, "evidence_id": "ef_d", "table_id": "table-2"}
    task = {
        "task_type": "selection",
        "target_measure": "revenue",
        "candidate_dimension": "region",
        "required_periods": ["2024"],
        "selection_direction": "max",
    }

    assert build_calculation_result(
        task,
        {"status": "complete"},
        [],
        evidence_frames=[first, second, duplicate_group_a, duplicate_group_b],
    ) is None


def test_structured_executor_precedes_text_row_parser(monkeypatch):
    monkeypatch.setenv("STRUCTURED_EXECUTOR_ENABLED", "true")
    task_spec = {
        "task_type": "calculation",
        "company": "Example Co",
        "required_periods": ["2024"],
        "required_fields": ["operating_income", "revenue"],
        "formula": "operating_income / revenue",
    }
    frames = [
        _evidence_frame("ef_income", "Operating income", "25"),
        _evidence_frame("ef_revenue", "Revenue", "100"),
    ]

    result = build_calculation_result(task_spec, {"status": "partial"}, [], evidence_frames=frames)

    assert result["executor"] == "evidence_frame"
    assert result["result"] == "0.25"
    assert result["operand_evidence_ids"] == ["ef_income", "ef_revenue"]


def test_formula_executor_rounding_field_satisfies_authoritative_contract(monkeypatch):
    monkeypatch.setenv("STRUCTURED_EXECUTOR_ENABLED", "true")
    task_spec = {
        "task_type": "calculation",
        "company": "Example Co",
        "required_periods": ["2024"],
        "period_order": ["2024"],
        "required_fields": ["operating_income", "revenue"],
        "target_measure": "operating margin",
        "formula": "operating_income / revenue",
        "operation": "divide",
        "operation_confidence": 1.0,
        "result_unit": "decimal",
        "rounding_decimal_places": 2,
    }
    frames = [
        _evidence_frame("ef_income", "Operating income", "25"),
        _evidence_frame("ef_revenue", "Revenue", "100"),
    ]

    result = build_calculation_result(task_spec, {"status": "partial"}, [], evidence_frames=frames)

    assert result["display_result"] == "0.25"
    assert result["execution_contract"]["rounding_match"] is True
    assert result["authoritative"] is True


def test_structured_executor_does_not_guess_missing_period(monkeypatch):
    monkeypatch.setenv("STRUCTURED_EXECUTOR_ENABLED", "true")
    task_spec = {
        "task_type": "calculation",
        "company": "Example Co",
        "required_periods": ["2024"],
        "required_fields": ["operating_income", "revenue"],
        "formula": "operating_income / revenue",
    }
    frames = [
        _evidence_frame("ef_income", "Operating income", "25", period=None),
        _evidence_frame("ef_revenue", "Revenue", "100", period=None),
    ]

    assert build_calculation_result(task_spec, {"status": "partial"}, [], evidence_frames=frames) is None


def test_structured_executor_averages_target_and_nearest_prior_explicit_period(monkeypatch):
    monkeypatch.setenv("STRUCTURED_EXECUTOR_ENABLED", "true")
    task_spec = {
        "task_type": "calculation",
        "company": "Example Co",
        "required_periods": ["2024"],
        "required_fields": ["net_income", "total_assets"],
        "formula": "net_income / average(total_assets)",
    }
    frames = [
        _evidence_frame("ef_income", "Net income", "30", period="2024"),
        _evidence_frame("ef_assets_2024", "Total assets", "120", period="2024"),
        _evidence_frame("ef_assets_2023", "Total assets", "80", period="2023"),
        _evidence_frame("ef_assets_2022", "Total assets", "60", period="2022"),
    ]

    result = build_calculation_result(task_spec, {"status": "partial"}, [], evidence_frames=frames)

    assert result["result"] == "0.3"
    assert result["operands"]["total_assets"]["value"] == ["120", "80"]
    assert result["operand_evidence_ids"] == ["ef_income", "ef_assets_2024", "ef_assets_2023"]


def test_structured_executor_disabled_preserves_existing_path(monkeypatch):
    monkeypatch.setenv("STRUCTURED_EXECUTOR_ENABLED", "false")
    task_spec = {
        "task_type": "calculation",
        "required_fields": ["operating_income", "revenue"],
        "formula": "operating_income / revenue",
    }
    frames = [
        _evidence_frame("ef_income", "Operating income", "25"),
        _evidence_frame("ef_revenue", "Revenue", "100"),
    ]

    assert build_calculation_result(task_spec, {"status": "partial"}, [], evidence_frames=frames) is None


def test_structured_coverage_blocks_execution_until_operands_are_validated(monkeypatch):
    monkeypatch.setenv("STRUCTURED_EXECUTOR_ENABLED", "true")
    monkeypatch.setenv("STRUCTURED_COVERAGE_ENABLED", "true")
    task_spec = {
        "task_type": "calculation",
        "company": "Example Co",
        "required_periods": ["2024"],
        "required_fields": ["operating_income", "revenue"],
        "formula": "operating_income / revenue",
    }
    frames = [
        _evidence_frame("ef_income", "Operating income", "25"),
        _evidence_frame("ef_revenue", "Revenue", "100"),
    ]

    assert build_calculation_result(task_spec, {"status": "partial", "operands_validated": False}, [], evidence_frames=frames) is None
    result = build_calculation_result(
        task_spec,
        {"status": "complete", "operands_validated": True},
        [],
        evidence_frames=frames,
    )
    assert result["executor"] == "evidence_frame"


def test_incomplete_frames_still_allow_validated_text_row_fallback(monkeypatch):
    monkeypatch.setenv("STRUCTURED_EXECUTOR_ENABLED", "true")
    monkeypatch.setenv("STRUCTURED_COVERAGE_ENABLED", "true")
    task_spec = {
        "task_type": "calculation",
        "required_fields": ["operating_income", "revenue"],
        "formula": "operating_income / revenue",
    }
    coverage = {
        "status": "partial",
        "base_status": "complete",
        "operands_validated": False,
        "field_evidence": {
            "operating_income": {"values": ["25"], "filename": "report.pdf", "page_number": 4},
            "revenue": {"values": ["100"], "filename": "report.pdf", "page_number": 4},
        },
    }

    result = build_calculation_result(task_spec, coverage, [], evidence_frames=[])

    assert result["result"] == "0.25"
    assert result.get("executor") is None


def test_calculation_result_applies_only_explicit_final_rounding():
    task_spec = {
        "task_type": "calculation",
        "required_fields": ["net_income", "total_assets"],
        "formula": "net_income / total_assets",
        "rounding_decimal_places": 2,
    }
    coverage = {
        "status": "complete",
        "field_evidence": {
            "net_income": {"values": ["-546"], "filename": "report.pdf", "page_number": 1},
            "total_assets": {"values": ["35663"], "filename": "report.pdf", "page_number": 2},
        },
    }

    result = build_calculation_result(task_spec, coverage)

    assert result["result"].startswith("-0.0153")
    assert result["display_result"] == "-0.02"


def test_build_calculation_result_uses_decimal_and_keeps_sources():
    task_spec = {
        "task_type": "calculation",
        "formula": "operating_income / revenue",
        "required_fields": ["operating_income", "revenue"],
    }
    coverage = {
        "status": "complete",
        "field_evidence": {
            "operating_income": {"values": ["25"], "filename": "a.pdf", "page_number": 4, "alias": "operating income"},
            "revenue": {"values": ["100"], "filename": "a.pdf", "page_number": 4, "alias": "revenue"},
        },
    }

    result = build_calculation_result(task_spec, coverage)

    assert result is not None
    assert result["result"] == "0.25"
    assert result["operands"]["revenue"]["page_number"] == 4


def test_build_calculation_result_rejects_ambiguous_values():
    task_spec = {
        "task_type": "calculation",
        "formula": "current_assets / current_liabilities",
        "required_fields": ["current_assets", "current_liabilities"],
    }
    coverage = {
        "status": "complete",
        "field_evidence": {
            "current_assets": {"values": ["100", "90"]},
            "current_liabilities": {"values": ["50"]},
        },
    }

    assert build_calculation_result(task_spec, coverage) is None


def test_build_calculation_result_rejects_one_value_reused_for_two_fields():
    task_spec = {
        "task_type": "calculation",
        "formula": "operating_income / revenue",
        "required_fields": ["operating_income", "revenue"],
    }
    coverage = {
        "status": "complete",
        "field_evidence": {
            "operating_income": {"values": ["100"], "filename": "a.pdf", "page_number": 4},
            "revenue": {"values": ["100"], "filename": "a.pdf", "page_number": 4},
        },
    }

    assert build_calculation_result(task_spec, coverage) is None


def test_structured_rows_calculate_fixed_asset_turnover_with_decimal_precision():
    task_spec = {
        "task_type": "calculation",
        "formula": "revenue / average(ppe)",
        "required_fields": ["revenue", "ppe"],
    }
    coverage = {"status": "complete", "field_evidence": {}}
    documents = [
        {"filename": "a.pdf", "page_number": 68, "text": "Property and equipment, net 253 282"},
        {"filename": "a.pdf", "page_number": 69, "text": "Net revenues 6,489 7,500 7,017"},
    ]

    result = build_calculation_result(task_spec, coverage, documents)

    assert result is not None
    assert result["result"].startswith("24.257943925")
    assert result["source"] == "structured_row_decimal"


def test_structured_rows_calculate_operating_working_capital():
    task_spec = {
        "task_type": "calculation",
        "formula": "accounts_receivable + inventory + other_current_assets - accounts_payable - other_accrued_liabilities",
        "required_fields": [
            "accounts_receivable",
            "inventory",
            "other_current_assets",
            "accounts_payable",
            "other_accrued_liabilities",
        ],
    }
    coverage = {"status": "complete", "field_evidence": {}}
    documents = [
        {
            "filename": "corning.pdf",
            "page_number": 59,
            "text": "\n".join(
                [
                    "Trade accounts receivable, net of doubtful accounts - $40 and $42 1,721 2,004",
                    "Inventories (Note 5) 2,904 2,481",
                    "Inventory turns 3.4 3.7",
                    "Other current assets (Notes 10 and 14) 1,157 1,026",
                    "Accounts payable 1,804 1,612",
                    "Other accrued liabilities (Notes 10 and 13) 3,147 3,139",
                ]
            ),
        }
    ]

    result = build_calculation_result(task_spec, coverage, documents)

    assert result is not None
    assert result["result"] == "831"


def test_structured_rows_prefer_net_receivables_and_total_current_liabilities():
    task_spec = {
        "task_type": "calculation",
        "formula": "(cash_and_equivalents + short_term_investments + accounts_receivable) / current_liabilities",
        "required_fields": [
            "cash_and_equivalents",
            "short_term_investments",
            "accounts_receivable",
            "current_liabilities",
        ],
    }
    coverage = {"status": "complete", "field_evidence": {}}
    documents = [
        {
            "filename": "verizon.pdf",
            "page_number": 55,
            "text": "\n".join(
                [
                    "Cash and cash equivalents $ 2,605 $ 2,921",
                    "Accounts receivable 25,332 24,742",
                    "Accounts receivable, net 24,506 23,846",
                    "Other current liabilities 12,097 11,025",
                    "Total current liabilities 50,171 47,160",
                    "Marketable securities 8 18",
                ]
            ),
        }
    ]

    result = build_calculation_result(task_spec, coverage, documents)

    assert result is not None
    assert result["operands"]["accounts_receivable"]["value"] == "24506"
    assert result["operands"]["current_liabilities"]["value"] == "50171"
    assert result["result"].startswith("0.540")


def test_cash_flow_capex_outflow_is_subtracted_once():
    task_spec = {
        "task_type": "calculation",
        "formula": "operating_income + depreciation_amortization - capital_expenditures",
        "required_fields": ["operating_income", "depreciation_amortization", "capital_expenditures"],
    }
    coverage = {"status": "complete", "field_evidence": {}}
    documents = [
        {"filename": "report.pdf", "page_number": 1, "text": "Operating profit 11,512"},
        {
            "filename": "report.pdf",
            "page_number": 2,
            "text": "Depreciation and amortization 2,763\nCapital spending (5,207)",
        },
    ]

    result = build_calculation_result(task_spec, coverage, documents)

    assert result is not None
    assert result["operands"]["capital_expenditures"]["value"] == "5207"
    assert result["result"] == "9068"


def test_revenue_operand_rejects_percentage_of_revenue_sentence():
    task_spec = {
        "task_type": "calculation",
        "formula": "depreciation_amortization / revenue",
        "required_fields": ["depreciation_amortization", "revenue"],
    }
    coverage = {"status": "complete", "field_evidence": {}}
    documents = [
        {"filename": "report.pdf", "page_number": 1, "text": "Depreciation and amortization 167"},
        {
            "filename": "report.pdf",
            "page_number": 2,
            "text": "International sales as a percentage of net revenue were 75% in 2015\nDeferred revenue 94",
        },
    ]

    assert build_calculation_result(task_spec, coverage, documents) is None


def test_operating_margin_comparison_records_validated_direction():
    task_spec = {
        "task_type": "calculation",
        "formula": "operating_income / revenue",
        "required_fields": ["operating_income", "revenue"],
        "compare_periods": True,
    }
    coverage = {"status": "complete", "field_evidence": {}}
    documents = [
        {
            "filename": "report.pdf",
            "page_number": 53,
            "text": "Operating income 6,098 5,802\nTotal revenue 17,606 15,785",
        }
    ]

    result = build_calculation_result(task_spec, coverage, documents)

    assert result is not None
    assert result["comparison"]["direction"] == "decreased"
    assert result["comparison"]["reported_order"] == "latest_then_prior"


def test_structured_rows_select_requested_year_column_instead_of_latest_column():
    task_spec = {
        "task_type": "calculation",
        "formula": "cash_from_operations / current_liabilities",
        "required_fields": ["cash_from_operations", "current_liabilities"],
        "required_periods": ["2015"],
    }
    coverage = {"status": "complete", "field_evidence": {}}
    documents = [
        {
            "filename": "ADOBE_2016_10K.pdf",
            "page_number": 60,
            "text": "\n".join(
                [
                    "ADOBE SYSTEMS INCORPORATED",
                    "CONSOLIDATED BALANCE SHEETS",
                    "December 2, 2016",
                    "November 27, 2015",
                    "Total current liabilities 2,811,635 2,213,556",
                ]
            ),
        },
        {
            "filename": "ADOBE_2016_10K.pdf",
            "page_number": 64,
            "text": "\n".join(
                [
                    "ADOBE SYSTEMS INCORPORATED",
                    "CONSOLIDATED STATEMENTS OF CASH FLOWS",
                    "Years Ended",
                    "December 2, 2016",
                    "November 27, 2015",
                    "November 28, 2014",
                    "Net cash provided by operating activities 2,199,728 1,469,502 1,287,010",
                ]
            ),
        },
    ]

    result = build_calculation_result(task_spec, coverage, documents)

    assert result is not None
    assert result["operands"]["cash_from_operations"]["value"] == "1469502"
    assert result["operands"]["current_liabilities"]["value"] == "2213556"
    assert result["operands"]["cash_from_operations"]["period"] == "2015"
    assert result["result"].startswith("0.663")


def test_structured_rows_reject_ambiguous_multi_column_row_without_requested_period_header():
    task_spec = {
        "task_type": "calculation",
        "formula": "cash_from_operations / current_liabilities",
        "required_fields": ["cash_from_operations", "current_liabilities"],
        "required_periods": ["2015"],
    }
    coverage = {"status": "complete", "field_evidence": {}}
    documents = [
        {"filename": "report.pdf", "page_number": 1, "text": "Total current liabilities 200 180"},
        {"filename": "report.pdf", "page_number": 2, "text": "Net cash provided by operating activities 100 90"},
    ]

    assert build_calculation_result(task_spec, coverage, documents) is None


def test_structured_rows_ignore_narrative_metric_mentions():
    task_spec = {
        "task_type": "calculation",
        "formula": "operating_income / revenue",
        "required_fields": ["operating_income", "revenue"],
        "required_periods": ["2022"],
        "compare_periods": True,
    }
    coverage = {"status": "complete", "field_evidence": {}}
    documents = [
        {
            "filename": "report.pdf",
            "page_number": 20,
            "text": "2022 2021\nForeign currency impacts decreased operating income by 271 and 103",
        },
        {
            "filename": "report.pdf",
            "page_number": 47,
            "text": "2022 2021\nOperating income 6,539 7,369\nNet sales 34,229 35,355",
        },
    ]

    result = build_calculation_result(task_spec, coverage, documents)

    assert result is not None
    assert result["operands"]["operating_income"]["value"] == "6539"
    assert result["comparison"]["values"][0].startswith("0.191")


def test_structured_rows_calculate_average_total_assets_for_roa():
    task_spec = {
        "task_type": "calculation",
        "formula": "net_income / average(total_assets)",
        "required_fields": ["net_income", "total_assets"],
        "required_periods": ["2022", "2021"],
    }
    coverage = {"status": "complete", "field_evidence": {}}
    documents = [
        {
            "filename": "AES_2022_10K.pdf",
            "page_number": 128,
            "text": "Consolidated Balance Sheets\n2022 2021\nTotal assets 38,237 34,122",
        },
        {
            "filename": "AES_2022_10K.pdf",
            "page_number": 130,
            "text": "Consolidated Statements of Operations\n2022 2021\nNet income (546) (409)",
        },
    ]

    result = build_calculation_result(task_spec, coverage, documents)

    assert result is not None
    assert result["operands"]["total_assets"]["value"] == ["38237", "34122"]
    assert result["result"].startswith("-0.015")


def test_structured_rows_prefer_target_annual_filing_over_later_quarterly_comparison():
    task_spec = {
        "task_type": "calculation",
        "formula": "operating_income - capital_expenditures",
        "required_fields": ["operating_income", "capital_expenditures"],
        "required_periods": ["2022"],
    }
    coverage = {"status": "complete", "field_evidence": {}}
    documents = [
        {
            "filename": "PEPSICO_2023Q1_EARNINGS.pdf",
            "page_number": 3,
            "text": "2023 2022\nOperating profit 6,000 5,267\nCapital spending 600 522",
        },
        {
            "filename": "PEPSICO_2022_10K.pdf",
            "page_number": 61,
            "text": "Consolidated Statement of Income\n2022 2021\nOperating profit 11,512 11,162",
        },
        {
            "filename": "PEPSICO_2022_10K.pdf",
            "page_number": 63,
            "text": "Consolidated Statement of Cash Flows\n2022 2021\nCapital spending (5,207) (4,625)",
        },
    ]

    result = build_calculation_result(task_spec, coverage, documents)

    assert result is not None
    assert result["operands"]["operating_income"]["value"] == "11512"
    assert result["operands"]["capital_expenditures"]["value"] == "5207"


def test_structured_rows_reject_parent_only_schedule_and_aocl_reclassification():
    task_spec = {
        "task_type": "calculation",
        "formula": "net_income / average(total_assets)",
        "required_fields": ["net_income", "total_assets"],
        "required_periods": ["2022", "2021"],
    }
    coverage = {"status": "complete", "field_evidence": {}}
    documents = [
        {
            "filename": "AES_2022_10K.pdf",
            "page_number": 180,
            "text": "Reclassifications out of AOCL\n2022 2021\nNet income attributable 44 254",
        },
        {
            "filename": "AES_2022_10K.pdf",
            "page_number": 213,
            "text": "SCHEDULE I CONDENSED FINANCIAL INFORMATION OF PARENT\n2022 2021\nTotal assets 7,575 7,525",
        },
        {
            "filename": "AES_2022_10K.pdf",
            "page_number": 83,
            "text": "Selected Financial Data\n2022 2021\nNet income attributable (546) (409)\nTotal assets 38,363 32,963",
        },
    ]

    result = build_calculation_result(task_spec, coverage, documents)

    assert result is not None
    assert result["operands"]["net_income"]["value"] == "-546"
    assert result["operands"]["total_assets"]["value"] == ["38363", "32963"]


def test_structured_rows_remove_allowance_amounts_before_receivable_columns():
    task_spec = {
        "task_type": "calculation",
        "formula": "accounts_receivable / current_liabilities",
        "required_fields": ["accounts_receivable", "current_liabilities"],
        "required_periods": ["2023"],
    }
    coverage = {"status": "complete", "field_evidence": {}}
    documents = [
        {
            "filename": "3M_2023Q2_10Q.pdf",
            "page_number": 4,
            "text": "\n".join(
                [
                    "Consolidated Balance Sheet",
                    "2023 2022",
                    "Accounts receivable — net of allowances of $160 and $174 4,947 4,532",
                    "Total current liabilities 10,936 9,523",
                ]
            ),
        }
    ]

    result = build_calculation_result(task_spec, coverage, documents)

    assert result is not None
    assert result["operands"]["accounts_receivable"]["value"] == "4947"
