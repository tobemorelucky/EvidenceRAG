from calculation_service import build_calculation_result


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
