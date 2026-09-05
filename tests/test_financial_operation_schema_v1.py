from backend.financial_operation_schema_v1 import build_financial_operation_schema_v1, find_required_operands_v1


def test_builds_supported_operation_schemas():
    current = build_financial_operation_schema_v1("What was the FY2023 current ratio for Example Co?")
    assert current["metric"] == "current ratio"
    assert current["formula"] == "total current assets / total current liabilities"
    inventory = build_financial_operation_schema_v1("How many times has Example sold its inventory in FY22? Calculate inventory turnover.")
    assert inventory["metric"] == "inventory turnover"
    assert inventory["period_requirement"]["explicit_periods"] == ["2022"]
    growth = build_financial_operation_schema_v1("Did adjusted EPS growth accelerate in FY2023?")
    assert growth["metric"] == "EPS growth"
    assert growth["operation_type"] == "percentage_change"
    reordered = build_financial_operation_schema_v1("Is growth in Example's adjusted EPS expected to accelerate in FY2023?")
    assert reordered["metric"] == "EPS growth"


def test_adjusted_ebit_does_not_match_ebitdar():
    schema = build_financial_operation_schema_v1(
        "What was MGM's interest coverage ratio using FY2022 Adjusted EBIT as the numerator and interest expense as the denominator?"
    )
    evidence = """Source: MGM.pdf | Page: 1
Adjusted EBITDAR $100
Interest expense $20
"""
    result = find_required_operands_v1(schema, evidence, [{"filename": "MGM.pdf", "page_number": 1, "company": "MGM"}])
    assert result["missing_operands"] == ["adjusted_ebit"]


def test_operand_matching_is_entity_scoped():
    schema = build_financial_operation_schema_v1("Does Boeing have an improving gross margin profile in FY2022?")
    evidence = """Source: ULTA.pdf | Page: 1
Total revenues $200
Cost of goods sold $100

Source: BOEING.pdf | Page: 2
Total revenues $300
"""
    metadata = [
        {"filename": "ULTA.pdf", "page_number": 1, "company": "ULTA"},
        {"filename": "BOEING.pdf", "page_number": 2, "company": "BOEING"},
    ]
    result = find_required_operands_v1(schema, evidence, metadata)
    assert "revenue" in result["found_operands"]
    assert result["missing_operands"] == ["cost_of_sales"]


def test_gross_margin_accepts_cost_of_products_disclosure():
    schema = build_financial_operation_schema_v1("Does Boeing have an improving gross margin profile in FY2022?")
    evidence = "Source: BOEING.pdf | Page: 2\nTotal revenues 66,608 62,286\nCost of products (53,969) (49,954)"
    metadata = [{"filename": "BOEING.pdf", "page_number": 2, "company": "BOEING"}]
    assert find_required_operands_v1(schema, evidence, metadata)["complete"] is True


def test_inventory_requires_two_balance_values():
    schema = build_financial_operation_schema_v1("Calculate inventory turnover for AES in FY2022.")
    one_period = "Source: AES.pdf | Page: 3\nCost of sales $90\nInventory $10"
    metadata = [{"filename": "AES.pdf", "page_number": 3, "company": "AES"}]
    assert find_required_operands_v1(schema, one_period, metadata)["missing_operands"] == ["average_inventory"]
    two_periods = "Source: AES.pdf | Page: 3\nCost of sales $90\nInventories 2022 10 2021 8"
    assert find_required_operands_v1(schema, two_periods, metadata)["complete"] is True


def test_unrecognized_question_returns_explicit_unsupported_schema():
    schema = build_financial_operation_schema_v1("Which companies were acquired?")
    assert schema["recognized"] is False
    assert schema["operation_type"] == "unsupported"
    assert schema["required_operands"] == []
