from copy import deepcopy

from scripts.audit_table_fact_extraction_failure_v1 import classify_failed_table


def _base_table():
    return {
        "table_id": "table-1",
        "document_id": "doc-1",
        "page_id": "doc-1:page:000001",
        "filename": "sample.pdf",
        "page_number": 1,
        "title": "Statement",
        "caption": "",
        "before_context": "",
        "after_context": "",
        "csv_text": "",
        "columns": ["Metric", "2023", "2022"],
        "rows": [
            {"_raw_line": "Year ended 2023 2022"},
            {"_raw_line": "Revenue 120 100"},
            {"_raw_line": "Operating income 25 20"},
        ],
        "quality_score": 0.9,
        "unit": "USD",
        "scale": "millions",
    }


def test_valid_table_is_not_classified_as_failure():
    result = classify_failed_table(_base_table())
    assert result["passed"] is True
    assert result["fact_count"] == 4


def test_header_year_recovery_failure_is_category_a():
    table = _base_table()
    table["title"] = "Statement for 2023"
    table["columns"] = ["Metric", "Current", "Prior"]
    table["rows"][0]["_raw_line"] = "Current period and prior period"

    result = classify_failed_table(table)
    assert result["primary_category"] == "A"


def test_duplicate_year_header_is_category_b():
    table = _base_table()
    table["rows"][0]["_raw_line"] = "Three and six months 2023 2022 2023 2022"

    result = classify_failed_table(table)
    assert result["primary_category"] == "B"
    assert result["features"]["duplicate_year"] is True


def test_unaligned_metric_rows_are_category_c():
    table = _base_table()
    table["rows"][1]["_raw_line"] = "Revenue 120 100 20%"
    table["rows"][2]["_raw_line"] = "Operating income 25 20 25%"

    result = classify_failed_table(table)
    assert result["primary_category"] == "C"
    assert result["features"]["metric_rows"] == 3


def test_narrative_shape_is_category_f_and_empty_is_e():
    narrative = _base_table()
    narrative["columns"] = ["Text"]
    narrative["rows"] = [
        ["This paragraph describes the business operations and management discussion without a numeric grid."],
        ["Additional narrative disclosure explains policies, risks, assumptions, and future expectations."],
    ]
    narrative["quality_score"] = 0
    empty = deepcopy(narrative)
    empty["table_id"] = "table-2"
    empty["rows"] = []
    empty["columns"] = []
    empty["title"] = ""

    assert classify_failed_table(narrative)["primary_category"] == "F"
    assert classify_failed_table(empty)["primary_category"] == "E"


def test_unit_binding_is_a_secondary_label_when_header_recovery_is_root_cause():
    table = _base_table()
    table["title"] = "USD millions for 2023"
    table["columns"] = ["Metric", "Current", "Prior"]
    table["rows"][0]["_raw_line"] = "Current period and prior period"
    table["unit"] = ""
    table["scale"] = ""

    result = classify_failed_table(table)
    assert result["primary_category"] == "A"
    assert "D" in result["diagnostic_labels"]
