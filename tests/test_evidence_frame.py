from backend.evidence_frame import build_evidence_frames


def test_balance_sheet_frames_preserve_headers_units_scope_and_negative_sign():
    table = {
        "table_id": "report::table::p7::1",
        "filename": "report.pdf",
        "page_number": 7,
        "title": "Consolidated Balance Sheets",
        "before_context": "USD in millions",
        "columns": ["Metric", "2023", "2022"],
        "rows": [
            {"Metric": "Cash and cash equivalents", "2023": "1,021", "2022": "(848)"},
            {"Metric": "Total assets", "2023": "14,450", "2022": "14,100"},
        ],
    }

    frames, trace = build_evidence_frames([table], company="generic_company")

    negative = next(frame for frame in frames if frame["row_label"] == "Cash and cash equivalents" and frame["period"] == "2022")
    assert negative["raw_value"] == "(848)"
    assert negative["normalized_value"] == "-848"
    assert negative["sign"] == "negative"
    assert negative["currency"] == "USD"
    assert negative["scale"] == "millions"
    assert negative["scope"] == "consolidated"
    assert negative["statement_type"] == "balance_sheet"
    assert "cash and cash equivalents" in negative["descriptor"].lower()
    assert "consolidated balance sheets" in negative["descriptor"].lower()
    assert "balance sheet" in negative["descriptor"].lower()
    assert negative["citation"] == "[source: report.pdf, page 7]"
    assert trace["evidence_frame_count"] == 4
    assert trace["frames_with_period"] == 4
    assert trace["frames_used_for_execution"] == 0


def test_income_statement_percentage_is_not_treated_as_decimal_or_money():
    table = {
        "table_id": "income-1",
        "filename": "income.pdf",
        "page_number": 4,
        "title": "Consolidated Statements of Income",
        "columns": ["Metric", "FY2024", "FY2023"],
        "rows": [
            {"Metric": "Operating margin", "FY2024": "12.5%", "FY2023": "11.2%"},
            {"Metric": "Net revenues", "FY2024": "$120", "FY2023": "$100"},
        ],
    }

    frames, _ = build_evidence_frames([table])

    margin = next(frame for frame in frames if frame["row_label"] == "Operating margin" and frame["period"] == "2024")
    assert margin["normalized_value"] == "12.5"
    assert margin["value_type"] == "percentage"
    assert margin["scale"] == "percent"
    assert margin["currency"] is None


def test_column_without_explicit_period_remains_unknown():
    table = {
        "table_id": "cash-flow-1",
        "filename": "cash-flow.pdf",
        "page_number": 8,
        "title": "Consolidated Statements of Cash Flows",
        "columns": ["Metric", "Current period", "Prior period"],
        "rows": [
            {"Metric": "Net cash provided by operating activities", "Current period": "300", "Prior period": "250"},
            {"Metric": "Capital expenditures", "Current period": "(80)", "Prior period": "(70)"},
        ],
    }

    frames, _ = build_evidence_frames([table])

    assert frames
    assert all(frame["period"] is None for frame in frames)
    assert all(frame["statement_type"] == "cash_flow" for frame in frames)


def test_generic_value_columns_recover_only_explicit_matched_page_headers():
    table = {
        "table_id": "balance-1",
        "filename": "report.pdf",
        "page_number": 10,
        "evidence_page_number": 9,
        "title": "See accompanying notes.",
        "columns": ["Metric", "value_1", "value_2"],
        "rows": [
            {"Metric": "Cash and cash equivalents", "value_1": "$120", "value_2": "$100"},
            {"Metric": "Total assets", "value_1": "500", "value_2": "450"},
        ],
        "evidence_page_context": (
            "Consolidated Balance Sheets\n"
            "(USD in millions)\n"
            "December 31, 2024\n"
            "December 31, 2023\n"
            "ASSETS\n"
            "Cash and cash equivalents $120 $100\n"
            "Total assets 500 450"
        ),
    }

    frames, _ = build_evidence_frames([table], company="Example Co")

    assert {frame["period"] for frame in frames} == {"2024", "2023"}
    assert all(frame["page_number"] == 9 for frame in frames)
    assert all(frame["currency"] == "USD" for frame in frames)
    assert all(frame["scale"] == "millions" for frame in frames)
    assert all(frame["scope"] == "consolidated" for frame in frames)
    assert all(frame["period_provenance"]["source"] == "matched_page_explicit_header" for frame in frames)


def test_narrative_years_do_not_define_generic_value_columns():
    table = {
        "table_id": "balance-2",
        "filename": "report.pdf",
        "page_number": 10,
        "title": "Consolidated Balance Sheets",
        "columns": ["Metric", "value_1", "value_2"],
        "rows": [
            {"Metric": "Cash and cash equivalents", "value_1": "120", "value_2": "100"},
            {"Metric": "Total assets", "value_1": "500", "value_2": "450"},
        ],
        "evidence_page_context": (
            "Consolidated Balance Sheets\n"
            "Revenue changed during 2024 compared with 2023.\n"
            "Cash and cash equivalents 120 100\n"
            "Total assets 500 450"
        ),
    }

    frames, _ = build_evidence_frames([table])

    assert frames
    assert all(frame["period"] is None for frame in frames)


def test_ambiguous_or_unsupported_table_does_not_create_frames():
    table = {
        "table_id": "note-1",
        "filename": "notes.pdf",
        "page_number": 20,
        "title": "Notes to Consolidated Financial Statements",
        "columns": ["Category", "2024", "2023"],
        "rows": [
            {"Category": "North", "2024": "10", "2023": "9"},
            {"Category": "South", "2024": "12", "2023": "11"},
        ],
    }

    frames, trace = build_evidence_frames([table])

    assert frames == []
    assert trace["evidence_frame_tables_accepted"] == 0
    assert trace["evidence_frame_skipped"]["unsupported_statement"] == 1


def test_existing_normalized_column_paths_are_preserved():
    table = {
        "table_id": "income-2",
        "filename": "income.pdf",
        "page_number": 5,
        "normalized": True,
        "normalized_title": "Consolidated Statements of Operations",
        "normalized_unit": "USD millions",
        "normalized_columns": ["Metric", "Three Months Ended, 2024", "Year Ended, 2024"],
        "column_schema": [
            {"label": "Three Months Ended, 2024", "path": ["Three Months Ended", "2024"], "value_type": "money", "unit": "USD millions"},
            {"label": "Year Ended, 2024", "path": ["Year Ended", "2024"], "value_type": "money", "unit": "USD millions"},
        ],
        "normalized_rows": [
            {"Metric": "Net revenues", "Three Months Ended, 2024": "30", "Year Ended, 2024": "120"},
            {"Metric": "Income from operations", "Three Months Ended, 2024": "6", "Year Ended, 2024": "20"},
        ],
    }

    frames, _ = build_evidence_frames([table])

    frame = next(item for item in frames if item["column_label"] == "Year Ended, 2024")
    assert frame["column_path"] == ["Year Ended", "2024"]
    assert frame["period"] == "2024"
    assert frame["period_provenance"]["source"] == "column_schema"


def test_multilevel_quarter_and_year_header_keeps_explicit_period_provenance():
    table = {
        "table_id": "quarterly-income",
        "filename": "income.pdf",
        "page_number": 5,
        "normalized": True,
        "normalized_title": "Consolidated Statements of Operations",
        "normalized_unit": "USD millions",
        "normalized_columns": ["Metric", "Q2 FY2024", "Q2 FY2023"],
        "column_schema": [
            {"label": "Q2 FY2024", "path": ["Three Months Ended", "Q2", "FY2024"], "unit": "USD millions"},
            {"label": "Q2 FY2023", "path": ["Three Months Ended", "Q2", "FY2023"], "unit": "USD millions"},
        ],
        "normalized_rows": [
            {"Metric": "Net revenues", "Q2 FY2024": "120", "Q2 FY2023": "100"},
            {"Metric": "Operating income", "Q2 FY2024": "30", "Q2 FY2023": "20"},
        ],
    }

    frames, _ = build_evidence_frames([table])

    assert {frame["period"] for frame in frames} == {"Q2 2024", "Q2 2023"}
    assert all(frame["period_provenance"]["source"] == "column_schema" for frame in frames)


def test_adjacent_same_table_continuation_can_inherit_explicit_header_with_provenance():
    header_page = {
        "table_id": "cash-flow-continuation",
        "filename": "report.pdf",
        "page_number": 10,
        "title": "Consolidated Statements of Cash Flows",
        "columns": ["Metric", "value_1", "value_2"],
        "rows": [
            {"Metric": "Net income", "value_1": "120", "value_2": "100"},
            {"Metric": "Depreciation and amortization", "value_1": "30", "value_2": "25"},
        ],
        "evidence_page_context": (
            "Consolidated Statements of Cash Flows\n"
            "Year Ended December 31, 2024 2023\n"
            "Net income 120 100\n"
            "Depreciation and amortization 30 25"
        ),
    }
    continuation = {
        "table_id": "cash-flow-continuation",
        "filename": "report.pdf",
        "page_number": 11,
        "title": "Consolidated Statements of Cash Flows (continued)",
        "columns": ["Metric", "value_1", "value_2"],
        "rows": [
            {"Metric": "Capital expenditures", "value_1": "(40)", "value_2": "(35)"},
            {"Metric": "Net cash provided by operating activities", "value_1": "150", "value_2": "130"},
        ],
    }

    frames, _ = build_evidence_frames([header_page, continuation])
    continued = [frame for frame in frames if frame["page_number"] == 11]

    assert {frame["period"] for frame in continued} == {"2024", "2023"}
    assert all(frame["period_provenance"]["source"] == "same_table_continuation" for frame in continued)
    assert all(frame["period_provenance"]["inherited_from_page"] == 10 for frame in continued)


def test_quality_rejected_table_can_be_recovered_only_with_stable_primary_statement_rows():
    table = {
        "table_id": "recovered-balance",
        "filename": "balance.pdf",
        "page_number": 10,
        "accepted": False,
        "reject_reason": "mostly_empty",
        "normalized": True,
        "normalized_title": "",
        "normalized_columns": ["Metric", "value_1", "value_2"],
        "normalized_rows": [
            {"Metric": "Total current assets", "value_1": "100", "value_2": "90"},
            {"Metric": "Total current liabilities", "value_1": "80", "value_2": "70"},
        ],
    }

    frames, trace = build_evidence_frames([table])

    assert len(frames) == 4
    assert all(frame["statement_type"] == "balance_sheet" for frame in frames)
    assert all(frame["period"] is None for frame in frames)
    assert trace["evidence_frame_tables_accepted"] == 1
