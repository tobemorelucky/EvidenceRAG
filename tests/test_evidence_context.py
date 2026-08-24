from backend.evidence_context import build_compact_evidence


def test_compact_evidence_keeps_relevant_lookup_line_and_source(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    documents = [
        {
            "filename": "report.pdf",
            "page_number": 12,
            "text": "Unrelated introduction.\nNet revenue increased to $500 million in FY2022.\nUnrelated closing.",
        }
    ]

    evidence, meta = build_compact_evidence(
        "What was net revenue in FY2022?",
        documents,
        {"task_type": "lookup", "required_fields": ["revenue"], "required_periods": ["2022"]},
    )

    assert "Net revenue increased to $500 million" in evidence
    assert "Source: report.pdf | Page: 12" in evidence
    assert meta["answer_context_unit_count"] == 1


def test_compact_evidence_keeps_calculation_rows_and_drops_irrelevant_page(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    monkeypatch.setenv("RAG_ANSWER_MAX_EVIDENCE_UNITS", "2")
    documents = [
        {
            "filename": "report.pdf",
            "page_number": 4,
            "text": "At December 31, 2022 2021\nCash and cash equivalents 100 90\nTotal current liabilities 200 180",
        },
        {"filename": "report.pdf", "page_number": 80, "text": "Board biographies and governance information."},
    ]
    calculation = {
        "operands": {
            "cash_and_equivalents": {"value": "100"},
            "current_liabilities": {"value": "200"},
        }
    }

    evidence, meta = build_compact_evidence(
        "Calculate the FY2022 liquidity ratio.",
        documents,
        {
            "task_type": "calculation",
            "required_fields": ["cash_and_equivalents", "current_liabilities"],
            "required_periods": ["2022"],
        },
        calculation,
    )

    assert "Cash and cash equivalents 100 90" in evidence
    assert "Total current liabilities 200 180" in evidence
    assert "Board biographies" not in evidence
    assert meta["answer_context_pages"] == [{"filename": "report.pdf", "page_number": 4}]


def test_compact_evidence_respects_context_budget(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    monkeypatch.setenv("RAG_ANSWER_MAX_EVIDENCE_UNITS", "6")
    monkeypatch.setenv("RAG_ANSWER_MAX_CONTEXT_CHARS", "2000")
    monkeypatch.setenv("RAG_ANSWER_MAX_UNIT_CHARS", "800")
    documents = [
        {"filename": "report.pdf", "page_number": page, "text": ("Revenue 2022 100. " * 200)}
        for page in range(10)
    ]

    evidence, meta = build_compact_evidence(
        "What was revenue in 2022?",
        documents,
        {"task_type": "lookup", "required_fields": ["revenue"], "required_periods": ["2022"]},
    )

    assert len(evidence) <= 2000
    assert meta["answer_context_unit_count"] <= 6
    assert meta["answer_context_reduction_ratio"] > 0.8


def test_compact_evidence_merges_complementary_chunks_on_the_same_page(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    documents = [
        {"filename": "stores.pdf", "page_number": 16, "text": "Total Stores at End of Second Quarter"},
        {"filename": "stores.pdf", "page_number": 16, "text": "Total 982 969"},
    ]

    evidence, meta = build_compact_evidence(
        "How did total store count change?",
        documents,
        {"task_type": "comparison", "required_fields": ["store_count"], "required_periods": []},
    )

    assert "Total Stores at End of Second Quarter" in evidence
    assert "Total 982 969" in evidence
    assert meta["answer_context_unit_count"] == 1


def test_compact_evidence_keeps_total_row_after_fragmented_table_header(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    document = {
        "filename": "stores.pdf",
        "page_number": 16,
        "text": "\n".join(
            [
                "Fiscal 2024 Fiscal 2023",
                "Total Stores at Beginning of Second Quarter",
                "Stores Opened",
                "Stores Closed",
                "Total Stores at End of Second Quarter",
                "Best Buy 908 - (1) 907 931 1 (2) 930",
                "Outlet Centers 20 1 (1) 20 16 2 - 18",
                "Pacific Sales 20 - - 20 21 - - 21",
                "Yardbird 18 4 - 22 9 4 - 13",
                "Total 966 5 (2) 969 977 7 (2) 982",
            ]
        ),
    }

    evidence, _ = build_compact_evidence(
        "Did total store count change between Q2 FY2024 and FY2023?",
        [document],
        {
            "task_type": "comparison",
            "required_fields": ["store_count"],
            "required_periods": ["2024", "2023", "Q2"],
        },
    )

    assert "Total 966 5 (2) 969 977 7 (2) 982" in evidence


def test_compact_evidence_keeps_total_row_below_metric_header(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    document = {
        "filename": "annual-report.pdf",
        "page_number": 33,
        "text": "\n".join(
            [
                "Employees as of December 31, Capital Spending",
                "Property, Plant and Equipment - net as of December 31",
                "(Millions, except Employees) 2022 2021 2022 2021 2022 2021",
                "Americas 54,000 56,000 $ 1,321 $ 1,046 $ 6,066 $ 5,864",
                "Asia Pacific 18,000 18,000 182 216 1,389 1,582",
                "Europe, Middle East and Africa 20,000 21,000 246 341 1,723 1,983",
                "Total Company 92,000 95,000 $ 1,749 $ 1,603 $ 9,178 $ 9,429",
            ]
        ),
    }

    evidence, _ = build_compact_evidence(
        "Assess capital intensity from FY2022 data.",
        [document],
        {
            "task_type": "judgment",
            "required_fields": ["capital_expenditures", "net_ppe"],
            "required_periods": ["2022"],
        },
    )

    assert "Total Company 92,000 95,000 $ 1,749" in evidence


def test_compact_evidence_filters_other_companies_when_target_company_matches(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    documents = [
        {"filename": "ADOBE_2022_10K.pdf", "page_number": 10, "text": "Adobe revenue 100"},
        {"filename": "AES_2022_10K.pdf", "page_number": 10, "text": "AES revenue 200"},
    ]

    evidence, meta = build_compact_evidence(
        "What was Adobe revenue?",
        documents,
        {"company": "adobe", "task_type": "lookup", "required_fields": ["revenue"]},
    )

    assert "Adobe revenue 100" in evidence
    assert "AES revenue 200" not in evidence
    assert meta["answer_context_company_filtered_count"] == 1


def test_filename_company_identity_beats_body_ticker_like_number(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    documents = [
        {"filename": "3M_2022_10K.pdf", "page_number": 10, "text": "3M operating margin was 19.1%."},
        {"filename": "AES_2022_10K.pdf", "page_number": 20, "text": "AES incurred $3 million of costs."},
    ]

    evidence, _ = build_compact_evidence(
        "What drove 3M operating margin?",
        documents,
        {"company": "3m", "task_type": "calculation", "required_fields": [], "required_periods": ["2022"]},
    )

    assert "3M operating margin" in evidence
    assert "AES incurred" not in evidence


def test_compact_evidence_preserves_split_table_header_for_requested_period(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    document = {
        "filename": "ADOBE_2016_10K.pdf",
        "page_number": 60,
        "text": "\n".join(
            [
                "ADOBE SYSTEMS INCORPORATED",
                "CONSOLIDATED BALANCE SHEETS",
                "(In thousands, except par value)",
                "December 2,",
                "2016",
                "November 27,",
                "2015",
                "ASSETS",
                "Current assets:",
                "Cash and cash equivalents 1,011,315 876,560",
                "LIABILITIES AND STOCKHOLDERS' EQUITY",
                "Current liabilities:",
                "Total current liabilities 2,811,635 2,213,556",
            ]
        ),
    }

    evidence, meta = build_compact_evidence(
        "What is Adobe's FY2015 current ratio?",
        [document],
        {
            "task_type": "calculation",
            "required_fields": ["cash_and_equivalents", "current_liabilities"],
            "required_periods": ["2015"],
        },
    )

    assert "(In thousands, except par value)" in evidence
    assert "December 2," in evidence and "2016" in evidence
    assert "November 27," in evidence and "2015" in evidence
    assert meta["answer_context_missing_required_fields"] == []


def test_compact_evidence_reserves_pages_covering_each_required_field(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    monkeypatch.setenv("RAG_ANSWER_MAX_EVIDENCE_UNITS", "3")
    monkeypatch.setenv("RAG_ANSWER_RANK_RESERVED_UNITS", "2")
    documents = [
        {"filename": "report.pdf", "page_number": 1, "text": "Revenue discussion and outlook 2022."},
        {"filename": "report.pdf", "page_number": 2, "text": "Revenue increased in 2022."},
        {"filename": "report.pdf", "page_number": 3, "text": "Net revenue 500 450"},
        {"filename": "report.pdf", "page_number": 4, "text": "Net income 50 40"},
    ]

    evidence, meta = build_compact_evidence(
        "Calculate FY2022 return using net income and revenue.",
        documents,
        {
            "task_type": "calculation",
            "required_fields": ["net_income", "revenue"],
            "required_periods": ["2022"],
        },
    )

    assert "Net revenue 500 450" in evidence
    assert "Net income 50 40" in evidence
    assert {page["page_number"] for page in meta["answer_context_pages"]} >= {3, 4}
    assert meta["answer_context_missing_required_fields"] == []


def test_compact_evidence_matches_acquisition_word_forms_after_period_header(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    monkeypatch.setenv("RAG_ANSWER_MAX_UNIT_CHARS", "800")
    document = {
        "filename": "report.pdf",
        "page_number": 43,
        "text": "\n".join(
            [
                "Fiscal 2023 fiscal 2022 fiscal 2021",
                *(f"Unrelated fiscal year disclosure {year}." for year in range(2000, 2010)),
                "In fiscal 2022, we acquired Current Health and Yardbird.",
                "Note 2, Acquisitions",
            ]
        ),
    }

    evidence, _ = build_compact_evidence(
        "What are the major acquisitions in FY2023, FY2022 and FY2021?",
        [document],
        {"task_type": "lookup", "required_fields": [], "required_periods": ["2023", "2022", "2021"]},
    )

    assert "acquired Current Health and Yardbird" in evidence


def test_compact_evidence_keeps_numeric_performance_rows(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    document = {
        "filename": "quarterly.pdf",
        "page_number": 17,
        "text": "\n".join(
            [
                "Domestic category results",
                "Consumer Electronics: 5.7% comparable sales decline",
                "Appliances: 16.1% comparable sales decline",
                "Entertainment: 9.0% comparable sales growth",
                "Services: 7.6% comparable sales growth",
            ]
        ),
    }

    evidence, _ = build_compact_evidence(
        "Which category performed best by top line?",
        [document],
        {"task_type": "lookup", "required_fields": [], "required_periods": []},
    )

    assert "Entertainment: 9.0% comparable sales growth" in evidence


def test_enumeration_context_limits_rank_reservation_for_relevant_pages(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    monkeypatch.setenv("RAG_ANSWER_MAX_EVIDENCE_UNITS", "4")
    monkeypatch.setenv("RAG_ANSWER_RANK_RESERVED_UNITS", "3")
    documents = [
        {"filename": "report.pdf", "page_number": 1, "text": "Glossary: Alpha means a product name."},
        {"filename": "report.pdf", "page_number": 2, "text": "Definitions: Beta means a product name."},
        {"filename": "report.pdf", "page_number": 3, "text": "Index and table of contents."},
        {"filename": "report.pdf", "page_number": 4, "text": "Note 2. Acquisitions. We acquired Alpha, Beta, and Gamma during the year."},
        {"filename": "report.pdf", "page_number": 5, "text": "Other unrelated disclosures."},
    ]

    evidence, meta = build_compact_evidence(
        "What are the main companies acquired during the year?",
        documents,
        {"task_type": "lookup", "required_fields": [], "required_periods": []},
    )

    assert "We acquired Alpha, Beta, and Gamma" in evidence
    assert meta["answer_context_rank_reserved_units"] == 3


def test_acquisition_action_outweighs_repeated_company_boilerplate(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    monkeypatch.setenv("RAG_ANSWER_MAX_UNIT_CHARS", "900")
    document = {
        "filename": "report.pdf",
        "page_number": 20,
        "text": "\n".join(
            ["Example Company filing disclosure and accounting policy." for _ in range(30)]
            + ["Note 2. Acquisitions", "We acquired Alpha and Beta during the year."]
        ),
    }

    evidence, _ = build_compact_evidence(
        "Which companies were acquired by Example Company?",
        [document],
        {"task_type": "lookup", "required_fields": [], "required_periods": []},
    )

    assert "We acquired Alpha and Beta" in evidence


def test_domestic_question_drops_explicit_international_page(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED", "true")
    documents = [
        {"filename": "report.pdf", "page_number": 1, "text": "International segment revenue categories\nEntertainment 2.5%"},
        {"filename": "report.pdf", "page_number": 2, "text": "Domestic comparable sales\nEntertainment 9.0% growth"},
    ]

    evidence, meta = build_compact_evidence(
        "Which category performed best in the domestic USA market?",
        documents,
        {"task_type": "selection", "required_fields": [], "required_periods": []},
    )

    assert "Entertainment 9.0% growth" in evidence
    assert "Entertainment 2.5%" not in evidence
    assert meta["answer_context_scope_filtered_count"] == 1
