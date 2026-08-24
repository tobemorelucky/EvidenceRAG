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
