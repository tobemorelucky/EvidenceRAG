from backend.row_relevance_selector import select_relevant_rows


def test_quick_ratio_synonyms_select_required_financial_rows():
    rows = [
        {"Metric": "Goodwill", "2024": "900"},
        {"Metric": "Cash and cash equivalents", "2024": "120"},
        {"Metric": "Accounts receivable, net", "2024": "80"},
        {"Metric": "Inventories", "2024": "30"},
        {"Metric": "Total current liabilities", "2024": "250"},
        {"Metric": "Long-term debt", "2024": "700"},
    ]

    selected, trace = select_relevant_rows(
        "What was the quick ratio in FY2024?",
        "Consolidated Balance Sheets",
        ["Metric", "2024"],
        rows,
        max_rows=4,
    )
    labels = {item["row"]["Metric"] for item in selected}

    assert "Cash and cash equivalents" in labels
    assert "Accounts receivable, net" in labels
    assert "Inventories" in labels
    assert "Total current liabilities" in labels
    assert "quick ratio" in trace["expanded_phrases"]
    assert trace["method"] == "bm25_lexical_finance_synonyms"


def test_lexical_bm25_selects_unlisted_metric_without_special_rule():
    rows = [
        {"Metric": "Revenue", "2024": "120"},
        {"Metric": "Research and development expense", "2024": "35"},
        {"Metric": "Interest expense", "2024": "8"},
    ]

    selected, _ = select_relevant_rows(
        "What was research and development expense in 2024?",
        "Operating expenses",
        ["Metric", "2024"],
        rows,
        max_rows=1,
    )

    assert selected[0]["row"]["Metric"] == "Research and development expense"


def test_inventory_turnover_expands_inventory_and_cost_rows():
    rows = [
        {"Metric": "Net sales", "2024": "500"},
        {"Metric": "Cost of goods sold", "2024": "300"},
        {"Metric": "Inventories", "2024": "75"},
    ]

    selected, trace = select_relevant_rows(
        "Calculate inventory turnover.",
        "Financial summary",
        ["Metric", "2024"],
        rows,
        max_rows=2,
    )

    assert {item["row"]["Metric"] for item in selected} == {"Cost of goods sold", "Inventories"}
    assert "inventory turnover" in trace["expanded_phrases"]
