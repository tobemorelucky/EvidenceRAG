from backend.financial_query_expander import expand_financial_bm25_query, explain_financial_query_expansion


def test_expands_abbreviation_and_metric_operands():
    result = explain_financial_query_expansion("What was PP&E and inventory turnover in FY2022?")
    assert "property plant and equipment" in result["expanded_query"]
    assert "cost of goods sold" in result["expanded_query"]
    assert "average inventory" in result["expanded_query"]
    assert {item["source"] for item in result["additions"]} >= {
        "accounting_abbreviation", "metric_synonym", "formula_operand", "fiscal_year_normalization",
    }


def test_normalizes_short_fiscal_quarter_without_changing_original_prefix():
    query = "Compare revenue in Q2 of FY'23."
    expanded = expand_financial_bm25_query(query)
    assert expanded.startswith(query)
    assert "fiscal year 2023 fiscal quarter 2 Q2 2023" in expanded


def test_unrelated_query_is_unchanged():
    query = "Who signed the annual report?"
    assert expand_financial_bm25_query(query) == query


def test_expansion_has_no_benchmark_or_answer_inputs():
    result = explain_financial_query_expansion("Calculate the current ratio in FY2022")
    assert set(result) == {"original_query", "expanded_query", "additions", "changed"}
    assert "financebench" not in result["expanded_query"].casefold()


def test_evaluator_source_keeps_dense_query_shared_and_has_no_answer_generation():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "scripts/evaluate_financial_bm25_shadow_v1.py").read_text(encoding="utf-8")
    assert "get_embeddings([question])" in source
    assert "generate_answer" not in source
    assert "judge_answer" not in source

