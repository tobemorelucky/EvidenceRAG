from copy import deepcopy

from backend.evidence_fact_store_v1 import build_fact_index, fact_text, facts_from_table


def _table():
    return {
        "table_id": "table-1",
        "document_id": "doc-1",
        "page_id": "doc-1:page:000010",
        "filename": "ACME_2023_10K.pdf",
        "page_number": 10,
        "entity": "ACME",
        "title": "Consolidated Statements of Income",
        "columns": ["Metric", "2023", "2022"],
        "rows": [
            {"Metric": "Year ended", "2023": "2023", "2022": "2022", "_raw_line": "Year ended 2023 2022"},
            {"Metric": "Revenue", "2023": "$1,250.0", "2022": "$1,100.0", "_raw_line": "Revenue $1,250.0 $1,100.0"},
            {"Metric": "Operating income", "2023": "(125)", "2022": "100", "_raw_line": "Operating income (125) 100"},
        ],
        "unit": "USD",
        "scale": "millions",
        "quality_score": 0.9,
    }


def test_clear_table_emits_period_aligned_facts_without_mutation():
    table = _table()
    original = deepcopy(table)

    facts, trace = facts_from_table(table)

    assert table == original
    assert trace["eligible"] is True
    assert [(fact.metric, fact.period, fact.value) for fact in facts] == [
        ("Revenue", "2023", "1250"),
        ("Revenue", "2022", "1100"),
        ("Operating income", "2023", "-125"),
        ("Operating income", "2022", "100"),
    ]
    assert all(fact.unit == "USD millions" for fact in facts)
    assert all(fact.entity == "ACME" for fact in facts)
    assert "Metric: Revenue" in fact_text(facts[0].to_dict())


def test_table_without_explicit_period_is_rejected():
    table = _table()
    table["columns"] = ["Metric", "Current", "Prior"]
    table["rows"][0]["_raw_line"] = "Current period Prior period"

    facts, trace = facts_from_table(table)

    assert facts == []
    assert trace["eligible"] is False
    assert trace["reason"] == "explicit_period_missing"


def test_table_without_aligned_numeric_rows_is_rejected():
    table = _table()
    table["rows"][1]["_raw_line"] = "Revenue not available"
    table["rows"][2]["_raw_line"] = "Operating income not available"

    facts, trace = facts_from_table(table)

    assert facts == []
    assert trace["eligible"] is False
    assert trace["reason"] == "no_aligned_numeric_rows"


def test_build_fact_index_reports_skips_and_is_deterministic():
    valid = _table()
    invalid = deepcopy(valid)
    invalid["table_id"] = "table-2"
    invalid["page_id"] = ""

    first, stats = build_fact_index([valid, invalid])
    second, _ = build_fact_index([valid, invalid])

    assert first == second
    assert stats["tables_seen"] == 2
    assert stats["tables_indexed"] == 1
    assert stats["facts"] == 4
    assert stats["table_reason_counts"]["missing_identity"] == 1
