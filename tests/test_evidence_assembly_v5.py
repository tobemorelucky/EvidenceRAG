from backend.evidence_assembly_v5 import EvidenceUnit, assemble_evidence_v5, build_evidence_units
from scripts.evaluate_evidence_assembly_v5 import summarize


def _page():
    return {
        "document_id": "doc_generic",
        "page_id": "doc_generic:page:000005",
        "filename": "generic.pdf",
        "page_number": 5,
        "company": "Generic Corp",
    }


def _chunks():
    return [
        {
            "filename": "generic.pdf",
            "page_number": 5,
            "chunk_id": "c1",
            "merged_rank": 1,
            "text": "Revenue was 120 in 2024 and 100 in 2023.",
        },
        {
            "filename": "generic.pdf",
            "page_number": 5,
            "chunk_id": "c2",
            "merged_rank": 2,
            "text": "Operating income increased during the period.",
        },
    ]


def _table():
    return {
        **_page(),
        "table_id": "table_1",
        "title": "Results",
        "columns": ["Metric", "2024", "2023"],
        "rows": [
            {"Metric": "Revenue", "2024": "120", "2023": "100"},
            {"Metric": "Employees", "2024": "40", "2023": "35"},
        ],
        "unit": "USD",
        "scale": "millions",
        "quality_score": 0.9,
    }


def test_evidence_unit_exposes_required_fields():
    unit = EvidenceUnit(
        document_id="doc", page_id="page", source_type="text", entity="Entity",
        period=["2024"], metric=None, value=["120"], unit=None,
        source_text="Revenue was 120.",
    )

    assert set(unit.to_dict()) >= {
        "document_id", "page_id", "source_type", "entity", "period",
        "metric", "value", "unit", "source_text",
    }


def test_builder_keeps_raw_chunks_and_adds_only_lexically_relevant_table_rows():
    units = build_evidence_units(
        "What was revenue in 2024?", _chunks(), pages=[_page()], tables=[_table()],
    )

    text_units = [unit for unit in units if unit.source_type == "text"]
    table_units = [unit for unit in units if unit.source_type == "table"]
    assert [unit.source_text for unit in text_units] == [item["text"] for item in _chunks()]
    assert len(table_units) == 1
    assert table_units[0].metric == "Revenue"
    assert "Header: Metric | 2024 | 2023" in table_units[0].source_text
    assert table_units[0].unit == "USD millions"


def test_assembler_respects_budget_and_returns_text_and_table_trace():
    context, units, trace = assemble_evidence_v5(
        "What was revenue in 2024?", _chunks(), pages=[_page()], tables=[_table()],
        max_context_chars=1500,
    )

    assert len(context) <= 1500
    assert {unit["source_type"] for unit in units} == {"text", "table"}
    assert trace["selected_text_unit_count"] == 2
    assert trace["selected_table_unit_count"] == 1
    assert "Revenue was 120" in context
    assert "Row: Revenue | 120 | 100" in context


def test_no_relevant_table_row_leaves_package_as_text_only():
    context, units, trace = assemble_evidence_v5(
        "Describe employee retention policy", _chunks(), pages=[_page()], tables=[_table()],
        max_context_chars=1500,
    )

    assert all(unit["source_type"] == "text" for unit in units)
    assert trace["selected_table_unit_count"] == 0
    assert "Revenue was 120" in context


def test_summary_reports_null_strict_judge_without_external_calls():
    metrics = {
        "answer_evidence_coverage": {"ratio": 0.5},
        "required_number_hit": True,
        "required_period_hit": True,
        "gold_page_hit": True,
        "all_gold_pages_hit": True,
        "context_chars": 100,
        "block_count": 2,
    }
    trace = {"selected_text_unit_count": 1, "selected_table_unit_count": 1}
    record = {
        "financebench_id": "generic_1",
        "group": "correct_regression10",
        "routes": {
            "current_chunk_retrieval": {"metrics": metrics},
            "evidence_assembly_v5": {"metrics": metrics, "trace": trace},
        },
    }

    result = summarize([record])

    assert result["current_chunk_retrieval"]["strict_judge"] is None
    assert result["evidence_assembly_v5"]["strict_judge"] is None
    assert result["external_calls"]["strict_judge"] == 0
