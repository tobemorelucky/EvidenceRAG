from copy import deepcopy

from backend.table_evidence_expander_v1 import expand_table_evidence_v1


class FakeStore:
    def __init__(self, tables):
        self.tables = tables

    def get_tables_by_ids(self, ids):
        return [table for table in self.tables if table["table_id"] in ids]

    def get_tables_by_page_ids(self, ids):
        return [table for table in self.tables if table["page_id"] in ids]


def chunk(index=1, *, table_id="", text="chunk text"):
    return {
        "chunk_id": f"c{index}", "document_id": "doc", "page_id": "page-1",
        "filename": "sample.pdf", "page_number": 1, "text": text, "rank": index,
        "metadata": {"table_id": table_id},
    }


def table(**updates):
    value = {
        "table_id": "t1", "document_id": "doc", "page_id": "page-1",
        "filename": "sample.pdf", "page_number": 1, "title": "Income Statement",
        "columns": ["Metric", "2023", "2022"],
        "rows": [{"Metric": "Revenue", "2023": "120", "2022": "100"}],
        "unit": "USD millions", "scale": "millions",
    }
    value.update(updates)
    return value


def test_direct_table_id_expands_exact_table():
    units, context, trace = expand_table_evidence_v1(
        [chunk(table_id="t1")], table_store=FakeStore([table()]), max_context_chars=1000,
    )
    expanded = [unit for unit in units if unit["source_type"] == "table"]
    assert [unit["table_id"] for unit in expanded] == ["t1"]
    assert "Header: Metric | 2023 | 2022" in context
    assert "Unit: USD millions" in context
    assert trace["expanded_table_count"] == 1


def test_missing_table_id_keeps_original_in_direct_mode():
    original = "original frozen context"
    units, context, trace = expand_table_evidence_v1(
        [chunk()], table_store=FakeStore([table()]), original_context=original,
    )
    assert context == original
    assert all(unit["source_type"] == "chunk" for unit in units)
    assert trace["expanded_table_count"] == 0


def test_page_fallback_requires_same_document_and_page():
    wrong = table(table_id="wrong", document_id="other")
    right = table()
    units, _, trace = expand_table_evidence_v1(
        [chunk()], table_store=FakeStore([wrong, right]), mode="page_table_fallback", max_context_chars=1000,
    )
    assert [unit["table_id"] for unit in units if unit["source_type"] == "table"] == ["t1"]
    assert trace["association_mismatch_count"] == 1


def test_oversized_first_row_keeps_a_row_fragment():
    units, _, _ = expand_table_evidence_v1(
        [chunk(table_id="t1")],
        table_store=FakeStore([table(rows=[{"Metric": "Revenue", "2023": "1" * 1000}])]),
        max_table_chars=220,
        max_context_chars=1000,
    )
    expanded = next(unit for unit in units if unit["source_type"] == "table")
    assert expanded["rows"]
    assert len(expanded["text"]) <= 220


def test_budget_removes_low_rank_non_anchor_and_does_not_mutate_input():
    chunks = [chunk(1, table_id="t1", text="anchor " * 40), chunk(9, text="low rank " * 80)]
    original = deepcopy(chunks)
    units, context, trace = expand_table_evidence_v1(
        chunks,
        table_store=FakeStore([table(rows=[
            {"Metric": "Revenue", "2023": "120", "2022": "100"},
            {"Metric": "Cost", "2023": "80", "2022": "70"},
        ])]),
        max_context_chars=700,
        max_table_chars=350,
    )
    assert len(context) <= 700
    assert chunks == original
    assert any(item["unit_id"] == "c9" for item in trace["removed_units"])
    assert "anchor" in context
    assert "Table ID: t1" in context
    assert "Header: Metric | 2023 | 2022" in context
    assert any(unit["source_type"] == "table" for unit in units)
