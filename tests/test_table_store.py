import os

os.environ["DATABASE_URL"] = "sqlite://"

import backend.table_store as table_store_module
from database import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


def _make_store(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(table_store_module, "SessionLocal", TestingSessionLocal)
    return table_store_module.TableStore()


def test_table_store_upsert_and_get_by_ids(monkeypatch):
    store = _make_store(monkeypatch)

    count = store.upsert_tables(
        [
            {
                "table_id": "t1",
                "filename": "demo.pdf",
                "doc_name": "demo",
                "page_number": 3,
                "table_index": 1,
                "title": "Revenue",
                "columns": ["Year", "Value"],
                "rows": [["2024", "100"]],
                "csv_text": "Year,Value\n2024,100",
            }
        ]
    )

    assert count == 1
    rows = store.get_tables_by_ids(["t1"])
    assert len(rows) == 1
    assert rows[0]["table_id"] == "t1"
    assert rows[0]["filename"] == "demo.pdf"
    assert rows[0]["title"] == "Revenue"


def test_table_store_upsert_updates_existing_record(monkeypatch):
    store = _make_store(monkeypatch)

    store.upsert_tables([{"table_id": "t1", "filename": "demo.pdf", "title": "Old"}])
    store.upsert_tables(
        [
            {
                "table_id": "t1",
                "filename": "demo.pdf",
                "title": "New",
                "columns": ["A"],
                "rows": [["1"]],
            }
        ]
    )

    rows = store.get_tables_by_ids(["t1"])
    assert len(rows) == 1
    assert rows[0]["title"] == "New"
    assert rows[0]["columns"] == ["A"]


def test_table_store_prefers_normalized_financial_structure(monkeypatch):
    store = _make_store(monkeypatch)

    store.upsert_tables(
        [
            {
                "table_id": "normalized-1",
                "filename": "report.pdf",
                "title": "Raw title",
                "columns": ["raw_a", "raw_b"],
                "rows": [{"raw_a": "Cash", "raw_b": "100"}],
                "normalized_title": "Consolidated Balance Sheets",
                "normalized_unit": "USD millions",
                "normalized_columns": ["Metric", "2024"],
                "normalized_rows": [{"Metric": "Cash", "2024": "100"}],
            }
        ]
    )

    table = store.get_tables_by_ids(["normalized-1"])[0]
    assert table["title"] == "Consolidated Balance Sheets"
    assert table["before_context"] == "USD millions"
    assert table["columns"] == ["Metric", "2024"]
    assert table["rows"] == [{"Metric": "Cash", "2024": "100"}]


def test_table_store_persists_evidence_assembly_identity_and_quality(monkeypatch):
    store = _make_store(monkeypatch)
    document_id = "doc_123"
    page_id = f"{document_id}:page:000003"

    store.upsert_tables([
        {
            "table_id": f"{page_id}:table:0001",
            "document_id": document_id,
            "page_id": page_id,
            "filename": "report.pdf",
            "page_number": 3,
            "start_page": 3,
            "end_page": 4,
            "table_index": 1,
            "parser_backend": "pdfplumber_words",
            "quality_score": 0.88,
            "columns": ["Metric", "2024"],
            "rows": [{"Metric": "Revenue", "2024": "100"}],
            "unit": "USD",
            "scale": "millions",
        }
    ])

    table = store.get_tables_by_page_ids([page_id])[0]
    assert table["document_id"] == document_id
    assert table["page_id"] == page_id
    assert table["start_page"] == 3
    assert table["end_page"] == 4
    assert table["parser_backend"] == "pdfplumber_words"
    assert table["quality_score"] == 0.88
    assert table["unit"] == "USD"
    assert table["scale"] == "millions"


def test_table_store_get_by_filename_orders_by_page_and_index(monkeypatch):
    store = _make_store(monkeypatch)

    store.upsert_tables(
        [
            {"table_id": "t3", "filename": "demo.pdf", "page_number": 2, "table_index": 2},
            {"table_id": "t1", "filename": "demo.pdf", "page_number": 1, "table_index": 2},
            {"table_id": "t2", "filename": "demo.pdf", "page_number": 1, "table_index": 1},
        ]
    )

    rows = store.get_tables_by_filename("demo.pdf")
    assert [row["table_id"] for row in rows] == ["t2", "t1", "t3"]


def test_table_store_delete_by_filename(monkeypatch):
    store = _make_store(monkeypatch)

    store.upsert_tables(
        [
            {"table_id": "t1", "filename": "demo.pdf"},
            {"table_id": "t2", "filename": "demo.pdf"},
            {"table_id": "t3", "filename": "other.pdf"},
        ]
    )

    deleted = store.delete_by_filename("demo.pdf")

    assert deleted == 2
    assert store.get_tables_by_filename("demo.pdf") == []
    assert [row["table_id"] for row in store.get_tables_by_filename("other.pdf")] == ["t3"]


def test_table_store_ignores_invalid_records(monkeypatch):
    store = _make_store(monkeypatch)

    count = store.upsert_tables(
        [
            {"table_id": "", "filename": "demo.pdf"},
            {"table_id": "t1", "filename": ""},
            {"table_id": "t2", "filename": "demo.pdf", "columns": "bad", "rows": "bad"},
        ]
    )

    rows = store.get_tables_by_filename("demo.pdf")
    assert count == 1
    assert len(rows) == 1
    assert rows[0]["table_id"] == "t2"
    assert rows[0]["columns"] == []
    assert rows[0]["rows"] == []
