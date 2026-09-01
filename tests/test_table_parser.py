from backend.table_parser import TableAwareParser


class _FakePage:
    def __init__(self, tables):
        self._tables = tables

    def extract_tables(self):
        return self._tables


class _FakePdf:
    def __init__(self, pages):
        self.pages = pages

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakePdfPlumber:
    def __init__(self, pages):
        self._pages = pages

    def open(self, _file_path):
        return _FakePdf(self._pages)


def test_extract_tables_returns_empty_when_disabled(monkeypatch):
    monkeypatch.setenv("TABLE_AWARE_INGESTION", "false")

    parser = TableAwareParser()

    assert parser.extract_tables("demo.pdf", "demo.pdf") == []


def test_extract_tables_returns_empty_when_dependencies_unavailable(monkeypatch):
    monkeypatch.setenv("TABLE_AWARE_INGESTION", "true")
    monkeypatch.setenv("TABLE_PARSER_BACKEND", "auto")
    monkeypatch.setattr(TableAwareParser, "_load_docling", staticmethod(lambda: None))
    monkeypatch.setattr(TableAwareParser, "_load_pdfplumber", staticmethod(lambda: None))

    parser = TableAwareParser()

    assert parser.extract_tables("demo.pdf", "demo.pdf") == []


def test_extract_tables_uses_pdfplumber_structure(monkeypatch):
    monkeypatch.setenv("TABLE_AWARE_INGESTION", "true")
    monkeypatch.setenv("TABLE_PARSER_BACKEND", "pdfplumber")
    monkeypatch.setattr(TableAwareParser, "_load_docling", staticmethod(lambda: None))
    monkeypatch.setattr(
        TableAwareParser,
        "_load_pdfplumber",
        staticmethod(
            lambda: _FakePdfPlumber(
                [
                    _FakePage(
                        [
                            [
                                ["Metric", "FY2022", "FY2021"],
                                ["Revenue", "100", "90"],
                                ["Operating margin", "22%", "20%"],
                            ]
                        ]
                    )
                ]
            )
        ),
    )

    parser = TableAwareParser()
    tables = parser.extract_tables("demo.pdf", "demo.pdf")

    assert len(tables) == 1
    assert tables[0]["page_number"] == 0
    assert tables[0]["page_id"].endswith(":page:000000")
    assert tables[0]["table_index"] == 1
    assert tables[0]["columns"] == ["Metric", "FY2022", "FY2021"]
    assert tables[0]["parser_backend"] == "pdfplumber"
    assert tables[0]["rows"] == [
        {"Metric": "Revenue", "FY2022": "100", "FY2021": "90"},
        {"Metric": "Operating margin", "FY2022": "22%", "FY2021": "20%"},
    ]
    assert "Revenue" in tables[0]["csv_text"]


def test_extract_tables_pdfplumber_backend_does_not_call_docling(monkeypatch):
    monkeypatch.setenv("TABLE_AWARE_INGESTION", "true")
    monkeypatch.setenv("TABLE_PARSER_BACKEND", "pdfplumber")
    monkeypatch.setattr(
        TableAwareParser,
        "_load_docling",
        staticmethod(lambda: (_ for _ in ()).throw(AssertionError("docling should not be called"))),
    )
    monkeypatch.setattr(
        TableAwareParser,
        "_load_pdfplumber",
        staticmethod(lambda: _FakePdfPlumber([_FakePage([[]])])),
    )

    parser = TableAwareParser()

    assert parser.extract_tables("demo.pdf", "demo.pdf") == []


def test_extract_tables_docling_backend_returns_empty_when_docling_unavailable(monkeypatch):
    monkeypatch.setenv("TABLE_AWARE_INGESTION", "true")
    monkeypatch.setenv("TABLE_PARSER_BACKEND", "docling")
    monkeypatch.setattr(TableAwareParser, "_load_docling", staticmethod(lambda: None))
    monkeypatch.setattr(
        TableAwareParser,
        "_load_pdfplumber",
        staticmethod(lambda: (_ for _ in ()).throw(AssertionError("pdfplumber should not be called"))),
    )

    parser = TableAwareParser()

    assert parser.extract_tables("demo.pdf", "demo.pdf") == []


def test_extract_tables_pdfplumber_words_backend(monkeypatch):
    monkeypatch.setenv("TABLE_AWARE_INGESTION", "true")
    monkeypatch.setenv("TABLE_PARSER_BACKEND", "pdfplumber_words")
    monkeypatch.setattr(
        TableAwareParser,
        "_extract_with_pdfplumber_words_debug",
        lambda self, file_path, filename, max_pages, include_rejected: [
            {
                "table_id": "demo.pdf::table::p1::1",
                "filename": filename,
                "page_number": 1,
                "table_index": 1,
                "columns": ["Metric", "FY2022"],
                "rows": [{"Metric": "Revenue", "FY2022": "100"}],
                "csv_text": "Metric,FY2022\nRevenue,100",
                "html": "<table></table>",
                "parser_backend": "pdfplumber_words",
            }
        ],
    )
    monkeypatch.setattr(
        TableAwareParser,
        "_load_docling",
        staticmethod(lambda: (_ for _ in ()).throw(AssertionError("docling should not be called"))),
    )
    monkeypatch.setattr(
        TableAwareParser,
        "_load_pdfplumber",
        staticmethod(lambda: (_ for _ in ()).throw(AssertionError("pdfplumber matrix parser should not be called"))),
    )

    parser = TableAwareParser()
    tables = parser.extract_tables("demo.pdf", "demo.pdf")

    assert len(tables) == 1
    assert tables[0]["parser_backend"] == "pdfplumber_words"


def test_extract_tables_pdfplumber_words_include_rejected_flag(monkeypatch):
    monkeypatch.setenv("TABLE_AWARE_INGESTION", "true")
    monkeypatch.setenv("TABLE_PARSER_BACKEND", "pdfplumber_words")
    monkeypatch.setattr(
        TableAwareParser,
        "_extract_with_pdfplumber_words_debug",
        lambda self, file_path, filename, max_pages, include_rejected: [
            {"table_id": "accepted", "parser_backend": "pdfplumber_words", "accepted": True},
            {"table_id": "rejected", "parser_backend": "pdfplumber_words", "accepted": False},
        ]
        if include_rejected
        else [{"table_id": "accepted", "parser_backend": "pdfplumber_words", "accepted": True}],
    )

    parser = TableAwareParser()

    accepted_only = parser.extract_tables("demo.pdf", "demo.pdf")
    with_rejected = parser.extract_tables("demo.pdf", "demo.pdf", include_rejected=True)

    assert len(accepted_only) == 1
    assert len(with_rejected) == 2
    assert with_rejected[0]["table_id"].endswith(":table:0001")
    assert with_rejected[1]["table_id"].endswith(":table:0002")
