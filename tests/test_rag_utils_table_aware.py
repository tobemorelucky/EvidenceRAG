import importlib.util
import sys
import types
from pathlib import Path


def _install_rag_utils_stubs():
    requests = types.ModuleType("requests")
    requests.post = lambda *args, **kwargs: None
    sys.modules["requests"] = requests

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv

    langchain_chat_models = types.ModuleType("langchain.chat_models")
    langchain_chat_models.init_chat_model = lambda *args, **kwargs: None
    sys.modules["langchain.chat_models"] = langchain_chat_models

    document_page_store = types.ModuleType("document_page_store")
    document_page_store.DocumentPageStore = type("DocumentPageStore", (), {})
    sys.modules["document_page_store"] = document_page_store

    parent_chunk_store = types.ModuleType("parent_chunk_store")
    parent_chunk_store.ParentChunkStore = type("ParentChunkStore", (), {})
    sys.modules["parent_chunk_store"] = parent_chunk_store

    embedding = types.ModuleType("embedding")

    class _FakeEmbeddingService:
        def get_all_embeddings(self, texts):
            return [[0.1, 0.2] for _ in texts], [{0: 1.0} for _ in texts]

    embedding.embedding_service = _FakeEmbeddingService()
    sys.modules["embedding"] = embedding

    finance = types.ModuleType("finance_rag_features")
    finance.COMPANY_ALIASES = {}
    finance.FINANCE_METRIC_HINTS = set()
    finance.extract_keyword_tokens = lambda text: set()
    finance.extract_metric_hints = lambda text: set()
    finance.extract_numbers = lambda text: set()
    finance.extract_years = lambda text: set()
    finance.infer_doc_type = lambda text: ""
    finance.normalize_doc_name = lambda text: text
    finance.parse_finance_query = lambda query: {"company": "", "years": [], "metrics": []}
    sys.modules["finance_rag_features"] = finance

    query_parser = types.ModuleType("query_parser")
    query_parser.FIELD_ALIASES = {}
    query_parser.FIELD_STATEMENT_TYPES = {}
    query_parser.company_aliases_for = lambda company: []
    query_parser.matches_company_text = lambda *args, **kwargs: False
    query_parser.infer_task_spec = lambda query: {
        "task_type": "lookup",
        "formula": "",
        "required_fields": [],
    }
    query_parser.build_required_field_query = lambda query: ""
    query_parser.build_finance_query_rewrite = lambda query: ""
    query_parser.build_supplemental_field_query = lambda query, coverage: ""
    query_parser.infer_statement_types = lambda query, required_fields=None: []
    query_parser.infer_page_statement_types = lambda text: []
    query_parser.match_required_fields_in_text = lambda required_fields, text: {}
    query_parser.assess_required_field_coverage = lambda *args, **kwargs: {
        "task_type": "lookup",
        "formula": "",
        "required_fields": [],
        "matched_fields": {},
        "missing_fields": [],
        "status": "complete",
        "supplemental_search_attempted": False,
    }
    sys.modules["query_parser"] = query_parser

    query_planner = types.ModuleType("query_planner")
    query_planner.plan_retrieval_queries = lambda question: {
        "enabled": False,
        "intent": "",
        "must_keep_terms": [],
        "semantic_queries": [],
        "evidence_field_queries": [],
        "table_heading_queries": [],
        "keyword_queries": [],
        "planner_validation_dropped_queries": [],
        "expected_evidence_type": "",
        "constraints": [],
        "parse_error": "",
    }
    sys.modules["query_planner"] = query_planner

    milvus_client = types.ModuleType("milvus_client")
    milvus_client.MilvusManager = type("MilvusManager", (), {})
    sys.modules["milvus_client"] = milvus_client

    table_store = types.ModuleType("table_store")
    table_store.TableStore = type("TableStore", (), {})
    sys.modules["table_store"] = table_store

    path = Path(__file__).resolve().parents[1] / "backend" / "rag_utils.py"
    spec = importlib.util.spec_from_file_location("rag_utils_table_aware_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_table_aware_retrieval_off_does_not_call_evidence_search(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("should not call table evidence retrieval when off")

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "off")

    doc, meta = module._build_table_context_doc("net sales")

    assert doc is None
    assert meta["table_aware_retrieval_mode"] == "off"
    assert meta["table_evidence_hit_count"] == 0


def test_extract_query_anchors_keeps_entities_and_drops_generic_finance_words():
    module = _install_rag_utils_stubs()

    anchors = module.extract_query_anchors("What was Adobe operating margin in fiscal year 2022?")

    assert "Adobe" in anchors
    assert "margin" not in [anchor.lower() for anchor in anchors]
    assert "fiscal" not in [anchor.lower() for anchor in anchors]


def test_retrieve_candidate_documents_uses_text_chunk_filter(monkeypatch):
    module = _install_rag_utils_stubs()

    captured = {}

    def _fake_retrieve_leaf_chunks(query, top_k, filter_expr, retrieval_scope):
        captured["filter_expr"] = filter_expr
        return {"docs": [], "meta": {"retrieval_mode": "global"}}

    monkeypatch.setattr(module, "_retrieve_leaf_chunks", _fake_retrieve_leaf_chunks)

    module.retrieve_candidate_documents("net sales", candidate_k=5)

    assert captured["filter_expr"] == '(chunk_level == 3) and (evidence_type == "text_chunk" or evidence_type == "")'


def test_table_aware_retrieval_auto_query_metric_or_number_triggers(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc("What was Amcor's net sales in fiscal year 2023?")

    assert doc is None
    assert meta["table_aware_retrieval_mode"] == "auto"
    assert meta["table_aware_auto_triggered"] is True
    assert "query_metric_or_number" in meta["table_aware_trigger_reason"]


def test_table_aware_retrieval_auto_query_contact_or_number_triggers(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc("What is the US and Canada conference call number?")

    assert doc is None
    assert meta["table_aware_auto_triggered"] is True
    assert "query_contact_or_number" in meta["table_aware_trigger_reason"]


def test_table_aware_retrieval_auto_summary_query_does_not_trigger(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("should not call table evidence retrieval when auto does not trigger")

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc("Summarize this document.")

    assert doc is None
    assert meta["table_aware_auto_triggered"] is False
    assert meta["table_aware_trigger_reason"] == []


def test_table_aware_retrieval_auto_business_query_does_not_trigger(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("should not call table evidence retrieval when auto does not trigger")

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc("Explain Amcor's business.")

    assert doc is None
    assert meta["table_aware_auto_triggered"] is False
    assert meta["table_aware_trigger_reason"] == []
    assert meta["table_context_skipped_reasons"] == ["non_table_query"]


def test_table_aware_retrieval_auto_date_only_query_does_not_trigger(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("date-only event query should not trigger table evidence retrieval")

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc("What happened from August 30, 2023 onward?")

    assert doc is None
    assert meta["table_aware_auto_triggered"] is False
    assert meta["table_context_skipped_reasons"] == ["non_table_query"]


def test_table_aware_retrieval_auto_retrieved_table_like_chunk_triggers(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc(
        "Explain this section.",
        retrieved_docs=[
            {
                "text": "Net sales 3,909 3,673 14,544 14,694\nCost of sales (3,115) (2,951) (11,724) (11,969)"
            }
        ],
    )

    assert doc is None
    assert meta["table_aware_auto_triggered"] is True
    assert "retrieved_table_like_chunk" in meta["table_aware_trigger_reason"]


def test_table_aware_retrieval_force_dedupes_ids_and_limits_tables(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            assert kwargs["filter_expr"] == 'evidence_type != "text_chunk"'
            return [
                {"table_id": "t1", "score": 0.9, "text": "row one", "evidence_type": "table_row", "row_id": "row_1", "page_number": 8},
                {"table_id": "t2", "score": 0.8, "text": "row two", "evidence_type": "table_row", "row_id": "row_2", "page_number": 9},
                {"table_id": "t1", "score": 0.7, "text": "row one duplicate", "evidence_type": "table_row", "row_id": "row_3", "page_number": 8},
                {"table_id": "", "score": 0.6, "text": "empty", "evidence_type": "table_row", "row_id": "", "page_number": 10},
            ]

    class _FakeTableStore:
        def __init__(self):
            self.called_with = None

        def get_tables_by_ids(self, table_ids):
            self.called_with = list(table_ids)
            return [
                {"table_id": "t1", "filename": "demo.pdf", "page_number": 8, "columns": ["Metric"], "rows": [{"Metric": "Net sales"}], "csv_text": ""},
            ]

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    fake_table_store = _FakeTableStore()
    monkeypatch.setattr(module, "_table_store", fake_table_store)
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")
    monkeypatch.setenv("TABLE_AWARE_EVIDENCE_TOP_K", "5")
    monkeypatch.setenv("TABLE_AWARE_MAX_TABLES", "1")
    monkeypatch.setenv("TABLE_AWARE_MAX_ROWS", "5")
    monkeypatch.setenv("TABLE_AWARE_MAX_CONTEXT_CHARS", "4000")

    doc, meta = module._build_table_context_doc("net sales")

    assert fake_table_store.called_with == ["t1"]
    assert meta["table_aware_retrieval_mode"] == "force"
    assert meta["table_evidence_hit_count"] == 2
    assert meta["table_context_table_count"] == 1
    assert meta["table_ids"] == ["t1"]
    assert meta["table_context_source"] == "global_fallback"
    assert "Additional structured table evidence:" in doc["text"]


def test_table_aware_retrieval_uses_candidate_filename_filter(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            assert (
                kwargs["filter_expr"]
                == 'evidence_type != "text_chunk" and filename in ["JPMORGAN_2021Q1_10Q.pdf", "JPMORGAN_2021_10K.pdf"]'
            )
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setattr(module, "_table_store", type("_FakeTableStore", (), {"get_tables_by_ids": lambda self, ids: []})())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")

    doc, meta = module._build_table_context_doc(
        "conference call number",
        retrieved_docs=[
            {"filename": "JPMORGAN_2021Q1_10Q.pdf"},
            {"filename": "JPMORGAN_2021Q1_10Q.pdf"},
            {"filename": "JPMORGAN_2021_10K.pdf"},
        ],
    )

    assert doc is None
    assert meta["table_candidate_filenames"] == ["JPMORGAN_2021Q1_10Q.pdf", "JPMORGAN_2021_10K.pdf"]


def test_table_aware_retrieval_candidate_filenames_are_ranked_deduped_and_limited(monkeypatch):
    module = _install_rag_utils_stubs()
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")
    monkeypatch.setenv("TABLE_AWARE_MAX_CANDIDATE_DOCS", "3")

    filenames = module._extract_table_candidate_filenames(
        [
            {"filename": "1.pdf", "score": 0.95},
            {"filename": "2.pdf", "score": 0.70},
            {"filename": "2.pdf", "score": 0.80},
            {"filename": "2.pdf", "score": 0.60},
            {"filename": "ADOBE_2022_10K.pdf"},
            {"filename": "3.pdf", "score": 0.85},
        ],
        max_files=3,
    )

    assert filenames == ["2.pdf", "1.pdf", "ADOBE_2022_10K.pdf"]


def test_table_aware_retrieval_falls_back_to_global_when_no_candidate_filenames(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            assert kwargs["filter_expr"] == 'evidence_type != "text_chunk"'
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setattr(module, "_table_store", type("_FakeTableStore", (), {"get_tables_by_ids": lambda self, ids: []})())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")

    doc, meta = module._build_table_context_doc("net sales", retrieved_docs=[{"filename": ""}, {"text": "chunk"}])

    assert doc is None
    assert meta["table_candidate_filenames"] == []


def test_table_aware_retrieval_prefers_retrieved_table_chunks(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("should not search table evidence when retrieved table chunk already exists")

    class _FakeTableStore:
        def get_tables_by_ids(self, table_ids):
            return [
                {
                    "table_id": "amd::table::1",
                    "filename": "AMD_2022_10K.pdf",
                    "page_number": 42,
                    "columns": ["Metric", "2022"],
                    "rows": [{"Metric": "Quick ratio", "2022": "1.84"}],
                    "csv_text": "",
                }
            ]

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc(
        "What was AMD's quick ratio in 2022?",
        retrieved_docs=[
            {
                "filename": "AMD_2022_10K.pdf",
                "page_number": 42,
                "table_id": "amd::table::1",
                "evidence_type": "table_row",
                "text": "Document: AMD_2022_10K.pdf\nRow Values: Metric: Quick ratio; 2022: 1.84",
            }
        ],
    )

    assert doc is not None
    assert meta["table_context_source"] == "retrieved_table_chunk"
    assert meta["table_ids"] == ["amd::table::1"]


def test_table_aware_retrieval_uses_same_page_table_before_search(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("should not search table evidence when same-page table is available")

    class _FakeTableStore:
        def get_tables_by_ids(self, table_ids):
            return []

        def get_tables_by_filename(self, filename):
            assert filename == "ADOBE_2022_10K.pdf"
            return [
                {
                    "table_id": "adobe::table::1",
                    "filename": "ADOBE_2022_10K.pdf",
                    "page_number": 88,
                    "columns": ["Metric", "2022"],
                    "rows": [{"Metric": "Operating margin", "2022": "35%"}],
                    "csv_text": "",
                }
            ]

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc(
        "What was Adobe operating margin in 2022?",
        retrieved_docs=[
            {
                "filename": "ADOBE_2022_10K.pdf",
                "page_number": 88,
                "text": "Adobe operating margin 35% 34% 33%\nRevenue 100 90 80",
            }
        ],
    )

    assert doc is not None
    assert meta["table_context_source"] == "same_page_table"
    assert meta["table_ids"] == ["adobe::table::1"]
    assert meta["table_candidate_pages"] == [{"filename": "ADOBE_2022_10K.pdf", "page_number": 88}]


def test_table_aware_retrieval_auto_document_scoped_fallback_requires_query_trigger(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("should not fall back to document-scoped table search for non-data query")

    class _FakeTableStore:
        def get_tables_by_ids(self, table_ids):
            return []

        def get_tables_by_filename(self, filename):
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc(
        "Explain Amcor's business.",
        retrieved_docs=[{"filename": "AMCOR_2023_10K.pdf", "page_number": 12, "text": "Business overview and strategy."}],
    )

    assert doc is None
    assert meta["table_context_source"] == "none"
    assert "non_table_query" in meta["table_context_skipped_reasons"]


def test_table_aware_retrieval_global_fallback_only_when_no_candidate_documents(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            assert kwargs["filter_expr"] == 'evidence_type != "text_chunk"'
            return [
                {"table_id": "t1", "score": 0.9, "text": "row one", "evidence_type": "table_row", "row_id": "row_1", "page_number": 8}
            ]

    class _FakeTableStore:
        def get_tables_by_ids(self, table_ids):
            return [
                {"table_id": "t1", "filename": "1.pdf", "page_number": 8, "columns": ["Metric"], "rows": [{"Metric": "Net sales"}], "csv_text": ""}
            ]

        def get_tables_by_filename(self, filename):
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")

    doc, meta = module._build_table_context_doc("What was revenue?", retrieved_docs=[])

    assert doc is not None
    assert meta["table_context_source"] == "global_fallback"


def test_table_aware_retrieval_quality_guard_skips_full_rows_for_bad_table(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise AssertionError("should not search when same-page candidate exists")

    class _FakeTableStore:
        def get_tables_by_ids(self, table_ids):
            return []

        def get_tables_by_filename(self, filename):
            return [
                {
                    "table_id": "bad::table::1",
                    "filename": filename,
                    "page_number": 4,
                    "title": "Chief executive officer commentary on strategic priorities and market conditions",
                    "columns": [
                        "Chief executive officer commentary on strategic priorities",
                        "market conditions and demand trends",
                        "portfolio simplification update",
                        "capital allocation overview",
                        "sustainability roadmap progress",
                        "regional operating context summary",
                    ],
                    "rows": [{"a": "narrative only", "b": "words only"}],
                    "csv_text": "narrative only,words only",
                }
            ]

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")

    doc, meta = module._build_table_context_doc(
        "What was revenue growth?",
        retrieved_docs=[{"filename": "1.pdf", "page_number": 4, "text": "Revenue 10 11 12\nCost of sales 5 6 7"}],
    )

    assert doc is not None
    assert "table_quality_rejected" in meta["table_context_skipped_reasons"]
    assert "(skipped: table_quality_rejected)" in doc["text"]


def test_table_context_respects_max_context_chars(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            return [{"table_id": "t1", "score": 0.9, "text": "x" * 500, "evidence_type": "table_row", "row_id": "row_1", "page_number": 8}]

    class _FakeTableStore:
        def get_tables_by_ids(self, table_ids):
            return [
                {
                    "table_id": "t1",
                    "filename": "demo.pdf",
                    "page_number": 8,
                    "columns": ["Metric", "2023"],
                    "rows": [{"Metric": "Net sales", "2023": "3,909"} for _ in range(20)],
                    "csv_text": "Net sales," + ("3,909," * 100),
                }
            ]

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")
    monkeypatch.setenv("TABLE_AWARE_MAX_CONTEXT_CHARS", "200")

    doc, meta = module._build_table_context_doc("net sales")

    assert meta["table_context_char_count"] <= 200
    assert "... table evidence truncated ..." in doc["text"]


def test_table_aware_retrieval_exception_does_not_break(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FailMilvus:
        def hybrid_retrieve(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(module, "_milvus_manager", _FailMilvus())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")

    doc, meta = module._build_table_context_doc("net sales")

    assert doc is None
    assert meta["table_aware_retrieval_mode"] == "force"
    assert meta["table_evidence_hit_count"] == 0


def test_debug_retrieval_pipeline_exposes_table_aware_trace_fields(monkeypatch):
    module = _install_rag_utils_stubs()

    monkeypatch.setattr(
        module,
        "retrieve_documents",
        lambda question, top_k=10: {
            "final_retrieved_docs": [],
            "context_docs": [],
            "meta": {
                "retrieval_mode": "baseline",
                "query_planner_enabled": True,
                "planner_intent": "numeric_lookup",
                "planner_must_keep_terms": ["Adobe", "2022"],
                "planner_dense_queries": ["Adobe operating margin 2022"],
                "planner_semantic_queries": ["Adobe operating margin 2022"],
                "planner_evidence_field_queries": ["Adobe revenue income from operations 2022"],
                "planner_table_heading_queries": ["Adobe statements of income 2022"],
                "planner_keyword_queries": ["Adobe margin 2022"],
                "planner_table_queries": [],
                "planner_validation_dropped_queries": [{"field": "semantic_queries", "query": "AES operating margin 2022", "reason": "validation_failed"}],
                "planner_parse_error": "",
                "per_query_retrieval_counts": [{"label": "original", "count": 3}],
                "rrf_fused_candidate_count": 4,
                "page_level_fusion_enabled": True,
                "fused_page_count": 2,
                "fused_top_pages": [{"filename": "demo.pdf", "page_number": 8, "page_score": 1.2, "matched_queries": ["Adobe margin 2022"], "contributing_routes": ["original", "semantic_1"]}],
                "fused_pages_after_anchor_guard": [{"filename": "demo.pdf", "page_number": 8, "page_score": 1.2, "matched_queries": ["Adobe margin 2022"], "contributing_routes": ["original", "semantic_1"]}],
                "page_anchor_filtered_count": 2,
                "page_contributing_routes": {"demo.pdf#page=8": ["original", "semantic_1"]},
                "page_fusion_used_for_final_context": True,
                "final_evidence_pack_source": "page_fusion",
                "table_aware_retrieval_mode": "force",
                "table_aware_auto_triggered": True,
                "table_aware_trigger_reason": ["force"],
                "query_anchors": ["Adobe"],
                "anchor_guard_applied": True,
                "anchor_filtered_count": 2,
                "table_context_source": "document_scoped_search",
                "table_evidence_hit_count": 2,
                "table_context_table_count": 1,
                "table_context_char_count": 321,
                "table_candidate_filenames": ["demo.pdf"],
                "table_candidate_pages": [{"filename": "demo.pdf", "page_number": 8}],
                "table_ids": ["t1"],
                "table_context_skipped_reasons": ["table_quality_rejected"],
                "latency_breakdown": {"total_retrieval_ms": 12.3},
            },
        },
    )

    result = module.debug_retrieval_pipeline("net sales", top_k=5)
    trace = result["rag_trace"]

    assert trace["query_planner_enabled"] is True
    assert trace["planner_intent"] == "numeric_lookup"
    assert trace["planner_must_keep_terms"] == ["Adobe", "2022"]
    assert trace["planner_dense_queries"] == ["Adobe operating margin 2022"]
    assert trace["planner_semantic_queries"] == ["Adobe operating margin 2022"]
    assert trace["planner_evidence_field_queries"] == ["Adobe revenue income from operations 2022"]
    assert trace["planner_table_heading_queries"] == ["Adobe statements of income 2022"]
    assert trace["planner_keyword_queries"] == ["Adobe margin 2022"]
    assert trace["planner_table_queries"] == []
    assert trace["planner_validation_dropped_queries"] == [
        {"field": "semantic_queries", "query": "AES operating margin 2022", "reason": "validation_failed"}
    ]
    assert trace["planner_parse_error"] == ""
    assert trace["per_query_retrieval_counts"] == [{"label": "original", "count": 3}]
    assert trace["rrf_fused_candidate_count"] == 4
    assert trace["page_level_fusion_enabled"] is True
    assert trace["fused_page_count"] == 2
    assert trace["fused_top_pages"][0]["filename"] == "demo.pdf"
    assert trace["fused_pages_after_anchor_guard"][0]["filename"] == "demo.pdf"
    assert trace["page_anchor_filtered_count"] == 2
    assert trace["page_contributing_routes"] == {"demo.pdf#page=8": ["original", "semantic_1"]}
    assert trace["page_fusion_used_for_final_context"] is True
    assert trace["final_evidence_pack_source"] == "page_fusion"
    assert trace["table_aware_retrieval_mode"] == "force"
    assert trace["table_aware_auto_triggered"] is True
    assert trace["table_aware_trigger_reason"] == ["force"]
    assert trace["query_anchors"] == ["Adobe"]
    assert trace["anchor_guard_applied"] is True
    assert trace["anchor_filtered_count"] == 2
    assert trace["table_context_source"] == "document_scoped_search"
    assert trace["table_evidence_hit_count"] == 2
    assert trace["table_context_table_count"] == 1
    assert trace["table_context_char_count"] == 321
    assert trace["table_candidate_filenames"] == ["demo.pdf"]
    assert trace["table_candidate_pages"] == [{"filename": "demo.pdf", "page_number": 8}]
    assert trace["table_ids"] == ["t1"]
    assert trace["table_context_skipped_reasons"] == ["table_quality_rejected"]


def test_table_aware_anchor_guard_filters_wrong_adobe_table(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            assert kwargs["filter_expr"] == 'evidence_type != "text_chunk"'
            return [
                {
                    "table_id": "aes::table::1",
                    "filename": "1.pdf",
                    "score": 0.9,
                    "text": "Document: 1.pdf\nRow Values: Metric: Operating margin; 2022: 12%",
                    "evidence_type": "table_row",
                    "row_id": "row_1",
                    "page_number": 8,
                }
            ]

    class _FakeTableStore:
        def get_tables_by_ids(self, table_ids):
            return [
                {
                    "table_id": "aes::table::1",
                    "filename": "1.pdf",
                    "page_number": 8,
                    "title": "Operating margin table",
                    "columns": ["Metric", "2022"],
                    "rows": [{"Metric": "Operating margin", "2022": "12%"}],
                    "csv_text": "Operating margin,12%",
                }
            ]

        def get_tables_by_filename(self, filename):
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")

    doc, meta = module._build_table_context_doc("What was Adobe operating margin in 2022?", retrieved_docs=[])

    assert doc is None
    assert meta["query_anchors"] == ["Adobe"]
    assert meta["anchor_guard_applied"] is True
    assert meta["anchor_filtered_count"] == 1
    assert meta["table_context_source"] == "none"
    assert "anchor_guard_filtered" in meta["table_context_skipped_reasons"]


def test_table_aware_anchor_guard_filters_wrong_3m_table(monkeypatch):
    module = _install_rag_utils_stubs()

    class _FakeMilvus:
        def hybrid_retrieve(self, **kwargs):
            assert kwargs["filter_expr"] == 'evidence_type != "text_chunk"'
            return [
                {
                    "table_id": "jnj::table::1",
                    "filename": "2.pdf",
                    "score": 0.9,
                    "text": "Document: 2.pdf\nRow Values: Metric: Segment margin; 2022: 18%",
                    "evidence_type": "table_row",
                    "row_id": "row_1",
                    "page_number": 12,
                }
            ]

    class _FakeTableStore:
        def get_tables_by_ids(self, table_ids):
            return [
                {
                    "table_id": "jnj::table::1",
                    "filename": "2.pdf",
                    "page_number": 12,
                    "title": "Segment margin table",
                    "columns": ["Metric", "2022"],
                    "rows": [{"Metric": "Segment margin", "2022": "18%"}],
                    "csv_text": "Segment margin,18%",
                }
            ]

        def get_tables_by_filename(self, filename):
            return []

    monkeypatch.setattr(module, "_milvus_manager", _FakeMilvus())
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "force")

    doc, meta = module._build_table_context_doc("How did 3M segment margin change in 2022?", retrieved_docs=[])

    assert doc is None
    assert "3M" in meta["query_anchors"]
    assert meta["anchor_guard_applied"] is True
    assert meta["anchor_filtered_count"] == 1
    assert "anchor_guard_filtered" in meta["table_context_skipped_reasons"]


def test_retrieve_documents_off_keeps_plain_text_context(monkeypatch):
    module = _install_rag_utils_stubs()

    plain_doc = {
        "filename": "1.pdf",
        "doc_name": "1",
        "page_number": 8,
        "chunk_id": "c1",
        "text": "Plain paragraph context.",
        "type": "chunk",
        "evidence_type": "text_chunk",
    }

    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "off")
    monkeypatch.setattr(module, "retrieve_candidate_documents", lambda query, candidate_k=None: {"docs": [plain_doc], "meta": {}})
    monkeypatch.setattr(
        module,
        "finalize_retrieved_documents",
        lambda query, candidate_docs, final_top_k=None, enable_page_merge=None, adjacent_page_window=None, adjacent_chunk_window=None: {
            "final_retrieved_docs": [plain_doc],
            "context_docs": [plain_doc],
            "meta": {},
        },
    )

    result = module.retrieve_documents("Explain this document.", top_k=5)

    assert result["context_docs"] == [plain_doc]
    assert result["meta"]["table_aware_retrieval_mode"] == "off"
    assert result["meta"]["evidence_unit_count"] == 0


def test_retrieve_documents_auto_attaches_same_page_table_from_context_doc(monkeypatch):
    module = _install_rag_utils_stubs()

    doc = {
        "filename": "1.pdf",
        "doc_name": "1",
        "page_number": 8,
        "chunk_id": "c1",
        "text": "Net sales 3,909 3,673 14,544 14,694\nCost of sales (3,115) (2,951) (11,724) (11,969)",
        "type": "chunk",
        "evidence_type": "text_chunk",
    }

    class _FakeTableStore:
        def get_tables_by_filename(self, filename):
            return [
                {
                    "table_id": "t1",
                    "filename": "1.pdf",
                    "page_number": 8,
                    "columns": ["Metric", "2023", "2022"],
                    "rows": [{"Metric": "Net sales", "2023": "3,909", "2022": "3,673"}],
                    "csv_text": "Metric,2023,2022\nNet sales,3,909,3,673",
                    "title": "Statements of Income",
                }
            ]

        def get_tables_by_ids(self, table_ids):
            return []

    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setattr(module, "retrieve_candidate_documents", lambda query, candidate_k=None: {"docs": [doc], "meta": {}})
    monkeypatch.setattr(
        module,
        "finalize_retrieved_documents",
        lambda query, candidate_docs, final_top_k=None, enable_page_merge=None, adjacent_page_window=None, adjacent_chunk_window=None: {
            "final_retrieved_docs": [doc],
            "context_docs": [doc],
            "meta": {},
        },
    )

    result = module.retrieve_documents("What was net sales?", top_k=5)

    assert len(result["context_docs"]) == 1
    assert "[Evidence Group 1]" in result["context_docs"][0]["text"]
    assert "Matched snippets:" in result["context_docs"][0]["text"]
    assert "Relevant table rows:" in result["context_docs"][0]["text"]
    assert result["meta"]["evidence_units_with_tables"] == 1
    assert result["meta"]["evidence_group_count"] == 1
    assert result["meta"]["selected_evidence_group_count"] == 1
    assert result["meta"]["final_evidence_pack_source"] == "evidence_groups"
    assert "chunk_table_like" in result["meta"]["table_attach_reasons"]


def test_retrieve_documents_auto_non_table_query_does_not_force_table_attachment(monkeypatch):
    module = _install_rag_utils_stubs()

    doc = {
        "filename": "1.pdf",
        "doc_name": "1",
        "page_number": 3,
        "chunk_id": "c2",
        "text": "Business overview and strategic priorities.",
        "type": "chunk",
        "evidence_type": "text_chunk",
    }

    class _FakeTableStore:
        def get_tables_by_filename(self, filename):
            return [
                {
                    "table_id": "t2",
                    "filename": "1.pdf",
                    "page_number": 3,
                    "columns": ["Narrative summary", "Strategic priorities", "Market trends", "Commentary", "Outlook", "Plan"],
                    "rows": [{"a": "words only", "b": "more words"}],
                    "csv_text": "words only,more words",
                    "title": "Chief executive commentary and business outlook",
                }
            ]

        def get_tables_by_ids(self, table_ids):
            return []

    monkeypatch.setenv("TABLE_AWARE_RETRIEVAL", "auto")
    monkeypatch.setattr(module, "_table_store", _FakeTableStore())
    monkeypatch.setattr(module, "retrieve_candidate_documents", lambda query, candidate_k=None: {"docs": [doc], "meta": {}})
    monkeypatch.setattr(
        module,
        "finalize_retrieved_documents",
        lambda query, candidate_docs, final_top_k=None, enable_page_merge=None, adjacent_page_window=None, adjacent_chunk_window=None: {
            "final_retrieved_docs": [doc],
            "context_docs": [doc],
            "meta": {},
        },
    )

    result = module.retrieve_documents("Explain Amcor's business.", top_k=5)

    assert result["context_docs"] == [doc]
    assert result["meta"]["evidence_units_with_tables"] == 0
    assert result["meta"]["table_context_source"] == "none"


def test_retrieve_candidate_documents_uses_only_original_and_finance_rewrite(monkeypatch):
    module = _install_rag_utils_stubs()
    calls = []

    monkeypatch.setenv("RAG_FIELD_AWARE_ENABLED", "true")
    rewrite = "What was Adobe operating margin in 2022?\nFinancial retrieval anchors: operating income; total revenue; 2022"
    monkeypatch.setattr(module, "build_finance_query_rewrite", lambda question: rewrite)

    def _fake_retrieve_leaf_chunks(query, top_k, filter_expr, retrieval_scope):
        calls.append({"query": query, "top_k": top_k, "filter_expr": filter_expr, "scope": retrieval_scope})
        docs_map = {
            "What was Adobe operating margin in 2022?": [
                {"filename": "1.pdf", "page_number": 8, "chunk_id": "c1", "text": "Adobe operating margin was 35% in 2022.", "score": 0.4}
            ],
            rewrite: [
                {"filename": "1.pdf", "page_number": 8, "chunk_id": "c3", "text": "Income from operations was $12.4 billion in 2022.", "score": 0.45}
            ],
        }
        return {"docs": docs_map.get(query, []), "meta": {"retrieval_mode": "hybrid"}}

    monkeypatch.setattr(module, "_retrieve_leaf_chunks", _fake_retrieve_leaf_chunks)

    result = module.retrieve_candidate_documents("What was Adobe operating margin in 2022?", candidate_k=6)

    assert [item["query"] for item in calls] == [
        "What was Adobe operating margin in 2022?",
        rewrite,
    ]
    assert result["meta"]["query_planner_enabled"] is False
    assert result["meta"]["planner_semantic_queries"] == []
    assert result["meta"]["rrf_fused_candidate_count"] == 2
    assert len(result["meta"]["per_query_retrieval_counts"]) == 2
    assert result["meta"]["per_query_retrieval_counts"][1]["category"] == "finance_rewrite"
    assert result["docs"][0]["chunk_id"] == "c1"


def test_statement_type_bonus_promotes_matching_financial_statement(monkeypatch):
    module = _install_rag_utils_stubs()

    class _Embedding:
        def get_embeddings(self, texts):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(module, "_embedding_service", _Embedding())
    monkeypatch.setattr(
        module,
        "infer_page_statement_types",
        lambda text: ["income_statement"] if "STATEMENTS OF INCOME" in text else ["balance_sheet"],
    )
    pages = [
        {
            "filename": "ADOBE_2022_10K.pdf",
            "page_number": 40,
            "page_text": "CONSOLIDATED STATEMENTS OF INCOME\nRevenue and operating income",
            "page_dense_embedding": [1.0, 0.0],
            "chunk_ids": ["income"],
        },
        {
            "filename": "ADOBE_2022_10K.pdf",
            "page_number": 41,
            "page_text": "CONSOLIDATED BALANCE SHEETS\nAssets and liabilities",
            "page_dense_embedding": [1.0, 0.0],
            "chunk_ids": ["balance"],
        },
    ]
    query_parse = {
        "company": "",
        "years": [],
        "doc_types": [],
        "statement_types": ["income_statement"],
    }

    scored, _ = module._score_candidate_pages(
        "operating margin",
        pages,
        [],
        query_parse,
        module.get_finance_rag_config(),
    )

    assert scored[0]["page_number"] == 40
    assert scored[0]["statement_type_match_score"] == 1.0


def test_page_first_returns_anchor_chunk_and_direct_neighbors(monkeypatch):
    module = _install_rag_utils_stubs()
    monkeypatch.setenv("RAG_FIELD_AWARE_ENABLED", "true")
    candidate = {"filename": "ADOBE_2022_10K.pdf", "page_number": 40, "chunk_id": "c1", "text": "candidate", "score": 1.0}
    page = {"filename": "ADOBE_2022_10K.pdf", "page_number": 40, "chunk_ids": ["c0", "c1", "c2"], "page_score": 1.0}
    chunks = [
        {"filename": page["filename"], "page_number": 40, "chunk_id": "c0", "chunk_idx": 0, "text": "context before"},
        {"filename": page["filename"], "page_number": 40, "chunk_id": "c1", "chunk_idx": 1, "text": "operating margin anchor"},
        {"filename": page["filename"], "page_number": 40, "chunk_id": "c2", "chunk_idx": 2, "text": "context after"},
    ]

    class _PageStore:
        def get_pages_by_filenames(self, filenames, warm_cache=False):
            return [page]

    class _Milvus:
        def get_chunks_by_ids(self, chunk_ids):
            return chunks

    monkeypatch.setattr(module, "_document_page_store", _PageStore())
    monkeypatch.setattr(module, "_milvus_manager", _Milvus())
    monkeypatch.setattr(module, "_select_doc_stage_documents", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(module, "_score_candidate_pages", lambda *args, **kwargs: ([page], {}))
    monkeypatch.setattr(
        module,
        "_extract_keyword_tokens",
        lambda text: {"operating", "margin"} if "operating margin" in text else set(),
    )

    docs, meta = module._build_page_first_candidates(
        "operating margin",
        [candidate],
        candidate_k=5,
        config=module.get_finance_rag_config(),
    )

    selected = {doc["chunk_id"]: doc["anchor_chunk_offset"] for doc in docs if "anchor_chunk_offset" in doc}
    assert selected == {"c0": -1, "c1": 0, "c2": 1}
    assert meta["page_first_anchor_window"] == 1


def test_legacy_planner_flag_cannot_reenable_generated_query_routes(monkeypatch):
    module = _install_rag_utils_stubs()
    calls = []

    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    monkeypatch.setenv("RAG_FIELD_AWARE_ENABLED", "true")
    rewrite = "What was Adobe operating margin in 2022?\nFinancial retrieval anchors: operating income; total revenue; 2022"
    monkeypatch.setattr(module, "build_finance_query_rewrite", lambda question: rewrite)

    def _fake_retrieve_leaf_chunks(query, top_k, filter_expr, retrieval_scope):
        calls.append(query)
        docs_map = {
            "What was Adobe operating margin in 2022?": [
                {"filename": "1.pdf", "page_number": 8, "chunk_id": "c1", "text": "Adobe operating margin was 35% in 2022.", "score": 0.6},
                {"filename": "2.pdf", "page_number": 4, "chunk_id": "c2", "text": "Ulta margin was 12% in 2022.", "score": 0.9},
            ],
            rewrite: [
                {"filename": "1.pdf", "page_number": 8, "chunk_id": "c4", "text": "Statements of income Adobe 2022 revenue income from operations.", "score": 0.4},
            ],
        }
        return {"docs": docs_map.get(query, []), "meta": {"retrieval_mode": "hybrid"}}

    monkeypatch.setattr(module, "_retrieve_leaf_chunks", _fake_retrieve_leaf_chunks)

    result = module.retrieve_candidate_documents("What was Adobe operating margin in 2022?", candidate_k=5)

    assert calls == ["What was Adobe operating margin in 2022?", rewrite]
    assert result["meta"]["query_planner_enabled"] is False
    assert result["meta"]["page_level_fusion_enabled"] is False
    assert result["meta"]["planner_semantic_queries"] == []


def test_finalize_retrieved_documents_prefers_page_fusion_for_final_context(monkeypatch):
    module = _install_rag_utils_stubs()
    monkeypatch.setenv("RAG_PAGE_LEVEL_FUSION", "true")

    docs = [
        {
            "filename": "1.pdf",
            "doc_name": "1",
            "page_number": 8,
            "chunk_id": "c1",
            "text": "Adobe operating margin was 35% in 2022.",
            "evidence_type": "text_chunk",
            "score": 0.9,
            "page_fused_rank": 1,
            "page_fused_score": 1.4,
            "page_contributing_routes": ["original", "semantic_1"],
            "page_matched_queries": ["What was Adobe operating margin in 2022?"],
        },
        {
            "filename": "1.pdf",
            "doc_name": "1",
            "page_number": 8,
            "chunk_id": "c2",
            "text": "Adobe income from operations and revenue are reported here.",
            "evidence_type": "text_chunk",
            "score": 0.8,
            "page_fused_rank": 1,
            "page_fused_score": 1.4,
            "page_contributing_routes": ["original", "evidence_field_1"],
            "page_matched_queries": ["Adobe revenue income from operations 2022"],
        },
        {
            "filename": "2.pdf",
            "doc_name": "2",
            "page_number": 4,
            "chunk_id": "c3",
            "text": "Ulta margin was 12% in 2022.",
            "evidence_type": "text_chunk",
            "score": 0.95,
            "page_fused_rank": 2,
            "page_fused_score": 0.7,
            "page_contributing_routes": ["original"],
            "page_matched_queries": ["What was Adobe operating margin in 2022?"],
        },
    ]

    result = module.finalize_retrieved_documents(
        "What was Adobe operating margin in 2022?",
        docs,
        final_top_k=2,
        enable_page_merge=False,
    )

    assert result["meta"]["page_fusion_used_for_final_context"] is True
    assert result["meta"]["final_evidence_pack_source"] == "page_fusion"
    assert result["meta"]["selected_pages"][0]["filename"] == "1.pdf"
    assert result["final_retrieved_docs"][0]["filename"] == "1.pdf"
    assert result["final_retrieved_docs"][1]["filename"] == "1.pdf"


def test_question_without_required_fields_uses_only_original_query(monkeypatch):
    module = _install_rag_utils_stubs()
    calls = []

    monkeypatch.setenv("RAG_QUERY_PLANNER_ENABLED", "true")
    monkeypatch.setattr(module, "build_finance_query_rewrite", lambda question: "")

    def _fake_retrieve_leaf_chunks(query, top_k, filter_expr, retrieval_scope):
        calls.append(query)
        return {
            "docs": [{"filename": "1.pdf", "chunk_id": "c1", "text": "Fallback original query result.", "score": 0.3}],
            "meta": {"retrieval_mode": "hybrid"},
        }

    monkeypatch.setattr(module, "_retrieve_leaf_chunks", _fake_retrieve_leaf_chunks)

    result = module.retrieve_candidate_documents("Summarize this document.", candidate_k=5)

    assert calls == ["Summarize this document."]
    assert result["meta"]["query_planner_enabled"] is False
    assert result["meta"]["planner_dense_queries"] == []
    assert result["meta"]["planner_semantic_queries"] == []
    assert result["meta"]["planner_evidence_field_queries"] == []
    assert result["meta"]["planner_table_heading_queries"] == []
    assert result["meta"]["planner_keyword_queries"] == []
    assert result["meta"]["planner_table_queries"] == []
    assert result["meta"]["planner_parse_error"] == ""
    assert result["docs"][0]["chunk_id"] == "c1"


def test_retrieve_candidate_documents_uses_original_when_finance_rewrite_is_empty(monkeypatch):
    module = _install_rag_utils_stubs()
    calls = []

    monkeypatch.delenv("RAG_QUERY_PLANNER_ENABLED", raising=False)

    def _fake_retrieve_leaf_chunks(query, top_k, filter_expr, retrieval_scope):
        calls.append(query)
        return {
            "docs": [{"filename": "1.pdf", "chunk_id": "c1", "text": "Original query result.", "score": 0.3}],
            "meta": {"retrieval_mode": "hybrid"},
        }

    monkeypatch.setattr(module, "build_finance_query_rewrite", lambda question: "")
    monkeypatch.setattr(module, "_retrieve_leaf_chunks", _fake_retrieve_leaf_chunks)

    result = module.retrieve_candidate_documents("What was revenue?", candidate_k=5)

    assert calls == ["What was revenue?"]
    assert result["meta"]["query_planner_enabled"] is False
    assert result["meta"]["planner_dense_queries"] == []
    assert result["meta"]["planner_semantic_queries"] == []
    assert result["meta"]["planner_evidence_field_queries"] == []
    assert result["meta"]["planner_table_heading_queries"] == []
    assert result["meta"]["planner_keyword_queries"] == []
    assert result["meta"]["planner_table_queries"] == []
    assert result["meta"]["per_query_retrieval_counts"] == [
        {
            "label": "original",
            "category": "original",
            "query": "What was revenue?",
            "count": 1,
            "retrieval_mode": "hybrid",
        }
    ]
