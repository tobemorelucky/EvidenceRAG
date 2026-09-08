import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path


def _install_api_stubs():
    fastapi = types.ModuleType("fastapi")

    class APIRouter:
        def post(self, *args, **kwargs):
            return lambda func: func

        def get(self, *args, **kwargs):
            return lambda func: func

        def delete(self, *args, **kwargs):
            return lambda func: func

    class BackgroundTasks:
        def add_task(self, *args, **kwargs):
            return None

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class UploadFile:
        filename = ""

        async def read(self, *_args, **_kwargs):
            return b""

    fastapi.APIRouter = APIRouter
    fastapi.BackgroundTasks = BackgroundTasks
    fastapi.Depends = lambda dependency=None: dependency
    fastapi.File = lambda default=None: default
    fastapi.HTTPException = HTTPException
    fastapi.UploadFile = UploadFile
    sys.modules["fastapi"] = fastapi

    fastapi_responses = types.ModuleType("fastapi.responses")
    fastapi_responses.StreamingResponse = object
    sys.modules["fastapi.responses"] = fastapi_responses

    sqlalchemy_orm = types.ModuleType("sqlalchemy.orm")
    sqlalchemy_orm.Session = object
    sys.modules["sqlalchemy.orm"] = sqlalchemy_orm

    agent = types.ModuleType("agent")
    agent.chat_with_agent = lambda *args, **kwargs: {"response": "ok"}

    async def _fake_stream(*args, **kwargs):
        if False:
            yield None

    agent.chat_with_agent_stream = _fake_stream
    agent.storage = types.SimpleNamespace(
        get_session_messages=lambda *args, **kwargs: [],
        list_session_infos=lambda *args, **kwargs: [],
        delete_session=lambda *args, **kwargs: True,
    )
    sys.modules["agent"] = agent

    conversation_memory = types.ModuleType("conversation_memory_v5")
    conversation_memory.conversation_memory_v5_enabled = lambda: False
    conversation_memory.conversation_memory_v5_service = types.SimpleNamespace()
    sys.modules["conversation_memory_v5"] = conversation_memory

    auth = types.ModuleType("auth")
    auth.authenticate_user = lambda *args, **kwargs: None
    auth.create_access_token = lambda *args, **kwargs: "token"
    auth.get_current_user = lambda: None
    auth.get_db = lambda: None
    auth.get_password_hash = lambda password: password
    auth.require_admin = lambda: None
    auth.resolve_role = lambda *args, **kwargs: "user"
    sys.modules["auth"] = auth

    document_loader = types.ModuleType("document_loader")
    document_loader.DocumentLoader = type("DocumentLoader", (), {})
    sys.modules["document_loader"] = document_loader

    document_page_store = types.ModuleType("document_page_store")
    document_page_store.DocumentPageStore = type(
        "DocumentPageStore",
        (),
        {"upsert_pages": lambda self, pages: None, "delete_by_filename": lambda self, filename: 0},
    )
    sys.modules["document_page_store"] = document_page_store

    embedding = types.ModuleType("embedding")
    embedding.embedding_service = object()
    sys.modules["embedding"] = embedding

    milvus_client = types.ModuleType("milvus_client")
    milvus_client.MilvusManager = type(
        "MilvusManager",
        (),
        {
            "init_collection": lambda self: None,
            "delete": lambda self, expr=None: {"delete_count": 0},
            "query_all": lambda self, **kwargs: [],
        },
    )
    sys.modules["milvus_client"] = milvus_client

    milvus_writer = types.ModuleType("milvus_writer")
    milvus_writer.MilvusWriter = type(
        "MilvusWriter",
        (),
        {"__init__": lambda self, *args, **kwargs: None, "write_documents": lambda self, docs, **kwargs: None},
    )
    sys.modules["milvus_writer"] = milvus_writer

    models = types.ModuleType("models")
    models.User = type("User", (), {})
    sys.modules["models"] = models

    parent_chunk_store = types.ModuleType("parent_chunk_store")
    parent_chunk_store.ParentChunkStore = type(
        "ParentChunkStore",
        (),
        {"upsert_documents": lambda self, docs: None, "delete_by_filename": lambda self, filename: 0},
    )
    sys.modules["parent_chunk_store"] = parent_chunk_store

    rag_utils = types.ModuleType("rag_utils")
    rag_utils.debug_retrieval_pipeline = lambda *args, **kwargs: {}
    sys.modules["rag_utils"] = rag_utils

    schemas = types.ModuleType("schemas")
    for name in (
        "AuthResponse",
        "ChatRequest",
        "ChatResponse",
        "CurrentUserResponse",
        "DebugRetrievalRequest",
        "DocumentBatchDeleteRequest",
        "DocumentBatchDeleteStartResponse",
        "DocumentDeleteJobResponse",
        "DocumentDeleteResponse",
        "DocumentDeleteStartResponse",
        "DocumentInfo",
        "DocumentListResponse",
        "DocumentUploadJobResponse",
        "DocumentUploadResponse",
        "DocumentUploadStartResponse",
        "LoginRequest",
        "MessageInfo",
        "RegisterRequest",
        "SessionDeleteResponse",
        "SessionInfo",
        "SessionListResponse",
        "SessionMessagesResponse",
        "ConversationChatRequest",
        "ConversationChatResponse",
        "ConversationCreateRequest",
        "ConversationCreateResponse",
        "ConversationDeleteResponse",
        "ConversationInfo",
        "ConversationListResponse",
        "ConversationMessageInfo",
        "ConversationMessagesResponse",
        "ConversationTraceListResponse",
    ):
        setattr(schemas, name, type(name, (), {"__init__": lambda self, **kwargs: None}))
    sys.modules["schemas"] = schemas

    table_store = types.ModuleType("table_store")
    table_store.TableStore = type(
        "TableStore",
        (),
        {"upsert_tables": lambda self, tables: len(tables), "delete_by_filename": lambda self, filename: 0},
    )
    sys.modules["table_store"] = table_store

    table_config = types.ModuleType("table_config")
    table_config.get_table_aware_config = lambda: types.SimpleNamespace(table_aware_ingestion=False)
    sys.modules["table_config"] = table_config

    table_indexer = types.ModuleType("table_indexer")
    table_indexer.build_table_evidence_docs = lambda _tables: []
    sys.modules["table_indexer"] = table_indexer

    upload_jobs = types.ModuleType("upload_jobs")
    upload_jobs.DELETE_STEPS = []
    upload_jobs.delete_job_manager = types.SimpleNamespace()
    upload_jobs.upload_job_manager = types.SimpleNamespace()
    sys.modules["upload_jobs"] = upload_jobs


def _load_api_module():
    original_sqlalchemy_orm = sys.modules.get("sqlalchemy.orm")
    _install_api_stubs()
    path = Path(__file__).resolve().parents[1] / "backend" / "api.py"
    spec = importlib.util.spec_from_file_location("api_table_evidence_test_module", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    if original_sqlalchemy_orm is not None:
        sys.modules["sqlalchemy.orm"] = original_sqlalchemy_orm
    else:
        sys.modules.pop("sqlalchemy.orm", None)
    return module


@dataclass
class _FakeConfig:
    table_aware_ingestion: bool


def test_prepare_table_evidence_docs_skips_build_when_disabled(monkeypatch):
    module = _load_api_module()
    build_called = {"value": False}

    class _FakeTableStore:
        def upsert_tables(self, tables):
            return len(tables)

    def _build(_tables):
        build_called["value"] = True
        return [{"text": "should not happen"}]

    monkeypatch.setattr(module, "table_store", _FakeTableStore())
    monkeypatch.setattr(module, "get_table_aware_config", lambda: _FakeConfig(table_aware_ingestion=False))
    monkeypatch.setattr(module, "build_table_evidence_docs", _build)

    stored_count, evidence_docs = module._prepare_table_evidence_docs("demo.pdf", [{"table_id": "t1"}])

    assert stored_count == 1
    assert evidence_docs == []
    assert build_called["value"] is False


def test_prepare_table_evidence_docs_builds_when_enabled(monkeypatch):
    module = _load_api_module()

    class _FakeTableStore:
        def upsert_tables(self, tables):
            return len(tables)

    built_docs = [{"text": "table evidence"}]
    monkeypatch.setattr(module, "table_store", _FakeTableStore())
    monkeypatch.setattr(module, "get_table_aware_config", lambda: _FakeConfig(table_aware_ingestion=True))
    monkeypatch.setattr(module, "build_table_evidence_docs", lambda tables: built_docs)

    stored_count, evidence_docs = module._prepare_table_evidence_docs("demo.pdf", [{"table_id": "t1"}])

    assert stored_count == 1
    assert evidence_docs == built_docs


def test_write_table_evidence_docs_safe_ignores_write_errors(monkeypatch):
    module = _load_api_module()

    class _FakeMilvusWriter:
        def write_documents(self, docs):
            raise RuntimeError("boom")

    monkeypatch.setattr(module, "milvus_writer", _FakeMilvusWriter())

    written_count = module._write_table_evidence_docs_safe("demo.pdf", [{"text": "table evidence"}])

    assert written_count == 0


def test_async_delete_removes_every_indexed_document_store(monkeypatch):
    module = _load_api_module()
    calls = []

    class _Jobs:
        def update_step(self, _job_id, step, *_args, **_kwargs):
            calls.append(("step", step))

        def complete_step(self, _job_id, step, *_args, **_kwargs):
            calls.append(("complete", step))

        def complete_job(self, *_args, **_kwargs):
            calls.append(("job", "completed"))

        def fail_job(self, *_args, **_kwargs):
            raise AssertionError("delete job unexpectedly failed")

    class _Store:
        def __init__(self, name, result=1):
            self.name = name
            self.result = result

        def delete_by_filename(self, filename):
            calls.append((self.name, filename))
            return self.result

    monkeypatch.setattr(module, "delete_job_manager", _Jobs())
    monkeypatch.setattr(module, "_remove_bm25_stats_for_filename", lambda name: calls.append(("bm25", name)))
    monkeypatch.setattr(module, "milvus_manager", types.SimpleNamespace(
        init_collection=lambda: calls.append(("milvus", "init")),
        delete=lambda _expr: {"delete_count": 3},
    ))
    monkeypatch.setattr(module, "parent_chunk_store", _Store("parent"))
    monkeypatch.setattr(module, "document_page_store", _Store("page", 2))
    monkeypatch.setattr(module, "table_store", _Store("table"))

    module._process_delete_job("job-1", "demo.pdf")

    assert ("parent", "demo.pdf") in calls
    assert ("page", "demo.pdf") in calls
    assert ("table", "demo.pdf") in calls
    assert ("complete", "page_store") in calls


def test_async_upload_writes_parent_pages_tables_and_vectors(monkeypatch):
    module = _load_api_module()
    calls = []

    class _Jobs:
        def update_step(self, _job_id, step, *_args, **_kwargs):
            calls.append(("step", step))

        def complete_step(self, _job_id, step, *_args, **_kwargs):
            calls.append(("complete", step))

        def complete_job(self, *_args, **_kwargs):
            calls.append(("job", "completed"))

        def fail_job(self, *_args, **_kwargs):
            raise AssertionError("upload job unexpectedly failed")

    bundle = {
        "chunks": [
            {"chunk_level": 1, "text": "parent"},
            {"chunk_level": 3, "text": "leaf"},
        ],
        "pages": [{"page_id": "p1", "text": "page"}],
        "tables": [{"table_id": "t1"}],
    }
    monkeypatch.setattr(module, "upload_job_manager", _Jobs())
    monkeypatch.setattr(module, "loader", types.SimpleNamespace(load_document_bundle=lambda *_args: bundle))
    monkeypatch.setattr(module, "milvus_manager", types.SimpleNamespace(
        init_collection=lambda: calls.append(("milvus", "init")),
        delete=lambda _expr: calls.append(("milvus", "delete")),
    ))
    monkeypatch.setattr(module, "_remove_bm25_stats_for_filename", lambda name: calls.append(("bm25", name)))
    monkeypatch.setattr(module, "parent_chunk_store", types.SimpleNamespace(
        delete_by_filename=lambda name: calls.append(("parent_delete", name)),
        upsert_documents=lambda docs: calls.append(("parent_write", len(docs))),
    ))
    monkeypatch.setattr(module, "document_page_store", types.SimpleNamespace(
        delete_by_filename=lambda name: calls.append(("page_delete", name)),
        upsert_pages=lambda pages: calls.append(("page_write", len(pages))),
    ))
    monkeypatch.setattr(module, "table_store", types.SimpleNamespace(
        delete_by_filename=lambda name: calls.append(("table_delete", name)),
    ))
    monkeypatch.setattr(module, "_prepare_table_evidence_docs", lambda *_args: (1, [{"text": "table"}]))
    monkeypatch.setattr(module, "_write_table_evidence_docs_safe", lambda *_args: 1)

    class _Writer:
        def write_documents(self, docs, progress_callback=None):
            calls.append(("vector_write", len(docs)))
            if progress_callback:
                progress_callback(len(docs), len(docs))

    monkeypatch.setattr(module, "milvus_writer", _Writer())

    module._process_upload_job("job-1", "demo.txt", "demo.txt")

    assert ("parent_write", 1) in calls
    assert ("page_write", 1) in calls
    assert ("vector_write", 1) in calls
    assert ("job", "completed") in calls
