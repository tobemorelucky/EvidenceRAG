import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_agent_tools():
    fake_page_store = types.ModuleType("document_page_store")
    fake_page_store.DocumentPageStore = type("DocumentPageStore", (), {})
    previous = sys.modules.get("document_page_store")
    sys.modules["document_page_store"] = fake_page_store
    path = ROOT / "backend" / "agent_tools.py"
    spec = importlib.util.spec_from_file_location("agent_tools_contracts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("document_page_store", None)
        else:
            sys.modules["document_page_store"] = previous
    return module


def _load_embedding_module():
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    previous_dotenv = sys.modules.get("dotenv")
    sys.modules["dotenv"] = dotenv
    huggingface = types.ModuleType("langchain_huggingface")
    huggingface.HuggingFaceEmbeddings = object
    previous_huggingface = sys.modules.get("langchain_huggingface")
    sys.modules["langchain_huggingface"] = huggingface
    path = ROOT / "backend" / "embedding.py"
    spec = importlib.util.spec_from_file_location("embedding_contracts", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_dotenv is None:
            sys.modules.pop("dotenv", None)
        else:
            sys.modules["dotenv"] = previous_dotenv
        if previous_huggingface is None:
            sys.modules.pop("langchain_huggingface", None)
        else:
            sys.modules["langchain_huggingface"] = previous_huggingface
    return module


def test_prompt_contract_requires_cited_grounded_answers():
    from backend import prompts

    assert "Answer only from the evidence" in prompts.ANSWER_SYSTEM_PROMPT
    assert "[source: filename, page N]" in prompts.ANSWER_SYSTEM_PROMPT
    assert "hidden chain-of-thought" in prompts.ANSWER_SYSTEM_PROMPT
    assert "Do not refuse merely because the final metric is not explicitly stated" in prompts.ANSWER_SYSTEM_PROMPT
    assert "same company, reporting period, unit, and accounting scope" in prompts.ANSWER_SYSTEM_PROMPT
    assert "including Corporate/Other rows and negative values" in prompts.ANSWER_SYSTEM_PROMPT
    assert "do not substitute a geographic, operating, or reportable segment value" in prompts.ANSWER_SYSTEM_PROMPT
    assert "Declare it irrelevant only when the supplied evidence explicitly supports" in prompts.ANSWER_SYSTEM_PROMPT
    assert "an explicitly labeled Total row" in prompts.ANSWER_SYSTEM_PROMPT
    assert "Keep full precision for intermediate arithmetic" in prompts.ANSWER_SYSTEM_PROMPT
    assert "do not apply a universal threshold across industries" in prompts.ANSWER_SYSTEM_PROMPT
    assert "never emit a draft conclusion followed by a correction" in prompts.ANSWER_SYSTEM_PROMPT
    assert "Best Buy" not in prompts.ANSWER_SYSTEM_PROMPT
    assert prompts.PROMPT_VERSION


def test_decimal_calculator_allows_only_safe_arithmetic():
    tools = _load_agent_tools()

    assert tools.calculate_decimal("(3909 - 3673) / 3673 * 100").startswith("6.425265450585")
    try:
        tools.calculate_decimal("__import__('os').system('whoami')")
        assert False, "unsafe expression must be rejected"
    except ValueError as exc:
        assert "invalid calculation" in str(exc)


def test_legacy_sparse_tokenizer_preserves_financial_numbers():
    module = _load_embedding_module()
    service = object.__new__(module.EmbeddingService)

    tokens = service.tokenize("Revenue was $1,234.50, up 12.5% in 2023.")

    assert "$1,234.50" in tokens
    assert "12.5%" in tokens
    assert "2023" in tokens


def test_chat_schema_exposes_evidence_metadata():
    from backend.schemas import ChatRequest, ChatResponse

    request = ChatRequest(message="test", profile="finance", execution_mode="auto")
    response = ChatResponse(
        response="grounded",
        execution_mode="static",
        route_reason="single_evidence_question",
        evidence_status="sufficient",
        citations=[{"id": "e1", "filename": "report.pdf", "page_number": 3}],
        trace_id="trace-1",
    )

    assert request.profile == "finance"
    assert response.citations[0].filename == "report.pdf"
