import ast
import importlib.util
from pathlib import Path

from backend import prompts
from backend.answer_generator import build_answer_messages, resolve_answer_prompt_mode


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_finance_reasoning_prompt_v1_1_shadow.py"


def load_shadow_module():
    spec = importlib.util.spec_from_file_location("finance_reasoning_prompt_v1_1_shadow", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v1_1_is_independent_mode_and_keeps_prior_modes():
    assert resolve_answer_prompt_mode("baseline") == "baseline"
    assert resolve_answer_prompt_mode("finance_reasoning") == "finance_reasoning"
    assert resolve_answer_prompt_mode("finance_reasoning_v1_1") == "finance_reasoning_v1_1"


def test_v1_1_prompt_contains_generic_contract_without_internal_plan_output():
    messages = build_answer_messages(
        "Question?", "Evidence", profile="clean_baseline", prompt_mode="finance_reasoning_v1_1"
    )
    system = messages[0].content
    assert prompts.FINANCE_REASONING_V1_1_PROMPT_VERSION in system
    for required in (
        "silently plan", "lookup", "calculation", "comparison", "trend analysis",
        "financial judgment", "PP&E", "SG&A", "COGS", "EPS", "Gross margin",
        "operating margin", "requested company and reporting period", "Do not reveal",
        "Do not refuse merely",
    ):
        assert required in system
    assert "FinanceBench" not in system


def test_resolution_diagnostic_is_conservative_and_non_judge():
    module = load_shadow_module()
    resolved = module.diagnose_resolution("calculation_failure", "0.83", "Result: 0.83")
    refused = module.diagnose_resolution("calculation_failure", "0.83", "I cannot calculate this.")
    excluded = module.diagnose_resolution("other", "Yes", "Yes")
    assert resolved["status"] == "likely_resolved"
    assert refused["status"] == "not_resolved"
    assert excluded["status"] == "not_applicable"


def test_resolution_diagnostic_rejects_wrong_selection_and_chinese_refusal():
    module = load_shadow_module()
    wrong = module.diagnose_resolution(
        "reasoning_failure",
        "Operations brought in the most cash flow.",
        "Investing brought in the most. Operating activities were also positive.",
    )
    refused = module.diagnose_resolution(
        "refusal_failure", "The ratio is zero.", "无法计算该比率，因为证据缺少数据。"
    )
    assert wrong["status"] == "not_resolved"
    assert refused["status"] == "not_resolved"


def test_resolution_diagnostic_requires_reference_qualifiers_and_names():
    module = load_shadow_module()
    wrong_sign = module.diagnose_resolution(
        "refusal_failure", "Adjusted EBIT is negative, so coverage is zero.",
        "Adjusted EBITDAR gives a coverage ratio of 5.88x.",
    )
    wrong_names = module.diagnose_resolution(
        "reasoning_failure", "Trillium, Array, and Therachon", "GBT, Biohaven, and Arena",
    )
    assert wrong_sign["status"] == "not_resolved"
    assert wrong_names["status"] == "not_resolved"


def test_incomplete_shadow_summary_handles_pending_modes():
    module = load_shadow_module()
    payload = {
        "manifest": {"prompt_modes": ["baseline", "finance_reasoning"]},
        "records": [{
            "results": {
                "baseline": {
                    "status": "ok", "usage": {"input_tokens": 100}, "latency_ms": 10,
                    "resolution": {"status": "likely_resolved"},
                }
            }
        }],
    }
    summary = module.build_summary(payload)
    assert summary["modes"]["finance_reasoning"]["average_input_tokens"] is None
    assert summary["modes"]["finance_reasoning"]["average_input_token_increase_vs_baseline"] is None


def test_summary_counts_known_baseline_failure_as_not_resolved():
    module = load_shadow_module()
    payload = {
        "manifest": {"prompt_modes": ["baseline"]},
        "records": [{
            "results": {
                "baseline": {
                    "status": "ok", "usage": {"input_tokens": 100}, "latency_ms": 10,
                    "resolution": {"status": "not_resolved"},
                }
            }
        }],
    }
    assert module.build_summary(payload)["modes"]["baseline"]["likely_resolved"] == 0


def test_shadow_script_has_no_retrieval_reranker_judge_or_langsmith_calls():
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imported & {
        "milvus_client", "rag_orchestrator", "shadow_rerankers_v1",
        "financebench_judge_common", "langsmith",
    }
    assert all(token not in source for token in ("JinaReranker(", "hybrid_retrieve(", "judge_answer("))
