import ast
from pathlib import Path

from backend import prompts
from backend.answer_generator import build_answer_messages, resolve_answer_prompt_mode
from scripts.run_finance_terminology_alignment_shadow_v1 import CASES, assess_answer


ROOT = Path(__file__).resolve().parents[1]


def test_new_prompt_adds_only_generic_terminology_and_trend_rules():
    messages = build_answer_messages(
        "中文财务问题", "English filing evidence", profile="finance",
        prompt_mode="finance_terminology_alignment_v1",
    )
    system = messages[0].content
    assert prompts.FINANCE_TERMINOLOGY_ALIGNMENT_V1_PROMPT_VERSION in system
    for required in (
        "total current liabilities = current liabilities total",
        "operating income = operating profit",
        "revenue = sales",
        "cash from operations = net cash provided by operating activities",
        "gross profit != operating income",
        "operating income != net income",
        "EBIT != EBITDA",
        "verify that the numerical comparison supports the stated direction",
    ):
        assert required in system
    assert "Adobe" not in system
    assert "FinanceBench" not in system


def test_explicit_clean_baseline_mode_is_preserved():
    messages = build_answer_messages(
        "Question", "Evidence", profile="finance", prompt_mode="clean_baseline_v1",
    )
    assert prompts.CLEAN_BASELINE_PROMPT_VERSION in messages[0].content
    assert prompts.CLEAN_BASELINE_ANSWER_SYSTEM_PROMPT in messages[0].content
    assert "Financial terminology normalization" not in messages[0].content
    assert resolve_answer_prompt_mode("clean_baseline_v1") == "clean_baseline_v1"


def test_shadow_acceptance_detects_recovery_refusal_and_direction():
    fy2015 = CASES[0]
    assert assess_answer("计算结果为 0.66。", fy2015)["acceptance_pass"]
    refused = assess_answer("无法计算；若计算则为 0.66。", fy2015)
    assert refused["required_number_hit"] and refused["unnecessary_refusal"]
    assert not refused["acceptance_pass"]
    trend = assess_answer("FY2021 为36.76%，FY2022 为34.63%，因此经营利润率下降。", CASES[2])
    assert trend["acceptance_pass"]
    assert not assess_answer("FY2021 为36.76%，FY2022 为34.63%，因此有所改善。", CASES[2])["acceptance_pass"]


def test_shadow_runner_has_no_retrieval_jina_judge_or_langsmith_calls():
    path = ROOT / "scripts" / "run_finance_terminology_alignment_shadow_v1.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imported & {
        "rag_orchestrator", "rag_utils", "query_rewriter", "shadow_rerankers_v1",
        "financebench_judge_common", "langsmith",
    }
    assert all(token not in source for token in ("JinaReranker(", "judge_answer(", "retrieve_profile_candidates("))
