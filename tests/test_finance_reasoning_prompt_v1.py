import ast
from pathlib import Path

import pytest

from backend import prompts
from backend.answer_generator import build_answer_messages, resolve_answer_prompt_mode
from scripts.run_finance_reasoning_prompt_smoke import select_sample


ROOT = Path(__file__).resolve().parents[1]


def test_default_mode_preserves_clean_baseline_prompt(monkeypatch):
    monkeypatch.delenv("ANSWER_PROMPT_MODE", raising=False)
    messages = build_answer_messages("Question?", "Evidence", profile="clean_baseline")
    assert prompts.CLEAN_BASELINE_PROMPT_VERSION in messages[0].content
    assert prompts.CLEAN_BASELINE_ANSWER_SYSTEM_PROMPT in messages[0].content
    assert "Finance Reasoning" not in messages[0].content


def test_finance_reasoning_mode_has_generic_financial_reasoning_contract():
    messages = build_answer_messages("Question?", "Evidence", profile="clean_baseline", prompt_mode="finance_reasoning")
    system = messages[0].content
    assert prompts.FINANCE_REASONING_PROMPT_VERSION in system
    for required in ("direct lookup", "calculation", "comparison", "trend analysis", "financial judgment",
                     "PP&E", "Property, Plant and Equipment", "SG&A", "EPS", "Preserve negative signs",
                     "numeric conclusion equals the displayed formula", "Do not refuse merely"):
        assert required in system
    assert "hidden chain-of-thought" in system
    assert messages[-1].content.count("Evidence") >= 1


def test_prompt_mode_environment_and_validation(monkeypatch):
    monkeypatch.setenv("ANSWER_PROMPT_MODE", "finance_reasoning")
    assert resolve_answer_prompt_mode() == "finance_reasoning"
    assert resolve_answer_prompt_mode("baseline") == "baseline"
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_answer_prompt_mode("unknown")


def test_smoke_sample_is_deterministic_and_requires_frozen_evidence():
    records = [{"financebench_id": str(i), "answer_status": "ok", "evidence": f"e{i}"} for i in range(5)]
    assert select_sample(records, 3, 7) == select_sample(records, 3, 7)
    with pytest.raises(ValueError):
        select_sample([{"answer_status": "ok", "evidence": ""}], 1, 7)


def test_smoke_runner_has_no_retrieval_reranker_or_judge_calls():
    tree = ast.parse((ROOT / "scripts/run_finance_reasoning_prompt_smoke.py").read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imported & {"milvus_client", "rag_orchestrator", "shadow_rerankers_v1", "financebench_judge_common"}
    source = (ROOT / "scripts/run_finance_reasoning_prompt_smoke.py").read_text(encoding="utf-8")
    assert all(token not in source for token in ("JinaReranker(", "hybrid_retrieve(", "judge_answer("))
