import ast
from pathlib import Path

from backend import prompts
from backend.answer_generator import build_answer_messages, resolve_answer_prompt_mode
from scripts.run_finance_evidence_focus_shadow_v1 import build_summary


ROOT = Path(__file__).resolve().parents[1]


def test_evidence_focus_prompt_inherits_terminology_and_adds_generic_focus_rules():
    messages = build_answer_messages(
        "Question", "Evidence", profile="finance", prompt_mode="finance_evidence_focus_v1",
    )
    system = messages[0].content
    assert prompts.FINANCE_EVIDENCE_FOCUS_V1_PROMPT_VERSION in system
    assert "Financial terminology normalization" in system
    for required in (
        "Identify the entity or company explicitly requested",
        "Ignore unrelated companies, documents, or metrics",
        "checking all relevant chunks for the target entity",
        "Do not mention this internal filtering process",
        "ensure the described direction matches the numerical change",
    ):
        assert required in system
    assert "Adobe" not in system
    assert "FinanceBench" not in system
    assert resolve_answer_prompt_mode("finance_evidence_focus_v1") == "finance_evidence_focus_v1"


def test_summary_counts_recovery_regression_and_tokens():
    def result(ok, tokens, refusal=False):
        return {
            "status": "ok", "usage": {"total_tokens": tokens}, "latency_ms": 10,
            "assessment": {"acceptance_pass": ok, "unnecessary_refusal": refusal},
        }
    payload = {"records": [
        {"question_id": "q1", "results": {
            "clean_baseline_v1": result(True, 100), "finance_terminology_alignment_v1": result(True, 110),
            "finance_evidence_focus_v1": result(True, 120),
        }},
        {"question_id": "q2", "results": {
            "clean_baseline_v1": result(False, 100, True), "finance_terminology_alignment_v1": result(False, 110, True),
            "finance_evidence_focus_v1": result(True, 120),
        }},
        {"question_id": "q3", "results": {
            "clean_baseline_v1": result(True, 100), "finance_terminology_alignment_v1": result(True, 110),
            "finance_evidence_focus_v1": result(False, 120, True),
        }},
    ]}
    summary = build_summary(payload)
    assert summary["recovered_vs_baseline"] == 1
    assert summary["regressions_vs_baseline"] == 1
    assert summary["focus_average_token_delta_vs_baseline"] == 20


def test_shadow_runner_has_no_retrieval_jina_judge_or_langsmith_calls():
    path = ROOT / "scripts" / "run_finance_evidence_focus_shadow_v1.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imported & {
        "rag_orchestrator", "rag_utils", "query_rewriter", "shadow_rerankers_v1",
        "financebench_judge_common", "langsmith",
    }
    assert all(token not in source for token in ("JinaReranker(", "judge_answer(", "retrieve_profile_candidates("))
