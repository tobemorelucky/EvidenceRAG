import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_evidence_focus_prompt_regression_shadow_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evidence_focus_regression", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_selection_is_seeded_and_contains_required_groups():
    module = load_module()
    rows, answers, judges, _ = module.load_frozen_inputs()
    first = module.select_experiment_rows(rows, answers, judges, seed=123)
    second = module.select_experiment_rows(rows, answers, judges, seed=123)
    assert [row["financebench_id"] for row in first] == [row["financebench_id"] for row in second]
    assert sum(row["selection_group"] == "answer_failure22" for row in first) == 22
    assert sum(row["selection_group"] == "correct_regression10" for row in first) == 10


def test_summary_computes_transitions_token_gate_and_failure_recovery():
    module = load_module()
    records = []
    for index in range(7):
        baseline_correct = index >= 5
        focus_correct = index not in {4, 6}
        records.append(
            {
                "question_id": f"q{index}",
                "selection_group": "answer_failure22" if not baseline_correct else "correct_regression10",
                "question": "q",
                "reference_answer": "r",
                "context_hit": True,
                "baseline_failure_category": "E" if index < 3 else "F",
                "baseline": {
                    "answer": "无法计算" if index == 0 else "old",
                    "usage": {"total_tokens": 1000},
                    "judge": {"score": int(baseline_correct)},
                    "refusal_language": index == 0,
                },
                "evidence_focus": {
                    "answer_status": "ok",
                    "answer": "new",
                    "usage": {"total_tokens": 1200},
                    "judge_status": "ok",
                    "judge": {"score": int(focus_correct)},
                    "refusal_language": False,
                },
            }
        )
    summary = module.summarize(records)
    assert summary["transitions"]["newly_correct"]["count"] == 4
    assert summary["transitions"]["regressed"]["count"] == 1
    assert summary["answer_tokens"]["average_delta"] == 200
    assert summary["unnecessary_refusal"] == {"baseline": 1, "evidence_focus": 0, "delta": -1}
    assert summary["failure_type_change"]["calculation_execution_error"]["recovered"] == 3


def test_runner_has_no_retrieval_or_context_builder_imports():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Call):
            target = node.func
            calls.append(target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else "")
    forbidden_imports = ("rag_utils", "rag_orchestrator", "query_rewriter", "retrieval")
    assert not any(any(word in name for word in forbidden_imports) for name in imported)
    assert not {"retrieve", "build_context", "rerank"}.intersection(calls)
