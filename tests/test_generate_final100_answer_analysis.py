import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/generate_final100_answer_analysis.py"


def load_module():
    spec = importlib.util.spec_from_file_location("final100_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_real_final100_report_contains_all_questions_and_complete_answers():
    module = load_module()
    inputs = module.load_inputs(module.DEFAULT_INPUT, module.DEFAULT_DATASET, module.DEFAULT_ATTRIBUTION)
    report, summary = module.build_report(inputs)
    assert summary["questions"] == 100
    assert summary["correct"] == 68
    assert summary["incorrect"] == 32
    assert summary["validation"] == {
        "all_100_questions_present_once": True,
        "correct_plus_incorrect_equals_100": True,
        "question_reference_model_answer_present": True,
    }
    assert report.count("### Question\n") == 100
    assert report.count("### Reference Answer\n") == 100
    assert report.count("### Model Answer\n") == 100


def test_stage_failure_fallback_uses_frozen_funnel_fields():
    module = load_module()
    assert module.stage_failure_category(
        {"context_metrics": {"context_hit": False}},
        {"retrieval_metrics": {"candidate_gold_page_hit": False}},
        {"rerank_metrics": {"context_hit": False}},
    ) == "retrieval_miss"
    assert module.stage_failure_category(
        {"context_metrics": {"context_hit": False}},
        {"retrieval_metrics": {"candidate_gold_page_hit": True}},
        {"rerank_metrics": {"context_hit": False}},
    ) == "rerank_miss"
    assert module.stage_failure_category(
        {"context_metrics": {"context_hit": False}},
        {"retrieval_metrics": {"candidate_gold_page_hit": True}},
        {"rerank_metrics": {"context_hit": True}},
    ) == "context_loss"


def test_script_does_not_import_or_call_external_pipeline_components():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            target = node.func
            calls.append(target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else "")
    forbidden = ("langchain", "openai", "jina", "retrieval", "rag_orchestrator", "answer_generator", "langsmith")
    assert not any(any(word in name.lower() for word in forbidden) for name in imports)
    assert not {"invoke", "retrieve", "rerank", "generate_answer", "build_context"}.intersection(calls)

