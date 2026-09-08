import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_evidence_focus_prompt_safety_regression_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("evidence_focus_safety", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_real_frozen_selection_is_exactly_remaining_58_correct_cases():
    module = load_module()
    rows, _, judges, excluded = module.load_inputs()
    selected = module.select_safety_rows(rows, judges, excluded)
    ids = {row["financebench_id"] for row in selected}
    assert len(ids) == 58
    assert ids.isdisjoint(excluded)
    assert all(judges[question_id]["judge"]["score"] == 1 for question_id in ids)


def test_regression_classifier_is_deterministic():
    module = load_module()
    assert module.classify_regression("无法计算", "") == "unnecessary_refusal"
    assert module.classify_regression("The metric increased", "direction is reversed") == "wrong_direction"
    assert module.classify_regression("answer", "used the wrong metric") == "wrong_metric"
    assert module.classify_regression("ratio is 2", "calculation is wrong") == "wrong_calculation"
    assert module.classify_regression("$4.2 billion per agreement", "does not provide the combined total") == "wrong_calculation"
    assert module.classify_regression("$8.738 billion", "does not match the reference answer") == "wrong_calculation"
    assert module.classify_regression("unsupported fact", "not supported") == "hallucination"


def test_runner_does_not_import_or_call_forbidden_pipeline_components():
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
    forbidden = ("retrieval", "rag_utils", "rag_orchestrator", "query_rewriter", "jina", "reranker")
    assert not any(any(word in name.lower() for word in forbidden) for name in imports)
    assert not {"retrieve", "rerank", "build_context", "rewrite_query"}.intersection(calls)
