import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "backend/evidence_focus_prompt_router_v1.py"
SCRIPT = ROOT / "scripts/evaluate_evidence_focus_prompt_gating_shadow_v1.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_router_exposes_all_generic_reasons():
    router = load(ROUTER, "focus_router")
    route = router.route_evidence_focus_prompt(
        "Compare the growth and operating margin trend",
        context_chars=16_000,
        candidate_evidence_chunks=8,
    )
    assert route["use_evidence_focus"] is True
    assert set(route["reasons"]) == {
        "trend_or_comparison_language",
        "calculation_metric_language",
        "long_context_gt_15000",
        "candidate_evidence_chunks_gte_8",
    }


def test_router_keeps_simple_short_lookup_on_baseline():
    router = load(ROUTER, "focus_router_short")
    assert router.route_evidence_focus_prompt(
        "Who signed the report?", context_chars=5_000, candidate_evidence_chunks=3
    ) == {"use_evidence_focus": False, "reasons": []}


def test_frozen_replay_has_exactly_90_unique_cases():
    module = load(SCRIPT, "focus_gating")
    records, _ = module.load_replay_inputs()
    assert len(records) == 90
    assert len({record["question_id"] for record in records}) == 90
    assert sum(record["baseline"]["judge"]["score"] == 1 for record in records) == 68


def test_evaluator_has_no_external_pipeline_or_model_calls():
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
    forbidden = ("retrieval", "query_rewrite", "jina", "rerank", "answer_generator", "judge_common", "langchain")
    assert not any(any(word in name.lower() for word in forbidden) for name in imports)
    assert not {"retrieve", "rerank", "build_context", "generate_answer", "invoke"}.intersection(calls)

