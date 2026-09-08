import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUTER = ROOT / "backend/evidence_focus_semantic_router_v2.py"
SCRIPT = ROOT / "scripts/evaluate_evidence_focus_semantic_gating_shadow_v2.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_reasoning_and_multiperiod_questions_trigger_focus():
    router = load(ROUTER, "semantic_router")
    assert router.route_evidence_focus_semantic("Calculate the operating margin growth") ["use_evidence_focus"] is True
    route = router.route_evidence_focus_semantic("What happened in FY2021 and FY2022?")
    assert route["use_evidence_focus"] is True
    assert "multiple_fiscal_periods" in route["reasons"]
    assert router.route_evidence_focus_semantic("从 FY2020 到 FY2022 是否改善？")["use_evidence_focus"] is True


def test_plain_lookup_stays_on_clean_baseline():
    router = load(ROUTER, "semantic_router_lookup")
    route = router.route_evidence_focus_semantic("What was revenue in FY2022?")
    assert route == {"use_evidence_focus": False, "reasons": [], "lookup_only": True}


def test_shadow_replay_uses_exact_frozen_90():
    module = load(SCRIPT, "semantic_gating")
    source, _ = module.load_replay_inputs()
    replayed = module.replay(source)
    assert len(replayed) == 90
    assert sum(record["baseline"]["judge"]["score"] == 1 for record in replayed) == 68


def test_shadow_script_has_no_model_or_pipeline_calls():
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

