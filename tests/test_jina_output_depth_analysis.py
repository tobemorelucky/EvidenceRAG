import json
from pathlib import Path

from scripts.analyze_jina_output_depth import INPUT_DEPTHS, OUTPUT_DEPTHS, analyze, markdown


ROOT = Path(__file__).resolve().parents[1]


def test_output_depth_analysis_replays_both_profiles_without_network():
    snapshot = json.loads((ROOT / "reports/reranker_shadow_v1_rrf_top120.json").read_text(encoding="utf-8"))
    cache = json.loads((ROOT / "reports/reranker_shadow_v1.json").read_text(encoding="utf-8"))
    report = analyze(snapshot, cache)
    assert tuple(report["jina_input_depths"]) == INPUT_DEPTHS
    assert tuple(report["output_depths"]) == OUTPUT_DEPTHS
    assert all(report["results"][str(input_k)][str(output_k)]["questions"] == 30
               for input_k in INPUT_DEPTHS for output_k in OUTPUT_DEPTHS)
    assert report["results"]["120"]["12"]["context_hit"] >= report["results"]["120"]["8"]["context_hit"]
    assert "JINA_INPUT_K=120" in markdown(report) and "JINA_INPUT_K=80" in markdown(report)


def test_output_depth_analyzer_contains_no_api_or_model_call():
    source = (ROOT / "scripts/analyze_jina_output_depth.py").read_text(encoding="utf-8")
    assert all(token not in source for token in ("requests.", "JinaReranker(", "generate_answer", "judge_answer"))
