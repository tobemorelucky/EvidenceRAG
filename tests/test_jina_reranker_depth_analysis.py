import json
from pathlib import Path

from scripts.analyze_jina_reranker_depth import DEPTHS, analyze, markdown


ROOT = Path(__file__).resolve().parents[1]


def test_cached_depth_analysis_is_complete_offline_and_top120_is_exact_cache_usage():
    snapshot = json.loads((ROOT / "reports/reranker_shadow_v1_rrf_top120.json").read_text(encoding="utf-8"))
    cache = json.loads((ROOT / "reports/reranker_shadow_v1.json").read_text(encoding="utf-8"))
    report = analyze(snapshot, cache, output_k=8)
    assert tuple(report["depths"]) == DEPTHS and report["questions"] == 30
    assert all(report["summary"][str(depth)]["questions"] == 30 for depth in DEPTHS)
    cached_tokens = sum((record["routes"]["jina"]["trace"].get("usage") or {}).get("total_tokens", 0)
                        for record in cache["records"])
    top120 = report["summary"]["120"]["estimated_token_cost"]
    assert top120["tokens"] == cached_tokens
    assert top120["relative_to_cached_top120"] == 1
    assert "未调用 Jina" in markdown(report)


def test_depth_analyzer_source_contains_no_network_or_model_client():
    source = (ROOT / "scripts/analyze_jina_reranker_depth.py").read_text(encoding="utf-8")
    assert all(token not in source for token in ("requests.", "JinaReranker(", "generate_answer", "judge_answer"))
