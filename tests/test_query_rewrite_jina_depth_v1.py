import json
from pathlib import Path

from scripts.analyze_query_rewrite_jina_depth_v1 import DEPTHS, analyze, markdown


ROOT = Path(__file__).resolve().parents[1]


def test_depth_replay_uses_complete_frozen_cache_without_network():
    candidates = json.loads((ROOT / "reports/query_rewrite_shadow_v1.json").read_text(encoding="utf-8"))
    jina = json.loads((ROOT / "reports/query_rewrite_jina_shadow_v1.json").read_text(encoding="utf-8"))
    report = analyze(candidates, jina)
    assert tuple(report["input_depths"]) == DEPTHS
    assert report["questions"] == 30 and report["output_k"] == 12
    assert report["network_calls"] == {"jina": 0, "llm": 0, "judge": 0}
    assert report["summary"]["120"]["token_saving_vs_120"] == 0
    assert "未调用Jina、LLM或Judge" in markdown(report)


def test_depth_analyzer_has_no_model_or_network_client():
    source = (ROOT / "scripts/analyze_query_rewrite_jina_depth_v1.py").read_text(encoding="utf-8")
    assert all(token not in source for token in ("requests.", "JinaReranker(", "generate_answer", "judge_answer", "init_chat_model"))
