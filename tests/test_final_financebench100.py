from pathlib import Path

from scripts.run_final_financebench100 import CONFIG, _rate, write_state


def test_final_config_is_frozen():
    assert CONFIG["query_rewrite"] == {"enabled": True, "max_rewrites": 2, "original_retained": True}
    assert CONFIG["retrieval"]["dense_top_k"] == CONFIG["retrieval"]["bm25_top_k"] == 240
    assert CONFIG["retrieval"]["rrf_top_k"] == 120
    assert CONFIG["reranker"]["input_k"] == 80 and CONFIG["reranker"]["output_k"] == 12
    assert CONFIG["context"] == {"top_k": 12, "max_chars": 28000}
    assert CONFIG["answer"]["prompt_mode"] == "baseline"


def test_state_is_only_written_at_ten_or_force(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.run_final_financebench100.manifest", lambda: {"stable": True})
    records = [{"status": "ok", "rewrite": {"usage": {"total_tokens": 1}}} for _ in range(9)]
    write_state(tmp_path, "recall", records, 1.0)
    assert not (tmp_path / "state.json").exists()
    records.append({"status": "error"})
    write_state(tmp_path, "recall", records, 2.0)
    assert (tmp_path / "state.json").exists()


def test_rate_ignores_missing_values():
    assert _rate([True, None, False]) == 0.5


def test_final_runner_has_no_production_retrieval_import():
    source = (Path(__file__).resolve().parents[1] / "scripts/run_final_financebench100.py").read_text(encoding="utf-8")
    assert "rag_utils" not in source and "rag_orchestrator" not in source
