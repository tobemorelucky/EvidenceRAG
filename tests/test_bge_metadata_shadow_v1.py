import copy
import inspect

import pytest

from scripts.bge_metadata_reranker_v1 import rerank, years, resolve_entities, BGE_WEIGHT, METADATA_WEIGHT
from scripts.evaluate_bge_metadata_shadow_v1 import input_view, summarize_comparison


def base(n):
    return [{"index": i, "score": float(n - i)} for i in range(n)]


def unit(company="Aster Labs", text="Throughput 2022: 400 units", report_year=2023):
    return {"company": company, "text": text, "report_year": report_year}


def test_period_parsing_and_report_year_is_not_fact_period():
    assert years("FY22 versus FY2021 and 2020") == {2020, 2021, 2022}
    _, trace = rerank("What was Aster Labs throughput in 2022?", [unit()], base(1))
    item = trace["units"][0]
    assert item["period_status"] == "local_cooccurrence"
    assert item["active_features"]["period"] == 1
    assert item["report_years"] == [2023]


def test_report_year_only_weak_and_missing_is_not_conflict():
    chunks = [unit(text="Throughput 400 units", report_year=2022), unit(text="Throughput 400 units", report_year=2023)]
    _, trace = rerank("Aster Labs throughput in 2022?", chunks, base(2))
    assert trace["units"][0]["active_features"]["period"] == 0.25
    assert trace["units"][1]["active_features"]["period"] == 0
    assert trace["units"][1]["period_status"] == "unknown_or_not_visible"


def test_company_matching_generic_and_no_substring_alias_invention():
    targets, _ = resolve_entities("What was Aster Labs Inc.'s throughput?", [unit(), unit("Other Co")])
    assert targets == {"asterlabs"}
    assert resolve_entities("Forecast outlook", [unit("Cast")])[0] == set()
    assert resolve_entities("AL throughput", [unit()])[0] == set()
    assert resolve_entities("Aster Labs compared with Other Co", [unit(), unit("Other Co")])[0] == {"asterlabs", "other"}


def test_entity_mismatch_and_unknown_distinct():
    _, trace = rerank("Aster Labs throughput?", [unit(), unit("Other"), unit("")], base(3))
    assert [t["active_features"]["entity"] for t in trace["units"]] == [1, -1, 0]


def test_all_candidates_retained_and_inputs_unchanged():
    chunks = [unit(company=f"Entity {i}") for i in range(120)]
    original = copy.deepcopy(chunks)
    ranked, trace = rerank("Entity 20 throughput in 2022?", chunks, base(120))
    assert len(ranked) == 120 and {r["index"] for r in ranked} == set(range(120))
    assert chunks == original
    assert BGE_WEIGHT == 0.75 and METADATA_WEIGHT == 0.25
    assert trace["model_calls"] == 0


def test_unknown_features_preserve_bge_order():
    chunks = [{"text": "Unrelated source"} for _ in range(4)]
    ranked, trace = rerank("the", chunks, list(reversed(base(4))))
    assert [r["index"] for r in ranked] == [0, 1, 2, 3]
    assert all(u["metadata_compatibility"] == 0 for u in trace["units"])


def test_gold_and_benchmark_fields_have_no_effect():
    chunks = [unit(), unit("Other")]
    expected = rerank("Aster Labs throughput in 2022?", chunks, base(2))
    for c in chunks:
        c.update(financebench_id="gold-rule", evidence_text="preferred answer", answer="400", gold_page=999)
    assert rerank("Aster Labs throughput in 2022?", chunks, base(2)) == expected
    assert set(input_view(chunks[0])) == {"text", "company", "report_year", "section", "table_title"}
    assert set(inspect.signature(rerank).parameters) == {"question", "chunks", "bge_ranked"}


def test_invalid_scores_rejected():
    with pytest.raises(ValueError):
        rerank("Q", [unit()], [{"index": 0, "score": float("nan")}])
    with pytest.raises(ValueError):
        rerank("Q", [unit(), unit()], [{"index": 0, "score": 1}])


def test_no_financial_alias_is_inferred():
    _, trace = rerank("What is the quasar ratio?", [unit(text="Cash 500 and liabilities 100")], base(1))
    assert trace["units"][0]["metric_relevance"] == 0


def test_gains_and_regressions_are_paired_not_net_only():
    def record(i, before, after):
        return {"question_id": i, "group": "selection_loss10", "routes": {
            "bge": {"metrics": {"context_hit": before}}, "bge_metadata_v1": {"metrics": {"context_hit": after}}}}
    result = summarize_comparison([record("gain", False, True), record("loss", True, False)])
    assert result["all"]["gains"] == ["gain"]
    assert result["all"]["regressions"] == ["loss"]


def test_no_model_or_network_or_production_constructors_in_entry():
    import ast
    from pathlib import Path
    tree = ast.parse(Path("scripts/evaluate_bge_metadata_shadow_v1.py").read_text(encoding="utf-8"))
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not calls & {"BGEReranker", "JinaReranker", "Client", "OpenAI", "MilvusManager"}
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(m and m.startswith(("backend", "langsmith", "transformers", "torch")) for m in imports)
