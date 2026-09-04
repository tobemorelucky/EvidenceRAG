import copy
import inspect
import re
from unittest.mock import patch

import pytest

from scripts.bge_reranker_input_builder_v1 import build_input, merge_spans, render, terms
from scripts.evaluate_bge_input_builder_v1 import check_resources, validate_prepared, visible_gold_span
from scripts.evaluate_reranker_shadow_v1 import digest, metrics


class WordTokenizer:
    def count(self, text):
        return len(re.findall(r"\S+", text))

    def pair_count(self, query, text):
        return self.count(query) + self.count(text) + 4


def test_short_input_exactly_unchanged_including_numbers_and_whitespace():
    text = "Report 2024\n\n  Amount ($ millions)   -1,234.56  (72.0)  3.5%\n"
    result = build_input("What changed in 2024?", text, WordTokenizer())
    assert result["text"] == text and not result["changed"]
    assert result["omitted_source_chars"] == 0


def test_late_fact_survives_with_header_and_exact_pair_budget():
    query = "What was the zephyr adjustment in 2024?"
    text = "Report 2024\nUnit: millions\n" + "Unrelated explanatory sentence.\n" * 500
    text += "2024 2023\nZephyr adjustment -1,234.56 72.0\nRelated note uses unchanged units.\n"
    result = build_input(query, text, WordTokenizer())
    assert result["changed"] and result["input_pair_tokens"] <= 1024
    assert "Zephyr adjustment -1,234.56 72.0" in result["text"]
    assert "Unit: millions" in result["text"] and "2024 2023" in result["text"]
    assert render(text, result["source_spans"]) == result["text"]
    assert result["omitted_source_chars"] > 0


def test_no_overlap_is_deterministic_and_not_empty():
    text = "Source facts 123 456.\n" * 1000
    a = build_input("Unknown concept", text, WordTokenizer())
    b = build_input("Unknown concept", text, WordTokenizer())
    assert a == b and a["text"].strip() and a["input_pair_tokens"] <= 1024


def test_long_single_line_split_preserves_number_strings():
    text = "Preface " * 1800 + "measure -8,123.456% " + "trailing " * 500
    result = build_input("What is the measure?", text, WordTokenizer())
    assert "-8,123.456%" in result["text"]
    assert result["forced_long_row_splits"] > 0
    assert result["input_pair_tokens"] <= 1024
    assert render(text, result["source_spans"]) == result["text"]


@pytest.mark.parametrize("query,text", [("", "text"), ("query", ""), ("word " * 1020, "x")])
def test_invalid_query_fails_instead_of_truncating_question(query, text):
    with pytest.raises(ValueError):
        build_input(query, text, WordTokenizer())


def test_span_merge_and_no_company_or_gold_input_interface():
    assert merge_spans([[5, 10], [0, 6], [10, 12], [20, 21]]) == [[0, 12], [20, 21]]
    assert set(inspect.signature(build_input).parameters) == {"question", "source_text", "tokenizer", "max_length"}
    assert "2024" in terms("What was the value in 2024?")


def prepared_fixture():
    view = build_input("Q", "Source text", WordTokenizer())
    view.update(index=0, chunk_id="c")
    views = [{**copy.deepcopy(view), "index": i, "chunk_id": str(i)} for i in range(120)]
    chunks = [{"chunk_id": str(i), "text": "Source text"} for i in range(120)]
    source = {"candidate_sha256": digest(chunks), "chunks": chunks}
    record = {"candidate_sha256": source["candidate_sha256"], "inputs": views, "inputs_sha256": digest(views)}
    return record, source


def test_prepared_view_tamper_and_identity_drift_rejected():
    record, source = prepared_fixture()
    validate_prepared(record, source)
    record["inputs"][0]["text"] = "Injected reference answer"
    with pytest.raises(ValueError, match="fingerprint"):
        validate_prepared(record, source)
    record["inputs_sha256"] = digest(record["inputs"])
    with pytest.raises(ValueError, match="verbatim"):
        validate_prepared(record, source)


def test_offline_gold_never_changes_builder_and_pages_are_zero_based():
    import json
    text = "A sufficiently long complete statement giving a measured amount for this period."
    chunk = {"filename": "sample.pdf", "page_number": 4, "text": text}
    row = {"evidence": json.dumps([{"doc_name": "sample", "evidence_page_num": 4, "evidence_text": text}])}
    view = build_input("What amount?", text, WordTokenizer())
    assert visible_gold_span(row, chunk, view["text"])
    assert not visible_gold_span(row, {**chunk, "page_number": 3}, view["text"])
    assert metrics(row, [chunk])["context_evidence_span_hit"]
    assert not metrics(row, [{**chunk, "text": "tiny input view"}])["context_evidence_span_hit"]


def test_resource_guard_never_stops_processes():
    from types import SimpleNamespace
    with patch("psutil.virtual_memory", return_value=SimpleNamespace(available=512 * 1024**2)):
        with pytest.raises(RuntimeError, match="checkpoint saved"):
            check_resources(4)


def test_eval_entry_has_no_remote_or_production_calls():
    import ast
    from pathlib import Path
    tree = ast.parse(Path("scripts/evaluate_bge_input_builder_v1.py").read_text(encoding="utf-8"))
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(m and (m.startswith("backend") or m.startswith("langsmith")) for m in imports)
    calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    assert "JinaReranker" not in calls


def test_local_tokenizer_real_pair_budget_and_offsets():
    from pathlib import Path
    from scripts.bge_reranker_input_builder_v1 import LocalPairTokenizer
    model_file = Path("models/bge-reranker-v2-m3/tokenizer.json")
    if not model_file.exists():
        pytest.skip("Optional local tokenizer weights not installed")
    tokenizer = LocalPairTokenizer(model_file)
    query = "What was the zephyr adjustment in 2024?"
    text = "Annual report 2024\nAmounts in millions\n" + "Unrelated explanatory passage.\n" * 700
    text += "2024 2023\nZephyr adjustment -1,234.56 72.0\n"
    visible = tokenizer.baseline_visible_text(query, text, 1024)
    assert "Zephyr adjustment" not in visible
    assert tokenizer.pair_count(query, text) > 1024  # truncation reset
    view = build_input(query, text, tokenizer)
    assert tokenizer.pair_count(query, view["text"]) == view["input_pair_tokens"] <= 1024
    assert "Zephyr adjustment -1,234.56 72.0" in view["text"]
    assert render(text, view["source_spans"]) == view["text"]
