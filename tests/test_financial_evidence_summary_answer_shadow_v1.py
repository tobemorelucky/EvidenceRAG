import importlib.util
from copy import deepcopy
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_financial_evidence_summary_answer_shadow_v1.py"
SPEC = importlib.util.spec_from_file_location("financial_summary_answer_shadow", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _fact():
    return {
        "entity": "Acme", "period": "2022", "metric": "revenue", "value": "$10",
        "unit": "USD million",
        "source_span": {
            "document": "ACME_2022_10K.pdf", "page": 7, "chunk_id": "chunk-1",
            "line_number": 12, "text": "Revenue $10 million",
        },
        "ambiguity_flags": ["operand_for_target_metric"],
    }


def test_summary_serialization_preserves_exact_source_span_and_location():
    text = MODULE.format_summary_context([_fact()])
    assert "Revenue $10 million" in text
    assert "ACME_2022_10K.pdf | Page: 7 | Chunk: chunk-1 | Line: 12" in text
    assert "operand_for_target_metric" in text


def test_combined_context_keeps_original_context_byte_for_byte_as_prefix():
    original = "Source: ACME_2022_10K.pdf | Page: 7\nOriginal evidence."
    facts = [_fact()]
    copied = deepcopy(facts)
    combined = MODULE.combine_context(original, facts)
    assert combined.startswith(original)
    assert "Financial Evidence Summary" in combined[len(original):]
    assert facts == copied


def test_empty_summary_does_not_change_context():
    original = "frozen context"
    assert MODULE.combine_context(original, []) == original


def test_comparison_marks_reference_aligned_answer_as_likely_recovered():
    result = MODULE.compare_answers(
        "calculation_failure", "The ratio was 1.25.",
        "The ratio was 0.75.", "The ratio was 1.25.",
    )
    assert result["status"] == "likely_recovered"
    assert result["recovered"] is True
    assert result["regression"] is False


def test_comparison_detects_new_refusal_and_numeric_regression():
    result = MODULE.compare_answers(
        "reasoning_failure", "Revenue increased by 12%.",
        "Revenue increased by 12%.", "The evidence is insufficient; cannot determine it.",
    )
    assert result["recovered"] is False
    assert result["regression"] is True
    assert "new_refusal" in result["regression_reasons"]


def test_runner_has_no_forbidden_service_imports_or_calls():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "JinaReranker(" not in source
    assert "judge_answer(" not in source
    assert "from langsmith" not in source
    assert "import langsmith" not in source
    assert "retrieve(" not in source
