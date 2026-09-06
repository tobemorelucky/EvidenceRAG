import importlib.util
from pathlib import Path

from backend.evidence_compression_shadow_v1 import compress_financial_evidence_v1


def _fact(entity, period, metric, value, line, *, flags=None):
    return {
        "entity": entity, "period": period, "metric": metric, "value": value, "unit": "USD million",
        "source_span": {"document": "ACME.pdf", "page": 4, "chunk_id": "c1", "line_number": line, "text": f"{metric} {value}"},
        "ambiguity_flags": flags or [],
    }


def test_filters_other_entity_unrelated_period_and_competing_metric():
    summary = {
        "target_metric": "gross margin",
        "financial_evidence_summary": [
            _fact("Acme", "2022", "gross margin", "30%", 1),
            _fact("Other", "2022", "gross margin", "40%", 2),
            _fact("Acme", "2019", "gross margin", "20%", 3),
            _fact("Acme", "2022", "operating margin", "10%", 4),
        ],
    }
    result = compress_financial_evidence_v1("Was Acme's gross margin improving in FY2022?", summary)
    assert len(result["kept_facts"]) == 1
    assert result["kept_facts"][0]["value"] == "30%"
    assert result["drop_reasons"] == {"other_entity": 1, "competing_metric": 1, "unrelated_period": 1}


def test_keeps_prior_period_from_same_financial_row():
    current = _fact("Acme", "2022", "revenue", "$10", 1, flags=["operand_for_target_metric"])
    prior = _fact("Acme", "2021", "revenue", "$8", 1, flags=["operand_for_target_metric"])
    prior["source_span"]["text"] = current["source_span"]["text"] = "Revenue 2022 $10 2021 $8"
    summary = {"target_metric": "revenue growth", "financial_evidence_summary": [current, prior]}
    result = compress_financial_evidence_v1("What was Acme's revenue growth in FY2022?", summary)
    assert {fact["period"] for fact in result["kept_facts"]} == {"2022", "2021"}
    assert result["kept_span_count"] == 1


def test_output_preserves_exact_source_span():
    fact = _fact("Acme", "2022", "gross margin", "30%", 7)
    summary = {"target_metric": "gross margin", "financial_evidence_summary": [fact]}
    result = compress_financial_evidence_v1("What was Acme's gross margin in FY2022?", summary)
    assert "gross margin 30%" in result["text"]
    assert "ACME.pdf | Page: 4 | Chunk: c1 | Line: 7" in result["text"]


def test_quarter_requirement_does_not_accept_bare_year():
    summary = {
        "target_metric": "business separation",
        "financial_evidence_summary": [_fact("Acme", "2023", "business separation", None, 1)],
    }
    result = compress_financial_evidence_v1("As of Q2 2023, is Acme spinning off a business?", summary)
    assert result["kept_facts"] == []
    assert result["drop_reasons"] == {"unrelated_period": 1}


def test_runner_contains_no_forbidden_service_calls():
    script = Path(__file__).resolve().parents[1] / "scripts/run_evidence_compression_shadow_v1.py"
    source = script.read_text(encoding="utf-8")
    assert "JinaReranker(" not in source
    assert "judge_answer(" not in source
    assert "from langsmith" not in source
    assert "retrieve(" not in source


def test_activity_choice_word_forms_are_equivalent():
    script = Path(__file__).resolve().parents[1] / "scripts/run_evidence_compression_shadow_v1.py"
    spec = importlib.util.spec_from_file_location("compression_shadow", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    result = module.compare_compressed_answers(
        "reasoning_failure",
        "In 2022, AMD brought in the most cashflow from Operations",
        "Investing activities brought in the most cash flow for AMD.",
        "Operating activities brought in the most cash flow for AMD in 2022.",
    )
    assert result["recovered"] is True
