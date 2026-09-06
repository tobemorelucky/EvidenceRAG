from backend.evidence_guidance_shadow_v1 import build_evidence_guidance_v1, estimate_guidance_tokens


def _fact(entity="Acme", period="2022", metric="gross margin", value="30%", line=1):
    return {
        "entity": entity, "period": period, "metric": metric, "value": value, "unit": "percent",
        "source_span": {"document": "ACME.pdf", "page": 4, "chunk_id": "c1", "line_number": line, "text": f"{metric} {value}"},
        "ambiguity_flags": [],
    }


def _frame():
    return {"frame": {
        "entity_candidates": [
            {"value": "Acme", "question_match": True}, {"value": "OtherCo", "question_match": False},
        ],
        "period_candidates": [
            {"value": "2022", "requested": True}, {"value": "2019", "requested": False},
        ],
    }}


def test_guidance_has_target_relevant_span_forbidden_confusion_and_formula():
    summary = {"target_metric": "gross margin", "financial_evidence_summary": [_fact()]}
    schema = {"schema": {
        "recognized": True, "operation_type": "divide", "formula": "gross profit / revenue * 100",
        "required_operands": [{"label": "gross profit"}, {"label": "revenue"}],
    }}
    result = build_evidence_guidance_v1("What was Acme's gross margin in FY2022?", summary, _frame(), schema)
    assert "entity=Acme" in result["text"]
    assert "source_span=\"gross margin 30%\"" in result["text"]
    assert "operating margin" in result["text"]
    assert "OtherCo" in result["text"]
    assert "2019" in result["text"]
    assert "gross profit / revenue * 100" in result["text"]


def test_guidance_excludes_unaligned_fact_from_relevant_evidence():
    summary = {
        "target_metric": "gross margin",
        "financial_evidence_summary": [_fact(), _fact(entity="OtherCo", value="SECRET_WRONG_ENTITY", line=2)],
    }
    result = build_evidence_guidance_v1("What was Acme's gross margin in FY2022?", summary, _frame(), {"schema": {"recognized": False}})
    assert "SECRET_WRONG_ENTITY" not in result["text"]
    assert len(result["selected_facts"]) == 1


def test_guidance_never_exceeds_800_estimated_tokens():
    facts = [_fact(value=str(index), line=index) for index in range(100)]
    summary = {"target_metric": "gross margin", "financial_evidence_summary": facts}
    result = build_evidence_guidance_v1("What was Acme's gross margin in FY2022?", summary, _frame(), {"schema": {"recognized": False}})
    assert result["estimated_tokens"] <= 800
    assert estimate_guidance_tokens(result["text"]) <= 800
    assert len(result["selected_facts"]) <= 6


def test_no_reference_or_gold_fields_are_consumed():
    summary = {
        "target_metric": "gross margin", "financial_evidence_summary": [_fact()],
        "reference_answer": "SHOULD_NOT_APPEAR", "gold_evidence": "GOLD_SHOULD_NOT_APPEAR",
    }
    result = build_evidence_guidance_v1("What was Acme's gross margin in FY2022?", summary, _frame(), {"schema": {"recognized": False}})
    assert "SHOULD_NOT_APPEAR" not in result["text"]
    assert "GOLD_SHOULD_NOT_APPEAR" not in result["text"]

