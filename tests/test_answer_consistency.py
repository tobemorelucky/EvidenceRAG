from calculation_service import validate_or_repair_structured_answer


def _authoritative(task_type, **extra):
    return {
        "authoritative": True,
        "task_type": task_type,
        "executor": "evidence_frame",
        "citations": ["[source: report.pdf, page 8]"],
        **extra,
    }


def test_calculation_conflict_is_repaired_without_another_model_call(monkeypatch):
    monkeypatch.setenv("ANSWER_CONSISTENCY_VALIDATOR_ENABLED", "true")
    answer, trace = validate_or_repair_structured_answer(
        "The result is 0.40.",
        {"task_type": "calculation"},
        _authoritative("calculation", result="0.25", formula="25 / 100", result_unit="decimal"),
    )

    assert "0.25" in answer
    assert "report.pdf" in answer
    assert trace["repaired"] is True
    assert trace["reason"] == "calculation_result_mismatch"


def test_matching_calculation_answer_is_not_rewritten(monkeypatch):
    monkeypatch.setenv("ANSWER_CONSISTENCY_VALIDATOR_ENABLED", "true")
    original = "The validated ratio is 25%."
    answer, trace = validate_or_repair_structured_answer(
        original,
        {"task_type": "calculation"},
        _authoritative("calculation", result="0.25", result_unit="percent"),
    )

    assert answer == original
    assert trace["checked"] is True
    assert trace["repaired"] is False


def test_comparison_direction_conflict_is_repaired(monkeypatch):
    monkeypatch.setenv("ANSWER_CONSISTENCY_VALIDATOR_ENABLED", "true")
    calculation = _authoritative(
        "comparison",
        comparison_direction="decreased",
        candidate_matrix=[
            {"period": "2024", "normalized_value": "90", "scale": "millions"},
            {"period": "2023", "normalized_value": "100", "scale": "millions"},
        ],
    )

    answer, trace = validate_or_repair_structured_answer(
        "Revenue increased in 2024.",
        {"task_type": "comparison"},
        calculation,
    )

    assert "decreased" in answer
    assert trace["reason"] == "comparison_direction_mismatch"


def test_selection_checks_the_conclusion_not_a_later_candidate_list(monkeypatch):
    monkeypatch.setenv("ANSWER_CONSISTENCY_VALIDATOR_ENABLED", "true")
    calculation = _authoritative(
        "selection",
        selected_entity="South region",
        result="120",
        unit="millions",
    )

    answer, trace = validate_or_repair_structured_answer(
        "North region was highest. South region reported 120.",
        {"task_type": "selection"},
        calculation,
    )

    assert answer.startswith("South region")
    assert trace["reason"] == "selection_entity_mismatch"


def test_validator_is_diagnostic_only_without_authoritative_result(monkeypatch):
    monkeypatch.setenv("ANSWER_CONSISTENCY_VALIDATOR_ENABLED", "true")
    original = "The evidence is incomplete."
    answer, trace = validate_or_repair_structured_answer(original, {"task_type": "comparison"}, None)

    assert answer == original
    assert trace["checked"] is False
    assert trace["repaired"] is False
