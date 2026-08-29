from calculation_service import validate_numeric_display, validate_or_repair_structured_answer


def _authoritative(task_type, **extra):
    return {
        "authoritative": True,
        "task_type": task_type,
        "executor": "evidence_frame",
        "citations": ["[source: report.pdf, page 8]"],
        "execution_contract": {"passed": True, "failure_reasons": []},
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


def test_percentage_change_validator_requires_numeric_result_and_direction(monkeypatch):
    monkeypatch.setenv("ANSWER_CONSISTENCY_VALIDATOR_ENABLED", "true")
    calculation = _authoritative(
        "comparison",
        operation="percentage_change",
        result="65.4",
        display_result="65.4",
        unit="percent",
        comparison_direction="increased",
        candidate_matrix=[
            {"period": "2015", "normalized_value": "903095", "scale": "thousands"},
            {"period": "2016", "normalized_value": "1493602", "scale": "thousands"},
        ],
    )

    answer, trace = validate_or_repair_structured_answer(
        "Operating income increased.",
        {"task_type": "comparison", "operation": "percentage_change"},
        calculation,
    )

    assert "65.4%" in answer
    assert "increased" in answer
    assert trace["reason"] == "comparison_result_mismatch"


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


def test_validator_ignores_failed_execution_contract_even_if_authoritative_flag_is_stale(monkeypatch):
    monkeypatch.setenv("ANSWER_CONSISTENCY_VALIDATOR_ENABLED", "true")
    original = "Operating income increased by 65.4%."
    calculation = _authoritative(
        "comparison",
        operation="compare",
        comparison_direction="decreased",
        execution_contract={"passed": False, "failure_reasons": ["operation_mismatch"]},
    )

    answer, trace = validate_or_repair_structured_answer(
        original,
        {"task_type": "comparison", "operation": "percentage_change"},
        calculation,
    )

    assert answer == original
    assert trace["checked"] is False
    assert trace["reason"] == "execution_contract_failed"


def test_numeric_display_validator_repairs_only_explicit_rounded_conclusion(monkeypatch):
    monkeypatch.setenv("NUMERIC_DISPLAY_VALIDATOR_ENABLED", "true")
    task = {
        "task_type": "calculation",
        "required_fields": ["capital_expenditures", "revenue"],
        "formula": "capital_expenditures / revenue",
        "operation": "divide",
        "result_unit": "percent",
        "rounding_decimal_places": 1,
    }
    calculation = {
        "status": "calculated",
        "source": "structured_row_decimal",
        "formula": "capital_expenditures / revenue",
        "result": "0.01915",
        "operands": {
            "capital_expenditures": {"value": "134", "filename": "report.pdf", "page_number": 72},
            "revenue": {"value": "7002", "filename": "report.pdf", "page_number": 69},
        },
    }

    answer, trace = validate_numeric_display(
        "**2.0%**\n\nCalculation: 134 / 7002 = 0.01915.",
        task,
        calculation,
    )

    assert answer.startswith("**1.9%**")
    assert "134 / 7002 = 0.01915" in answer
    assert trace["eligible"] is True
    assert trace["repaired"] is True


def test_numeric_display_validator_does_not_apply_without_explicit_rounding(monkeypatch):
    monkeypatch.setenv("NUMERIC_DISPLAY_VALIDATOR_ENABLED", "true")
    task = {"task_type": "lookup", "required_fields": ["ppe"], "rounding_decimal_places": None}
    calculation = {
        "status": "calculated",
        "formula": "ppe",
        "result": "8.738",
        "operands": {"ppe": {"value": "8738", "filename": "report.pdf", "page_number": 57}},
    }

    answer, trace = validate_numeric_display("The result is **8.738 billion**.", task, calculation)

    assert answer == "The result is **8.738 billion**."
    assert trace["eligible"] is False
    assert trace["reason"] == "explicit_rounding_not_requested"


def test_numeric_display_validator_does_not_replace_operand_when_conclusion_is_implicit(monkeypatch):
    monkeypatch.setenv("NUMERIC_DISPLAY_VALIDATOR_ENABLED", "true")
    task = {
        "task_type": "calculation",
        "required_fields": ["revenue"],
        "formula": "revenue",
        "rounding_decimal_places": 1,
    }
    calculation = {
        "status": "calculated",
        "formula": "revenue",
        "result": "100.04",
        "operands": {"revenue": {"value": "100.04", "filename": "report.pdf", "page_number": 8}},
    }
    original = "Revenue of 100.04 was reported on page 8."

    answer, trace = validate_numeric_display(original, task, calculation)

    assert answer == original
    assert trace["repaired"] is False
    assert trace["reason"] == "explicit_conclusion_number_not_found"


def test_numeric_display_validator_never_overlaps_authoritative_result(monkeypatch):
    monkeypatch.setenv("NUMERIC_DISPLAY_VALIDATOR_ENABLED", "true")
    answer, trace = validate_numeric_display(
        "**2.0%**",
        {"rounding_decimal_places": 1},
        {"authoritative": True, "status": "calculated", "formula": "a/b", "result": "0.019"},
    )

    assert answer == "**2.0%**"
    assert trace["eligible"] is False
    assert trace["reason"] == "authoritative_result_uses_structured_validator"
