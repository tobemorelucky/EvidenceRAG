from evidence_coverage import assess_structured_coverage


def _frame(evidence_id, label, value, *, period="2024", currency="USD", scale="millions", scope="consolidated"):
    return {
        "evidence_id": evidence_id,
        "company": "Example Co",
        "table_id": "statement-1",
        "row_label": label,
        "period": period,
        "normalized_value": value,
        "currency": currency,
        "scale": scale,
        "scope": scope,
    }


def test_calculation_complete_requires_valid_structured_operands():
    task = {
        "task_type": "calculation",
        "company": "Example Co",
        "required_fields": ["operating_income", "revenue"],
        "required_periods": ["2024"],
        "formula": "operating_income / revenue",
    }
    frames = [_frame("ef_1", "Operating income", "25"), _frame("ef_2", "Revenue", "100")]

    result = assess_structured_coverage(task, [{"filename": "report.pdf"}], frames, {"status": "partial"})

    assert result["page_supported"] is True
    assert result["row_supported"] is True
    assert result["period_supported"] is True
    assert result["unit_scale_supported"] is True
    assert result["scope_supported"] is True
    assert result["operands_validated"] is True
    assert result["answerable"] is True
    assert result["status"] == "complete"


def test_unknown_period_and_units_remain_partial_not_complete():
    task = {
        "task_type": "calculation",
        "company": "Example Co",
        "required_fields": ["operating_income", "revenue"],
        "required_periods": ["2024"],
        "formula": "operating_income / revenue",
    }
    frames = [
        _frame("ef_1", "Operating income", "25", period=None, currency=None, scale=None, scope=None),
        _frame("ef_2", "Revenue", "100", period=None, currency=None, scale=None, scope=None),
    ]

    result = assess_structured_coverage(task, [{"filename": "report.pdf"}], frames, {"status": "complete"})

    assert result["row_supported"] is True
    assert result["period_supported"] is False
    assert result["unit_scale_supported"] is False
    assert result["scope_supported"] is False
    assert result["operands_validated"] is False
    assert result["answerable"] is False
    assert result["status"] == "partial"


def test_comparison_requires_every_requested_period_for_every_field():
    task = {
        "task_type": "comparison",
        "company": "Example Co",
        "required_fields": ["revenue"],
        "required_periods": ["2024", "2023"],
    }
    complete = [_frame("ef_1", "Revenue", "100", period="2024"), _frame("ef_2", "Revenue", "90", period="2023")]
    incomplete = complete[:1]

    complete_result = assess_structured_coverage(task, [{}], complete, {"status": "complete"})
    incomplete_result = assess_structured_coverage(task, [{}], incomplete, {"status": "complete"})

    assert complete_result["answerable"] is True
    assert incomplete_result["period_supported"] is False
    assert incomplete_result["answerable"] is False


def test_text_lookup_preserves_existing_coverage_semantics_without_frames():
    result = assess_structured_coverage(
        {"task_type": "lookup", "required_fields": ["revenue"]},
        [{"filename": "report.pdf"}],
        [],
        {"status": "complete", "matched_fields": {"revenue": "revenue"}},
    )

    assert result["coverage_basis"] == "text_lookup"
    assert result["answerable"] is True
    assert result["row_supported"] is None


def test_selection_requires_more_than_one_structured_candidate():
    task = {"task_type": "selection", "required_fields": ["revenue"]}
    one = [_frame("ef_1", "Revenue", "100", period="2024")]
    two = [*one, _frame("ef_2", "Revenue", "90", period="2023")]

    assert assess_structured_coverage(task, [{}], one, {"status": "complete"})["answerable"] is False
    assert assess_structured_coverage(task, [{}], two, {"status": "complete"})["answerable"] is True
