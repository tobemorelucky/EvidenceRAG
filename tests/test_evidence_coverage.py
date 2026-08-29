from evidence_coverage import (
    assess_stage_coverage,
    assess_structured_coverage,
    coverage_transition_reason,
    protect_selected_page_slots,
)


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
    assert result["base_status"] == "partial"


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
    assert result["base_answerable"] is True
    assert result["structured_answerable"] is False
    assert result["structured_execution_ready"] is False
    assert result["structured_status"] == "partial"
    assert result["answerable"] is True
    assert result["status"] == "complete"


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
    assert incomplete_result["structured_answerable"] is False
    assert incomplete_result["answerable"] is True


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


def test_selection_requires_more_than_one_structured_candidate(monkeypatch):
    monkeypatch.setenv("FRAME_ALIGNMENT_ENABLED", "true")
    task = {
        "task_type": "selection",
        "target_measure": "revenue",
        "candidate_dimension": "region",
        "selection_direction": "max",
        "required_periods": ["2024"],
    }
    one = [_frame("ef_1", "Revenue", "100", period="2024")]
    one[0]["row_label"] = "North revenue"
    one[0]["descriptor"] = "North revenue | region revenue"
    second = _frame("ef_2", "South revenue", "90", period="2024")
    second["descriptor"] = "South revenue | region revenue"
    two = [*one, second]

    one_result = assess_structured_coverage(task, [{}], one, {"status": "complete"})
    two_result = assess_structured_coverage(task, [{}], two, {"status": "complete"})
    assert one_result["structured_answerable"] is False
    assert one_result["answerable"] is True
    assert two_result["structured_answerable"] is True
    assert two_result["answerable"] is True


def test_missing_frames_preserve_complete_base_path():
    result = assess_structured_coverage(
        {
            "task_type": "calculation",
            "required_fields": ["operating_income", "revenue"],
            "required_periods": ["2024"],
            "formula": "operating_income / revenue",
        },
        [{"filename": "report.pdf"}],
        [],
        {"status": "complete", "matched_fields": {"operating_income": "operating income", "revenue": "revenue"}},
    )

    assert result["base_answerable"] is True
    assert result["structured_answerable"] is False
    assert result["structured_execution_ready"] is False
    assert result["answerable"] is True
    assert result["status"] == "complete"


def test_advisory_mode_can_be_disabled_for_exact_legacy_structured_semantics(monkeypatch):
    monkeypatch.setenv("STRUCTURED_COVERAGE_ADVISORY_ENABLED", "false")
    result = assess_structured_coverage(
        {"task_type": "calculation", "required_fields": ["revenue"], "formula": "revenue"},
        [{"filename": "report.pdf"}],
        [],
        {"status": "complete"},
    )

    assert result["base_answerable"] is True
    assert result["structured_answerable"] is False
    assert result["answerable"] is False
    assert result["status"] == "insufficient"


def test_stage_coverage_tracks_requirement_loss_without_gold_pages():
    task = {
        "task_type": "comparison",
        "required_fields": ["revenue"],
        "required_concepts": ["revenue"],
        "required_periods": ["2024", "2023"],
        "target_measure": "revenue",
        "target_measure_explicit": True,
    }
    candidates = [
        {"filename": "report.pdf", "page_number": 4, "text": "FY2024 Revenue $120 million."},
        {"filename": "report.pdf", "page_number": 5, "text": "FY2023 Revenue $100 million."},
    ]
    selected = [candidates[0]]

    candidate = assess_stage_coverage(task, candidates, stage="candidate")
    final = assess_stage_coverage(task, selected, stage="selected_page")

    assert candidate["status"] == "complete"
    assert candidate["coverage_basis"] == "query_spec_runtime_no_gold"
    assert candidate["diagnostic_confidence"] == "field_bound"
    assert final["status"] == "partial"
    assert final["missing_comparison_sides"] == ["2023"]
    assert coverage_transition_reason(candidate, final) == "coverage_lost"


def test_candidate_diagnosis_distinguishes_document_miss_from_page_requirement_miss():
    task = {
        "task_type": "lookup",
        "company": "example_co",
        "company_confidence": 0.98,
        "required_fields": ["revenue"],
        "required_concepts": ["revenue"],
    }
    wrong_document = assess_stage_coverage(
        task,
        [{"filename": "other_company_2024_10K.pdf", "page_number": 2, "text": "Revenue $100"}],
        stage="candidate",
    )
    right_document_wrong_page = assess_stage_coverage(
        task,
        [{"filename": "example_co_2024_10K.pdf", "page_number": 2, "text": "Corporate overview"}],
        stage="candidate",
    )

    assert wrong_document["candidate_miss_diagnosis"] == "target_document_not_hit"
    assert right_document_wrong_page["candidate_miss_diagnosis"] == "target_document_hit_requirement_page_not_hit"


def test_protected_page_slot_recovers_comparison_side_without_growing_page_budget(monkeypatch):
    monkeypatch.setenv("PROTECTED_PAGE_SLOTS_ENABLED", "true")
    task = {
        "task_type": "comparison",
        "required_fields": ["revenue"],
        "required_concepts": ["revenue"],
        "required_periods": ["2024", "2023"],
        "target_measure": "revenue",
        "target_measure_explicit": True,
    }
    candidates = [
        {"filename": "report.pdf", "page_number": 4, "text": "FY2024 Revenue $120 million.", "score": 1.0},
        {"filename": "report.pdf", "page_number": 5, "text": "FY2023 Revenue $100 million.", "score": 0.8},
    ]
    selected = [
        candidates[0],
        {"filename": "report.pdf", "page_number": 9, "text": "Corporate background.", "score": 0.9},
    ]

    protected, trace = protect_selected_page_slots(task, candidates, selected)

    assert len({(item["filename"], item["page_number"]) for item in protected}) == 2
    assert {item["page_number"] for item in protected} == {4, 5}
    assert trace["protected_page_count"] == 1
    assert trace["protected_page_replacements"][0]["replaced_page"] == "report.pdf#page=9"
    assert trace["coverage_after"]["status"] == "complete"


def test_lookup_without_explicit_target_does_not_protect_page(monkeypatch):
    monkeypatch.setenv("PROTECTED_PAGE_SLOTS_ENABLED", "true")
    task = {
        "task_type": "lookup",
        "required_fields": [],
        "required_concepts": ["customer"],
        "target_measure": "customer",
        "target_measure_explicit": False,
    }
    candidates = [{"filename": "report.pdf", "page_number": 7, "text": "Primary customer was government."}]
    selected = [{"filename": "report.pdf", "page_number": 2, "text": "Overview."}]

    protected, trace = protect_selected_page_slots(task, candidates, selected)

    assert protected == selected
    assert trace["protected_page_count"] == 0


def test_calculation_does_not_require_or_protect_derived_target_phrase(monkeypatch):
    monkeypatch.setenv("PROTECTED_PAGE_SLOTS_ENABLED", "true")
    task = {
        "task_type": "calculation",
        "required_fields": ["net_income", "total_assets"],
        "required_concepts": ["net income", "total assets", "on assets"],
        "target_measure": "on assets",
        "target_measure_explicit": True,
        "required_periods": ["2022", "2021"],
        "formula": "net_income / average(total_assets)",
    }
    operand_page = {
        "filename": "report.pdf",
        "page_number": 8,
        "text": "2022 2021 Net income $100 $90 Total assets $1000 $900",
    }
    derived_phrase_page = {
        "filename": "report.pdf",
        "page_number": 20,
        "text": "A narrative discussion about return on assets.",
    }

    coverage = assess_stage_coverage(task, [operand_page], stage="selected_page")
    protected, trace = protect_selected_page_slots(task, [operand_page, derived_phrase_page], [operand_page])

    assert coverage["status"] == "complete"
    assert all("concept_2" not in item for item in coverage["missing_requirements"])
    assert protected == [operand_page]
    assert trace["protected_page_count"] == 0


def test_stage_coverage_uses_generic_token_overlap_for_question_derived_measure():
    task = {
        "task_type": "comparison",
        "required_fields": [],
        "required_concepts": ["year-over-year change in unadjusted operating income"],
        "required_periods": ["2015", "2016"],
    }
    document = {
        "filename": "report.pdf",
        "page_number": 61,
        "text": "Income statement 2016 2015 Operating income 1,493,602 903,095",
    }

    coverage = assess_stage_coverage(task, [document], stage="candidate")

    assert coverage["status"] == "complete"
    assert coverage["missing_concepts"] == []
    assert coverage["diagnostic_confidence"] == "lexical_concept_only"


def test_selection_without_explicit_measure_never_protects_question_phrase_page(monkeypatch):
    monkeypatch.setenv("PROTECTED_PAGE_SLOTS_ENABLED", "true")
    task = {
        "task_type": "selection",
        "required_fields": [],
        "required_concepts": ["which business performed best"],
        "target_measure": "which business performed best",
        "target_measure_explicit": False,
    }
    candidates = [{"filename": "report.pdf", "page_number": 7, "text": "Business performance overview."}]
    selected = [{"filename": "report.pdf", "page_number": 2, "text": "Corporate overview."}]

    protected, trace = protect_selected_page_slots(task, candidates, selected)

    assert protected == selected
    assert trace["protected_page_count"] == 0


def test_explicit_formula_operands_recover_missing_page_without_growing_budget(monkeypatch):
    monkeypatch.setenv("EXPLICIT_FORMULA_ADVISORY_ENABLED", "true")
    monkeypatch.setenv("PROTECTED_PAGE_SLOTS_ENABLED", "true")
    task = {
        "task_type": "comparison",
        "required_fields": ["inventory"],
        "required_periods": ["2023", "2024"],
        "explicit_formula_confidence": 1.0,
        "explicit_formula_source": "question_explicit_definition",
        "explicit_formula_periods": ["2023", "2024"],
        "explicit_formula_operands": [
            {
                "key": "accounts_payable", "field": "accounts_payable",
                "label": "accounts payable", "aliases": ["accounts payable"],
                "periods": ["2023", "2024"], "transform": "average",
            },
            {
                "key": "cogs", "field": "", "label": "COGS", "aliases": ["COGS"],
                "periods": ["2024"], "transform": "direct",
            },
            {
                "key": "inventory", "field": "inventory", "label": "inventory",
                "aliases": ["inventory", "inventories"],
                "periods": ["2023", "2024"], "transform": "change",
            },
        ],
    }
    combined_page = {
        "filename": "report.pdf", "page_number": 10,
        "text": "2024 2023 Accounts payable 120 100; 2024 COGS 900", "score": 1.0,
    }
    inventory_page = {
        "filename": "report.pdf", "page_number": 11,
        "text": "2024 2023 Inventories 80 70", "score": 0.8,
    }
    noise_page = {
        "filename": "report.pdf", "page_number": 2,
        "text": "2024 2023 Corporate overview", "score": 0.9,
    }

    candidate = assess_stage_coverage(task, [combined_page, inventory_page], stage="candidate")
    before = assess_stage_coverage(task, [combined_page, noise_page], stage="selected_page")
    protected, trace = protect_selected_page_slots(
        task, [combined_page, inventory_page], [combined_page, noise_page],
    )

    assert candidate["status"] == "complete"
    assert before["status"] == "partial"
    assert "inventory" in before["missing_operands"]
    assert trace["coverage_after"]["status"] == "complete"
    assert trace["protected_page_count"] == 1
    assert trace["protected_page_replacements"][0]["protected_page_reason"] == "operand:inventory"
    assert len({(item["filename"], item["page_number"]) for item in protected}) == 2


def test_explicit_formula_advisory_flag_preserves_original_coverage_when_disabled(monkeypatch):
    monkeypatch.setenv("EXPLICIT_FORMULA_ADVISORY_ENABLED", "false")
    task = {
        "task_type": "comparison",
        "required_fields": ["inventory"],
        "required_periods": ["2024"],
        "explicit_formula_confidence": 1.0,
        "explicit_formula_operands": [{
            "key": "accounts_payable", "field": "accounts_payable",
            "label": "accounts payable", "aliases": ["accounts payable"], "periods": ["2024"],
        }],
    }

    coverage = assess_stage_coverage(
        task,
        [{"filename": "report.pdf", "page_number": 8, "text": "2024 Inventory 10"}],
        stage="candidate",
    )

    assert coverage["status"] == "complete"
    assert coverage["explicit_formula_advisory_enabled"] is False
    assert coverage["required_fields"] == ["inventory"]
    assert coverage["required_operands"] == []
