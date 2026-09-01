from scripts.evaluate_evidence_assembly_ab import (
    _contains_all,
    _numbers,
    _periods,
    _required_numbers,
    _trusted_table_evidence,
    build_page_text_evidence,
    classify_question,
    gold_row_hit,
    select_rows,
    summarize,
)


def _row(identifier: str, question: str, reasoning: str = "nan", evidence_text: str = "Revenue 100"):
    return {
        "financebench_id": identifier,
        "question": question,
        "question_reasoning": reasoning,
        "evidence": (
            '[{"doc_name":"GENERIC_2024_10K","evidence_page_num":4,'
            f'"evidence_text":"{evidence_text}"}}]'
        ),
    }


def test_question_classification_is_deterministic_and_multi_label():
    row = _row(
        "calc",
        "Compare the FY2024 and FY2023 operating margin and calculate the percentage change.",
        "Numerical reasoning",
        "Metric 2024 2023 Operating margin 20% 18%",
    )

    assert classify_question(row) == {"table_likely", "calculation", "comparison"}


def test_auto_selection_uses_only_frozen_ids_and_stays_within_limit():
    rows = [
        _row("calc", "Calculate the revenue ratio.", "Numerical reasoning"),
        _row("compare", "Which year had higher revenue?"),
        _row("lookup", "What was the disclosed customer name?", evidence_text="Customer Example Corp"),
        _row("outside", "What was revenue?"),
    ]

    selected = select_rows(rows, ["compare", "lookup", "calc"], limit=3)

    assert {item["financebench_id"] for item in selected} == {"calc", "compare", "lookup"}
    assert len(selected) == 3


def test_page_text_baseline_respects_same_context_ceiling():
    pages = [
        {"filename": "a.pdf", "page_number": 1, "page_id": "p1", "page_text": "a" * 1000},
        {"filename": "a.pdf", "page_number": 2, "page_id": "p2", "page_text": "b" * 1000},
    ]

    evidence = build_page_text_evidence(pages, max_context_chars=500)

    assert len(evidence) <= 500
    assert "Page ID: p1" in evidence
    assert "Page ID: p2" in evidence


def test_deterministic_gold_metrics_cover_row_numbers_and_periods():
    gold = [{"evidence_text": "Operating income 24 18"}]
    evidence = "Header 2024 2023\nOperating income 24 18"

    assert gold_row_hit(gold, evidence, required_numbers=["24", "18"]) is True
    assert _contains_all(["24", "18"], evidence, _numbers) is True
    assert _contains_all(["FY2024", "FY2023"], evidence, _periods) is False
    assert _periods("Compare FY2024, FY23 and Q1") == ["2024", "2023", "Q1"]


def test_required_numbers_use_justification_operands_not_derived_result():
    row = {
        "justification": "Quick ratio = (37857 - 2388 - 8358) / 50171 = 0.5403719",
        "answer": "The ratio was 0.54.",
    }

    assert _required_numbers(row) == ["37857", "2388", "8358", "50171"]


def test_gold_row_hit_has_no_whole_page_word_overlap_fallback():
    gold = [{"evidence_text": "Operating income 24 18"}]

    assert gold_row_hit(gold, "Operating income was discussed elsewhere") is False


def test_trusted_table_evidence_excludes_page_fallback_units():
    evidence = (
        "Source: a.pdf, internal page 1\n[Trusted Table Evidence]\nRevenue: 24\n\n"
        "Source: a.pdf, internal page 2\n[Page Text Evidence]\nRevenue: 18"
    )

    trusted = _trusted_table_evidence(evidence)

    assert "Revenue: 24" in trusted
    assert "Revenue: 18" not in trusted


def test_summary_reports_table_coverage_fallback_and_question_types():
    record = {
        "question_types": ["table_likely", "lookup"],
        "trusted_tables": ["t1"],
        "rejected_tables": [],
        "page_text_fallback_count": 1,
        "selected_page_count": 2,
        "baseline_evidence_chars": 100,
        "assembly_evidence_chars": 80,
        "gold_row_table_hit": {"baseline_page_text": True, "assembly_v1": True},
        "required_number_hit": {"baseline_page_text": True, "assembly_v1": True},
        "required_period_hit": {"baseline_page_text": None, "assembly_v1": None},
    }

    summary = summarize([record])

    assert summary["table_evidence_coverage"] == 1.0
    assert summary["page_fallback_ratio"] == 0.5
    assert summary["question_types"]["lookup"]["questions"] == 1
    assert summary["question_types"]["calculation"]["questions"] == 0
