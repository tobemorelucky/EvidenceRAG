from backend.evidence_intent_alignment_v1 import (
    align_context_chunk_v1,
    build_frozen_context_chunks_v1,
    classify_context_alignment_v1,
    extract_question_intent_v1,
)


def test_extracts_entity_period_metric_and_calculation_type():
    intent = extract_question_intent_v1("What was MGM's interest coverage ratio in FY22?")
    assert intent["entity_candidates"][0]["value"] == "MGM"
    assert intent["period_candidates"][0]["value"] == "2022"
    assert intent["metric_candidates"][0]["value"] == "interest coverage"
    assert intent["calculation_type"] == "calculation"


def test_builds_chunks_only_from_actual_frozen_evidence_text():
    evidence = "Source: MGM.pdf | Page: 2\nFY2022 interest expense was $20."
    metadata = [{"filename": "MGM.pdf", "page_number": 2, "company": "MGM", "report_year": 2022, "text": "SECRET"}]
    chunks = build_frozen_context_chunks_v1(evidence, metadata)
    assert len(chunks) == 1
    assert chunks[0]["company"] == "MGM"
    assert "SECRET" not in chunks[0]["text"]


def test_aligned_chunk_matches_all_three_dimensions():
    intent = extract_question_intent_v1("What was MGM's interest coverage ratio in FY2022?")
    chunk = {"chunk_id": "c1", "document": "MGM.pdf", "page": 2, "company": "MGM", "report_year": 2022,
             "text": "FY2022 Adjusted EBIT was $100 and interest expense was $20."}
    result = align_context_chunk_v1(intent, chunk)
    assert result["entity_match"] and result["period_match"] and result["metric_match"]
    assert result["aligned"] is True
    assert result["alignment_score"] == 1.0


def test_classifies_present_but_wrong_entity_as_misaligned():
    intent = extract_question_intent_v1("What was MGM's interest coverage ratio in FY2022?")
    chunks = [{"chunk_id": "c1", "document": "OTHER.pdf", "page": 1, "company": "OTHER", "report_year": 2022,
               "text": "FY2022 interest coverage was 4.0."}]
    result = classify_context_alignment_v1(intent, chunks)
    assert result["classification"] == "B_evidence_present_but_misaligned"


def test_classifies_absent_and_aligned_contexts():
    intent = extract_question_intent_v1("What was MGM's interest coverage ratio in FY2022?")
    absent = [{"chunk_id": "a", "document": "MGM.pdf", "page": 1, "company": "MGM", "report_year": 2022, "text": "Revenue was $10."}]
    assert classify_context_alignment_v1(intent, absent)["classification"] == "A_evidence_absent"
    aligned = [{"chunk_id": "b", "document": "MGM.pdf", "page": 2, "company": "MGM", "report_year": 2022, "text": "Interest expense was $2."}]
    assert classify_context_alignment_v1(intent, aligned)["classification"] == "C_evidence_aligned_but_reasoning_failure"


def test_quarter_year_is_preserved_at_quarter_precision():
    intent = extract_question_intent_v1("Was there a change in Q2 of FY2024?")
    assert [item["value"] for item in intent["period_candidates"]] == ["2024Q2"]


def test_generic_entity_syntax_and_document_quarter_are_supported():
    assert extract_question_intent_v1("Did Ulta Beauty's wages expense increase?")["entity_candidates"][0]["value"] == "Ulta Beauty"
    assert extract_question_intent_v1("Which companies were acquired by Pfizer mentioned here?")["entity_candidates"][0]["value"] == "Pfizer"
    intent = extract_question_intent_v1("As of Q2'2023, is Pfizer spinning off a segment?")
    chunks = [{"chunk_id": "p", "document": "Pfizer_2023Q2_10Q.pdf", "page": 1, "company": "PFIZER", "report_year": 2023,
               "text": "The planned separation includes a business segment."}]
    assert classify_context_alignment_v1(intent, chunks)["aligned_evidence_present"] is True


def test_metric_intent_does_not_expand_from_operand_words_alone():
    intent = extract_question_intent_v1("What is the operating cash flow ratio using current liabilities?")
    assert [item["value"] for item in intent["metric_candidates"]] == ["operating cash flow ratio"]
    wages = extract_question_intent_v1("Did wages expense as a percent of net sales increase?")
    assert [item["value"] for item in wages["metric_candidates"]] == ["wages expense as percent of sales"]
