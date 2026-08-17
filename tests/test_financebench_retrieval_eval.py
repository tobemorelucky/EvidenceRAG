import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_financebench_retrieval.py"
SPEC = importlib.util.spec_from_file_location("financebench_retrieval_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_gold_evidence_adds_pdf_suffix_and_page():
    row = {
        "doc_name": "Fallback_2023_10K",
        "evidence": json.dumps([{"evidence_page_num": 7, "evidence_text": "Revenue was 10."}]),
    }

    assert MODULE.parse_gold_evidence(row) == [
        {"filename": "Fallback_2023_10K.pdf", "page_number": 7, "text": "Revenue was 10."}
    ]


def test_failure_classification_distinguishes_pdf_zero_based_page_offset():
    gold = [{"filename": "Acme_2023_10K.pdf", "page_number": 7, "text": "x"}]

    assert MODULE.classify_failure(gold, [{"filename": "Acme_2023_10K.pdf", "page_number": 6}]) == "gold_page_retrieved_offset_only"
    assert MODULE.classify_failure(gold, [{"filename": "Other.pdf", "page_number": 7}]) == "gold_document_not_retrieved"


def test_development_split_is_stable_and_has_twenty_rows():
    rows = [
        {"financebench_id": f"id-{index:03d}", "question_type": "a" if index % 2 else "b"}
        for index in range(100)
    ]

    selected = MODULE.select_development_rows(rows)
    assert len(selected) == 20
    assert [row["financebench_id"] for row in selected] == [
        row["financebench_id"] for row in MODULE.select_development_rows(rows)
    ]
