import pytest

from backend.agent_tools import calculate, find_evidence, select_pages


def test_find_evidence_prioritizes_query_overlap():
    docs = [
        {"filename": "a.pdf", "page_number": 1, "text": "Retail stores increased."},
        {"filename": "b.pdf", "page_number": 2, "text": "Operating income and revenue determine operating margin."},
    ]
    result = find_evidence("What was operating margin?", docs, limit=1)
    assert result[0]["filename"] == "b.pdf"


def test_select_pages_deduplicates_document_pages():
    pages = select_pages(
        [
            {"filename": "a.pdf", "page_number": 1},
            {"filename": "a.pdf", "page_number": 1},
            {"filename": "a.pdf", "page_number": 2},
        ],
        limit=3,
    )
    assert pages == [{"filename": "a.pdf", "page_number": 1}, {"filename": "a.pdf", "page_number": 2}]


def test_calculate_uses_decimal_arithmetic():
    result = calculate("(5121.3 / 7491.5) * 100")
    assert result["operands"] == "5121.3, 7491.5, 100"
    assert result["result"].startswith("68.361")


@pytest.mark.parametrize("expression", ["__import__('os')", "1 / 0", "a + 1"])
def test_calculate_rejects_unsafe_or_invalid_expressions(expression):
    with pytest.raises(ValueError):
        calculate(expression)
