from scripts.evaluate_oracle_evidence_block import (
    answer_evidence_coverage,
    build_oracle_evidence_blocks,
    summarize,
)


def _row() -> dict:
    return {
        "financebench_id": "generic-id",
        "question": "What was revenue in 2024?",
        "answer": "120",
        "evidence": '[{"doc_name":"GENERIC","evidence_page_num":7,"evidence_text":"Revenue | 120 | 100","evidence_text_full_page":"Results 2024\\nRevenue | 120 | 100\\nCosts | 80 | 70"}]',
    }


def test_oracle_block_uses_direct_page_contract_snippet_and_full_page():
    context, blocks = build_oracle_evidence_blocks(_row(), max_context_chars=1000)

    assert "internal page 7" in context
    assert "Revenue | 120 | 100" in context
    assert "Costs | 80 | 70" in context
    assert blocks[0]["source_pages"][0] == {"filename": "GENERIC.pdf", "page_number": 7}


def test_answer_evidence_coverage_reports_line_ratio():
    gold = [{"evidence_text": "Revenue | 120 | 100\nCosts | 80 | 70"}]
    coverage = answer_evidence_coverage(gold, "Revenue | 120 | 100")
    assert coverage == {"matched_lines": 1, "total_lines": 2, "ratio": 0.5}


def test_summary_compares_paired_strict_judge_routes():
    def route(score: int, coverage: float) -> dict:
        return {
            "judge_result": {"score": score},
            "metrics": {
                "answer_evidence_coverage": {"ratio": coverage},
                "required_number_hit": True,
                "required_period_hit": True,
                "gold_page_hit": True,
                "all_gold_pages_hit": True,
            },
            "answer_input_tokens": 100,
            "latency_ms": 10,
            "error": "",
        }
    records = [{
        "financebench_id": "generic-id",
        "group": "selection_loss10",
        "routes": {"evidence_block_v1": route(0, 0.5), "oracle_evidence_block": route(1, 1.0)},
    }]
    for group in ("candidate_miss10", "correct_regression10"):
        records.append({
            "financebench_id": group,
            "group": group,
            "routes": {"evidence_block_v1": route(1, 1.0), "oracle_evidence_block": route(1, 1.0)},
        })

    summary = summarize(records)

    assert summary["strict_judge_delta"] == 0.3333
    assert summary["groups"]["selection_loss10"]["oracle_gains"] == ["generic-id"]
