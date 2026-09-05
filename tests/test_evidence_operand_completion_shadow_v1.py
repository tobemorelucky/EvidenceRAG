from backend.evidence_operand_completion_shadow_v1 import search_missing_operands_v1


def _schema(entity="JNJ", period="2022"):
    return {
        "recognized": True, "entity_requirement": entity,
        "period_requirement": {"explicit_periods": [period]},
        "required_operands": [{
            "key": "average_inventory", "label": "average inventory",
            "aliases": ["average inventory", "inventories", "inventory"], "min_values": 2,
        }],
    }


def _chunk(text, *, company="JOHNSON_JOHNSON", year=2022, rank=5, chunk_id="c1"):
    return {
        "text": text, "company": company, "report_year": year, "rrf_rank": rank,
        "chunk_id": chunk_id, "filename": f"{company}_{year}.pdf", "page_number": rank,
        "content_hash": chunk_id, "dense_rank": rank, "bm25_rank": None,
    }


def test_finds_operand_only_for_same_entity_and_period():
    chunks = [
        _chunk("Inventories 100 80"),
        _chunk("Inventories 900 800", company="OTHER", chunk_id="other"),
        _chunk("Inventories 70 60", year=2021, chunk_id="old"),
    ]
    result = search_missing_operands_v1(_schema(), ["average_inventory"], chunks)
    candidates = result["found_candidates"]["average_inventory"]
    assert [item["chunk_id"] for item in candidates] == ["c1"]
    assert result["all_missing_operands_recoverable"] is True


def test_rejects_cash_flow_change_as_average_inventory_balance():
    result = search_missing_operands_v1(
        _schema(), ["average_inventory"], [_chunk("Increase in inventories (2,527) (1,248) (265)")]
    )
    assert result["found_candidates"]["average_inventory"] == []
    assert result["all_missing_operands_recoverable"] is False


def test_adjusted_ebit_does_not_match_ebitdar():
    schema = {
        "recognized": True, "entity_requirement": "MGM", "period_requirement": {"explicit_periods": ["2022"]},
        "required_operands": [{"key": "adjusted_ebit", "aliases": ["adjusted ebit"], "min_values": 1}],
    }
    result = search_missing_operands_v1(
        schema, ["adjusted_ebit"], [_chunk("Adjusted EBITDAR $3,497", company="MGMRESORTS", chunk_id="mgm")]
    )
    assert result["found_candidates"]["adjusted_ebit"] == []


def test_existing_context_candidate_is_not_recoverable():
    candidate = _chunk("Inventories 100 80")
    result = search_missing_operands_v1(_schema(), ["average_inventory"], [candidate], [candidate])
    assert result["candidate_has_operand"]["average_inventory"] is True
    assert result["outside_context_candidate_has_operand"]["average_inventory"] is False
    assert result["all_missing_operands_recoverable"] is False
