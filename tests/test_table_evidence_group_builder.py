from scripts.table_evidence_group_builder import (
    TableRecord,
    build_groups,
    evaluate_gold_coverage,
    score_link,
)


def table(table_id, page, *, title="", columns=("2024", "2023"), before="", after="", document="doc"):
    return TableRecord(
        table_id=table_id,
        document_id=document,
        page_id=f"{document}:p{page}",
        filename=f"{document}.pdf",
        page_number=page,
        start_page=page,
        end_page=page,
        title=title,
        caption="",
        columns=columns,
        before_context=before,
        after_context=after,
        quality_score=0.8,
    )


def test_adjacent_matching_title_and_headers_form_group():
    groups = build_groups([
        table("a", 10, title="Consolidated balance sheets"),
        table("b", 11, title="Consolidated balance sheets (continued)"),
    ])
    group = next(item for item in groups if len(item["table_ids"]) == 2)
    assert group["member_pages"] == [10, 11]
    assert group["cross_page"] is True
    assert group["links"][0]["continuation"] is True


def test_non_adjacent_or_different_document_tables_do_not_group():
    groups = build_groups([
        table("a", 10, title="Balance sheets"),
        table("b", 12, title="Balance sheets"),
        table("c", 11, title="Balance sheets", document="other"),
    ])
    assert len(groups) == 3
    assert all(len(item["table_ids"]) == 1 for item in groups)


def test_adjacent_unrelated_tables_do_not_group():
    left = table("a", 10, title="Revenue by geography", columns=("Region", "Revenue"))
    right = table("b", 11, title="Debt maturity schedule", columns=("Maturity", "Principal"))
    assert score_link(left, right) is None
    assert len(build_groups([left, right])) == 2


def test_three_page_chain_and_group_id_are_deterministic():
    tables = [
        table("a", 10, title="Statements of cash flows"),
        table("b", 11, title="Statements of cash flows continued"),
        table("c", 12, title="Statements of cash flows continued"),
    ]
    first = build_groups(tables)
    second = build_groups(list(reversed(tables)))
    group = next(item for item in first if len(item["table_ids"]) == 3)
    assert group["member_pages"] == [10, 11, 12]
    assert group["table_group_id"] == next(item for item in second if len(item["table_ids"]) == 3)["table_group_id"]


def test_coverage_distinguishes_evidenced_recovery_from_adjacent_upper_bound():
    groups = build_groups([
        table("a", 5, title="Debt schedule"),
        table("b", 8, title="Balance sheets continued", before="Continued from previous page"),
    ])
    records = [
        {"financebench_id": "direct", "gold_pages": [["doc.pdf", 5]]},
        {"financebench_id": "continuation", "gold_pages": [["doc.pdf", 7]]},
        {"financebench_id": "adjacent-only", "gold_pages": [["doc.pdf", 4]]},
    ]
    result = evaluate_gold_coverage(groups, records)
    assert result["direct_table_member_coverage"]["count"] == 1
    assert result["continuation_evidenced_group_coverage"]["count"] == 2
    assert result["adjacent_table_upper_bound"]["count"] == 3
