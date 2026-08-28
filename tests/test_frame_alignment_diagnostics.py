from scripts.evaluate_financebench_frame_alignment import _summarize


def test_alignment_summary_reports_the_full_utilization_funnel():
    records = [
        {
            "evidence_frame_count": 8,
            "relevant_frame_count": 3,
            "evidence_coverage": {"structured_answerable": True, "structured_execution_ready": True},
            "frames_used_for_execution": 2,
            "operand_resolution_failure_reason": "",
            "supplemental_triggered": False,
            "supplemental_effective": False,
        },
        {
            "evidence_frame_count": 4,
            "relevant_frame_count": 0,
            "evidence_coverage": {"structured_answerable": False, "structured_execution_ready": False},
            "frames_used_for_execution": 0,
            "operand_resolution_failure_reason": "no_related_frames",
            "supplemental_triggered": True,
            "supplemental_effective": False,
        },
    ]

    summary = _summarize(records)

    assert summary["evidence_frame_questions"] == 2
    assert summary["queryspec_related_frame_questions"] == 1
    assert summary["structured_answerable_questions"] == 1
    assert summary["structured_execution_ready_questions"] == 1
    assert summary["structured_executions"] == 1
    assert summary["operand_resolution_failures"] == {"no_related_frames": 1}
