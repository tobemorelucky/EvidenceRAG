from scripts.merge_financebench_retry_results import merge_records


def test_merge_records_replaces_answer_and_matching_judge_in_base_order():
    base_answers = [
        {"financebench_id": "a", "langsmith_trace_id": "run-a"},
        {"financebench_id": "b", "langsmith_trace_id": "run-b"},
    ]
    base_judges = [{"run_id": "run-a", "score": 0}, {"run_id": "run-b", "score": 1}]
    retry_answers = [{"financebench_id": "a", "langsmith_trace_id": "retry-a"}]
    retry_judges = [{"run_id": "retry-a", "score": 1}]

    answers, judges = merge_records(base_answers, base_judges, retry_answers, retry_judges)

    assert [record["langsmith_trace_id"] for record in answers] == ["retry-a", "run-b"]
    assert [record["run_id"] for record in judges] == ["retry-a", "run-b"]
