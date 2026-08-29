from scripts.financebench_judge_common import _parse_verdict


def test_parse_local_judge_verdict_accepts_json_payload():
    result = _parse_verdict('{"score": 1, "verdict": "correct", "reason": "matches"}')

    assert result == {"score": 1, "verdict": "correct", "reason": "matches"}


def test_parse_local_judge_verdict_rejects_non_json():
    result = _parse_verdict("not json")

    assert result["score"] == 0
    assert result["verdict"] == "invalid_judge_output"
