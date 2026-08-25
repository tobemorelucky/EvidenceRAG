from pathlib import Path

from backend.finance_policy import (
    POLICY_DIR,
    SUPPORTED_TASK_TYPES,
    clear_finance_policy_cache,
    load_finance_policy,
)


def test_finance_policy_disabled_returns_empty_legacy_payload(monkeypatch):
    monkeypatch.setenv("FINANCE_POLICY_ENABLED", "false")

    policy = load_finance_policy("calculation")

    assert policy["enabled"] is False
    assert policy["task_type"] == "calculation"
    assert policy["policy_file"] == ""
    assert policy["text"] == ""
    assert policy["estimated_tokens"] == 0


def test_all_finance_policies_are_generic_and_below_budget(monkeypatch):
    monkeypatch.setenv("FINANCE_POLICY_ENABLED", "true")
    clear_finance_policy_cache()
    forbidden = {
        "financebench_id",
        "best buy",
        "apple",
        "amazon",
        "inventory turnover",
        "gross margin",
        "return on assets",
        "roa",
    }

    for task_type in SUPPORTED_TASK_TYPES:
        policy = load_finance_policy(task_type)
        lowered = policy["text"].lower()

        assert policy["enabled"] is True
        assert policy["task_type"] == task_type
        assert policy["policy_file"] == f"{task_type}.json"
        assert policy["estimated_tokens"] < 500
        assert "not a factual source" in lowered
        assert "supported by the evidence" in lowered
        assert not any(term in lowered for term in forbidden)
        assert (POLICY_DIR / policy["policy_file"]).is_file()


def test_finance_policy_loader_uses_memory_cache(monkeypatch):
    monkeypatch.setenv("FINANCE_POLICY_ENABLED", "true")
    clear_finance_policy_cache()

    first = load_finance_policy("comparison")
    second = load_finance_policy("comparison")

    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert first["text"] == second["text"]


def test_unknown_task_type_falls_back_to_lookup(monkeypatch):
    monkeypatch.setenv("FINANCE_POLICY_ENABLED", "true")
    clear_finance_policy_cache()

    policy = load_finance_policy("unsupported")

    assert policy["task_type"] == "lookup"
    assert policy["policy_file"] == "lookup.json"


def test_policy_files_contain_only_expected_schema():
    import json

    for path in Path(POLICY_DIR).glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert set(payload) == {"task_type", "instructions"}
        assert payload["task_type"] == path.stem
        assert len(payload["instructions"]) == 4


def test_judgment_policy_requires_direct_conclusion_when_supported(monkeypatch):
    monkeypatch.setenv("FINANCE_POLICY_ENABLED", "true")
    clear_finance_policy_cache()

    policy = load_finance_policy("judgment")

    assert "conclusion directly" in policy["text"]
    assert "when inputs are available" in policy["text"].lower()


def test_comparison_policy_requires_directional_consistency(monkeypatch):
    monkeypatch.setenv("FINANCE_POLICY_ENABLED", "true")
    clear_finance_policy_cache()

    policy = load_finance_policy("comparison")

    assert "match the computed direction" in policy["text"]
    assert "nonzero increase or decrease" in policy["text"]
