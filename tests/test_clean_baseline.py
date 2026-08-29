import os

from backend import prompts
from backend.evidence_context import build_baseline_evidence
from backend.runtime_profile import (
    CLEAN_BASELINE_OVERRIDES,
    apply_runtime_profile,
    feature_state,
)


def test_clean_profile_authoritatively_disables_experimental_features(monkeypatch):
    for name in CLEAN_BASELINE_OVERRIDES:
        monkeypatch.setenv(name, "true")

    assert apply_runtime_profile("clean_baseline") == "clean_baseline"
    state = feature_state("clean_baseline")

    assert state["profile"] == "clean_baseline"
    assert state["execution_mode"] == "static"
    assert state["retrieval_mode"] == "baseline"
    assert not any(state["modules"].values())
    assert state["field_aware"] is False
    assert state["page_first"] is False
    assert state["step_back"] is False
    assert os.environ["RERANK_REMOTE_MAX_ATTEMPTS"] == "2"


def test_clean_prompt_contains_only_generic_grounding_rules():
    prompt = prompts.CLEAN_BASELINE_ANSWER_SYSTEM_PROMPT.casefold()
    assert "answer only from the evidence" in prompt
    assert "[source: filename, page n]" in prompt
    for forbidden in (
        "quick ratio", "working capital", "gross margin", "operating margin",
        "store count", "section 12(b)", "acquisition", "financial institution",
        "validated calculation contract", "corporate/other",
    ):
        assert forbidden not in prompt


def test_baseline_evidence_does_not_use_task_or_metric_rules(monkeypatch):
    monkeypatch.setenv("RAG_ANSWER_MAX_EVIDENCE_UNITS", "2")
    evidence, trace = build_baseline_evidence(
        "What changed in 2023?",
        [
            {"filename": "A.pdf", "page_number": 1, "text": "2022 was stable.\nRevenue changed in 2023."},
            {"filename": "A.pdf", "page_number": 1, "text": "duplicate page"},
            {"filename": "B.pdf", "page_number": 4, "text": "The 2023 report explains the change."},
        ],
    )

    assert "Source: A.pdf | Page: 1" in evidence
    assert "Source: B.pdf | Page: 4" in evidence
    assert trace["answer_context_strategy"] == "clean_baseline_generic_v1"
    assert trace["answer_context_task_rules_used"] is False
    assert trace["answer_context_unit_count"] == 2
