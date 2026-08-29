"""Runtime profiles that make experimental RAG features explicit and auditable."""

from __future__ import annotations

import os
from typing import Mapping


CLEAN_BASELINE_PROFILE = "clean_baseline"

# These values are authoritative for the clean baseline. They intentionally
# override stale values from .env so an experiment cannot be polluted by a
# forgotten feature flag.
CLEAN_BASELINE_OVERRIDES: Mapping[str, str] = {
    "RAG_PROFILE": CLEAN_BASELINE_PROFILE,
    "RAG_EXECUTION_MODE": "static",
    "RAG_RETRIEVAL_MODE": "baseline",
    "RAG_QUERY_PLANNER_ENABLED": "false",
    "FINANCE_RAG_ENABLE_STEP_BACK": "false",
    "RAG_FIELD_AWARE_ENABLED": "false",
    "RAG_PAGE_FIRST_ENABLED": "false",
    "RAG_PAGE_LEVEL_FUSION_ENABLED": "false",
    "RAG_SUPPLEMENTAL_SEARCH_ENABLED": "false",
    "FINANCE_RAG_ENABLE_PAGE_MERGE": "false",
    "FINANCE_RAG_ADJACENT_PAGE_WINDOW": "0",
    "FINANCE_RAG_ADJACENT_CHUNK_WINDOW": "0",
    "RAG_CONTEXT_PAGE_WINDOW": "0",
    "AUTO_MERGE_ENABLED": "false",
    "TABLE_AWARE_RETRIEVAL": "off",
    "RAG_EVIDENCE_GROUPING_ENABLED": "false",
    "FINANCE_POLICY_ENABLED": "false",
    "EVIDENCE_FRAME_ENABLED": "false",
    "STRUCTURED_EXECUTOR_ENABLED": "false",
    "STRUCTURED_COVERAGE_ENABLED": "false",
    "STRUCTURED_COVERAGE_ADVISORY_ENABLED": "false",
    "FRAME_ALIGNMENT_ENABLED": "false",
    "STRUCTURED_TASK_EXECUTOR_ENABLED": "false",
    "ANSWER_CONSISTENCY_VALIDATOR_ENABLED": "false",
    "RAG_PROTECTED_EVIDENCE_SLOTS_ENABLED": "false",
    "STAGE_AWARE_COVERAGE_ENABLED": "false",
    "PROTECTED_PAGE_SLOTS_ENABLED": "false",
    "NUMERIC_DISPLAY_VALIDATOR_ENABLED": "false",
    "ANSWER_REQUIRED_FACETS_ENABLED": "false",
    "EXPLICIT_FORMULA_ADVISORY_ENABLED": "false",
    "SUPPLEMENTAL_FIND_ENABLED": "false",
    "ENABLE_FINANCE_FORMULA_EXPANSION": "false",
    "RERANK_REMOTE_MAX_ATTEMPTS": "2",
    "LOCAL_RERANK_ENABLED": "true",
    "RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED": "true",
}


FEATURE_LABELS: tuple[tuple[str, str], ...] = (
    ("Finance Policy", "FINANCE_POLICY_ENABLED"),
    ("Structured Coverage", "STRUCTURED_COVERAGE_ENABLED"),
    ("Structured Coverage Advisory", "STRUCTURED_COVERAGE_ADVISORY_ENABLED"),
    ("Structured Executor", "STRUCTURED_TASK_EXECUTOR_ENABLED"),
    ("Answer Validator", "ANSWER_CONSISTENCY_VALIDATOR_ENABLED"),
    ("Protected Evidence", "RAG_PROTECTED_EVIDENCE_SLOTS_ENABLED"),
    ("Protected Pages", "PROTECTED_PAGE_SLOTS_ENABLED"),
    ("Stage Coverage", "STAGE_AWARE_COVERAGE_ENABLED"),
    ("Formula Advisory", "EXPLICIT_FORMULA_ADVISORY_ENABLED"),
    ("Answer Facets", "ANSWER_REQUIRED_FACETS_ENABLED"),
    ("Supplemental Retrieval", "SUPPLEMENTAL_FIND_ENABLED"),
    ("Agent/Planner", "RAG_QUERY_PLANNER_ENABLED"),
)


def normalize_profile(profile: str | None = None) -> str:
    return (profile or os.getenv("RAG_PROFILE", "finance")).strip().lower()


def is_clean_baseline(profile: str | None = None) -> bool:
    return normalize_profile(profile) == CLEAN_BASELINE_PROFILE


def apply_runtime_profile(profile: str | None = None) -> str:
    """Apply authoritative settings for profiles that require hard isolation."""
    resolved = normalize_profile(profile)
    if resolved == CLEAN_BASELINE_PROFILE:
        os.environ.update(CLEAN_BASELINE_OVERRIDES)
    return resolved


def _enabled(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes", "on"}


def feature_state(profile: str | None = None) -> dict[str, object]:
    resolved = normalize_profile(profile)
    modules = {label: _enabled(name) for label, name in FEATURE_LABELS}
    modules["Structured Executor"] = (
        _enabled("STRUCTURED_EXECUTOR_ENABLED")
        or _enabled("STRUCTURED_TASK_EXECUTOR_ENABLED")
    )
    return {
        "profile": resolved,
        "execution_mode": os.getenv("RAG_EXECUTION_MODE", ""),
        "retrieval_mode": os.getenv("RAG_RETRIEVAL_MODE", ""),
        "modules": modules,
        "field_aware": _enabled("RAG_FIELD_AWARE_ENABLED"),
        "page_first": _enabled("RAG_PAGE_FIRST_ENABLED"),
        "step_back": _enabled("FINANCE_RAG_ENABLE_STEP_BACK"),
        "auto_merge": _enabled("AUTO_MERGE_ENABLED"),
        "table_aware": os.getenv("TABLE_AWARE_RETRIEVAL", "off").strip().lower() != "off",
    }


def feature_summary_lines(profile: str | None = None) -> list[str]:
    state = feature_state(profile)
    lines = [
        f"RAG Profile: {state['profile']}",
        f"Execution Mode: {state['execution_mode'] or 'default'}",
        f"Retrieval Mode: {state['retrieval_mode'] or 'default'}",
    ]
    lines.extend(
        f"{label}: {'ON' if enabled else 'OFF'}"
        for label, enabled in state["modules"].items()
    )
    return lines


def print_feature_summary(profile: str | None = None) -> None:
    print("[feature-summary]", flush=True)
    for line in feature_summary_lines(profile):
        print(f"  {line}", flush=True)
