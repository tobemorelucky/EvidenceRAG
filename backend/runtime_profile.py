"""Runtime profiles that make experimental RAG features explicit and auditable."""

from __future__ import annotations

import os
from typing import Mapping


CLEAN_BASELINE_PROFILE = "clean_baseline"
EXPLICIT_FORMULA_SKILL_PROFILE = "clean_baseline_formula_skill"
FINANCE_SKILLS_V1_PROFILE = "finance_skills_v1"
RAG_CORE_V2_PROFILE = "rag_core_v2"
RAG_CORE_V2_SKILLS_PROFILE = "rag_core_v2_skills"
RAG_CORE_V3_PROFILE = "rag_core_v3"
RAG_CORE_V3_SKILLS_PROFILE = "rag_core_v3_skills"
RETRIEVAL_ABLATION_STRUCTURAL_PROFILE = "retrieval_ablation_structural"
RETRIEVAL_ABLATION_FIELD_AWARE_PROFILE = "retrieval_ablation_field_aware"
RETRIEVAL_DENSE_PRIMARY_PROFILE = "retrieval_dense_primary"
RETRIEVAL_DENSE_PRIMARY_NEIGHBORS_PROFILE = "retrieval_dense_primary_neighbors"

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
    "EXPLICIT_FORMULA_SKILL_ENABLED": "false",
    "CANONICAL_FINANCE_METRIC_SKILL_ENABLED": "false",
    "SUPPLEMENTAL_FIND_ENABLED": "false",
    "ENABLE_FINANCE_FORMULA_EXPANSION": "false",
    "RERANK_REMOTE_MAX_ATTEMPTS": "2",
    "LOCAL_RERANK_ENABLED": "true",
    "RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED": "true",
}

RAG_CORE_V2_OVERRIDES: Mapping[str, str] = {
    **CLEAN_BASELINE_OVERRIDES,
    "RAG_PROFILE": RAG_CORE_V2_PROFILE,
    "FINANCE_RAG_CANDIDATE_K": "60",
    "FINANCE_RAG_FINAL_TOP_K": "16",
    "RERANK_REMOTE_CANDIDATE_K": "18",
    "RAG_CORE_V2_DOCUMENT_TOP_K": "4",
    "RAG_CORE_V2_PAGE_POOL_K": "10",
    "RAG_CORE_V2_FINAL_PAGE_K": "6",
    "RAG_CORE_V2_GLOBAL_ESCAPE_PAGES": "2",
    "RAG_CORE_V2_MAX_CONTEXT_CHARS": "28000",
    "RAG_CORE_V2_MAX_TABLE_CHARS": "5000",
    "RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED": "false",
}

RAG_CORE_V3_OVERRIDES: Mapping[str, str] = {
    **RAG_CORE_V2_OVERRIDES,
    "RAG_PROFILE": RAG_CORE_V3_PROFILE,
    "RAG_CORE_V3_DOCUMENT_TOP_K": "4",
    "RAG_CORE_V3_PAGE_POOL_K": "12",
    "RAG_CORE_V3_FINAL_PAGE_K": "8",
    "RAG_CORE_V3_GLOBAL_ESCAPE_PAGES": "2",
    "RAG_CORE_V3_MAX_CONTEXT_CHARS": "28000",
    "RAG_CORE_V3_MAX_TABLE_CHARS": "5000",
    "RAG_CORE_V3_MIN_PAGE_CHARS": "2200",
    "RAG_CORE_V3_DOCUMENT_LOCAL_RETRIEVAL": "false",
}

RETRIEVAL_ABLATION_STRUCTURAL_OVERRIDES: Mapping[str, str] = {
    **CLEAN_BASELINE_OVERRIDES,
    "RAG_PROFILE": RETRIEVAL_ABLATION_STRUCTURAL_PROFILE,
    "FINANCE_RAG_CANDIDATE_K": "60",
    "FINANCE_RAG_FINAL_TOP_K": "8",
    "RERANK_REMOTE_CANDIDATE_K": "18",
    "RAG_PAGE_FIRST_ENABLED": "true",
    "FINANCE_RAG_ENABLE_PAGE_MERGE": "true",
    "FINANCE_RAG_ADJACENT_PAGE_WINDOW": "1",
    "FINANCE_RAG_ADJACENT_CHUNK_WINDOW": "1",
    "RAG_CONTEXT_PAGE_WINDOW": "1",
    "RAG_ANSWER_CONTEXT_COMPRESSION_ENABLED": "false",
}

RETRIEVAL_ABLATION_FIELD_AWARE_OVERRIDES: Mapping[str, str] = {
    **RETRIEVAL_ABLATION_STRUCTURAL_OVERRIDES,
    "RAG_PROFILE": RETRIEVAL_ABLATION_FIELD_AWARE_PROFILE,
    "RAG_FIELD_AWARE_ENABLED": "true",
    "RAG_SUPPLEMENTAL_SEARCH_ENABLED": "true",
    "FINANCE_RAG_ADJACENT_PAGE_WINDOW": "2",
    "RAG_CONTEXT_PAGE_WINDOW": "2",
}

RETRIEVAL_DENSE_PRIMARY_OVERRIDES: Mapping[str, str] = {
    **RAG_CORE_V3_OVERRIDES,
    "RAG_PROFILE": RETRIEVAL_DENSE_PRIMARY_PROFILE,
    "EXPLICIT_FORMULA_SKILL_ENABLED": "true",
    "CANONICAL_FINANCE_METRIC_SKILL_ENABLED": "true",
    "RAG_CORE_V4_DENSE_K": "120",
    "RAG_CORE_V4_BM25_K": "30",
    "RAG_CORE_V4_NEIGHBOR_WINDOW": "0",
    "RAG_CORE_V4_PAGE_JINA_ENABLED": "false",
}

RETRIEVAL_DENSE_PRIMARY_NEIGHBORS_OVERRIDES: Mapping[str, str] = {
    **RETRIEVAL_DENSE_PRIMARY_OVERRIDES,
    "RAG_PROFILE": RETRIEVAL_DENSE_PRIMARY_NEIGHBORS_PROFILE,
    "RAG_CORE_V4_NEIGHBOR_WINDOW": "1",
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
    ("Explicit Formula Skill", "EXPLICIT_FORMULA_SKILL_ENABLED"),
    ("Canonical Finance Metric Skill", "CANONICAL_FINANCE_METRIC_SKILL_ENABLED"),
    ("Answer Facets", "ANSWER_REQUIRED_FACETS_ENABLED"),
    ("Supplemental Retrieval", "SUPPLEMENTAL_FIND_ENABLED"),
    ("Agent/Planner", "RAG_QUERY_PLANNER_ENABLED"),
)


def normalize_profile(profile: str | None = None) -> str:
    return (profile or os.getenv("RAG_PROFILE", "finance")).strip().lower()


def is_clean_baseline(profile: str | None = None) -> bool:
    return normalize_profile(profile) == CLEAN_BASELINE_PROFILE


def uses_clean_baseline_path(profile: str | None = None) -> bool:
    return normalize_profile(profile) in {
        CLEAN_BASELINE_PROFILE, EXPLICIT_FORMULA_SKILL_PROFILE, FINANCE_SKILLS_V1_PROFILE,
        RAG_CORE_V2_PROFILE, RAG_CORE_V2_SKILLS_PROFILE, RAG_CORE_V3_PROFILE, RAG_CORE_V3_SKILLS_PROFILE,
    }


def uses_rag_core_v2_path(profile: str | None = None) -> bool:
    return normalize_profile(profile) in {RAG_CORE_V2_PROFILE, RAG_CORE_V2_SKILLS_PROFILE}


def uses_rag_core_v3_path(profile: str | None = None) -> bool:
    return normalize_profile(profile) in {RAG_CORE_V3_PROFILE, RAG_CORE_V3_SKILLS_PROFILE}


def apply_runtime_profile(profile: str | None = None) -> str:
    """Apply authoritative settings for profiles that require hard isolation."""
    resolved = normalize_profile(profile)
    if resolved == CLEAN_BASELINE_PROFILE:
        os.environ.update(CLEAN_BASELINE_OVERRIDES)
    elif resolved == EXPLICIT_FORMULA_SKILL_PROFILE:
        os.environ.update(CLEAN_BASELINE_OVERRIDES)
        os.environ.update({
            "RAG_PROFILE": EXPLICIT_FORMULA_SKILL_PROFILE,
            "EXPLICIT_FORMULA_SKILL_ENABLED": "true",
        })
    elif resolved == FINANCE_SKILLS_V1_PROFILE:
        os.environ.update(CLEAN_BASELINE_OVERRIDES)
        os.environ.update({
            "RAG_PROFILE": FINANCE_SKILLS_V1_PROFILE,
            "EXPLICIT_FORMULA_SKILL_ENABLED": "true",
            "CANONICAL_FINANCE_METRIC_SKILL_ENABLED": "true",
        })
    elif resolved == RAG_CORE_V2_PROFILE:
        os.environ.update(RAG_CORE_V2_OVERRIDES)
    elif resolved == RAG_CORE_V2_SKILLS_PROFILE:
        os.environ.update(RAG_CORE_V2_OVERRIDES)
        os.environ.update({
            "RAG_PROFILE": RAG_CORE_V2_SKILLS_PROFILE,
            "EXPLICIT_FORMULA_SKILL_ENABLED": "true",
            "CANONICAL_FINANCE_METRIC_SKILL_ENABLED": "true",
        })
    elif resolved == RAG_CORE_V3_PROFILE:
        os.environ.update(RAG_CORE_V3_OVERRIDES)
    elif resolved == RAG_CORE_V3_SKILLS_PROFILE:
        os.environ.update(RAG_CORE_V3_OVERRIDES)
        os.environ.update({
            "RAG_PROFILE": RAG_CORE_V3_SKILLS_PROFILE,
            "EXPLICIT_FORMULA_SKILL_ENABLED": "true",
            "CANONICAL_FINANCE_METRIC_SKILL_ENABLED": "true",
        })
    elif resolved == RETRIEVAL_ABLATION_STRUCTURAL_PROFILE:
        os.environ.update(RETRIEVAL_ABLATION_STRUCTURAL_OVERRIDES)
    elif resolved == RETRIEVAL_ABLATION_FIELD_AWARE_PROFILE:
        os.environ.update(RETRIEVAL_ABLATION_FIELD_AWARE_OVERRIDES)
    elif resolved == RETRIEVAL_DENSE_PRIMARY_PROFILE:
        os.environ.update(RETRIEVAL_DENSE_PRIMARY_OVERRIDES)
    elif resolved == RETRIEVAL_DENSE_PRIMARY_NEIGHBORS_PROFILE:
        os.environ.update(RETRIEVAL_DENSE_PRIMARY_NEIGHBORS_OVERRIDES)
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
