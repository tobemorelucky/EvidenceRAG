"""Minimal registry for opt-in EvidenceRAG skills."""

from __future__ import annotations

from skills.canonical_finance_metric.skill import CanonicalFinanceMetricSkill
from skills.explicit_formula.schema import SkillResult
from skills.explicit_formula.skill import ExplicitFormulaSkill


_SKILLS = (ExplicitFormulaSkill(), CanonicalFinanceMetricSkill())


def execute_matching_skill(
    question: str,
    baseline_documents: list[dict],
    candidate_documents: list[dict],
    enabled_skill_names: tuple[str, ...],
) -> SkillResult:
    for skill in _SKILLS:
        if skill.name not in enabled_skill_names:
            continue
        if skill.can_handle(question):
            return skill.execute(question, baseline_documents, candidate_documents)
    return SkillResult(trace={
        "skill_detected": False,
        "skill_name": "none",
        "skill_success": False,
        "skill_applied": False,
        "fallback_to_clean_baseline": True,
        "fallback_reason": "no_registered_skill_matched",
        "skill_latency_ms": 0,
        "skill_dense_bm25_calls": 0,
        "skill_jina_calls": 0,
        "skill_llm_calls": 0,
    })
