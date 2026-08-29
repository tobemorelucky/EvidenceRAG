"""Minimal registry for opt-in EvidenceRAG skills."""

from __future__ import annotations

from skills.explicit_formula.skill import ExplicitFormulaSkill


_SKILLS = (ExplicitFormulaSkill(),)


def execute_matching_skill(question: str, baseline_documents: list[dict], candidate_documents: list[dict]):
    for skill in _SKILLS:
        if skill.can_handle(question):
            return skill.execute(question, baseline_documents, candidate_documents)
    # Execute once to return the standard non-detected trace.
    return _SKILLS[0].execute(question, baseline_documents, candidate_documents)
