"""Question-only requirement signals and an offline Packing v1 input adapter.

No domain dictionary, reference labels, model calls or production integration.
Signals are conservative surface-language proxies, not an answerability proof.
"""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass

from backend.evidence_packing_v1 import _numbers, _rank, _score, _terms


_YEAR = re.compile(r"(?<!\w)(?:FY\s*[-/]?\s*)?((?:19|20)\d{2})(?!\d)|\bFY\s*[-/]?\s*(\d{2})(?!\d)", re.I)
_QUARTER = re.compile(r"\bQ[1-4]\b", re.I)
_TEMPORAL = re.compile(r"\b(?:fiscal|annual|quarter|year|period|month|previous|prior|current|latest)\b", re.I)
_NUMERIC = re.compile(r"\bhow (?:much|many)\b|\b(?:quantity|amount|number|percent|percentage|ratio|rate|million|millions|billion|billions)\b|%", re.I)
_COMPARISON = re.compile(r"\b(?:compar\w*|versus|vs|differ\w*|chang\w*|increas\w*|decreas\w*|improv\w*|grow\w*|grew|drop\w*|higher|lower)\b", re.I)
_SUPERLATIVE = re.compile(r"\b(?:highest|lowest|largest|smallest|most|least|maximum|minimum)\b", re.I)
_CALCULATE = re.compile(r"\b(?:calculate|compute|sum|subtract|multiply|divide|average)\b", re.I)
_EXPLANATION = re.compile(r"^\s*(?:why\b|explain\b|what (?:drove|caused|explains)\b|how (?:did|does) .+? (?:happen|occur)\b)", re.I)
_JUDGMENT = re.compile(r"^\s*(?:is|are|was|were|does|do|did|has|have|can|could|would|should)\b", re.I)
_ENTITY_STOP = {"what", "which", "how", "why", "does", "did", "is", "are", "was", "were", "has", "have", "among", "calculate", "compute", "compare", "roughly", "real", "fy", "q", "usd"}


def periods(text: str) -> set[str]:
    """Surface period tokens, not row-to-year binding. FYyy uses a 1950/2050 pivot."""
    result = set()
    for match in _YEAR.finditer(text):
        full, short = match.groups()
        result.add(full or str((2000 if int(short) < 50 else 1900) + int(short)))
    result.update(value.casefold() for value in _QUARTER.findall(text))
    return result


@dataclass(frozen=True)
class QueryRequirement:
    answer_type: str
    requires_numeric_evidence: bool
    requires_period: bool
    requires_entity: bool
    requires_comparison: bool
    requires_calculation: bool
    explicit_periods: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def parse_query_requirement(question: str) -> QueryRequirement:
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    # Keep primary instructions; a conditional explanatory fallback is not its task type.
    primary = re.split(r"(?<=[?;.])\s*if\b|,\s*if\b", question.strip(), maxsplit=1, flags=re.I)[0]
    explanation = bool(_EXPLANATION.search(primary))
    selection = bool(_SUPERLATIVE.search(primary) and re.search(r"\b(?:which|who|among|select|identify)\b", primary, re.I))
    comparison = bool(not explanation and (_COMPARISON.search(primary) or selection))
    calculation = bool(not explanation and (
        _CALCULATE.search(primary)
        or (re.search(r"\bhow much\b", primary, re.I) and comparison)
        or re.search(r"\b(?:what|which) (?:percent|percentage|fraction|proportion) of\b", primary, re.I)
    ))
    numeric = bool(not explanation and (_NUMERIC.search(primary) or calculation or comparison))
    if explanation:
        kind = "explanation"
    elif selection:
        kind = "selection"
    elif _JUDGMENT.search(primary):
        kind = "judgment"
    elif numeric:
        kind = "numeric"
    else:
        kind = "lookup"
    capitalized = re.findall(r"\b[A-Z][A-Za-z0-9]*\b", _YEAR.sub(" ", primary))
    entity = bool(re.search(r"\b(?:which|who|whose)\b|['’]s\b", primary, re.I) or
                  any(token.casefold() not in _ENTITY_STOP and not _QUARTER.fullmatch(token) for token in capitalized))
    explicit = periods(primary)
    return QueryRequirement(kind, numeric, bool(explicit or _TEMPORAL.search(primary)), entity,
                            comparison, calculation, tuple(sorted(explicit)))


def requirement_support(question: str, requirement: QueryRequirement, unit: dict) -> dict:
    """Positive support only. Missing metadata is unknown, never a hard exclusion.

    Multi-number presence does not prove a valid calculation or comparison pair.
    Text/metadata period tokens do not prove cell/period alignment.
    """
    text = str(unit.get("source_text") or "")
    # Strip period tokens first so FY22 and Q4 are not counted as operands.
    numeric = _numbers(_QUARTER.sub(" ", _YEAR.sub(" ", text)))
    unit_periods = periods(text) | periods(" ".join(map(str, unit.get("period") or [])))
    requested = set(requirement.explicit_periods)
    entity_terms = _terms(unit.get("entity")) - {"inc", "corp", "corporation", "ltd", "company", "unknown"}
    query_terms = _terms(question)
    supports = {}
    if requirement.requires_numeric_evidence:
        supports["numeric"] = float(bool(numeric))
    if requirement.requires_period:
        supports["period"] = len(requested & unit_periods) / len(requested) if requested else float(bool(unit_periods))
    if requirement.requires_entity:
        supports["entity"] = len(entity_terms & query_terms) / len(entity_terms) if entity_terms else 0.0
    if requirement.requires_comparison:
        supports["comparison"] = min(len(numeric), 2) / 2
    if requirement.requires_calculation:
        supports["calculation"] = min(len(numeric), 2) / 2
    # The sole intervention: bounded requirement bonus, no new generic ranker.
    # Lexical gating prevents unrelated numeric blocks from receiving a full bonus.
    lexical = len(query_terms & _terms(text)) / len(query_terms) if query_terms else 0.0
    mean_support = sum(supports.values()) / len(supports) if supports else 0.0
    return {"active_support": supports, "lexical_gate": lexical,
            "support_fraction": mean_support, "multiplier": 1 + lexical * mean_support}


def requirement_guided_inputs(question: str, units: list[dict]) -> tuple[list[dict], dict]:
    """Copy whitelisted Unit fields; change only effective score for shadow packing.

    Original ranks, metadata and source text remain unchanged. Returned selection
    must be resolved back to original Units by rank before rendering/evaluation.
    """
    from backend.evidence_assembly_v5 import EvidenceUnit

    requirement = parse_query_requirement(question)
    ranks = [_rank(unit) for unit in units]
    if any(rank <= 0 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("Candidate ranks must be unique and positive")
    adapted, traces = [], []
    for unit in units:
        value = {key: copy.deepcopy(unit[key]) for key in (*EvidenceUnit.__dataclass_fields__, "current_ranking")}
        support = requirement_support(question, requirement, unit)
        value["current_ranking"]["score"] = _score(unit) * support["multiplier"]
        adapted.append(value)
        traces.append({"rank": _rank(unit), "original_score": _score(unit),
                       "effective_score": value["current_ranking"]["score"], **support})
    return adapted, {"requirement": requirement.to_dict(), "unit_support": traces,
                     "score_formula": "original_score * (1 + lexical_gate * mean_active_requirement_support)",
                     "interpretation": "Soft question-conditioned support proxy, not validated operands or answerability"}
