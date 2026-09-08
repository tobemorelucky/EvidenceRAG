"""Semantic-only shadow router for the Evidence Focus answer prompt."""

from __future__ import annotations

import re


REASONING_INDICATORS = re.compile(
    r"\b(?:calculate|ratio|margin|percentage|growth|change|increase|decrease|improve|decline|trend|"
    r"compare|comparison|highest|lowest)\b|"
    r"计算|比率|利润率|百分比|增长|变化|增加|上升|下降|减少|改善|恶化|趋势|比较|最高|最低",
    re.IGNORECASE,
)
FY_PATTERN = re.compile(r"\bFY\s*20\d{2}\b", re.IGNORECASE)
FROM_TO_PERIOD_PATTERN = re.compile(
    r"\bfrom\s+(?:FY\s*)?(?:19|20)\d{2}\s+to\s+(?:FY\s*)?(?:19|20)\d{2}\b|"
    r"从\s*(?:FY\s*)?(?:19|20)\d{2}\s*(?:年)?\s*(?:到|至)\s*(?:FY\s*)?(?:19|20)\d{2}",
    re.IGNORECASE,
)
WHETHER_CHANGE_PATTERN = re.compile(
    r"\bwhether\b[^?]{0,120}\b(?:improve|change|increase|decrease|decline)\b|"
    r"是否[^？?]{0,120}(?:改善|变化|增加|上升|下降|减少|恶化)",
    re.IGNORECASE,
)
LOOKUP_INDICATORS = re.compile(
    r"\b(?:revenue|cash|income|expense|amount|value|number)\b|收入|现金|收益|利润|费用|金额|数值|数字",
    re.IGNORECASE,
)


def route_evidence_focus_semantic(question: str) -> dict:
    """Route only questions that show explicit financial reasoning semantics."""
    text = question or ""
    reasons: list[str] = []
    if REASONING_INDICATORS.search(text):
        reasons.append("calculation_or_reasoning_indicator")
    if len(FY_PATTERN.findall(text)) >= 2:
        reasons.append("multiple_fiscal_periods")
    if FROM_TO_PERIOD_PATTERN.search(text):
        reasons.append("from_to_period_comparison")
    if WHETHER_CHANGE_PATTERN.search(text):
        reasons.append("whether_change_pattern")
    lookup_only = bool(LOOKUP_INDICATORS.search(text)) and not reasons
    return {
        "use_evidence_focus": bool(reasons),
        "reasons": reasons,
        "lookup_only": lookup_only,
    }

