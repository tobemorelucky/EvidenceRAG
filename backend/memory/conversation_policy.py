"""Deterministic execution policy for Conversation-aware RAG shadows.

The conversation-understanding model supplies semantic observations.  This
module owns execution decisions so model booleans are never forwarded directly
to retrieval or memory infrastructure.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from backend.conversation_understanding_v3 import ConversationDecisionV3


_CHALLENGE = re.compile(
    r"\b(are you sure|verify|recheck|double[- ]check|check again|that seems wrong|incorrect)\b|"
    r"(确定吗|核实|重新检查|再检查|似乎不对|不正确)",
    re.IGNORECASE,
)
_CITATION = re.compile(
    r"\b(citation|cite|source|page|evidence|where.*from)\b|(引用|来源|页码|证据|哪一页)",
    re.IGNORECASE,
)
_CALCULATION_EXPLANATION = re.compile(
    r"\b(how (?:did you calculate|(?:was|is|were|are) .{0,80} calculat(?:ed|e))|show (?:the )?formula|explain (?:the )?calculation|walk me through)\b|"
    r"(怎么算|计算过程|解释.*计算|列出公式)",
    re.IGNORECASE,
)
_CONDITION_MODIFICATION = re.compile(
    r"\b(instead|rather than|change (?:the )?(?:comparison|period|year|entity|company)? ?to|use (?:fy|q[1-4]|fiscal)|only include|exclude|"
    r"what about .* instead|same .* for)\b|(改为|换成|只看|仅看|排除|同样.*用于)",
    re.IGNORECASE,
)
_NON_RAG = re.compile(
    r"^(thanks|thank you|hello|hi|stop|cancel|好的|谢谢|你好|停止|取消)[.!。！ ]*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConversationExecutionPolicy:
    retrieval: bool
    rewrite: bool
    retrieval_query: str
    response_mode: str
    reuse_previous_evidence: bool
    memory_scope: str
    policy_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_query(user_input: str, standalone_query: str, rewrite: bool) -> str:
    original = " ".join(str(user_input or "").split())
    resolved = " ".join(str(standalone_query or "").split())
    if rewrite and resolved:
        return resolved[:1000]
    return original[:1000]


def build_conversation_policy(
    user_input: str,
    decision: "ConversationDecisionV3",
    *,
    has_previous_evidence: bool,
) -> ConversationExecutionPolicy:
    """Translate v3 observations into a bounded, deterministic execution plan.

    ``decision.need_retrieval`` and the free-form ``response_mode`` are
    intentionally not used as executable commands.
    """
    text = " ".join(str(user_input or "").split())
    if not text:
        raise ValueError("user_input must not be empty")

    challenge = bool(_CHALLENGE.search(text))
    citation = bool(_CITATION.search(text))
    calculation_explanation = bool(_CALCULATION_EXPLANATION.search(text))
    condition_modification = bool(_CONDITION_MODIFICATION.search(text))
    independent = not decision.depends_on_history
    non_rag = not decision.need_rag and bool(_NON_RAG.search(text))

    if challenge:
        retrieval, rewrite, reuse = True, decision.query_resolution_needed, has_previous_evidence
        mode, reason = "verify", "challenge_requires_fresh_verification"
    elif calculation_explanation:
        reuse = has_previous_evidence
        retrieval, rewrite = not reuse, not reuse and decision.query_resolution_needed
        mode, reason = "explain_calculation", "calculation_explanation_reuses_previous_evidence" if reuse else "calculation_explanation_missing_previous_evidence"
    elif citation:
        reuse = has_previous_evidence
        retrieval, rewrite = not reuse, not reuse and decision.query_resolution_needed
        mode, reason = "explain_citations", "citation_followup_reuses_previous_evidence" if reuse else "citation_followup_missing_previous_evidence"
    elif condition_modification:
        retrieval, rewrite, reuse = True, True, False
        mode, reason = "answer_with_modified_conditions", "condition_modification_requires_resolved_retrieval"
    elif independent and not non_rag:
        retrieval, rewrite, reuse = True, False, False
        mode, reason = "answer_with_evidence", "independent_question_requires_retrieval"
    elif non_rag:
        retrieval, rewrite, reuse = False, False, False
        mode, reason = "conversational", "explicit_non_rag_turn"
    else:
        reuse = has_previous_evidence and decision.depends_on_history
        retrieval = not reuse
        rewrite = retrieval and decision.query_resolution_needed
        mode = "continue_with_context"
        reason = "dependent_followup_reuses_previous_evidence" if reuse else "dependent_followup_requires_retrieval"

    return ConversationExecutionPolicy(
        retrieval=retrieval,
        rewrite=rewrite,
        retrieval_query=_safe_query(text, decision.standalone_query, rewrite),
        response_mode=mode,
        reuse_previous_evidence=reuse,
        memory_scope=decision.required_memory_scope if decision.depends_on_history else "none",
        policy_reason=reason,
    )
