"""Deterministic memory budgeting without tokenizer/model dependencies."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryBudget:
    max_tokens: int = 7000
    message_tokens: int = 1600
    summary_tokens: int = 400


class TokenBudgetManager:
    def __init__(self, budget: MemoryBudget | None = None):
        self.budget = budget or MemoryBudget(
            max_tokens=int(os.getenv("CONVERSATION_MEMORY_MAX_TOKENS", "7000")),
            message_tokens=int(os.getenv("CONVERSATION_MEMORY_MESSAGE_TOKENS", "1600")),
            summary_tokens=int(os.getenv("CONVERSATION_MEMORY_SUMMARY_TOKENS", "400")),
        )
        if min(self.budget.max_tokens, self.budget.message_tokens, self.budget.summary_tokens) < 0:
            raise ValueError("memory budgets must be non-negative")
        if self.budget.message_tokens + self.budget.summary_tokens > self.budget.max_tokens:
            raise ValueError("message and summary budgets exceed total budget")

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return (len(str(text or "")) + 3) // 4

    @staticmethod
    def _clip(text: str, token_limit: int, *, keep_tail: bool = False) -> str:
        char_limit = max(0, token_limit) * 4
        text = str(text or "")
        if len(text) <= char_limit:
            return text
        return text[-char_limit:] if keep_tail else text[:char_limit]

    def select_recent_messages(self, messages: list[dict]) -> list[dict]:
        selected, used = [], 0
        for message in reversed(messages):
            cost = self.estimate_tokens(message.get("content", ""))
            remaining = self.budget.message_tokens - used
            if remaining <= 0:
                break
            item = dict(message)
            if cost > remaining:
                item["content"] = self._clip(item.get("content", ""), remaining, keep_tail=True)
                selected.append(item)
                break
            selected.append(item)
            used += cost
        return list(reversed(selected))

    def build_context(
        self,
        messages: list[dict],
        *,
        summary: str = "",
        evidence: str = "",
        evidence_char_budget: int | None = None,
    ) -> dict:
        recent = self.select_recent_messages(messages)
        clipped_summary = self._clip(summary, self.budget.summary_tokens)
        fixed_tokens = sum(self.estimate_tokens(item.get("content", "")) for item in recent)
        fixed_tokens += self.estimate_tokens(clipped_summary)
        if evidence_char_budget is None:
            evidence_tokens = max(0, self.budget.max_tokens - fixed_tokens)
            clipped_evidence = self._clip(evidence, evidence_tokens)
        else:
            evidence_chars = max(0, int(evidence_char_budget))
            clipped_evidence = str(evidence or "")[:evidence_chars]
            evidence_tokens = self.estimate_tokens(clipped_evidence)
        return {
            "messages": recent, "summary": clipped_summary, "evidence": clipped_evidence,
            "estimated_tokens": fixed_tokens + self.estimate_tokens(clipped_evidence),
            "evidence_truncated": len(clipped_evidence) < len(str(evidence or "")),
            "evidence_char_budget": evidence_char_budget,
        }
