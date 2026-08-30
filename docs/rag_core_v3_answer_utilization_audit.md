# RAG Core v3 Answer Utilization Audit

This is an offline review of frozen Core v2 + Skills runs where the exact FinanceBench gold page was already present in the answer context but the strict Judge returned incorrect. Gold evidence and reference answers are not used by runtime code.

## Summary

- Reviewed: 9 questions
- Additional LLM calls: 0
- Runtime prompt changes in this phase: none
- Main pattern: evidence presence alone is insufficient when competing documents, derived outputs, or multi-part requirements are involved.

| Failure pattern | Count | Examples | General observation |
|---|---:|---|---|
| Unnecessary refusal despite usable evidence | 4 | `01911`, `00790`, `00917`, `04481` | The model required the final metric or exact wording instead of using supported operands or a directly stated driver. |
| Wrong document/period emphasis | 2 | `00601`, `00651` | Relevant gold evidence coexisted with a competing company or a later guidance period, and generation followed the wrong context. |
| Required derived facet omitted | 1 | `01936` | The response listed 81 and 93 but omitted the requested 87% conclusion. |
| Unsupported or contradictory addition | 1 | `01226` | The supported margin decline and drivers were followed by an unnecessary conclusion not required by the question. |
| Incomplete list | 1 | `00494` | The response reported only one supported program although the question required all forecast production-rate changes. |

## Minimal future prompt candidates

These are audit findings only and are not enabled in RAG Core v3 evidence-flow development:

1. Use directly stated causal evidence even when the question names the corresponding ratio rather than the statement line item.
2. When all requested operands are present, calculate the derived result instead of refusing because the final metric is absent.
3. Cover every explicit part or listed entity requested by the question.
4. Prefer evidence matching the question's entity and period when competing context is present.
5. Do not add a qualitative conclusion that is not requested or supported.

Any future prompt experiment must be isolated from retrieval changes and tested on this fixed set plus correct lookup regressions. No item above authorizes an ID-, company-, or metric-specific runtime rule.
