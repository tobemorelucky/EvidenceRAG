# RAG Core v3: Evidence Flow

RAG Core v3 is an isolated, question-only corpus RAG profile built on frozen commit `7cfd9cd5acf41fa36fc10e3d8f22326f3748a38e`. It does not change `clean_baseline`, `rag_core_v2`, or `rag_core_v2_skills`.

## Profiles

- `rag_core_v3`: generic Page Selector v3 and fair context allocation; no Skills.
- `rag_core_v3_skills`: the same evidence flow plus the two frozen Skills (`explicit_formula` and `canonical_finance_metric`).

Both profiles keep Finance Policy, QuerySpec, task classification, Agent, planner, HyDE, step-back, structured execution, and supplemental retrieval disabled.

## Page Selector v3

The selector uses only the original question and existing hybrid/RRF/Jina candidates:

- strongest chunk score;
- capped second/multi-chunk support;
- explicit terms, entities, periods, quarters, and numbers from the question;
- soft document aggregation;
- greedy new-information and document diversity;
- text redundancy penalty;
- two slots taken directly from global page ranking, independent of the soft document shortlist.

It contains no FinanceBench IDs, company registry, task type, canonical metric, required operand, or reference-answer rules.

The frozen page count is eight. In the fixed `selection_loss10 + correct10` replay, six pages produced 6/20 exact selected-page hits while eight pages produced 7/20. Correct-regression exact hits remained 6/10, and average selected-page redundancy was approximately 0.084.

## Context Budget v3

The total context remains capped at 28,000 characters. Every selected page first receives a 2,200-character contiguous window, after which remaining budget is distributed by page rank. Same-page table title, context/unit, columns/year header, and selected rows are attached as a complete block rather than truncated after page prose.

The real 20-question retrieval-only check produced:

- candidate hit: 17/20;
- selected hit: 7/20;
- context hit: 7/20;
- selected-to-context losses: 0;
- average context: 26,820 characters;
- table attached: 20/20.

## Operand safety

Operand parsing now distinguishes a leading `(1)` or `[1]` note marker only when removing it yields exactly one value per explicit year header. Genuine values `1` and `(1)` remain valid year-column values. Unexplained extra leading small integers cause fallback.

Statement-constrained authoritative operands also reject pages headed `Notes to Consolidated Financial Statements`. Such pages can contain cross-reference columns such as `Affected Line Item in the Consolidated Statements of Operations` but are not the primary statement.

The AES safety regression changed the selected cost-of-sales operand from the AOCL note value `(1)` on page 180 to consolidated `Total cost of sales (10,069)` on page 131. Explicit Formula remains 8/8 executed and 7/8 strict Judge.

## Document-local retrieval experiment

An optional interface implements:

`global original-query retrieval → soft top-3 documents → original-query document-local retrieval → RRF merge → one Jina rerank`

It adds no LLM call and records global/scoped provenance. On `candidate-miss10 + correct10`, candidate exact hit improved from 7/20 to 12/20, but selected/context hit stayed 6/20. Average retrieval latency increased from about 0.26 seconds to 3.49 seconds and the 20 questions sent about 560k Jina input characters.

Therefore `RAG_CORE_V3_DOCUMENT_LOCAL_RETRIEVAL=false` remains the frozen default. The implementation is retained for independent metadata-aware and upper-bound experiments, but is not part of the question-only Core v3 formal score.

## Optional retrieval context

`prepare_rag_response` accepts an optional `retrieval_context` object:

```json
{
  "company": "...",
  "period": "...",
  "document_type": "...",
  "selected_documents": ["..."]
}
```

It is ignored by ordinary question-only Core v3. When the optional document-local experiment is enabled, it contributes only soft document boosts; it never supplies evidence and never filters by benchmark page, evidence text, answer, or justification.

## Development artifacts

- `reports/rag_core_v3_error_flow_audit.md`
- `reports/rag_core_v3_error_flow_audit.json`
- `tests/fixtures/rag_core_v3_diagnostic_ids.json`
- `reports/rag_core_v3_page_selector_replay_v2.json`
- `reports/rag_core_v3_selection_correct_retrieval_diagnostic.json`
- `reports/rag_core_v3_candidate_correct_document_local_weighted_diagnostic.json`
- `docs/rag_core_v3_answer_utilization_audit.md`

## Formal run contract

Run both profiles exactly once from the same frozen commit. Do not enable `--document-local-retrieval` for the main question-only result.

```powershell
conda run --no-capture-output -n rag python -u scripts\run_financebench_rag_core_v3.py `
  --split all `
  --output reports\evidencerag-rag-core-v3-all100-final_answers.jsonl `
  --judge-output reports\evidencerag-rag-core-v3-all100-final_judge.jsonl

conda run --no-capture-output -n rag python -u scripts\run_financebench_rag_core_v3.py `
  --with-skills `
  --split all `
  --output reports\evidencerag-rag-core-v3-skills-all100-final_answers.jsonl `
  --judge-output reports\evidencerag-rag-core-v3-skills-all100-final_judge.jsonl
```

No source, configuration, prompt, or parameter change is permitted between these two runs.
