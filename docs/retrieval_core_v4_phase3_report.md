# Retrieval Core v4 Phase 3 Report

## Scope

- Profile: `retrieval_document_local`
- Dataset: fixed diagnostic30 only
- Groups: candidate miss 10, selection loss 10, correct regression 10
- Jina calls: 0
- Answer/LLM/Judge calls: 0
- Context Budget v3: unchanged, maximum 28,000 characters

## Results

| Variant | Candidate hit | Selected hit | Context hit | Mean gold page rank |
|---|---:|---:|---:|---:|
| A — global | 96.67% | 33.33% | 33.33% | 36.24 |
| B — document-local | 70.00% | 33.33% | 33.33% | 14.38 |
| C — global + local | 96.67% | 26.67% | 26.67% | 36.17 |

### Group context hit

| Group | A | B | C |
|---|---:|---:|---:|
| Candidate miss 10 | 10% | 0% | 0% |
| Selection loss 10 | 30% | 40% | 20% |
| Correct regression 10 | 60% | 60% | 60% |

## Document and page ranks

- Mean gold-document rank before local retrieval: 1.63.
- Mean gold-page rank before local retrieval: 36.24.
- Mean gold-page rank after document-local retrieval: 14.38.
- Mean gold-page rank after global/local merge: 36.17.

## Cost

- Dense calls: 4/question (1 global + 3 document-local).
- BM25 calls: 4/question (1 global + 3 document-local).
- Total calls over diagnostic30: 120 Dense + 120 BM25.
- Mean global retrieval latency: 88.17 ms/question.
- Mean added document-local retrieval latency: 33.44 ms/question.
- Mean total diagnostic pipeline latency: 1,486.75 ms/question.
- Jina characters and calls: 0 / 0.

## Migrations relative to A

Document-local B recovered:

- `financebench_id_00540`
- `financebench_id_01351`

Document-local B regressed:

- `financebench_id_00605`
- `financebench_id_01328`

Global/local C recovered no context hits and regressed:

- `financebench_id_00605`
- `financebench_id_01328`

The correct-regression group remains 6/10 under all three variants, but individual
migrations show B is not a safe replacement for A.

## Gate decision

| Gate | Target | Result | Status |
|---|---:|---:|---|
| Mean gold page rank | `<15` | 14.38 | Pass |
| Context hit | `>50%` | 33.33% | Fail |
| Added local retrieval latency | acceptable | +33.44 ms | Pass |

Overall Phase 3 status: **failed**. Document-local retrieval improves the rank of
pages it finds, but its Top30 local candidate capacity loses too many global
candidate hits and does not improve final context coverage. The next page-level
Jina phase must not start under the configured gate.

Detailed per-question traces are stored in
`reports/retrieval_document_local_diagnostic30.json`.
