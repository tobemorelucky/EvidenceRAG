# Retrieval Core v4

Retrieval Core v4 is an isolated, configuration-selected retrieval experiment.
It does not change Core v3, answer prompts, answer contracts, finance skills, or
Context Budget v3. Gold pages are used only by offline evaluation scripts.

## Fixed diagnostic set

The fixed 30-question fixture contains 10 Core v3 candidate misses, 10 page
selection losses, and 10 previously correct regression checks. Full 100-question
runs are prohibited until all three profiles pass their stage gates.

## Phase 1: Dense Primary Retrieval

Profile: `retrieval_dense_primary`

- Dense Top120 is retained in its original order.
- BM25 Top30 only appends chunks not already present in Dense.
- BM25 never promotes or replaces a Dense result.
- Every chunk records `dense_rank`, `bm25_rank`, and `merged_rank`.
- The existing single chunk-level Jina call, Core v3 page selector, and Context
  Budget v3 remain unchanged for this phase.

Fixed diagnostic30 result:

| Metric | Core v3 + Skills | Dense Primary |
|---|---:|---:|
| Candidate hit | 56.67% | 93.33% |
| Selected hit | 23.33% | 30.00% |
| Context hit | 23.33% | 30.00% |
| Average context chars | 26,725 | 26,326 |
| Input/context token indicator | 6,529 actual input | 6,582 estimated context |

The phase gate passed: candidate coverage improved materially, context did not
decline, and the frozen context budget did not expand. The low selected/context
rate confirms that candidate-to-page selection is now the primary bottleneck.

## Phase 2: Page Neighbor Expansion

Profile: `retrieval_dense_primary_neighbors`

- Every merged page opens only the same document's page −1/current/page +1.
- Existing offline page embeddings, lexical overlap, and merged source rank
  produce a deterministic page order; there is no query-time page embedding.
- Document aggregation keeps seven of eight final slots in the strongest
  document and reserves one generic global escape slot.
- Jina calls are zero and Context Budget v3 remains capped at 28,000 chars.

Fixed diagnostic30 result:

| Metric | Dense Primary | Page Neighbors |
|---|---:|---:|
| Candidate hit | 93.33% | 96.67% |
| Selected hit | 30.00% | 33.33% |
| Context hit | 30.00% | 33.33% |
| Selection-loss group context | 40.00% | 30.00% |
| Correct-regression group context | 50.00% | 60.00% |
| Average context chars | 26,326 | 26,034 |
| Jina calls | 30 | 0 |

Phase 2 did **not** pass its gate. Aggregate context improved and no Core v3
context hit regressed, but the dedicated selection-loss group became worse than
Profile 1. Consequently Profile 3 page-level Jina is intentionally not run.
This prevents remote spend and avoids tuning page weights against the fixed set.

## Phase 3: Document-local Page Retrieval

Profile: `retrieval_document_local`

The global Dense Primary result is aggregated into a Top3 document shortlist
using only Dense/BM25 ranks, chunk count, and distinct-page coverage. The query
embedding is reused. Each shortlisted document receives one Dense Top30 and one
BM25 Top30 search; the final local capacity reserves 20 Dense slots and up to 10
BM25-only supplement slots. No Jina, answer model, or Judge is called.

The fixed diagnostic30 compares:

- A: current global page selection;
- B: document-local-only page selection;
- C: local-first chunks plus the global candidate pool.

| Metric | A global | B document-local | C global+local |
|---|---:|---:|---:|
| Candidate hit | 96.67% | 70.00% | 96.67% |
| Selected hit | 33.33% | 33.33% | 26.67% |
| Context hit | 33.33% | 33.33% | 26.67% |
| Average gold page rank | 36.24 | 14.38 | 36.17 |
| Average candidate pages | 291.57 | 80.70 | 356.40 |
| Average context chars | 26,034 | 26,380 | 26,374 |

Rank improved below the `<15` target, but context did not exceed 50%. B recovered
`financebench_id_00540` and `financebench_id_01351` while losing
`financebench_id_00605` and `financebench_id_01328`. C retained global candidate
coverage but its enlarged mixed page pool worsened final selection. Average
document-local Milvus latency was 33.44 ms, with four Dense and four BM25 calls
per question including global discovery. The phase gate therefore failed and
page-level Jina remains disabled.
