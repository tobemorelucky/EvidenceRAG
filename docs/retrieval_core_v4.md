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
