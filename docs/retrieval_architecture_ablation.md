# Retrieval Architecture Ablation

This stage is based on frozen commit `0d0898ee04e7250b0010691a148450a40780277e`.
It is retrieval-only: no answer model, Judge, Agent, planner, HyDE, step-back,
new Skill, or answer prompt is used.  Existing Core v3 profiles and Context
Budget v3 remain unchanged.

## Reproducible commands

```powershell
conda run --no-capture-output -n rag python -u scripts\audit_retrieval_funnel.py --max-k 120

conda run --no-capture-output -n rag python -u scripts\evaluate_recall_k_ablation.py

conda run --no-capture-output -n rag python -u scripts\run_retrieval_profile_ablation.py `
  --profile retrieval_ablation_structural

conda run --no-capture-output -n rag python -u scripts\run_retrieval_profile_ablation.py `
  --profile retrieval_ablation_field_aware

conda run --no-capture-output -n rag python -u scripts\evaluate_page_level_jina_ablation.py `
  --skip-jina --page-candidate-k 30 --representation-chars 1200 `
  --output reports\page_level_candidate_pool_diagnostic30.json

conda run --no-capture-output -n rag python scripts\compare_retrieval_ablations.py
```

The page-level Jina implementation is an experimental prototype. One remote
smoke request verified the API path. The 30-question remote run was intentionally
stopped because the pre-Jina Top-30 page pool retained only 36.67% of gold pages;
Jina cannot recover pages that are absent from its input.

## Main findings

- Dense is the dominant recall route. Across all 100 seen regression questions,
  Dense-only/BM25-only/both/neither counts are 34/2/54/10.
- On diagnostic30, increasing K from 40 to 120 raises RRF gold-page hit from
  46.67% to 83.33% with little Milvus latency growth.
- Equal RRF is not automatically beneficial at high K. Dense@100 is 85% on all
  100 questions while RRF@100 is 81%; 6 Dense hits within 120 fall outside
  RRF Top-120.
- The frozen Jina input cutoff is a larger issue than Jina downranking: 20 gold
  pages are in frozen RRF candidates but outside Jina input, while only one
  enters Jina and then disappears from its returned output.
- Structural page-first is the strongest tested architecture signal. On
  diagnostic30 it raises context hit from Core v3's 23.33% to 76.67%.
- Field-aware retrieval has the same aggregate context hit as structural but
  requires 44 rather than 30 retrieval routes and has higher latency. It is not
  promoted.
- The assumed `100 chunks -> 20-30 pages` compression does not hold for this
  index: RRF Top-100 contains about 96 unique pages on average. A plain page
  cutoff therefore destroys recall before reranking.

## Decision

Do not run a new 100-question answer experiment yet. Freeze the current results,
keep Context Budget v3 and Skills unchanged, and use the structural finding to
design the next candidate-page generator. The next design must retain the
document-level page discovery benefit without scanning/reranking an excessive
number of pages or adding field-aware query routes.
