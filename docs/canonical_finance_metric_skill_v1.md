# Canonical Finance Metric Skill v1

## Scope

This stage adds one isolated, opt-in skill on top of the frozen clean RAG and
explicit-formula skill. It supports only four registry-defined metrics:

- quick ratio;
- inventory turnover;
- gross margin;
- operating margin.

The production registry contains no company names, FinanceBench IDs, years, or
reference values. It is not sent to the answer model.

## Runtime contract

The new profile is `finance_skills_v1`:

```text
clean baseline
  -> explicit_formula (highest priority)
  -> canonical_finance_metric
  -> unchanged clean RAG fallback
```

The existing `clean_baseline` and `clean_baseline_formula_skill` profiles remain
independent. A canonical metric is executed only when the metric, company
document, fiscal periods, consolidated scope, currency, scale, and unique
operands are all validated. Missing or ambiguous evidence returns to the clean
path. The skill reuses `operand_search` and `decimal_calculator`; it makes no
LLM or Jina call.

Pure numeric and two-period trend results use a cited deterministic answer.
Quick-ratio health questions inject a short verified numeric result into the
existing single answer call; no liquidity threshold is hard-coded.

## Metric definitions

| Metric | Definition | Important fail-closed rule |
|---|---|---|
| Quick ratio | validated quick assets / current liabilities | An absent optional quick asset is not zero unless a complete current-assets statement proves it is not separately disclosed. |
| Inventory turnover | COGS / average(beginning, ending inventory) | Ending inventory alone is never substituted for average inventory. |
| Gross margin | gross profit / revenue; fallback `(revenue - COGS) / revenue` | Operands must share company, period, statement scope, currency, and compatible scale. |
| Operating margin | operating income / revenue | Cause/driver questions are excluded. |

Parent-only, guarantor, and cross-guarantee statement scopes are excluded from
group-level operands. Fiscal shorthand such as `FY22` and common dynamic
filename abbreviations are handled without a company registry.

## Validation

### Synthetic and repository tests

- Canonical synthetic tests: 38 cases covering routing, all four formulas,
  trend inference, optional-field safety, scale normalization, negative values,
  ambiguity, cross-document rejection, scope exclusion, and fallback behavior.
- Canonical + explicit focused regression: 50 passed.
- Full repository suite: 372 passed before the final target/full evaluation;
  the final code is rerun under the same suite before commit.

### Explicit-formula frozen regression

| Measure | Result |
|---|---:|
| Questions | 8 |
| Explicit skill success | 8/8 |
| Judge | 7/8 |
| Routed to canonical skill | 0 |
| Extra Jina / LLM | 0 / 0 |

AES ROA remains the standard Decimal result `-0.01`; no reference-answer
special case was added.

### Canonical alias target regression

The alias-derived target set contains 14 questions. Four causal questions are
intentionally excluded, ten enter the canonical contract, and three execute.

| Metric | Alias targets | Detected | Executed | Executed correct |
|---|---:|---:|---:|---:|
| Quick ratio | 4 | 4 | 1 | 1 |
| Inventory turnover | 2 | 2 | 1 | 0 |
| Gross margin | 4 | 2 | 1 | 1 |
| Operating margin | 4 | 2 | 0 | 0 |
| Total | 14 | 10 | 3 | 2 |

Paired target score changed from 2/14 to 3/14: one wrong-to-correct and zero
correct-to-wrong. Seven detected questions failed closed because a target
document or required operand was not uniquely available.

The inventory-turnover execution is arithmetically correct under the registered
standard formula, but FinanceBench uses ending inventory for that example. The
standard result is retained and recorded as a benchmark-definition disagreement.

### Full 100-question A/B

| Profile | Strict Judge | Total answer tokens | Avg latency |
|---|---:|---:|---:|
| Clean baseline | 33/100 | 134,394 | — |
| Explicit formula skill | 38/100 | 124,915 | 3,185.99 ms |
| Finance skills v1 | 39/100 | 122,475 | 3,117.34 ms |

Against the explicit-formula profile, the final single run has three
wrong-to-correct and two correct-to-wrong results, for net +1. The only target
migration is quick ratio wrong-to-correct; there is no target
correct-to-wrong. The other migrations are answer/Judge variability on
unchanged non-target prompts.

All 86 non-target questions have identical selected pages, final pages, answer
context pages, citations, and answer input token counts between B and C.

Canonical overhead in the full run:

- 42 additional bounded Dense+BM25 operand searches;
- 0 additional Jina calls;
- 0 additional LLM calls;
- average canonical skill latency across alias targets: 606.93 ms;
- total answer tokens decreased by 2,440 because authoritative numeric answers
  bypass answer generation.

## Reproduction

Target regression:

```powershell
conda run --no-capture-output -n rag python -u scripts\run_financebench_canonical_metric_skill.py
```

Full local 100-question run:

```powershell
conda run --no-capture-output -n rag python -u scripts\run_financebench_canonical_metric_skill.py `
  --split all `
  --experiment-prefix evidencerag-finance-skills-v1-all100-v2 `
  --output reports\evidencerag-finance-skills-v1-all100-v2_answers.jsonl `
  --judge-output reports\evidencerag-finance-skills-v1-all100-v2_judge.jsonl
```

The runner uses local evaluation storage and automatically invokes the separate
DeepSeek-V4-Pro judge after successful answer generation.

## Overfitting and promotion decision

No benchmark-specific production logic was introduced. Every runtime decision
is based on metric aliases, question semantics, document metadata, statement
scope, period, and auditable operands. The new profile remains opt-in rather
than changing either frozen profile.

No metric is removed from the registry in v1. Inventory turnover's benchmark
disagreement is a definition mismatch, not a calculation defect; execution is
still safe when beginning and ending inventory are present. Operating margin
has zero executions because the current evidence did not satisfy the strict
contract, which is preferable to relaxing scope or uniqueness constraints.

## Artifacts

- `reports/evidencerag-finance-skills-v1-all100-v2-summary.json`
- `reports/evidencerag-finance-skills-v1-all100-v2-summary.md`
- `reports/evidencerag-finance-skills-v1-all100-v2-ab.json`
- `reports/evidencerag-finance-skills-v1-all100-v2-resolved-answers.md`
- `reports/evidencerag-skill-canonical-finance-metric-v1-target-v4-ab.json`
- `reports/evidencerag-skill-explicit-formula-v1-fixed8-canonical-regression-ab.json`
