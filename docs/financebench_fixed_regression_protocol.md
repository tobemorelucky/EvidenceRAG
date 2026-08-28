# FinanceBench 固定回归与 Oracle 协议

## 定位

当前 100 题均为已经查看过的固定回归集。历史 `dev20` 与 `holdout80` 名称仅用于 ID 对齐；报告必须使用 `fixed_seen_regression`，不能把结果描述为未见数据泛化能力。

结构化增强不改变现有 Dense/BM25、RRF、Jina rerank、页面选择和 evidence compression。正式路径为：

`Question → QuerySpec → Hybrid Retrieval → Jina → EvidenceFrame alignment → Advisory Coverage / Decimal Executor → Protected-slot Compression → Answer → Local Consistency Check`

coverage 为 partial/insufficient 时，`SUPPLEMENTAL_FIND_ENABLED` 最多在已经选定的文档集合内补搜一次，并打开命中页及 ±1 页。它不会搜索互联网、扩大公司范围或进入 Agent 循环。

分阶段开关均默认关闭（advisory 语义本身默认开启，但只有 `STRUCTURED_COVERAGE_ENABLED=true` 时生效）：`FRAME_ALIGNMENT_ENABLED`、`STRUCTURED_TASK_EXECUTOR_ENABLED`、`ANSWER_CONSISTENCY_VALIDATOR_ENABLED`、`RAG_PROTECTED_EVIDENCE_SLOTS_ENABLED`、`SUPPLEMENTAL_FIND_ENABLED`。

## 0. 前置检查

```powershell
conda run -n rag python -m pytest tests -q
conda run -n rag python scripts/backfill_finance_tables.py
```

第二条默认 dry-run。确认恰好选择 FinanceBench 对应 PDF 且没有 parser error 后，写入现有 PostgreSQL 表存储：

```powershell
conda run --no-capture-output -n rag python -u scripts/backfill_finance_tables.py --execute
```

这一步不生成 embedding，也不修改 Milvus。

## 1. Oracle 诊断

先跑 1 题 smoke；它跳过 retrieval/Jina，直接使用 benchmark gold page：

```powershell
conda run --no-capture-output -n rag python -u scripts/evaluate_financebench_oracle.py --split dev --limit 1 --skip-judge --evidence-frame --structured-executor --structured-coverage --output reports/oracle_structured_smoke.jsonl
```

正式 Oracle 100 题（回答模型与独立 Judge 都会产生 token）：

```powershell
conda run --no-capture-output -n rag python -u scripts/evaluate_financebench_oracle.py --split all --evidence-frame --structured-executor --structured-coverage --judge-interval-seconds 2 --output reports/evidencerag_structured_oracle_all100.jsonl
```

## 2. LangSmith 固定 100 题回归

先跑 1 题端到端 smoke，成功后脚本会自动 Judge：

```powershell
conda run --no-capture-output -n rag python -u scripts/run_financebench_langsmith_experiment.py --split dev --limit 1 --experiment-prefix evidencerag-alignment-smoke --max-concurrency 1 --thinking disabled --max-completion-tokens 512 --enable-rerank --evidence-frame --structured-executor --structured-coverage --frame-alignment --structured-task-executor --answer-consistency-validator --protected-evidence-slots --output reports/evidencerag_alignment_smoke_answers.jsonl --judge-output reports/evidencerag_alignment_smoke_judge.jsonl
```

完整 100 题可以一次运行 `--split all`。为了保留历史 20/80 文件结构并降低中断损失，也可以分别运行：

```powershell
conda run --no-capture-output -n rag python -u scripts/run_financebench_langsmith_experiment.py --split dev --experiment-prefix evidencerag-structured-fixed20-v1 --max-concurrency 1 --thinking disabled --max-completion-tokens 512 --enable-rerank --evidence-frame --structured-executor --structured-coverage --supplemental-find --output reports/evidencerag_structured_fixed20_answers.jsonl --judge-output reports/evidencerag_structured_fixed20_judge.jsonl

conda run --no-capture-output -n rag python -u scripts/run_financebench_langsmith_experiment.py --split holdout --experiment-prefix evidencerag-structured-regression80-v1 --max-concurrency 1 --thinking disabled --max-completion-tokens 512 --enable-rerank --evidence-frame --structured-executor --structured-coverage --supplemental-find --output reports/evidencerag_structured_regression80_answers.jsonl --judge-output reports/evidencerag_structured_regression80_judge.jsonl
```

若中断，使用完全相同的参数并增加 `--resume`。远程 Jina 会最多尝试两次并优先复用缓存，两次都失败后才使用本地 fallback；每个结果 trace 会保留实际 provider 和失败原因。

## 3. 生成 100 题汇总

```powershell
conda run --no-capture-output -n rag python -u scripts/summarize_financebench_experiment.py --split fixed20 reports/evidencerag_structured_fixed20_answers.jsonl reports/evidencerag_structured_fixed20_judge.jsonl --split regression80 reports/evidencerag_structured_regression80_answers.jsonl reports/evidencerag_structured_regression80_judge.jsonl --oracle-summary reports/evidencerag_structured_oracle_all100.summary.json --baseline-summary reports/evidencerag-finance-v14-general-all100-resolved-summary.json --output reports/evidencerag_structured_all100_summary.json
```

脚本同时生成同名 Markdown，统计：accuracy、task type、candidate/context gold-page hit、candidate→context loss、Oracle gap、answer token、延迟、Jina 输入与 fallback、EvidenceFrame 使用、structured execution，以及补搜触发/修复数。

## 判读顺序

1. Oracle accuracy 明显高于端到端：优先修 retrieval/page selection 或 candidate→context loss。
2. Gold page 已有但 EvidenceFrame `period/unit/scope` 不完整：优先提升通用表头恢复，不能增加某家公司或某题规则。
3. EvidenceFrame 完整但 executor 未使用：检查约束或通用 formula binding。
4. Executor 结果正确但 Judge 错：检查 answer contract/生成，而不是继续扩大检索。
5. lookup 回退：保持结构化功能关闭或限制在数值任务，不用单题 prompt 补丁修复。
