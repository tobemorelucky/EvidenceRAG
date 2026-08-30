# RAG Core v2

RAG Core v2 是在冻结 clean baseline 之上新增的独立实验路径。它只改善证据流，不启用 planner、Agent、query expansion、EvidenceFrame、政策层或历史实验模块。

## Profiles

- `rag_core_v2`：Core v2 检索与上下文，使用 clean baseline 的通用证据约束提示词，不启用 Skill。
- `rag_core_v2_skills`：完全相同的 Core v2，再挂载已冻结的 `explicit_formula` 和 `canonical_finance_metric` Skill。Skill 定义与内部执行逻辑未修改。

历史 `clean_baseline`、`clean_baseline_formula_skill`、`finance_skills_v1` 和 `finance` profile 保持原行为。

## 固定流程

1. 现有 BGE-M3 Dense + Milvus 原生 BM25 混合召回。
2. RRF 得到 60 个叶子 chunk 候选。
3. Jina 在最多 18 个候选上重排，保留 16 个 reranked chunk。
4. 对候选 chunk 聚合 document/page 分数；软选 4 份文档，建立最多 10 页 page pool，并记录 Jina 全局逃逸页。
5. 最终回答固定最多 6 页，不做 hard document filter，不增加检索轮次。
6. 最高 1–2 页保留合理长度的完整页；其他页保留围绕命中 chunk 的连续页内窗口。
7. 从 TableStore 读取最终页面上的结构化表格，保留标题、单位/上下文、列头和相关行；不调用新的表格解析器，也不把表格写入新的向量索引。
8. 使用最小化通用回答提示词生成一次答案。Skills profile 只在既有 Skill 高置信命中时执行其冻结逻辑。

## 冻结参数

| 参数 | 值 |
|---|---:|
| `FINANCE_RAG_CANDIDATE_K` | 60 |
| `FINANCE_RAG_FINAL_TOP_K` | 16 |
| `RERANK_REMOTE_CANDIDATE_K` | 18 |
| `RAG_CORE_V2_DOCUMENT_TOP_K` | 4 |
| `RAG_CORE_V2_PAGE_POOL_K` | 10 |
| `RAG_CORE_V2_FINAL_PAGE_K` | 6 |
| `RAG_CORE_V2_GLOBAL_ESCAPE_PAGES` | 2 |
| `RAG_CORE_V2_MAX_CONTEXT_CHARS` | 28,000 |
| `RAG_CORE_V2_MAX_TABLE_CHARS` | 5,000 |

旧 clean baseline 的“每页按问题词选最多 8 行”不会在 Core v2 使用。

## Trace

Core v2 新增或统一记录：

- `initial_dense_candidates` / `initial_bm25_candidates`：Milvus hybrid API 不暴露两个分支的独立实际结果数，因此值为 `null`，另记录 requested 数，禁止伪造观察值；
- `rrf_candidates`、`reranked_chunks`；
- `document_scores`、`selected_documents`；
- `page_scores`、`selected_pages`、`final_selected_pages`、`global_escape_pages`；
- `tables_available_on_selected_pages`、`tables_attached`、`table_ids`、`table_rows_attached`、`table_context_chars`、`table_attach_reason`；
- 完整的 context chars、token 和阶段 latency。

## 开发期验证

固定诊断集由冻结 clean baseline 的历史 trace 自动选择，仅保存在 `tests/fixtures/rag_core_v2_diagnostic_ids.json`：5 个 candidate miss、5 个 candidate-to-context loss、5 个 gold-context refusal、5 个 table/calculation，共 20 个不重复 ID。ID 和 gold evidence 仅用于离线评价，从不进入运行时排序。

首轮 Core v2 fixed20：

- candidate page hit：80%（同一集合历史 clean baseline 60%）；
- context page hit：40%（历史 35%）；
- citation document hit：90%（历史 80%）；
- Judge：9/20（历史 clean baseline fixed20 为 4/20）；
- 平均回答输入：6,127.75 token；
- 平均总延迟：5.03 秒；
- 空检索：0；
- 19/20 题的最终页面存在并附加了结构化表格。

一次 PageStore 完整页词面复评分消融没有提高 selected/context hit，且造成一题回退，因此没有进入正式运行路径。强制用 Jina 逃逸页替换最终页面也造成回退，同样撤销；逃逸页只保留在可审计 page pool。

`rag_core_v2_skills` 的 explicit8 为 7/8，与冻结 Skill 的两个历史 explicit8 结果一致；唯一失败仍是 AES ROA 的既有参考舍入口径差异。

## 命令

审计：

```powershell
conda run --no-capture-output -n rag python -u scripts\audit_rag_core_v2.py
```

fixed20 检索-only（不会调用回答模型或 Judge）：

```powershell
conda run --no-capture-output -n rag python -u scripts\run_financebench_rag_core_v2.py --diagnostic-fixed20 --retrieval-only --output reports\rag_core_v2_fixed20_retrieval_answers.jsonl
```

fixed20 完整回答与本地 Judge：

```powershell
conda run --no-capture-output -n rag python -u scripts\run_financebench_rag_core_v2.py --diagnostic-fixed20 --output reports\rag_core_v2_fixed20_answers.jsonl --judge-output reports\rag_core_v2_fixed20_judge.jsonl
```

explicit8 Skills 回归：

```powershell
conda run --no-capture-output -n rag python -u scripts\run_financebench_rag_core_v2.py --with-skills --explicit8 --output reports\rag_core_v2_skills_explicit8_answers.jsonl --judge-output reports\rag_core_v2_skills_explicit8_judge.jsonl
```

正式 100 题只能在代码和参数提交冻结后各运行一次；结果命令和提交 SHA 将写入最终报告。
