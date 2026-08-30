# RAG Core v2 存储与基线审计

审计日期：2026-08-30  
审计提交：`39d7743`  
审计性质：只读检查；未重建索引，未修改正式检索算法，未运行新的 100 题实验。

## 结论

当前索引足以继续开发 RAG Core v2，无需重建。页面存储完整，TableStore 已覆盖 39/40 份文档；主要缺口不是表格数量，而是历史 clean baseline 没有消费 TableStore，并把约 74% 的已选页面文本在回答前删掉了。

索引没有 L1/L2 parent chunk。Milvus 只有 L3 叶子块，PostgreSQL 的 parent chunk 数为 0。因此 Core v2 应先使用现有叶子候选、完整页面和页内连续窗口，不应假设 Auto-Merging/Parent-Child 当前可用。以后若要恢复真正的三级 Parent-Child，需要单独重建索引，但这不是本阶段的必要条件。

## 存储现状

### Milvus

| 项目 | 结果 |
|---|---:|
| collection | `embeddings_collection` |
| 总记录 | 6,542 |
| 文档 | 40 |
| L3 叶子块 | 6,542 |
| L1/L2 块 | 0 |
| 平均文本长度 | 2,816 chars |
| 中位文本长度 | 2,774 chars |
| `text_chunk` | 6,542 |
| `table_summary/table_row/table_raw` | 0 |

Milvus 当前只保存文本叶子块。表格不在 Milvus 中做独立向量召回，因此 Core v2 的首版表格策略应为“先选页面，再从 TableStore 附加同页表格”，而不是假装已有 table evidence 向量索引。

### PostgreSQL 页面与父块

| 项目 | 结果 |
|---|---:|
| DocumentPage | 5,200 |
| 文档 | 40 |
| 平均页文本 | 3,377 chars |
| 中位页文本 | 3,372 chars |
| 非空 page_text | 5,185 / 5,200（99.71%） |
| filename/page/company/year/type 完整率 | 100% |
| 非空 table_text | 314 / 5,200（6.04%） |
| ParentChunk | 0 |

页面正文和金融元数据足够完整。15 个空白页占比很低，可在运行时跳过。`table_text` 不是主要表格来源；结构化表格应读取独立 TableStore。

### TableStore

| 项目 | 结果 |
|---|---:|
| DocumentTable 记录 | 1,936 |
| 有表格的文档 | 39 / 40 |
| 有表格的页面 | 1,376 |
| 总表格行 | 15,177 |
| 有 columns 的表 | 1,936 |
| 有 csv_text 的表 | 1,936 |

TableStore 覆盖率较高，不建议为 Core v2 重建索引。需要说明的是，`DocumentTable` 表没有保存 parser backend 或 accepted 标记；`upsert_tables()` 保存传入的表格，但无法仅从存量记录严格证明历史 parser 与验收状态。当前配置后端是 `pdfplumber_words`，`TABLE_AWARE_INGESTION=false`，只能说明当前运行配置，不能把它当作历史导入后端的审计证据。

重建脚本当前明确使用 `include_parent_chunks=False` 并关闭 table-aware Milvus ingestion，这与“只有 L3 文本块、无父块、无 Milvus 表格证据”的实测一致。

## clean baseline 为什么只有约 1.2k 输入 token

历史 clean baseline 共 100 题：

| 项目 | 结果 |
|---|---:|
| 平均回答输入 token | 1,163.87 |
| 平均原始已选页面文本 | 16,489 chars |
| 平均送入回答的 evidence | 4,171 chars |
| 平均文本保留率 | 25.86% |
| 平均文本删除率 | 74.14% |
| 中位保留率 | 25.91% |
| 平均证据页/单元 | 4.98 |

根因在 `build_baseline_evidence()`：每页先按换行/句子切分，`_baseline_snippet()` 用问题词做 lexical overlap 排序，每页最多只保留 8 行，同时每页最多 2,000 chars。最终上下文虽然配置上限为 24,000 chars，实际平均只有约 4,171 chars。

这种方法会破坏财务表的连续结构：表头、年份、单位、指标行和相邻操作数常被分开；没有与问题共享词面的必要行也会被删除。它节省 token，但解释了 candidate gold page hit 65%、context gold page hit 45%，以及 20 题从候选到上下文丢失的现象。

## Core v2 的直接设计约束

1. 不重建索引，复用 Dense + Milvus BM25 + RRF + Jina。
2. 不依赖不存在的 L1/L2 parent chunk。
3. 候选 chunk 聚合为 document/page 分数，使用软文档选择并保留全局逃逸页面，禁止硬 document filter。
4. 最高质量的 1–2 页保留完整页面；其他页面保留命中 chunk 对应的连续页内窗口，禁止 8 行 lexical compression。
5. 只对已选页面查询 TableStore，附加同页表头、年份列和相关行；不扩大为新的表格向量索引。
6. 将回答输入控制在约 4k–6.5k token，复杂题最高约 8k；通过固定总字符预算控制，而不是删除非重合行。
7. `rag_core_v2` 使用最小化通用提示词；`rag_core_v2_skills` 只在相同 Core 上挂载已冻结的 explicit/canonical Skills。

## 可复现命令

```powershell
conda run --no-capture-output -n rag python -u scripts\audit_rag_core_v2.py
```

机器可读结果写入 `reports/rag_core_v2_store_audit.json`。审计脚本只读取 PostgreSQL、Milvus 和既有 clean baseline 报告。
