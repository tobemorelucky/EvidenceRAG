# EvidenceRAG 底层数据结构审计

> 审计日期：2026-09-01  
> 审计范围：当前仓库 `HEAD=56c1454`、当前 PostgreSQL/Milvus 只读状态、仓库已有 retrieval diagnostic 报告。  
> 约束：本次未修改生产代码，未提交 commit，未运行 FinanceBench，未调用 Jina、LLM 或其他外部 API。

## 结论摘要

1. 当前 FinanceBench 主索引是 **PDF 页内 L3 文本块索引**：800 个估算 token、128 overlap，只把 L3 写入 Milvus；`parent_chunks` 当前为 0。
2. Dense 和 Milvus 原生 BM25 都检索同一个 `text` 字段。当前不存在独立的 `raw_text`、`embedding_text` 或结构化 `search_text`。
3. PostgreSQL 已有 5,200 个 `DocumentPage` 和 1,936 个 `DocumentTable`，但 Milvus 的 6,542 条记录全部是 `text_chunk`；表格摘要/表格行没有进入 Dense 或 BM25 索引。
4. 当前 chunk 没有 `document_id`、`section`、`bbox`、token 长度或表格外键。`filename + page_number` 是实际使用的关联键。
5. 存在一个必须优先解决的数据契约问题：文本页号由 PDF loader 产生，**从 0 开始**；pdfplumber 表格页号 **从 1 开始**。当前按相同 `filename + page_number` 关联会把表格接到下一张物理 PDF 页。
6. 表格恢复能力是“选中页面后按页查 PostgreSQL”的旁路，不是可检索的一等证据。抽样显示表格质量不稳定，既有可用财务行，也有严重错列和正文误识别。
7. 已有报告的 30 题中，Dense@100 为 27/30（90.0%），RRF@100 为 23/30（76.7%）：5 题被融合实质性挤出，BM25 只救回 1 题；另有 11 题在冻结候选阶段命中后未进入最终选择。
8. 下一阶段应优先选择 **方案 A：保留当前 chunk embedding，修复页号契约、page selection 和 table evidence reconstruction**。方案 B 的结构增强 `retrieval_text` 应作为独立 shadow index 的 30 题 A/B，而不是先重建正式索引。

---

## 一、当前索引和 chunk 结构

### 1.1 PDF 解析入口

入口位于 `backend/document_loader.py`：

- 默认 `PDF_TEXT_BACKEND=pdfium`，使用 `pypdfium2.PdfDocument` 和 `page.get_textpage().get_text_range()`。
- 若 `pypdfium2` 不可用，回退到 LangChain `PyPDFLoader`。
- PDFium loader 为每页生成 `Document(page_content=text, metadata={"source": ..., "page": page_index})`。
- `page_index` 从 0 开始；`_resolve_page_number()` 不做 `+1`，因此 `DocumentPage` 和文本 chunk 的 `page_number` 是 **0-based PDF physical index**。
- 文本经过 `sanitize_text()`，没有保存 PDF 字符/词的坐标信息。

FinanceBench 重建入口是 `scripts/rebuild_financebench_index.py`：

- 固定 `DocumentLoader(chunk_size=800, chunk_overlap=128, include_parent_chunks=False)`。
- 明确设置 `TABLE_AWARE_INGESTION=false`。
- 只导入测试集对应的 40 个 PDF。
- 重建时删除原有 `DocumentTable`、`DocumentPage`、`ParentChunk`，随后写入页面和 L3 文本块。

当前数据库只读计数：

| 存储 | 当前数量 | 含义 |
|---|---:|---|
| PostgreSQL `document_pages` | 5,200 | 完整页面文本及页面 embedding |
| PostgreSQL `document_tables` | 1,936 | 后续 backfill 得到的结构化表 |
| PostgreSQL `parent_chunks` | 0 | 当前冻结索引没有 L1/L2 父块 |
| Milvus `embeddings_collection` | 6,542 | 全部为 L3 `text_chunk` |

### 1.2 chunk 生成逻辑

`DocumentLoader` 定义了三级 splitter，但 FinanceBench 重建关闭父块，只直接运行 L3 splitter：

| 层级 | 最大估算 token | overlap | 当前 FinanceBench 是否生成/保存 |
|---|---:|---:|---|
| L1 | 1,600 | 256 | 否 |
| L2 | 800 | 128 | 否 |
| L3 | 800 | 128 | 是，写入 Milvus |

长度函数不是 BGE tokenizer，而是本地正则估算：金额/数值/百分比、英文词、中文字符及其他非空白字符各计一个 token。分隔符依次为双换行、换行、句点、分号、逗号、空格和字符级兜底。

chunk ID 格式：

```text
{filename}::p{page_number}::l{level}::{page_local_index}
```

`chunk_idx` 是文件内递增序号；`content_hash` 是 chunk 文本的 SHA-256。

### 1.3 embedding 生成逻辑

实现位于 `backend/embedding.py`：

- 默认模型：`BAAI/bge-m3`。
- 默认维度：1,024。
- CUDA 可用时使用 GPU，默认 FP16；向量做 L2 normalize。
- Transformers 兼容 fallback 使用 tokenizer/model，取 CLS 向量后归一化。

FinanceBench 重建一次性生成两类向量：

1. 页面向量：`_page_retrieval_text(page)`；保存到 PostgreSQL `DocumentPage.page_dense_embedding`。
2. L3 chunk 向量：直接对 `leaf["text"]` embedding；保存到 Milvus `dense_embedding`。

页面检索文本最多默认 3,000 字符，拼接方式为：页面头部 1/2 + 启发式 `table_text` 1/4 + 页面尾部 1/4。它仍然是截断的原始文本组合，不是结构化字段描述。

一个细节：重建脚本先对原 `leaf["text"]` 生成向量，随后 writer 才把写入 Milvus 的文本裁到最多 7,500 字符。正常 800-token chunk 通常不会触发，但 schema 没有保证“embedding 输入”和“BM25/存储文本”在极端长 token 情况下绝对一致。

### 1.4 BM25 索引生成逻辑

当前 `MILVUS_SPARSE_MODE=milvus_bm25`：

- Milvus collection 的 `text` 字段启用 `standard` analyzer 和 match。
- Milvus Function `text_bm25` 以 `text` 为输入、`sparse_embedding` 为输出。
- 写入时不再维护应用侧自制 BM25 语料统计。
- 查询时把原 query 文本直接传给 Milvus BM25。

因此当前：

```text
Dense document text = L3 chunk text
BM25 document text   = 同一个 L3 chunk text
```

不存在独立的财务结构化 BM25 文本。

### 1.5 当前真实 Milvus schema

以下来自当前 collection 的只读 `describe_collection`，不是根据 model 推测：

| 字段 | 类型/限制 | 当前用途 |
|---|---|---|
| `id` | INT64 auto primary | Milvus 主键 |
| `dense_embedding` | FLOAT_VECTOR(1024) | BGE-M3 dense |
| `text` | VARCHAR(8192), analyzer enabled | 原 chunk 文本，同时作为 BM25 输入 |
| `sparse_embedding` | SPARSE_FLOAT_VECTOR | Milvus BM25 Function 输出 |
| `filename` | VARCHAR(255) | 文档标识/过滤 |
| `file_type` | VARCHAR(50) | PDF 等 |
| `file_path` | VARCHAR(1024) | 来源路径；retrieval 默认不返回它 |
| `page_number` | INT64 | 当前文本索引为 0-based |
| `chunk_idx` | INT64 | 文件内 chunk 序号 |
| `company` | VARCHAR(255) | 从文件名推导 |
| `report_year` | INT64 | 从文件名首个年份推导 |
| `financial_document_type` | VARCHAR(50) | 10-K/10-Q/earnings/financial_document |
| `location` | VARCHAR(255) | `page:{page_number}` |
| `content_hash` | VARCHAR(64) | chunk 文本 SHA-256 |
| `chunk_id` | VARCHAR(512) | 稳定 chunk ID |
| `parent_chunk_id` | VARCHAR(512) | 当前 FinanceBench 均为空 |
| `root_chunk_id` | VARCHAR(512) | 当前 FinanceBench 均为空 |
| `chunk_level` | INT64 | 当前检索数据为 3 |
| `evidence_type` | VARCHAR(50) | 当前 6,542 条均为 `text_chunk` |
| `table_id` | VARCHAR(512) | 当前文本块均为空 |
| `row_id` | VARCHAR(512) | 当前文本块均为空 |
| `table_title` | VARCHAR(1024) | 当前文本块均为空 |

用户指定字段核对：

| 期望字段 | 当前是否保存 | 当前真实形态/缺口 |
|---|---|---|
| `document_id` | 否 | 实际以 `filename` 代替，不是稳定独立 ID |
| `file_name` | 是（异名） | 字段名是 `filename` |
| `company` | 是 | 文件名中年份之前的前缀 |
| `page_number` | 是 | 文本为 0-based；表格为 1-based |
| `chunk_id` | 是 | filename/page/level/local index |
| `section` | 否 | 未抽取、未存储 |
| `table_id` | schema 有，当前 chunk 无 | 所有当前 Milvus 记录为空 |
| `bbox` | 否 | PDF loader 只取纯文本 |
| `source_type` | 否 | 最接近的是 `evidence_type=text_chunk` |
| token 长度 | 否 | 生成时估算，但不保存 |
| 原始文本 | 部分 | `DocumentPage.page_text` 保存页文本；chunk 只保存清洗后的 `text` |
| embedding 文本 | 否（独立字段） | L3 embedding 输入为 chunk `text`；页面 embedding 输入不落盘 |

### 1.6 PostgreSQL 页面 schema

`DocumentPage` 保存：`id, doc_name, filename, file_type, file_path, page_number, company, report_year, financial_document_type, location, content_hash, page_text, table_text, chunk_ids, embedding_cache_key, page_dense_embedding, page_tokens, page_numbers, page_years, page_metric_tokens, updated_at`。

这里的 `table_text` 只是行级启发式抽取：包含 `|`、tab，或同时含数字和两个以上连续空格的行。5 页抽样中该字段全部为空，说明 PDFium 提取后的排版空格经常不足以触发此规则。

---

## 二、TableStore 审计

### 2.1 是否存在表格解析

存在，入口为 `backend/table_parser.py`：

- `docling`：可选；支持关闭 OCR 和超时配置。
- `pdfplumber.extract_tables()`：矩阵提取。
- `pdfplumber_words`：`backend/table_reconstructor.py` 按 word 坐标重建行列，并经过质量门控和标准化。
- `auto`：先 Docling，失败后 pdfplumber；具体 fallback 见配置。

当前 1,936 张表不是 FinanceBench 主重建产生的，而是后续 `scripts/backfill_finance_tables.py` 回填。主重建会先清空表，再以 `TABLE_AWARE_INGESTION=false` 导入页面/chunk。

### 2.2 table_id 与保存字段

table ID 已生成：

```text
{filename}::table::p{page_number}::{table_index}
```

`DocumentTable` 当前真实 schema：

| 信息 | 是否保存 | 说明 |
|---|---|---|
| table title | 是 | `title`；优先保存 normalized title |
| caption | 是 | `caption` |
| page 范围 | 否 | 只有单个 `page_number`，没有 start/end 或跨页关系 |
| row | 是 | `rows` JSON；优先保存 normalized rows |
| column | 是 | `columns` JSON；优先保存 normalized columns |
| cell | 间接 | cell 是 row dict 的值，没有独立 cell ID/schema |
| header | 部分 | 最终表头折叠为 `columns`；header rows/层级没有单独持久化 |
| unit | 部分 | `normalized_unit` 被拼到 `before_context`，没有独立字段 |
| scale | 部分 | 与 unit 相同，没有独立 scale 字段 |
| bbox | 否 | parser 中间态可使用 word 坐标，但 DB 不保存 table/cell bbox |
| raw matrix/lines | 否 | reconstructor 中间态存在，TableStore 未持久化 |
| parser backend/quality | 否 | 解析器产生，TableStore 未持久化 |
| HTML/CSV | 是 | `html`, `csv_text` |
| before/after context | 是 | `before_context`, `after_context` |

### 2.3 表格是否进入检索索引

仓库有 `build_table_evidence_docs()`，能构造：

- `table_summary`
- `table_row`
- `table_raw`

这些文档的文本确实是结构增强格式，如 Document/Page/Table ID/Title/Columns/Row Values。但当前实际 Milvus 只读统计是：

```text
text_chunk: 6542
table_summary: 0
table_row: 0
table_raw: 0
```

因此当前表格：

- **未进入 Dense 索引**；
- **未进入 BM25 索引**；
- 仅保存在 PostgreSQL，页面已经被选中后才按 `filename + page_number` 加载；
- 表格不能帮助发现正确文档或正确页面。

上传 API 中存在写表和构造 table evidence 的代码路径，但当前 `table_indexer.py` 的注释也把它称为 debug/dry-run evidence；实际冻结 FinanceBench collection 没有这些记录。

### 2.4 5 个确定性随机样本

抽样方法：从当前 `DocumentTable` 的不同 `filename + page_number` 中，按其字符串 MD5 排序取前 5 个；没有重新解析 PDF。以下只展示短摘录，完整数据仍在 PostgreSQL/Milvus。

#### 样本 1：JPMORGAN 2021Q1，page 92

- 页面原文：关于 fair value、MSR、DVA 的脚注和解释。
- chunk：2 个 L3 chunk，`p92::l3::0/1`；均无 `table_id`。
- 对应 TableStore：`p92::5`，1 个 `Metric` 列、5 行，行数据把正文词语和数值错配为大量动态列。
- 判断：表格结构不可用于可靠计算；chunk 无法直接指向该表。

#### 样本 2：AMERICANEXPRESS 2022 10-K，page 134

- 页面原文：derivative assets/liabilities 及 fair value 表。
- chunk：1 个 L3 chunk；保留原页面展平表格文本，无 `table_id`。
- 对应 TableStore：`p134::5`，正文被错误拆成大量列，4 行。
- 判断：页面文本可检索，但结构化表不可安全作为 cell evidence。

#### 样本 3：AMCOR 2020 10-K，page 111

- 页面原文：季度净销售、毛利、净利润和 EPS。
- chunk：1 个 L3 chunk；文本中数值序列保留较完整。
- 对应 TableStore：`p111::1` 的行却是 Cash/receivables/inventories 等内容，与该页面原文不一致。
- 判断：这是页号错位的直接样本；当前 same-page table attach 会附错表。

#### 样本 4：JPMORGAN 2021Q1，page 17

- 页面原文：Business Segment Results 的说明。
- chunk：1 个 L3 chunk。
- 对应 TableStore：TCE reconciliation，包含 Goodwill、Other intangible assets 等较规整行。
- 判断：表本身可用，但不属于当前 `DocumentPage.page_number=17` 的页面文本，仍是页号错位。

#### 样本 5：PEPSICO 2023Q1 Earnings，page 6

- 页面原文：Cash Flows continued，Financing Activities。
- chunk：1 个 L3 chunk。
- 对应 TableStore：Cash Flow 主表中的 Net income、Depreciation、Juice Transaction 等，29 行。
- 判断：表行质量较好，但更像前一物理页的主表；按相同页号关联仍不可靠。

### 2.5 `chunk → page → complete table` 是否成立

当前只能部分成立：

```text
chunk.filename + chunk.page_number
    → DocumentPage（可靠，来自同一 loader）
    → DocumentTable（不可靠，另一套 1-based 页号）
```

不能认为 TableStore 已足够支持稳定恢复完整表，原因是：

1. chunk 没有 `table_id` 或 bbox，只有页面级弱关联；
2. 文本页号 0-based、表格页号 1-based；
3. 没有跨页表的 page range/continuation ID；
4. 表格结构质量门控后的结果仍有正文误识别和错列；
5. header hierarchy、cell 坐标、unit/scale、解析质量没有结构化持久化；
6. 当前表格不参与候选检索，只有页面已经选中后才可能被加载。

---

## 三、当前 Retrieval 流程

### 3.1 需要区分的两条真实路径

仓库同时存在：

1. **冻结 Core v3/Skills 在线回答路径**：`rag_orchestrator` 调用 hybrid RRF、chunk rerank、Core v3 page selector、Context Budget v3。
2. **Retrieval Core v4 实验原语**：`rag_core_v4.py` 中的 Dense-primary、neighbor expansion、document-local retrieval，由诊断脚本直接调用。

`runtime_profile.py` 已注册 v4 profile 名，但 `uses_clean_baseline_path()` 和 `uses_rag_core_v3_path()` 并未把 v4 profile 纳入在线 Core v3 回答分支。因此，不能把“v4 诊断脚本已验证”自动等同于“聊天 API 已切换到 v4”。本审计不修改这一状态。

### 3.2 冻结 hybrid/Core v3 回答路径

```text
Query: 原始问题字符串
  ↓ BGE-M3 query embedding + 原始 query text
Milvus Dense Top(2K) + Milvus BM25 Top(2K)
  ↓ Milvus RRFRanker(k=60)，截为 candidate_k
Candidate L3 chunks
  ↓ 可选 page-first；冻结 Core v3 profile 默认关闭
Chunk rerank/finalize（远端 Jina 或本地降级，按运行配置）
  ↓ candidate chunks + reranked chunks
Core v3 page aggregation / document score / diversity selection
  ↓ 最多 8 个 page keys
PostgreSQL open_pages
  ↓ 完整 page_text
按 selected filename/page 加载 TableStore
  ↓ same-page table（当前存在页号错位风险）
Context Budget v3 组装 evidence（默认最多 28,000 chars）
  ↓ evidence + citations
LLM
```

各阶段 metadata：

| 阶段 | 输入 | 输出 | metadata 变化/丢失 |
|---|---|---|---|
| Dense | query vector；Milvus `dense_embedding` | L3 hits | 返回 chunk/page/company/year/type/hash/table 字段；不返回 `file_path` |
| BM25 | 原始 query text；Milvus `text` analyzer | L3 hits | 同上 |
| RRF | Dense/BM25 rank lists | 融合 chunk list | 保留实体字段和融合 score，但默认结果没有独立 dense rank/BM25 rank；诊断脚本需分路重跑才得到 |
| candidate fusion | 一条或多条 query route | 去重 chunks | 当前 planner 默认关；field-aware 由 profile 决定 |
| chunk rerank | question + chunk `text` | reranked chunks | 增加 rerank score/rank；候选截断会丢失未送入 rerank 的页 |
| page aggregation | candidate + reranked chunks | ranked pages/documents | 使用 `filename/page_number` 聚合；chunk 文本只保留在 `best_chunk` 等内部结构 |
| page selection | ranked pages | selected page keys | 截至 final page budget；未选页全部丢失 |
| open page | page keys | PostgreSQL full page records | 恢复 `page_text`、page embedding/tokens 等页面字段 |
| table attach | selected page keys | PostgreSQL tables | 仅同数字页；当前 0/1-based 不一致 |
| context construction | full pages + tables | evidence string | 转为文本块和引用；大量结构 metadata 不会全部呈现给 LLM |

### 3.3 Retrieval Core v4 实验路径

当前 v4 原语不使用等权 RRF：

```text
Query
  ↓
Dense Top120（保持顺序）
  + BM25 Top30 中 Dense 未出现的 chunk 追加
  ↓
merged chunks（dense_rank / bm25_rank / merged_rank）
  ↓
可选同文档 page ±1 expansion
  ↓ PostgreSQL 预计算 page embedding + lexical + seed rank
page candidates
  ↓
document ranking / document-local Dense+BM25 / global-local merge（实验 profile）
  ↓
现有 page selector / Context Budget（由评测脚本连接）
```

`retrieval_document_local` 对 top 3 documents 分别运行一次 Dense 和 BM25；每个文档默认保留 20 个 Dense slot，再用 BM25 补到 30。它仍是实验代码，没有 Page-level Jina。

### 3.4 Dense/BM25 实际检索文本

| 检索对象 | Dense 使用文本 | BM25 使用文本 | 是否结构增强 |
|---|---|---|---|
| L3 chunk | `leaf["text"]` | Milvus 同一 `text` | 否 |
| Page vector | head + heuristic table text + tail，最多 3,000 字符 | 无独立 page BM25 | 否 |
| Table | 当前索引无 table embedding | 当前索引无 table BM25 | 不适用 |

结论：二者不是未经任何处理的 PDF binary/raw layout，而是 `sanitize_text` 后的展平文本；但从检索 schema 看，它们都属于同一 raw-like chunk text，没有 `Section/Table/Rows` 形式的结构化 `search_text`。

---

## 四、BM25/RRF 诊断

### 4.1 数据来源和口径

本节只读取已有文件：

- `reports/retrieval_recall_k_ablation.json`：固定 30 题、K=40/60/80/100/120。
- `reports/retrieval_funnel_audit.json`：已有 100 题只读漏斗和逐题 rank，用于解释用户提到的 Dense@100=85%、RRF@100=81%。

没有运行新 retrieval。30 题是既有诊断集，不是本次重新抽样。

### 4.2 30 题 recall

| K | Dense page hit | Dense+BM25 RRF candidate hit | BM25 page hit |
|---:|---:|---:|---:|
| 40 | 63.3% | 46.7% | 20.0% |
| 60 | 73.3% | 63.3% | 30.0% |
| 80 | 80.0% | 70.0% | 30.0% |
| 100 | **90.0% (27/30)** | **76.7% (23/30)** | 33.3% |
| 120 | 93.3% | 83.3% | 36.7% |

### 4.3 gold page rank 分类

这些类别用于定位不同阶段，**不是互斥的总和分区**；A 是融合影响，D 是融合后的下游损失，同一题可能同时出现。

| 类别 | 定义 | 固定 30 题 | 比例 | 结论 |
|---|---|---:|---:|---|
| A：Dense 找到、融合实质丢失 | Dense rank ≤100，但 RRF rank >100/不存在 | 5 | 16.7% | BM25 弱排名经等权 RRF 后把正确页挤出 |
| A2：Dense rank 被降低 | Dense rank ≤100，且 RRF rank > Dense rank | 21 | 70.0% | 融合普遍稀释 Dense 排名；不一定都越过 cutoff |
| B：BM25 补充成功 | Dense rank >100/不存在，但 RRF rank ≤100 | 1 | 3.3% | sparse 的真正救回收益很小 |
| C：Dense/BM25 都失败 | 两路 rank 均 >100/不存在 | 2 | 6.7% | 需要结构化检索文本或其他召回实验 |
| D：候选命中后选择丢失 | 冻结候选阶段命中，但最终 selected page 未命中 | 11 | 36.7% | 当前最大可操作瓶颈仍在 page ranking/selection |

A 的 5 题 rank：

| FinanceBench ID | Dense rank | BM25 rank | RRF rank |
|---|---:|---:|---:|
| `00222` | 56 | 未命中 | 107 |
| `00438` | 92 | 未命中 | 170 |
| `00464` | 65 | 未命中 | 102 |
| `00299` | 72 | 未命中 | 141 |
| `00302` | 83 | 未命中 | 146 |

B 的 1 题：`01930`，Dense 未进 Top100，BM25 rank 54，RRF rank 100。

C 的 2 题：`00216`、`00720`。其中 `00216` 的 Dense rank 为 102，属于阈值边缘；`00720` 两路均未召回。

D 的 11 题：`01275, 01351, 00678, 00215, 01028, 00540, 00702, 00711, 00603, 00566, 03882`。

### 4.4 为什么全量已有报告是 Dense 85%、RRF 81%

已有 100 题漏斗中：

- Dense@100：85/100。
- BM25@100：52/100。
- 独立 Dense+BM25 RRF@100：81/100。
- Dense@100 被 RRF 挤出：6 题。
- Dense 未进 Top100、由 BM25/RRF 救回：2 题。
- 两路都未进 Top100：12 题。
- 48 题的 RRF rank 比 Dense rank 更差。
- 冻结候选命中但最终未选择：26 题。

净变化正好是 `-6 + 2 = -4`，解释了 85% → 81%。原因不是 BM25 实现报错，而是 **等权 RRF 假设两路同等可靠**；当前 BM25 对展平财务文本的 page recall 明显低于 Dense，导致 Dense-only gold page 得到的融合分不足。

已有漏斗还记录：20 题的正确页在冻结 RRF 中存在，但位于 Jina 输入 cutoff 之外；7 题已进入 page ranking 但未被最终选择。这进一步说明“扩大或重建检索”不是唯一问题，candidate cutoff 和 page selection 同样关键。

---

## 五、下一阶段方案比较

### 方案 A：保留当前 chunk embedding，增强页面聚合与表格证据恢复

内容：

1. 先统一页号契约，明确所有 internal page number 都用 0-based 或 1-based，并在 ingestion 边界只转换一次。
2. 为 TableStore 增加可验证的 page linkage；在实验数据中校验 `table page ↔ DocumentPage text`。
3. 保持 Dense-primary 宽召回，优先解决 document 内 page ranking。
4. 固定最终页面/token budget；选中页面后恢复完整表或相关行，而不是增加页面数。
5. 对低质量、错列、错页表设置 deterministic quality gate；失败时回退原页面文本，不把错误结构作为权威证据。

优势：

- 直接对应 30 题中 11 个 downstream selection loss。
- 不需要先重建 6,542 个向量。
- 不破坏已经验证的 Dense recall。
- 能在固定 retrieval 输出上单独验证 table/text assembly，因果归因清晰。

风险：

- 当前表格页号和结构质量必须先修，否则 table reconstruction 会主动注入错误证据。
- 对 C 类真正 candidate miss 的帮助有限。

### 方案 B：重构 structure-enhanced `retrieval_text` 后重新 embedding/BM25

示例：

```text
Document: <filename/company/report year>
Section: <statement/section>
Table: <title>
Columns: <periods>
Rows: <row labels>
Content: <original evidence text>
```

优势：

- 可能改善表标题、财务行名、年份和 statement type 的 BM25/Dense 匹配。
- 对 C 类两路都失败、以及 document 内表格页面定位有潜在帮助。
- 可让表格成为可检索的一等证据。

风险：

- 当前没有可靠 `section`，表格 title/header/unit 也不稳定；直接重建会把错误结构编码进两种索引。
- 若替换而不是并行保留 raw text，可能损害叙述性 MD&A/脚注查询。
- 重建成本大，且无法单独证明改进来自结构文本还是新的表格 parser 数据。

### 优先级判断

**优先方案 A，但把“统一页号与表格质量契约”作为 Phase 0。**

依据不是主观偏好，而是当前数据：

- Dense@120 已有 93.3%，说明宽召回底座不是第一瓶颈。
- BM25 真正救回很少，等权融合反而产生明显负迁移。
- 候选到最终页面的损失大于两路完全 miss。
- 当前表格完全不在索引中，且 same-page attach 存在确定性 off-by-one。

方案 B 不应取消，而应在方案 A 建立可靠结构数据之后，用 shadow collection 做 30 题 A/B。结构文本应与原始文本共存，例如 `raw_text` 和 `retrieval_text` 分列/分 evidence type，而不是不可逆覆盖。

---

## 六、最小 A/B 实验计划

### Phase 0：数据契约验收（不做答案评测）

目的：避免在错误页号和错表上做后续实验。

固定抽样 30 个表格页，验证：

- `DocumentPage.page_number` 与 table parser page number 的转换；
- table title/row token 是否真实出现在对应页或允许的跨页窗口；
- 表格是否跨页；
- quality score、parser backend、unit/scale 是否可追溯；
- chunk/page/table 三者是否能用稳定 ID 关联。

通过标准建议：页号关联错误为 0；低质量表不进入 authoritative assembly；原页面文本 fallback 始终存在。

### 实验 1：当前 embedding vs structure-enhanced embedding

只跑固定 30 题 retrieval-only，不调用 LLM/Jina，不跑 100 题。

| 组别 | 索引文本 | 其他条件 |
|---|---|---|
| A1 | 当前 L3 `text` | 当前 BGE-M3、同 K、同过滤 |
| B1 | `metadata header + original chunk text` | shadow collection，不覆盖正式索引 |
| B2（可选） | table summary/row 独立 evidence + 原 text chunk | 同一 shadow collection，以 evidence type 区分 |

结构字段只能来自已验证 ingestion 数据，不从问题或 benchmark gold 注入。

指标：

- candidate page hit@40/100/120；
- gold page rank / MRR；
- document hit/rank；
- table hit@K：gold evidence page 上正确 table/row 是否命中；
- Dense-only、BM25-only、融合分别报告，避免只看融合总分；
- 索引条数、构建时间、查询 latency。

成功条件：结构增强在 candidate/page/table hit 上稳定提升，并且叙述性/非表格回归题不下降；不能只改善少数已知指标词。

### 实验 2：当前 page context vs table/text evidence assembly

固定实验 1 的 retrieval 和 page selection 结果，不重新检索，使差异只来自 evidence assembly。

| 组别 | Context |
|---|---|
| A2 | 当前完整页面窗口 + Context Budget v3 |
| B2 | 同一页面集合；相关完整表/相关行 + 必要表头 + 剩余页面文本，保持同预算 |

规则：

- 不增加 final page K；
- 不增加总 token/char budget；
- 表格低质量或关联不确定时自动回退 A2；
- 每题最多使用现有的一次答案 LLM 调用，不新增调用；
- answer prompt、skills、contract 完全冻结。

指标：

- context hit：gold evidence 是否实际出现在最终 context；
- table context hit：所需表头、row label、period、value 是否同时存在；
- 输入 token、context chars；
- 截断/丢弃的 page/table evidence；
- 固定 judge 或 deterministic numeric evaluator 的 answer accuracy；
- 新增正确、回退题和表格 fallback 率。

### 推荐开发顺序

1. **页号和 ID 契约**：统一 0/1-based，定义 `document_id/page_id/table_id` 关系；先写审计/迁移工具，不直接改正式索引。
2. **表格质量与可追溯字段**：保留 parser backend、quality、page range、header hierarchy、unit/scale、raw lines，确保低质量可拒绝。
3. **方案 A 的固定结果实验**：page selection 后 table/text assembly，先证明 context 恢复收益。
4. **Dense-primary 与 page ranking**：继续使用 v4 30 题诊断，不使用等权 RRF 覆盖 Dense 顺序。
5. **方案 B shadow index**：结构增强文本与 raw text 并存，只跑 retrieval-only 30 题。
6. 只有 30 题同时满足 candidate/context 提升、回归可控、预算不增，才考虑完整 benchmark。

---

## 审计依据

主要代码：

- `backend/document_loader.py`
- `scripts/rebuild_financebench_index.py`
- `backend/embedding.py`
- `backend/milvus_writer.py`
- `backend/milvus_client.py`
- `backend/models.py`
- `backend/table_parser.py`
- `backend/table_reconstructor.py`
- `backend/table_store.py`
- `backend/table_indexer.py`
- `backend/rag_utils.py`
- `backend/rag_core_v3.py`
- `backend/rag_core_v4.py`
- `backend/rag_orchestrator.py`
- `backend/runtime_profile.py`

已有报告：

- `reports/retrieval_recall_k_ablation.json`
- `reports/retrieval_funnel_audit.json`

本次只读运行：PostgreSQL 行数/样本 SELECT、Milvus collection schema/记录 SELECT、已有 JSON 的本地统计。没有生成新的 benchmark 结果。
