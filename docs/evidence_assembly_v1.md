# Evidence Assembly v1 开发报告

> 完成日期：2026-09-01  
> 基于提交：`56c1454 feat: add document local retrieval prototype`  
> 本阶段没有修改 Retrieval Core、Dense、BM25、RRF、Jina、Skills、Prompt 或 Answer Contract；没有运行 FinanceBench。

## 1. 完成结果

Evidence Assembly v1 已完成并接入回答前的证据组装阶段：

```text
selected pages
    ↓ page_id
DocumentTable by page_id
    ↓ deterministic quality gate
trusted table evidence ──否──→ original page_text fallback
    ↓ 是
title + header/columns + target rows + unit/scale + nearby text
    ↓
existing 28,000-character context ceiling
```

检索返回的页面集合和排序保持不变。新逻辑只发生在 selected pages 之后。

## 2. 统一 ID 体系

新增 `backend/evidence_identity.py`。

### document_id

- 优先基于 PDF 文件内容 SHA-256 生成。
- 对 digest 再做一次 SHA-256，截取 60 个十六进制字符，加 `doc_` 前缀，总长 64。
- 只有文件不可读时才使用规范化 filename 的哈希作为兼容 fallback。

格式：

```text
doc_<60 hex characters>
```

### page_id

由 `document_id + 0-based page_number` 唯一生成：

```text
{document_id}:page:{page_number:06d}
```

### table_id

由 `page_id + table_index` 唯一生成：

```text
{page_id}:table:{table_index:04d}
```

Evidence Assembly v1 只使用 `page_id` 加载表格，不再用 `filename + page_number` 作为关联键。旧方法仍保留给未迁移调用方兼容，但不参与新 evidence path。

## 3. 页码统一

内部页码统一采用 PDF loader 的 0-based physical page index。

- PDFium/PyPDF 文本页面：保持原 0-based。
- pdfplumber：在 `TableAwareParser.extract_tables()` 返回边界统一执行 `external page - 1`。
- pdfplumber_words：同样只在 parser 边界转换一次。
- Docling：当前适配层也在统一边界转换，不在 TableStore/Evidence Builder 二次转换。

`start_page`、`end_page` 和 `page_number` 均遵守相同 0-based 合约。

## 4. DocumentTable schema

新增并已迁移：

| 字段 | 含义 |
|---|---|
| `document_id` | 内容稳定文档 ID |
| `page_id` | 表格主页面 ID |
| `start_page` | 跨页表起始页，0-based |
| `end_page` | 跨页表结束页，0-based |
| `parser_backend` | `pdfplumber`、`pdfplumber_words`、`docling` 或 legacy 标记 |
| `quality_score` | parser 质量或迁移期 deterministic structural score |
| `unit` | 表格单位 |
| `scale` | thousands/millions/billions 等尺度 |

`DocumentPage` 同时新增 `document_id` 和 `page_id`。

## 5. 表格质量门控

新增 `backend/table_quality.py`。一张表只有同时满足以下条件才进入 answer evidence：

1. `table.document_id == page.document_id`；
2. `table.page_id == page.page_id`；
3. selected page 位于 `start_page..end_page`；
4. `columns` 和 `rows` 非空；
5. `quality_score >= 0.65`；
6. 表格 title/columns/rows 与实际 `page_text` 的 token match score `>= 0.35`。

任何一项失败，整页回退到原始 `page_text`。门控不会推测或修复 cell，不会把低质量表作为权威证据。

迁移前没有持久化 parser 原始 quality score，因此旧表使用结构完整性生成 conservative score；新解析表直接保存 parser score。页面匹配分始终在组装时基于实际 `page_text` 重新计算。

## 6. Evidence Builder

新增 `backend/evidence_assembly_v1.py`，输入是 question、selected pages 和按 `page_id` 取得的 tables。

可信表输出：

- source、internal page 和 page ID；
- table ID；
- table title；
- header/columns；
- 与问题词项匹配度最高的 target rows；
- unit 和 scale；
- 最多 600 字符的前后附近文本。

无可信表输出：

- 完整原始 `page_text`，在既有上下文预算内裁剪；
- trace 记录 fallback page ID 和每张被拒绝表的原因。

新增 trace：

- `evidence_assembly_version`
- `trusted_table_count/trusted_table_ids`
- `rejected_table_count/rejected_tables`
- `page_text_fallback_count/page_text_fallback_page_ids`
- `quality_threshold/page_match_threshold`
- `answer_context_chars/answer_context_max_chars`

## 7. Context Budget

没有扩大上下文。

- 默认仍读取 `RAG_CORE_V3_MAX_CONTEXT_CHARS=28000`。
- 多个 selected pages 按现有总 ceiling 分配槽位。
- 最终证据再次执行总字符上限检查。
- 单元测试验证最终 evidence 不超过显式 budget。

## 8. 数据迁移与关联审计

新增 `scripts/migrate_evidence_assembly_v1.py`：

- 默认 dry-run；
- `--execute` 才执行 PostgreSQL 事务；
- 不访问或修改 Milvus；
- 不修改 retrieval 配置。

本次已在 dry-run 通过后执行迁移：

```text
DocumentPage migrated: 5,200
DocumentTable migrated: 1,936
```

### 100 张确定性随机表

抽样 key 使用 `filename + corrected internal page + table_index` 的 SHA-256 排序，保证迁移前后样本稳定。

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| page → table 关联错误 | 100/100 | 0/100 |
| 关联错误率 | **100%** | **0%** |
| 通过完整质量门控 | 不允许进入新链路 | 28/100 |
| table evidence 可用比例 | — | **28.0%** |

修复前 100% 的原因是确定性的 page base 不一致：现存回填表由 pdfplumber_words 的 1-based `enumerate(..., start=1)` 产生，而 `DocumentPage` 使用 loader 的 0-based index。相同数字键必然指向下一张物理页。

### 全部 1,936 张表

| 指标 | 结果 |
|---|---:|
| 新 ID 关联失败 | 0 |
| 新 ID 关联错误率 | **0%** |
| 通过质量与页面匹配门控 | 447 |
| table evidence 可用比例 | **23.09%** |
| 自动 page_text fallback | 1,489（76.91%） |

23.09% 是 conservative 可用率，不是 table parser recall。其余表不会丢失页面证据，而是使用原始 `page_text`。

## 9. 测试

新增测试覆盖：

- 100 个随机化 document/page/table ID 唯一关联；
- parser 1-based → internal 0-based 转换；
- DocumentTable 新字段持久化和 page ID 查询；
- trusted table evidence；
- target row、unit、scale 和 nearby text；
- 低质量表原始 page text fallback；
- wrong page ID 拒绝；
- 最终 context budget 不扩大。

运行结果：

```text
407 passed, 12 warnings in 66.45s
```

12 个 warning 均为既有 `datetime.utcnow()` deprecation warning，没有测试失败。

## 10. 修改文件

新增：

- `backend/evidence_identity.py`
- `backend/table_quality.py`
- `backend/evidence_assembly_v1.py`
- `scripts/migrate_evidence_assembly_v1.py`
- `tests/test_evidence_assembly_v1.py`
- `docs/evidence_assembly_v1.md`

修改：

- `backend/models.py`
- `backend/document_loader.py`
- `backend/document_page_store.py`
- `backend/table_parser.py`
- `backend/table_store.py`
- `backend/rag_orchestrator.py`，仅替换 selected-pages 之后的 table lookup/evidence assembly
- `tests/test_table_parser.py`
- `tests/test_table_store.py`

明确未修改：

- `backend/rag_core_v4.py`
- `backend/rag_core_v3.py`
- `backend/rag_utils.py`
- `backend/milvus_client.py`
- `backend/embedding.py`
- reranker/Jina 代码
- Skills、Prompt、Answer Contract

## 11. 当前边界

Evidence Assembly v1 已解决 ID、页码、可信表门控和 fallback 数据链路，但没有尝试提高 table parser 的产出率。当前只有 23.09% 的旧表能作为可信结构证据，属于预期的保守行为。

本阶段到此停止。后续若继续，应先针对被拒绝原因做 parser-quality 诊断；不应通过降低门槛来人为扩大 table evidence 数量，也不应在没有 30 题固定实验前修改 retrieval。
