# Finance Online Profile Alignment v1

## 目标

线上 `profile=finance` 使用与 final100 一致的查询改写、检索、Jina 重排和回答上下文装配方式，同时保留现有 Answer、Prompt 和 Memory 行为。`mode=auto` 仍负责会话层是否检索，但不再把 finance 计算题切换到另一套 agentic 检索参数。

## 固定配置

配置来源：`configs/production/finance_online_v1.json`。

| 阶段 | 修改前线上值 | finance_online_v1 |
|---|---:|---:|
| Query Rewrite | 未执行 | 原问题 + 最多 2 条改写 |
| Dense TopK | 未单独记录（hybrid candidate 40） | 240/query |
| BM25 TopK | 未单独记录（hybrid candidate 40） | 240/query |
| RRF TopK | 40 | 120 |
| Jina | 16→5（部分旧 trace 为 40→5） | 80→12 |
| Answer context budget | 未固化/旧默认 24K | 28,000 chars |
| Local reranker fallback | 可发生 | finance_online_v1 禁用 |

配置按请求显式传递，没有通过临时修改全局环境变量实现，因此并发的 `general` 和 `finance` 请求不会相互污染。

## 调用链

`Conversation policy → prepare_rag_response(profile=finance, mode=auto) → DeepSeek Query Rewrite → per-query Dense240 + BM25240 → RRF120 → existing Jina reranker 80→12 → ranked raw-chunk context (28K) → existing baseline Answer prompt`

`auto` 不再把 finance 复杂题改成内部 agentic loop；显式 `agentic` 模式和其他 profile 的原行为未改。

## Block / Adobe trace 重放

本次重放只调用 Query Rewrite、Milvus 检索和 Jina；没有调用 Answer、Judge，也没有运行 FinanceBench100。

| Query | 修改前 | 修改后 | 关键证据页 |
|---|---|---|---|
| Block FY2020 operating cash flow | 无 rewrite；RRF40；Jina 40→5 | rewrite 成功；Dense/BM25 240；RRF120；Jina 80→12；28K | `BLOCK_2020_10K.pdf` p.89 位于最终 12 页 |
| Adobe FY2015 operating cash-flow ratio | 无 rewrite；RRF40；Jina 16→5 | rewrite 成功；Dense/BM25 240；RRF120；Jina 80→12；28K | 现金流和负债表相关 chunk 均进入最终上下文；包含 `1,469,502` 与 `2,213,556` |

两个重放 trace 均由诊断器判定 `matches_financebench_pipeline=true`。详细结果：

- `reports/online_rag_trace/finance_alignment_v1_before/`
- `reports/online_rag_trace/finance_alignment_v1_after/`

注意：重放确认的是检索与上下文链路对齐，不等同于重新执行 Answer/Judge。

## Adobe 线上失败的进一步诊断

首次对齐只保证了正确页面进入 Jina Top12，但线上随后把命中 chunk 展开为完整页面，再由 compact evidence builder 按单元上限压缩。`ADOBE_2015_10K.pdf` 的资产负债表页面虽然已被选中，但压缩后的证据只保留页面前半段的流动资产，后半段的 `Total current liabilities 2,213,556` 被裁掉。因此 Answer 模型基于实际收到的证据拒绝计算是合理行为。

finance profile 现改为与 final100 基线一致的 `ranked_raw_chunks` 装配：按 Jina 排名顺序直接放入最多 12 个原始 chunk，并严格限制总长度为 28,000 字符；不再先打开整页再进行二次语义压缩。其他 profile 的 Evidence Assembly 行为不变。

修复后 Adobe 冻结重放结果：

- 上下文长度：24,758 chars；
- builder：`ranked_raw_chunks`；
- 同时存在经营活动现金流 `1,469,502` 和流动负债合计 `2,213,556`；
- 复用该上下文调用 Flash，得到 `1,469,502 / 2,213,556 = 0.6639`，两位小数为 `0.66`；
- 验证文件位于 `reports/online_rag_trace/adobe_raw_chunk_fix_validation/`。

## Trace 字段

新 trace 明确保存：

- `profile_config=finance_online_v1`
- `query_rewrite_enabled`
- `query_rewrite_executed`、`retrieval_queries`、rewrite token usage
- `dense_top_k`、`bm25_top_k`、`rrf_top_k`
- `jina_input_k`、`jina_output_k`
- `context_budget`

持久化 trace 中这些字段分别进入 policy/config 和 rerank 参数区，旧 trace 保持可读并显示 `unknown`，不会用当前环境反推历史值。

## 验证

- 定向测试：30 passed。
- 全仓测试：933 passed，12 warnings。
- warnings 均为已有 `datetime.utcnow()` 弃用提示。
