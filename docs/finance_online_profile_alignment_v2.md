# Finance Online Profile Alignment v2

## 目标

`profile=finance` 的 `auto` 与 `static` 模式使用一条独立、可审计的在线推理链，并对齐已冻结的 final100 首轮回答配置。`agentic` 仍保留为实验模式，但 trace 会明确标记其不属于 final100 对齐链路。

## 固定配置

配置来源：`configs/production/finance_online_v2.json`。

| 阶段 | v2 配置 |
| --- | --- |
| Query Rewrite | 开启；保留原问题；最多 2 个改写 |
| Dense | Top 240 |
| BM25 | Top 240 |
| RRF | Top 120，`rrf_k=60` |
| Jina | 输入 80，输出 12，完整 chunk（`input_max_chars=0`） |
| Context | 最多 28,000 字符 |
| Prompt | `clean_baseline_v1` |
| Answer directives | 关闭 |
| Deterministic calculation contract | 关闭 |
| Finance policy injection | 关闭 |
| Answer model | `deepseek-v4-flash-ga-260731` |
| Generation | temperature `0.1`，thinking `disabled`，max tokens `1024` |

`FINANCE_JINA_MAX_CHARS` 只覆盖 finance profile 的 Jina 输入长度。`0` 表示不截断；通用链路的 `RERANK_REMOTE_MAX_CHARS` 不受影响。

## 首轮调用链

```text
用户问题
  -> Conversation Understanding（只决定是否检索/是否解析追问）
  -> 原问题 + 最多 2 个 Query Rewrite
  -> Dense 240 + BM25 240
  -> RRF Top120
  -> Jina：完整 chunk，Input80 -> Output12
  -> 按 Jina 顺序构建 <=28K 的原始 chunk evidence
  -> Clean Baseline Prompt
  -> DeepSeek-V4-Flash（0.1 / thinking disabled / 1024）
```

首轮没有历史消息时，Memory 层为 evidence 单独保留 28,000 字符；历史消息和摘要使用各自预算，不再从本轮 evidence 配额中扣除。多轮问题仍可使用历史，但不会因此把当前检索 evidence 压到共享 token 余量内。

## Jina 失败语义

- `success`：远程 Jina 或合法的 Jina cache 命中，继续回答。
- `degraded`：仅供其他链路诊断；finance v2 不接受本地 reranker 降级结果。
- `failed`：保留 RRF 候选及失败 trace，但阻止 answer generation，避免把未重排结果伪装成 final100 对齐答案。

Jina cache key 已升级到 `rerank:v2`，身份字段包括：

- `jina_model`
- `jina_input_k`
- `jina_output_k`
- `jina_input_max_chars`
- query 与候选内容哈希

因此旧的 1600 字符缓存不会污染完整 chunk 实验。

## Trace 示例

```json
{
  "profile": "finance",
  "profile_config": "finance_online_v2",
  "retrieval_config": {
    "dense_top_k": 240,
    "bm25_top_k": 240,
    "rrf_top_k": 120,
    "rrf_k": 60
  },
  "jina_config": {
    "provider": "jina",
    "input_k": 80,
    "output_k": 12,
    "input_max_chars": 0
  },
  "rerank_status": "success",
  "prompt_config": {
    "name": "clean_baseline_v1",
    "answer_directives": false,
    "calculation_contract": false,
    "finance_policy": false
  },
  "answer_config": {
    "model": "deepseek-v4-flash-ga-260731",
    "temperature": 0.1,
    "thinking": "disabled",
    "max_tokens": 1024
  },
  "context_budget": 28000,
  "before_memory_context_chars": 24680,
  "after_memory_context_chars": 24680
}
```

`before_memory_context_chars == after_memory_context_chars` 表示本次检索 evidence 没有被 Memory 阶段二次裁剪。实际长度小于 28,000 是正常的，表示 Jina Top12 本身未用满预算。

## 在线核验

服务重启后，可从返回的 `trace_id` 检查单次线上调用：

```powershell
conda run --no-capture-output -n rag python -u scripts\analyze_online_rag_trace.py --trace-id <TRACE_ID>
```

报告中的 `matches_financebench_pipeline=true` 只表示配置与数据通道对齐，不等价于该题答案一定正确。
