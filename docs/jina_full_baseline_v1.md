# Jina Full Baseline v1

## 检索与重排深度环境变量

该独立实验 profile 支持下列正整数环境变量；未设置时使用括号中的默认值：

- `DENSE_TOP_K`（240）：独立 Dense 召回深度。
- `BM25_TOP_K`（240）：独立 Milvus BM25 召回深度。
- `RRF_TOP_K`（120）：实验脚本本地 RRF 融合后冻结的候选数。
- `JINA_INPUT_K`（120）：从 RRF 有序候选前缀中送入 Jina 的数量。
- `JINA_OUTPUT_K`（8）：重排后交给既有 28K context builder 的 chunk 数量。

这些参数只作用于 `scripts/run_jina_full_baseline_v1.py`。为保证 Dense/BM25 深度可以独立配置，实验脚本分别调用现有只读检索接口并在脚本内做 RRF；`backend/milvus_client.py` 和生产路由未修改。约束为 `JINA_OUTPUT_K <= JINA_INPUT_K <= RRF_TOP_K`。

PowerShell 示例：

```powershell
$env:DENSE_TOP_K = "240"
$env:BM25_TOP_K = "120"
$env:RRF_TOP_K = "120"
$env:JINA_INPUT_K = "60"
$env:JINA_OUTPUT_K = "8"
conda run --no-capture-output -n rag python -u scripts\run_jina_full_baseline_v1.py --stage check
```

## 离线 Jina 深度分析

下列命令只读取固定30题的统一 RRF Top120 snapshot 与已有 Jina cache，不访问网络，也不调用 LLM/Judge：

```powershell
conda run --no-capture-output -n rag python -u scripts\analyze_jina_reranker_depth.py
```

输出：

- `reports/jina_reranker_depth_analysis_30.json`
- `reports/jina_reranker_depth_analysis_30.md`

分析会把缓存的完整 Jina 排名分别过滤到原始 RRF 前 `40/60/80/100/120`，固定 `JINA_OUTPUT_K` 和 28000 chars context，统计 candidate recall、gold page rank、context hit。Token 成本使用每题既有 Top120 reported tokens 按输入字符比例校准估算，不代表真实 tokenizer 计数或货币账单。

### 收敛后的两个profile

- `jina_full_baseline_input120_v1`：质量主基线，默认 `JINA_INPUT_K=120`, `JINA_OUTPUT_K=12`。
- `jina_full_baseline_input80_v1`：成本对照，默认 `JINA_INPUT_K=80`, `JINA_OUTPUT_K=10`。

输出深度报告由以下命令离线生成：

```powershell
conda run --no-capture-output -n rag python -u scripts\analyze_jina_output_depth.py
```

结果位于 `reports/jina_output_depth_analysis_30.md`。正式100题主基线使用input120 profile；input80只作为独立成本A/B，不能将两者结果混合汇总。

## 定位

这是独立、可恢复、可审计的强RAG基线，不修改生产路由，也不删除Core v3、BGE、Skills和所有shadow实验。它是后续Evidence/Fusion/Answer实验的固定参照，不是理论上限。

固定链路：

```text
FinanceBench question
  → 现有BGE-M3 query embedding
  → Milvus Dense + 原生BM25 + RRF Top120
  → jina-reranker-v3一次重排全部120个原始chunks
  → 按Jina顺序Top8原始chunks（带文件/内部0-based页码，≤28000 chars）
  → 既有clean-baseline回答提示词
  → DeepSeek-V4-Flash正式版
  → 全部回答成功后，DeepSeek-V4-Pro strict Judge
  → 本地JSON/JSONL/Markdown报告
```

profile为`configs/experiments/jina_full_baseline_v1.json`。v1默认关闭Skills、Agent、Planner、LangSmith、query rewrite和本地reranker fallback。原因是建立可归因的纯RAG参照；旧Skills实验完整保留。如要比较Jina + Skills，应另建v2 profile，不能覆盖v1。

默认 Jina 候选与先前30题shadow契约一致：每题RRF Top120、一次Jina请求，API为获得完整可验证排列使用 `top_n=JINA_INPUT_K`；随后仅将重排前 `JINA_OUTPUT_K`（默认8）交给28K context builder。不会把结果送回旧page selector，但会增加来源标题供回答引用。

## 模型与预算

- Reranker：`jina-reranker-v3`，官方HTTPS endpoint；无fallback；成功请求最小间隔60秒。
- Answer：`deepseek-v4-flash-ga-260731`，temperature 0.1，thinking disabled，max completion 1024，timeout 60秒，应用层不重试。
- Judge：`deepseek-v4-pro-ga-260813`，temperature 0，thinking disabled，max completion 512，timeout 60秒；仅在全部答案成功后逐题运行。
- Answer/Judge模型固定在profile，不依赖`.env`中的旧MODEL名称；API地址仍使用现有火山方舟兼容地址。
- 旧30题Jina缓存只有在question、完整candidate hash、model和官方endpoint全部一致时才能复用；否则重新调用，不按ID套用旧排名。
- 历史30题Jina成功响应报告2879092 token，约95970 token/题。粗略外推：100题最多约960万rerank token；若30题均严格命中缓存，新调用约70题、约672万token。实际文档长度不同，且失败请求计费未知，最终以`summary.json`记录的成功响应usage为准。
- 回答prompt最多28K字符，粗略约7K token/题；100题约70万输入token量级，但具体取决于模型tokenizer、实际context和引用标题。它与Jina token为两个独立计量口径。

## 安全与恢复契约

- `check`阶段不访问网络，只验证profile与凭据是否已配置，且不显示密钥。
- `recall`必须显式传`--allow-retrieval`；只执行本地embedding和Milvus查询，不调用Jina/Answer/Judge。
- `run`必须显式传`--allow-paid`；缺少冻结快照时拒绝运行，绝不隐式重建。
- 快照固定dataset、代码、embedding/Milvus配置、collection、问题集合和每题120个chunk的内容/顺序hash。
- Jina成功后立即checkpoint，再构建context和调用回答模型。回答失败后续跑复用Jina，不重复计费。
- 所有答案成功后才开始Judge；Judge失败后续跑不重复Jina或回答。
- Jina HTTP/网络失败不降级BGE或identity，也不伪装“无相关内容”；错误类型安全保存，避免把请求对象或密钥写入报告。
- profile、快照、dataset、核心脚本、prompt、endpoint或问题集合变化会拒绝混用旧checkpoint，应使用新output目录。
- 正式accuracy只在100个Judge全部完成时计算。未完成/invalid Judge记pending/error，绝不自动算incorrect。

## 推荐运行顺序

### 1. 离线检查

```powershell
conda run --no-capture-output -n rag python -u scripts\run_jina_full_baseline_v1.py --stage check
```

### 2. 一题smoke（复用已验证的固定30题快照）

这一步不会重新检索；只要候选hash一致就复用已有Jina排名，因此通常只产生1次Answer和1次Judge调用：

```powershell
conda run --no-capture-output -n rag python -u scripts\run_jina_full_baseline_v1.py `
  --stage run `
  --scope diagnostic30 `
  --limit 1 `
  --snapshot reports\reranker_shadow_v1_rrf_top120.json `
  --output-dir reports\jina_full_baseline_v1_smoke1 `
  --allow-paid
```

### 3. 冻结全100题候选

首次执行：

```powershell
conda run --no-capture-output -n rag python -u scripts\run_jina_full_baseline_v1.py `
  --stage recall `
  --scope all100 `
  --limit 0 `
  --output-dir reports\jina_full_baseline_v1_all100 `
  --allow-retrieval
```

中断后运行同一命令，已保存的问题跳过。不要删除或手工编辑`recall.json`。

### 4. 运行/续跑全100题Jina、回答与Judge

```powershell
conda run --no-capture-output -n rag python -u scripts\run_jina_full_baseline_v1.py `
  --stage run `
  --scope all100 `
  --limit 0 `
  --output-dir reports\jina_full_baseline_v1_all100 `
  --allow-paid
```

同一命令用于恢复。不要并发启动两个相同output目录的runner；文件级原子写不能替代跨进程锁。

### 5. 不调用任何模型，仅重新导出报告

```powershell
conda run --no-capture-output -n rag python scripts\run_jina_full_baseline_v1.py `
  --stage report `
  --output-dir reports\jina_full_baseline_v1_all100
```

## 输出

每个output目录包含：

- `recall.json`：100题冻结RRF Top120完整原文、metadata、分数与hash。
- `state.json`：逐题Jina完整120项排名、context、回答、Judge、usage、latency与错误历史。
- `answers.jsonl`：已成功生成的本地答案，可兼容现有导出/分析思路。
- `judge.jsonl`：完成的strict Judge结果。
- `summary.json`：完成数、正式accuracy（不完整时null）、缓存/新Jina次数及token、Answer/Judge token和错误数。
- `answers.md`：**每题问题、参考答案、模型答案、引用、Judge及必要指标**；即使中断也会输出当前状态。

注意：Markdown含benchmark参考答案，只能用于离线分析，绝不能回流到retrieval、reranker、prompt或回答输入。

## 验收标准

正式基线必须满足：

1. 100题candidate和Jina route都成功，Jina provider为官方API，无fallback。
2. 100题回答与Judge全部完成，strict accuracy非null。
3. 每个引用可解析到冻结candidate的真实文件、0-based页码和chunk ID。
4. 每题context≤28000字符、最多8个chunks；回答输入与Judge输入用途分离。
5. `summary.json`、JSONL与Markdown数量一致；报告必须同时包含参考答案，不能只保存Judge标签。
6. 后续实验必须复用该baseline的冻结候选/答案设置，且只改变被研究变量；不能用Jina context hit 63.33%替代正式100题accuracy。
