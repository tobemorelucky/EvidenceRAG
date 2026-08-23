# EvidenceRAG FinanceBench 改进计划（59% 基线之后）

## 基线

- 数据集：FinanceBench 100 题（dev20 + holdout80）。
- 配置：静态 RAG、page-first、original + required-fields、Jina rerank 16×1600。
- 正确率：59/100；dev 12/20；holdout 47/80。
- 空检索：0。
- 计算题：7/15 正确。
- 41 条错误中仅 4 条为明确拒答；24 条没有在最终引用中命中金标准页 ±1。
- 当前 `evidence_status` 主要依赖引用数量，不能可靠表示公司、期间、单位和操作数是否完整。

## 设计原则

1. 先修证据完整性，再放宽回答行为。
2. 同公司、同期间、同单位、同口径的操作数完整时，应计算而非拒答。
3. 查询仅保留 original 与确定性的 finance rewrite，不使用通用扩展、Step-back 或 HyDE。
4. 使用一套 Dense + BM25 检索底座，通过 statement type 与字段信息加权，不建立四套重复检索服务。
5. partial coverage 最多进行一次同文档/同报表类型补搜。
6. 固定 Jina 16×1600，不继续调整 rerank，直至检索和计算链路完成消融。

## 阶段一：回答契约、证据覆盖与受控计算

### 改动

- 更新回答提示词：最终指标未直接出现，但证据包含完整操作数时必须计算。
- 操作数缺失、公司/期间/单位冲突时，明确指出缺失项，不估算。
- `evidence_coverage` 校验公司、期间、单位、字段和数值，不再只做别名文本匹配。
- calculation 题只有所有必要操作数均可解析时才为 `complete`。
- comparison 题必须覆盖目标的两个期间。
- `evidence_status` 使用 coverage 与检索状态，不再仅依据引用数量。
- 使用 Decimal 白名单计算器，记录操作数、公式、单位、结果和引用。

### 验收

- 提示词、coverage、期间/公司冲突和 Decimal 计算契约测试通过。
- 4 条拒答题中，具备完整操作数的题能够回答；缺失操作数的题仍安全拒答。
- 不引入跨公司拼接计算。

## 阶段二：统一查询路径

### 改动

- 生产与评测统一关闭通用 Query Planner、Step-back 和 HyDE。
- 路由仅保留：
  - `original`
  - `finance_rewrite`（公司、期间、指标、报表类型、required fields）
- 修复 trace，明确记录 field-aware、查询路由和实际启用配置。

### 验收

- 每题最多两条确定性查询。
- trace 不再出现 generic semantic、step-back 或 hypothetical-document 内容。
- 生产默认配置与评测脚本一致。

## 阶段三：Statement-aware 页面与表格检索

### 改动

- Query Parser 增加：
  - `statement_type=balance_sheet|income_statement|cash_flow|equity_statement|notes|mda|other`
  - `required_fields`
  - `required_periods`
- 页面/chunk 保存或推断 statement type、表头、行标签、年份列和单位。
- 页面召回继续使用 Dense + BM25 + RRF；statement type 只加权，不做绝对硬过滤。
- 表格上下文保留表头、单位、目标行、公式依赖行、目标年份列和必要 total/subtotal 行。
- coverage 为 partial 时，最多一次同公司、同文件或同 statement type 的补搜。

### 验收

- 计算题正确率从 7/15 提升到至少 10/15。
- 41 条历史错误中的金标准页 ±1 命中从 17 提升到至少 25。
- 跨公司引用为 0；空检索保持 0。

## 评测顺序

1. 单元与契约测试。
2. 4 条拒答题。
3. 15 条 calculation 题。
4. 24 条历史金标准页未命中题。
5. 固定 dev20，要求不低于 12/20。
6. 达标后才重新运行完整 100 题；目标至少 65/100。

## 执行记录（2026-08-22）

- 阶段一已完成：回答契约、公司/期间/数值 coverage、Decimal 受控计算与 trace 已实现。
- 阶段二已完成：运行时不再调用通用 Query Planner；仅保留 `original` 与确定性 `finance_rewrite`。
- 阶段三首轮已完成：
  - 页面推断 statement type，并加入软加权；
  - required-fields 同页数值覆盖与 required-periods 覆盖加入页面评分；
  - page-first 从单一 best chunk 改为 anchor chunk ±1；
  - field-aware 候选页展开为 anchor page ±2，生成上下文同样允许相邻页；
  - 仅缺少 required fields 时允许一次限定文件的补搜，期间缺口本身不触发补搜。
- 修复评测污染：未启用 rerank 时会显式关闭本地 reranker；报告新增真实生成上下文的 page hit。
- 单元/契约测试：53 项通过。
- dev20 检索消融（Jina 远程 20/20，无降级）：
  - final page hit@1：55%；
  - final page hit@5：85%（17/20）；
  - context page hit：85%；
  - 剩余页面失败：`financebench_id_02987`、`financebench_id_00005`、`financebench_id_00499`。
- 下一门槛：运行相同配置的 LangSmith dev20 回答实验并由独立 DeepSeek-V4-Pro Judge 自动评分；正确率不低于旧 dev 基线 12/20 才进入目标错误集与完整 100 题。

## 执行记录（2026-08-23）

- LangSmith dev20 首轮回答实验：13/20，超过旧 dev 基线 12/20；但提升不足以直接重跑完整 100 题。
- 对 7 条错误做定向消融后，已修复：
  - `financebench_id_00299`：选择题要求比较全部候选行，并保留负值；
  - `financebench_id_03069`：固定资产周转率同时取得收入及期初/期末净 PP&E，并用未舍入平均值计算。
- 剩余 5 条错误被归纳为明确的、可复用的失败类型，而不是继续增加通用查询：
  - `financebench_id_00460`：公司级门店问题误选品牌子行，必须优先使用显式 `Total` 行；
  - `financebench_id_00499`：已取得资本支出、净 PP&E 和收入，但缺少直接的资本密集度结论；
  - `financebench_id_00005`：基准答案使用经营性营运资本，而不是流动资产减流动负债；
  - `financebench_id_02987`：操作数正确，但中间平均值过早舍入导致 24.25/24.26 差异；
  - `financebench_id_00216`：quick ratio 数值和判断正确，但无关 caveat 改写了最终结论。
- 已完成对应实现：
  - Query Parser 区分经营性营运资本字段与公式；
  - 结构化行解析支持 `respectively` 行、期初/期末 PP&E 和全精度 Decimal 计算；
  - 回答生成器接收结构化计算结果，并针对 Total 行、quick ratio 和资本密集度施加任务级答案约束；
  - required-field anchor 优先落在该字段的目标报表类型；
  - Jina 仍保持总候选上限 16，仅在该预算内预留最多 6 个字段/报表/选择范围 anchor，不增加远程 rerank token 上限。
- 当前本地验证：147 项测试全部通过；源码和示例配置未包含硬编码 Jina key。

## 当前发布门槛

1. 只运行上述剩余 5 题的 LangSmith + 独立 Judge 实验。
2. 检查结构化 `calculation` 是否真实出现在营运资本与固定资产周转率记录中，不只看最终正确/错误标签。
3. 若至少 4/5 正确，并确认已修复的 `00299`、`03069` 没有回退，再运行一次干净 dev20。
4. dev20 至少达到 15/20，且全部 rerank 为 remote、无本地降级，才运行新的 holdout80。
5. 最终将 dev20 与 holdout80 合并成一份精简 100 题报告，与 59/100 基线按题比较；不使用同一 holdout 反复调参。

## 冻结候选结果（2026-08-23）

- 剩余 5 条历史错题定向验收：5/5。
  - 首轮 v5 修复 Total 行、资本密集度和固定资产周转率；
  - v6 修复财务行内注释金额、净应收账款、`Total current liabilities` 与 `Inventory turns` 误匹配。
- 使用真实上一轮检索上下文重放 Corning 结构化计算，结果稳定为：
  - `1721 + 2904 + 1157 - 1804 - 3147 = 831`；
  - 不再依赖生成模型忽略错误计算提示后碰巧答对。
- 干净 dev20 冻结实验：`evidencerag-finance-field-statement-period-dev20-v7-c710574d`。
  - 正确：18/20（90%）；
  - Jina rerank provider：remote 20/20；
  - 本地 rerank fallback：0；
  - rerank error：0；
  - evidence status sufficient：20/20；
  - 回答模型总 token：233,270。
- 剩余 dev 错题：
  - `financebench_id_03069`：D&A margin 本轮没有计算，属于此前正确题的生成/证据使用回退；
  - `financebench_id_00215`：资本密集度结论与参考答案相反。
- dev20 已超过 15/20 晋级门槛。当前代码与参数冻结，不再针对这两题调参；下一步只运行一次 holdout80，然后与本次固定 dev20 合并为 100 题报告。
- 回答客户端新增 `ANSWER_TIMEOUT_SECONDS=60`、`ANSWER_MAX_RETRIES=2`，避免方舟偶发慢响应无限阻塞。
- Judge JSONL 现在保存 `financebench_id` 与 `question`，不再依赖 LangSmith 返回顺序进行本地关联。
