# EvidenceRAG FinanceBench 69% 之后的泛化与成本计划

## 冻结结果

- 版本：EvidenceRAG prompt `2026-08-23.v5`，static finance，original + deterministic finance rewrite，page-first，Jina remote rerank。
- dev20：18/20（90%）。
- holdout80：51/80（63.75%）。
- 合计：69/100；旧基线为 59/100。
- 100 个唯一 FinanceBench ID，空检索 0，最终 resolved 记录全部使用 Jina remote，无本地 rerank 降级。
- 引用命中：文档 94/100；金标准页 exact 72/100；金标准页 ±1 为 76/100。
- 31 条错误中，金标准页 ±1 命中 17 条；其余 14 条首先属于页面发现失败。

本次 80 题 holdout 已被查看，不能再作为未见测试集。后续修改可以将旧 100 题用作回归集，但不得把在同一 holdout 上获得的提升称为泛化提升。

## 过拟合审计

dev 与 holdout 相差 26.25 个百分点，不能只用样本量波动解释，应按存在开发集过拟合或规则选择偏差处理。

需要移除或泛化的规则：

1. `Best Buy` 字面指令改成与公司无关的 total/subtotal/segment scope resolver。
2. 普通“working capital”默认采用披露口径或 `current assets - current liabilities`；只有问题明确要求 operating working capital 时才使用经营性公式。
3. 删除“低个位数资本支出占收入且 PP&E 低于收入即不资本密集”的硬编码结论。资本密集度需要可解释的多指标、行业上下文或明确证据，不能从 3 道题拟合阈值。
4. Quick ratio、fixed asset turnover 等保留通用公式注册表，但公式操作数必须绑定同一表头、期间、单位与净额/总额口径。

当前 Query Planner 保持关闭。问题不在通用 query expansion；不得恢复 generic QA expansion、Step-back 或 HyDE 来掩盖页面定位问题。

## 错误结构

- calculation：12/16（75%）。
- comparison：16/22（72.7%）。
- judgment：1/3（33.3%）。
- lookup：34/53（64.2%），贡献 19 条错误，是最大错误来源。
- selection：6/6（100%）。

`evidence_status=sufficient` 在 100 题中全部为 sufficient，但仍有 31 条错误，因此该状态目前过于宽松。它只能说明形式字段通过，不能代表正确行、正确期间列、净额/总额口径或回答可用性已验证。

## 阶段一：降低上下文和 rerank 成本

当前每题平均约 14.7 个最终 chunk，回答模型平均输入约 11.4k token；输出平均只有约 209 token。成本主要来自证据上下文，而不是答案长度。

改动：

1. 计算题只向生成器传递结构化操作数、表头、目标行与来源，不再附带 15 个完整 chunk。
2. lookup/comparison 在 rerank 后进行 evidence-unit 压缩：目标句/行、必要相邻句、表头和单位；默认最多 6 个证据单元。
3. anchor ±1 继续用于候选发现，但不等于全部进入生成上下文。
4. Jina 从 16×1600 字符做受控消融：先比较 12×1200、10×1200；字段/报表 anchor 仍在总预算内。

验收：

- 平均回答输入不超过 7,000 token，P95 不超过 9,000。
- Jina 输入字符相对 v7 降低至少 35%。
- 在新开发集上 page hit、计算题和答案正确率无显著回退。

## 阶段二：页面定位与结构化行解析

1. 文档命中后增加轻量 page localizer，使用年份、季度、statement heading、metric row 和问题实体重新排序页面。
2. `find` 不再只做字符串包含；返回目标行、表头年份列、单位、total/subtotal scope 与 page offset。
3. 对净额/总额、注释金额、百分比、负数括号、`respectively`、跨页表头建立统一 row resolver。
4. coverage 分为：
   - `page_supported`
   - `row_supported`
   - `operands_validated`
   - `answerable`
5. qualitative lookup 必须检查问题要求的年份、实体列表或驱动因素是否实际存在，不能因引用数量达到阈值就标记 sufficient。

验收：

- 新盲测 citation page exact ≥80%，page ±1 ≥85%。
- calculation ≥80%，lookup ≥75%。
- 错误答案不得被标记为 `operands_validated`。

## 阶段三：建立新的未见评测集

1. 旧 FinanceBench 100 题冻结为 `regression_v7`，只用于发现回退。
2. 从不同公司、不同年份及不同文件类型构建新的开发集与盲测集；不得复用本轮看过的答案和具体规则。
3. 新开发集至少覆盖 lookup、comparison、calculation、selection、judgment，以及单页/跨页证据。
4. 在新开发集完成消融后，只运行一次新 blind holdout。
5. 报告同时列出：新盲测、旧 100 回归、page hit、任务类型、token、Jina 字符量和 P95 延迟。

## 下一次实现顺序

1. 移除三个 benchmark-specific 答案规则并建立通用 scope/formula policy。
2. 实现 evidence-unit context compressor；先在旧 dev20 做 token/回退测试。
3. 实现 page localizer 与严格 coverage 状态。
4. 建立新的未见金融 QA 评测集。
5. 再评估 auto/static 与受限 Agentic；Agent 只处理多页缺口和多操作数问题，不用于简单 lookup。

## 2026-08-23 执行记录：v13 dev20 门禁

已完成：

1. 删除 Best Buy、Corning 和资本密集度的 benchmark-specific 答案规则，改为通用 total/subtotal/company scope 与标准公式策略。
2. 实现回答证据压缩器；保留检索前 6 页的排名覆盖，并按字段、期间、操作数和表格窗口选择最多 10 个证据单元。
3. 所有 static/agentic 路径在回答前读取已召回前 10 个唯一页的完整 PostgreSQL `page_text`；不扩大文档范围。
4. 增加同页表格窗口、精确字段别名、目标公司优先、跨期方向校验、SEC 12(b) 注册证券语义，以及负号资本支出处理。
5. finance rewrite 已限制为公司、指标、报表类型和期间锚点；保留 original query，未恢复 generic expansion、Step-back 或 HyDE。
6. 生成失败不会再自动进入 Judge；只有实际产生答案的完整 run 才触发 Judge。

最终 dev20 resolved：19/20（95%），空检索 0，Jina remote 20/20。本地合并只替换了因跨公司 PP&E anchor 缺陷而重试的 `financebench_id_02987`；第一次重试的火山方舟超时记录未纳入 resolved 文件。

- Answer input：99,336 tokens。
- Answer output：2,306 tokens。
- Answer total：101,642 tokens。
- Jina input：489,131 characters。
- 相比 v7 dev20 的 233,270 answer tokens，下降 56.4%。
- 唯一错误 `financebench_id_00005` 使用标准 working capital（current assets - current liabilities）得到 2,278；FinanceBench 参考答案采用经营性 working capital 得到 831。该差异保留，不恢复样本特例。

Resolved 文件：

- `reports/evidencerag-finance-v13-final-dev20-resolved_answers.jsonl`
- `reports/evidencerag-finance-v13-final-dev20-resolved_judge.jsonl`
- `reports/evidencerag-finance-v13-final-dev20-resolved-summary.json`

下一门禁是手动运行旧 holdout80 回归。由于该 holdout 已经看过，结果只能称为旧回归集表现，不能称为新的泛化结果。真正的泛化结论必须来自阶段三的新盲测集。
