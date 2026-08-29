# Explicit Formula Skill v1

## 研究问题

在完全普通的 RAG 基线上，当问题明确给出计算公式时，按需加载一个能够主动补齐操作数并调用确定性计算工具的技能，能否在不影响其他问题的情况下稳定提高金融问答准确率？

## 隔离边界

- 冻结提交 `f1a0489` 的 `clean_baseline` 行为不变。
- 新 profile 为 `clean_baseline_formula_skill`，继承 clean baseline 的全部强制开关，只额外启用 `Explicit Formula Skill`。
- 不启用 Formula Advisory、标准金融指标公式、Query Planner、Agent、EvidenceFrame、额外 reranker 或额外 LLM。
- 未检测到显式公式时，不执行技能检索；技能失败时，Evidence、引用和 clean prompt 不变。

## 执行流程

1. 执行原 clean baseline Dense + BM25、RRF、rerank 与通用 evidence compression。
2. 只识别问题自己定义的公式，并编译为受限表达式树。
3. 先从原基线页面解析逐期操作数。
4. 缺失时按操作数进行至多 4 次 document-scoped 或 entity + period + operand 检索。
5. 校验公司、期间、年度/季度、币种、scale、报表类型、scope 与唯一性。
6. 将已识别的 thousands/millions/billions 统一到基础量级，用 Decimal AST 执行 `+ - * / average()`。
7. 仅在问题明确要求时用 `ROUND_HALF_UP` 舍入，并直接生成确定性带引用答案。

## 固定 8 题 A/B

| 指标 | clean baseline | explicit formula skill |
|---|---:|---:|
| Judge correct | 2/8 | 7/8 |
| 错误 → 正确 | — | 5 |
| 正确 → 错误 | — | 0 |
| Skill detected / executed | 0 | 8 / 8 |
| 操作数覆盖（工具前 → 后） | — | 59.09% → 100% |
| 额外 Dense/BM25 | 0 | 9 |
| 额外 Jina | 0 | 0 |
| 额外 LLM | 0 | 0 |
| Answer token | 10,074 | 0 |
| 平均总延迟 | 3,954.80 ms | 558.89 ms |
| 平均 Skill 本地延迟 | — | 175.84 ms |

唯一未被 Judge 接受的是 AES ROA：题面显式操作数 `net income` 对应合并净亏损 -505，Decimal 结果按两位小数为 -0.01；参考答案 -0.02 使用不同净利润口径或显示约定。本实现没有为参考答案修改操作数或舍入规则。

## 完整 100 题

| 指标 | clean baseline v1 | skill-explicit-formula-v1 | 变化 |
|---|---:|---:|---:|
| Judge correct | 33/100 | 38/100 | +5 |
| Answer token | 134,394 | 124,915 | -9,479 |
| 平均延迟 | 约 4,450 ms | 3,185.99 ms | 下降 |

8 道技能题产生 5 个错误→正确、0 个正确→错误。其余 92 题的引用、上下文页面与输入 token 均逐题一致，技能补搜为 0；Judge 中出现 3 个上升和 3 个下降，净变化为 0，属于相同输入下回答模型的生成波动。

## 结论

该实验对研究问题给出肯定结果：在严格触发和严格失败回退下，普通 RAG 加按需显式公式技能能够稳定修复一组计算问题，同时不改变非目标问题的检索与 prompt。FinanceBench 100 题已被反复查看，因此 +5 只能视为固定回归集上的工程证据，不能单独证明对未见数据的泛化；下一项 Skill 应使用新的、公司无关的验证集单独评估。
