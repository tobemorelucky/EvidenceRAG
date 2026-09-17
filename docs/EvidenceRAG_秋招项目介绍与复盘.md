# EvidenceRAG：秋招项目介绍、技术实现与实验复盘

> 文档用途：简历撰写、项目答辩、技术面试和后续开发复盘。  
> 数据口径：以仓库内冻结的 FinanceBench 100 题报告为准。Shadow 实验与正式结果分开说明，不把局部实验成绩当作生产成绩。

## 1. 一句话介绍

EvidenceRAG 是一个面向金融财报的、可检索、可引用、可验证、可追踪的 RAG 系统。系统使用 BGE-M3、Milvus 原生 BM25、RRF、Jina Reranker 和 DeepSeek 构建金融问答链路，并提供多轮会话、证据检查、文档管理、运行 Trace 和离线评测能力。

项目的重点不只是“接入向量数据库和大模型”，而是围绕真实金融文档中的长表格、跨页证据、年份与单位对齐、计算问题和错误归因，持续建立可复现实验，并依据实验结果决定哪些方案进入正式链路、哪些方案应停止。

## 2. 面试时的 60 秒版本

我把一个原本面向简单数据集的通用 RAG 重构成了金融财报 RAG。最初系统有三级 Parent-Child、Auto-Merging、Planner、Evidence Grader、Step-back 和 HyDE 等较复杂的设计，但切换到 FinanceBench 的真实年报后效果很差，而且很难判断究竟是召回、页面选择、上下文构造还是回答推理出错。

我首先冻结了一条没有题目特例的干净基线，并建立分阶段评测漏斗；随后用 GPU BGE-M3、Dense + Milvus BM25 + RRF 做宽召回，用 Jina 对候选重排，再将证据压缩到 28,000 字符内交给 DeepSeek-V4-Flash 回答，用独立 DeepSeek-V4-Pro Judge 评估。最终 100 题实验达到 68% Strict Judge，RRF 候选页命中率 94%，Jina Top12 命中率 82%，最终 Context 命中率 81%。Oracle Gold Evidence 实验达到 78%，证明剩余瓶颈主要在证据利用和金融推理，而不是单纯继续扩大召回。

工程上我还做了 Jina 缓存和断点恢复、离线可复现实验、Redis/PostgreSQL 多轮记忆、RAG Trace、Vue 证据检查器、多格式文档适配与 OCR fallback。这个项目让我形成的核心经验是：RAG 优化不能只看最终正确率，必须用候选召回、重排、上下文、回答四级漏斗定位问题；复杂模块也不能凭直觉接入，必须通过 Shadow A/B 和回退门槛验证。

## 3. 项目背景与问题定义

### 3.1 为什么通用 RAG 在金融数据上失效

简单知识库问题通常可以通过一个语义相似片段直接回答，但金融财报问题有明显不同：

- 同一指标可能存在中英文、缩写和会计表达差异，例如 `operating income` 与“经营利润”。
- 一个答案可能需要从不同表格行或不同页面取得两个操作数，再执行比率或增长率计算。
- 年度、季度、币种、单位、缩放口径和公司范围必须同时一致。
- 表头、单位、目标行和脚注可能位于不同 chunk，甚至跨页。
- 检索到“相关文档”不等于检索到“正确事实”，页面命中也不等于回答模型正确利用了证据。
- 长上下文中存在其他公司、其他年份和竞争指标，模型容易选错证据或不必要拒答。

因此，项目目标从“返回语义相似文本”升级为：

1. 找到正确文档和页面；
2. 保留足以支持答案的完整证据；
3. 对金融事实的实体、期间、指标、单位和符号保持一致；
4. 对计算题执行正确公式；
5. 输出可定位到文件和页码的引用；
6. 能够追踪每个阶段，解释错误发生在哪里。

### 3.2 数据与评测

- 数据集：FinanceBench 固定 100 题。
- 文档：测试集对应的 40 份真实 PDF 财报。
- 开发方式：历史上使用 fixed20/dev20 做快速开发，holdout80 做阶段验证；最终再运行完整 100 题。
- 主指标：DeepSeek-V4-Pro Strict Judge。
- 检索指标：Candidate hit、Jina Top-K hit、Context hit、Gold page rank。
- 工程指标：输入/输出 token、Jina token、平均延迟、API 失败数量、缓存命中和断点恢复。
- Judge 与回答模型分离：回答使用 DeepSeek-V4-Flash，Judge 使用 DeepSeek-V4-Pro，避免同一模型既生成又自评。

需要诚实说明：这 100 题在研发过程中被多次用于诊断，属于固定回归集，不等价于完全未见的外部测试集。项目用禁止公司名、FinanceBench ID 和参考答案特例、保留 holdout80、Shadow A/B 和回退检测来降低过拟合风险，但真正的泛化能力仍应在新的公司、新的报告和新的问法上验证。

## 4. 当前系统架构

### 4.1 离线文档处理链路

```text
PDF / TXT
  └─ 原有解析器（保持兼容）

DOCX / XLSX / PPTX / CSV
  └─ AnyDoc Adapter

扫描 PDF / 图片
  └─ 原解析文本不足时，可选 PaddleOCR fallback

统一 Document / metadata
  ↓
FinanceBench: 约 800 token chunk，128 token overlap
  ↓
BGE-M3 GPU 批量向量化（1024 维、归一化）
  ↓
Milvus Dense Vector + Milvus 原生 BM25
  ↓
PostgreSQL 保存文档、页面、表格、会话和 Trace
Redis 保存缓存和短期会话状态
```

PDF/TXT 继续走原解析链路，新增格式通过 Adapter 统一成 LangChain Document，因此没有修改 Chunk、Embedding 和 Milvus 写入接口。OCR 默认关闭，只在文本为空、字符数低或明显为扫描文档时启用，避免影响正常 PDF 的解析质量和速度。

### 4.2 当前金融在线问答链路

当前 `finance_online_v2` 的核心参数是：

| 阶段 | 配置 |
|---|---|
| Query Rewrite | 保留原问题，最多生成 2 个改写查询 |
| Dense Recall | Top 240 |
| BM25 Recall | Top 240 |
| RRF | Top 120，`rrf_k=60` |
| Jina Rerank | 输入 80，输出 12，完整 chunk 文本 |
| Context | 最多 28,000 字符 |
| Answer | DeepSeek-V4-Flash，temperature 0.1，thinking disabled，max tokens 1024 |
| Prompt | `clean_baseline_v1`；Evidence Focus 为独立可控模式 |
| Citation | 文件名、页码和证据列表 |

完整流程为：

```text
用户问题
  ↓
多轮语义理解（auto 模式只决定是否检索、是否依赖历史）
  ↓
保留原查询 + 最多两个 Query Rewrite
  ↓
每个查询分别 Dense / BM25 召回
  ↓
RRF 融合并截取 Top120
  ↓
Jina 单次重排：Input80 → Output12
  ↓
证据上下文构造，预算 28,000 字符
  ↓
DeepSeek-V4-Flash 基于证据生成答案
  ↓
返回流式正文、引用、Trace 和完成事件
```

`auto` 模式不再覆盖金融检索参数。第一轮问题走固定金融链路；多轮情况下只由 Conversation Understanding 和确定性 Policy 决定是否复用历史证据、是否重写为独立问题、是否重新检索。`agentic` 被保留为实验模式，没有作为当前默认方案。

### 4.3 多轮记忆与可观测性

多轮链路采用分层状态：

- Redis：会话状态、近期消息、近期证据状态和快速缓存。
- PostgreSQL：Conversation、Message、Summary 和 RAG Trace 的持久化。
- Conversation Understanding：使用结构化 JSON 描述 `need_rag`、`need_retrieval`、`depends_on_history`、`query_resolution_needed`、`required_memory_scope`、`standalone_query` 和 `response_mode`。
- Conversation Policy：将模型输出转换为受控执行决策，避免让 LLM 直接控制检索系统。
- Trace：记录 profile、独立查询、是否检索、Dense/BM25/RRF 数量、Jina 参数、证据、模型、Prompt、token 和各阶段延迟。

这种设计把“理解用户在多轮中想做什么”与“允许系统执行什么”分开，兼顾灵活性和工程可控性。

### 4.4 前端与接口

前端是 Vue 3 单页工作台，包含：

- 登录和管理员/普通用户权限；
- 会话创建、流式问答、历史恢复和删除；
- Markdown、表格和重点数字展示；
- 文件与页码引用卡片；
- 默认折叠的高级检索过程；
- 管理员知识库上传、索引状态和删除；
- 响应式布局与移动端证据抽屉。

Conversation API 提供创建、聊天、流式聊天、列表、历史消息、Trace 和删除能力，同时保留旧 `/chat/stream` SSE 协议兼容。流式事件包含 `status`、`content`、`citation`、`trace`、`error` 和 `done`。

## 5. 核心实现细节

### 5.1 Dense + BM25 + RRF

Dense 使用 BGE-M3 获取语义相关内容，BM25 用于捕获财务术语、年份、金额、百分比和缩写等精确词项。两路结果通过 Reciprocal Rank Fusion 合并，而不是直接比较不同量纲的分数：

```text
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

离线构建时使用 GPU 批量 Embedding，并通过 FP16、合理 batch 和一次加载模型提高吞吐。线上只向量化短查询，模型在进程启动时加载，避免逐题重复初始化。

### 5.2 Query Rewrite

改写不会替代原问题，而是始终保留原问题，并最多增加两个检索表达。这样既可以覆盖中英文术语和财报表达差异，也能在改写失败时保留原查询的召回能力。改写只影响检索，不携带参考答案和 Gold Evidence。

### 5.3 Jina 重排与成本控制

Jina 对 RRF 候选执行一次重排。所有成功结果按查询、候选内容、模型、Input K、Output K 和文本截断配置缓存，支持断点恢复，避免重跑已经成功的请求。

深度实验表明，Jina Input120 的 100 题报告 token 约为 952.6 万；最终采用 Input80 后约为 632.3 万，约减少 33.6%，同时结合 Query Rewrite 将 Candidate hit 从 85% 提升到 94%、Context hit 从 73% 提升到 81%。

这里的 Jina token 是服务侧估算口径，不能和 LLM token 价格直接混为一谈；它主要用于同一 reranker 配置间的相对成本比较。

### 5.4 证据预算与引用

最终上下文预算固定为 28,000 字符，没有通过无限增加 Top-K 来换分。证据项保留文档、页码、chunk 和来源文本，回答后输出文件/页码引用。系统区分：

- 找到相关证据；
- 证据足以回答；
- 模型实际正确使用证据。

前端只展示“已找到相关证据”，不把检索成功误导为“答案一定正确”。

### 5.5 稳定 ID 与表格关联

早期 TableStore 依赖 `filename + page_number` 关联，而且 PDF 文本页码和 pdfplumber 页码存在 0/1-based 混用，造成表格错误关联。Evidence Assembly v1 引入：

- `document_id`
- `page_id`
- `table_id`
- `start_page` / `end_page`
- `parser_backend`
- `quality_score`

所有内部页码统一为 PDF loader 的 0-based，外部 parser 只在入口转换一次。迁移审计中，5,200 个页面和 1,936 个表格完成 ID 回填；随机 100 个 page-table 关联样本的错误率从旧契约下的 100% 降为 0%，全量新 ID 关联失败为 0。

表格必须同时满足页面关联正确、行列非空、结构有效、质量与页面匹配阈值，才作为可信表格证据；否则回退到 page text。该策略优先保证证据正确性，而不是强行扩大表格覆盖率。

### 5.6 回答 Prompt 与路由

干净基线 Prompt 坚持只依据证据回答、保留年份/单位/符号并给出引用。后续 Prompt 实验没有添加公司名、FinanceBench ID 或单题答案，而是只尝试通用规则：

- 金融术语归一化，例如 `total current liabilities` 与“流动负债合计”；
- 区分 `gross profit`、`operating income`、`net income`、EBIT 和 EBITDA；
- 计算前确认实体、期间、单位和操作数；
- 趋势结论必须与计算后的数值方向一致；
- 长上下文中优先目标公司、目标期间和目标指标，忽略无关公司。

Evidence Focus Prompt 的语义路由在冻结的 90 题 Shadow 集上由 68 个正确提升到 74 个正确，修复 8 题、回退 2 题、净提升 6 题，达到了 Shadow 门槛。它通过 feature flag 接入，不能对外宣称为正式 100 题 74%。

## 6. 迭代过程与关键结果

### 6.1 从复杂通用 RAG 回到可解释基线

最初系统包含三级 Parent-Child、Auto-Merging、问题复杂度路由、多子问题并行、Evidence Grader、Step-back、HyDE、HITL 和主 Agent。该架构在简单问题上尚可，但金融数据上出现三个问题：

1. 分支太多，错误归因困难；
2. 每题可能产生多次 LLM 调用，延迟和 token 较高；
3. 页面和表格证据还不可靠时，上层 Agent 只是在放大底层噪声。

因此后续先关闭 Planner、Step-back、HyDE、自动表格分支和多数结构化增强，冻结 `clean-baseline-v1`。这个基线只有 33/100，但它建立了一个重要的可信起点：结果差可以接受，不能接受的是不知道为什么差。

### 6.2 主要版本结果

| 版本/实验 | Strict Judge | Candidate hit | Context hit | 说明 |
|---|---:|---:|---:|---|
| Clean Baseline v1 | 33/100 | 65% | 45% | 无题目特例的冻结干净基线 |
| Core v2 + Skills | 54/100 | 74% | 47% | 显式公式和通用金融指标 Skill 带来提升 |
| Core v3 + Skills | 55/100 | 74% | 52% | Context 提升，但答案只增加 1 题 |
| v7 历史版本 | 69/100 | — | — | dev20 90%，holdout80 63.75%，存在明显开发集偏差风险 |
| v14 历史版本 | 68/100 | — | — | 通过 Evidence Compression 将回答 token 从约 115.7 万降到约 51.4 万 |
| Jina120 旧完整基线 | 68/100 | 85% | 73% | 高质量重排基线，但成本较高 |
| Final100：Rewrite + Jina80 | **68/100** | **94%** | **81%** | 当前正式、可复现 100 题结果 |
| Oracle Gold Evidence | **78/100** | 100% Gold Evidence | 100% Gold Evidence | 回答能力上限诊断，不是生产结果 |

从 Clean Baseline 的 33% 到 Final100 的 68%，提升来自检索、重排、证据组织、通用金融能力和工程稳定性的组合，而不是针对某一道题写答案规则。

### 6.3 Final100 漏斗

```text
100 题
├─ RRF Top120 命中：94 题
│  └─ Retrieval miss：6 题
├─ Jina Top12 命中：82 题
│  └─ Rerank miss：12 题
├─ 最终 Context 命中：81 题
│  └─ Context loss：1 题
└─ Strict Judge 正确：68 题
   └─ Gold context 已进入但回答错误：22 题
```

这个漏斗是项目最重要的诊断结论：最终阶段的主要瓶颈已经不是 Candidate Recall，而是 Jina 后的证据排序和 Answer Utilization。Query Rewrite + Jina80 相比旧 Jina120 将 Candidate hit 提升 9 个百分点、Context hit 提升 8 个百分点，但 Strict Judge 仍为 68%，说明继续只优化召回不会自动提升最终答案。

### 6.4 成本与延迟

Final100 冻结报告的平均数据：

| 阶段 | 平均 token/题 | 平均延迟/题 |
|---|---:|---:|
| Query Rewrite | 162 | 1.70 秒（包含检索阶段） |
| Jina Rerank | 63,232（服务报告口径） | 58.93 秒 |
| Flash Answer | 7,953 | 3.03 秒 |
| Pro Judge | 415 | 1.88 秒 |

Jina 是评测链路的主要延迟来源。线上对单问题仍需结合网络、Jina API 状态、Query Rewrite 和本地检索测量，不能把上表简单相加后当作固定 SLA。Judge 仅用于离线评测，不属于用户在线请求。

## 7. 失败尝试与停止结论

项目的价值很大一部分来自“哪些方法没有用，以及为什么及时停止”。

### 7.1 直接堆叠复杂 Agent/RAG 模块

尝试方向：Planner、Step-back、HyDE、Evidence Grader、Auto-Merging、多子问题并行和 HITL。

问题：

- 底层页面与表格发现尚未稳定，Agent 无法弥补错误证据；
- 多次调用显著增加 token 和延迟；
- 模块之间互相影响，无法进行清晰消融；
- 对简单单页题可能产生不必要改写和回退。

结论：Agentic RAG 可以作为长期复杂题通道，但前提是检索、证据身份和工具契约可靠。当前默认仍是确定性金融 RAG，不把多 Agent 当作“先进但无证据的升级”。

### 7.2 Page Selector v1/v2

v1 尝试使用 best chunk、多 chunk 支持、标题、表格结构和年份给页面打分；v2 改为相邻页面 group 和 coverage-aware greedy selection。两者都未通过固定 30 题门槛。

原因：chunk 转 page/page group 本身会丢失细粒度事实身份；通用页面特征无法稳定区分同一财报中多个高度相似的财务表页。

结论：停止调权重。排序失败并不一定能通过增加一个手工线性公式修复。

### 7.3 Document-local Retrieval

Dense Primary 将 30 题 Candidate hit 从 56.67% 提升到 93.33%，相邻页扩展进一步到 96.67%。Document-local search 将部分 Gold page 排名从约 36 改善到约 14，但 Context hit 没有达到 50% 验收线，Global + Local Merge 还出现 Selected/Context 回退。

结论：文档定位得到改善，但同一文档内部页面语义过于相似；“先找文档再搜页面”不是充分解法。

### 7.4 Table Retrieval、跨页 Table Group 与 Fact Store

Table-aware Retrieval 没有提高 Gold page hit；Table Evidence Group 将跨页表合并后，30 题 TableStore Gold-page coverage 仍为 12/30。早期 Fact Store 只有 74/1,936 个表格能按严格结构生成事实，暴露了多级表头、行列对齐、单位绑定和 text-like table 等问题。

结论：`table found` 不等于 `target row found`。在表格抽取质量不足时，继续优化 table retriever 或扩大 table group 只会检索更完整的错误结构。先修页码契约和表格身份，再谈表格检索。

### 7.5 Evidence Fusion v2/v3

将可信表格替换 page text 导致 Gold row hit 从 30% 降到 6.67%；改为 page text + table 后，Gold row hit 提升到 33.33%，Required number hit 提升到 29.17%，但收益有限。

结论：结构化证据适合作为补充，不能粗暴替换原文；固定预算下，每增加表格都可能挤掉有用文本事实。

### 7.6 Evidence Ranking、Packing、Bundle 与 Set Selector

尝试过 point-wise ranking、utility/length packing、replacement guard、same-page/adjacent-page bundle、Answerability Ranking、动态 Set Selector、Query Requirement 和 Requirement-Evidence Matching。

现象：

- Packing v1 能提高总体 coverage，但产生逐题回退；
- threshold、anchor protection 和 replacement budget 没有找到稳定 guard；
- Bundle 提前绑定 Evidence Unit，错误组合反而难以替换；
- Set Selector 和 Answerability Ranking 没有稳定改善 selection-loss；
- Query Requirement 有正向收益，但分类准确率不是主要瓶颈。

结论：Coverage 指标不等于 Fact-compatible Coverage。只看字符利用率、页面命中或包含某个数字，可能选到错误实体、错误期间或竞争指标。下一步若继续该方向，必须先解决 Evidence Identity，而不是继续微调 packing 权重。

### 7.7 本地 BGE Reranker

固定 30 题结果：

| Reranker | Context hit |
|---|---:|
| Identity / 原 RRF | 20.00% |
| BGE-reranker-v2-m3 raw | 23.33% |
| BGE Input Builder | 23.33% |
| BGE Metadata-aware | 33.33% |
| Jina | 63.33% |

BGE raw 有 577 个 pair 被 tokenizer 截断。优化输入后消除了截断，但 Context hit 没有提升，说明截断不是根因；加入通用 metadata 后提高到 33.33%，仍明显落后 Jina。

结论：停止继续调本地 BGE 权重和规则。这个实验避免了因为“本地模型更便宜”而长期投入到低上限路线。

### 7.8 单纯更换更强 Answer 模型

只把 Answer 从 DeepSeek-V4-Flash 换成 DeepSeek-V4-Pro，其他上下文不变：Flash 68/100，Pro 67/100；Pro 新修复 8 题，但回退 9 题。

结论：模型更强或更贵不代表在当前证据和 Prompt 下稳定更好。模型升级必须做全量回归，而不是展示几个成功案例。

### 7.9 Operation Planner

Planner v2 在完整 100 题中从 68 提升到 69：修复 5 题、回退 4 题，净提升只有 1，未达到净提升至少 5 题的门槛。

结论：问题结构化可以帮助部分计算题，但额外 LLM 调用和回退风险尚不值得进入默认生产链路。

### 7.10 检索提升没有转化为答案提升

旧链路 Candidate/Context 为 85%/73%，Final100 为 94%/81%，Strict Judge 都是 68%。Gold Evidence 下 Flash 为 78%。

结论：

- 召回仍有约 6 题的理论缺口；
- Jina/Context 还有约 13 题证据选择缺口；
- 即使证据正确，回答模型仍有约 22% 的错误；
- 后续应该分别处理 Evidence Identity、可验证计算和 Answer Utilization，而不是把所有错误统称为“检索不好”。

## 8. 从失败中形成的方法论

### 8.1 建立漏斗，而不是只看最终分数

最终准确率无法告诉我们错误在哪。项目固定记录：

```text
Gold document/page 是否进入候选
→ 是否被 reranker 保留
→ 是否进入最终 context
→ 模型是否正确回答
```

只有这样才能判断下一轮应该改 Retriever、Reranker、Evidence Assembly 还是 Answer。

### 8.2 每次实验只改变一个变量

Query Rewrite、Jina Input K、Output K、Prompt、Answer 模型和 Packing 都分别做 Shadow 对比。固定相同候选、相同 Context 或相同答案，可以减少实验噪声，避免“同时改五处然后不知道是谁起作用”。

### 8.3 为实验设置停止条件

例如：

- Local BGE 明显落后 Jina，停止调参；
- Document-local Context hit 未超过 50%，不接入；
- Planner 净提升小于 5，保持 Shadow；
- Prompt Router 要求净提升至少 5 且回退不超过 2；
- 未通过 30 题诊断前，不运行完整 100 题。

停止条件可以防止为单题不断加补丁，也节省 API token。

### 8.4 高命中率不等于高质量事实

Page hit、Gold row hit、Required number hit 和 Strict Judge 衡量的是不同层次。一个 context 即使包含参考页，也可能：

- 缺少表头和单位；
- 命中同页的错误行；
- 包含正确数字但实体或年份错误；
- 操作数齐全但公式执行错误；
- 模型得出正确数字却写反趋势方向。

因此需要 Fact-compatible Coverage 和答案一致性验证，而不是只追求更高的 Page hit。

### 8.5 结构化证据要“可回退”

EvidenceFrame、TableStore 和 Executor 在 metadata 不完整时不能覆盖原始文本路径。结构化结果只有在实体、期间、币种、单位、范围和操作数唯一性全部通过时，才能成为 authoritative result；否则应作为诊断或辅助证据，并回退到原文回答。

### 8.6 缓存、断点恢复和可审计报告是算法的一部分

长实验会遇到 Jina 限流、LangSmith 网络超时、Judge 空返回和本地中断。如果没有逐题 JSONL、cache key、失败记录和 resume，实验成本会重复消耗，结果也无法复现。因此项目把：

- 成功项跳过；
- 失败项可重试；
- 每 10 题保存状态；
- Answer/Judge 分阶段运行；
- 本地 Markdown/JSON 报告；

作为评测链路的正式功能，而不是临时脚本。

## 9. 项目中可以重点讲的工程问题

### 9.1 GPU Embedding 性能

最初重建索引速度慢，根因不是 BGE-M3 本身，而是调用粒度和模型加载方式。解决方式包括：模型进程内单例、CUDA/FP16、批量编码、避免逐 chunk 调用和避免在脚本子流程中重复初始化。面试时可以强调：GPU 利用率取决于 batch 和数据管线，不是把 `device=cuda` 写上就会自动跑满。

### 9.2 页码契约 Bug

FinanceBench `evidence_page_num` 已对应内部 PDF 页码，但历史 evaluator 曾错误执行 `-1`。同时不同 PDF parser 的页码基准不一致。最终通过：

- 全仓搜索页码使用点；
- 内部统一 0-based；
- parser 边界一次转换；
- 建立 `document_id/page_id/table_id`；
- 增加给定 N 必须定位内部 page N 的回归测试；

修复。这是一个很适合面试讲的真实数据契约问题。

### 9.3 API 网络与第三方服务

Jina 和 LangSmith 曾出现限流、超时或 VPN 路由问题。处理方式不是吞掉异常并伪装成“无相关内容”，而是区分：

- 远程重排成功；
- 降级使用本地/原 RRF；
- 重排失败；

并在 Trace 中保存状态。LangSmith 到期后，评测仍可完全落到本地 JSONL/Markdown，不让可观测平台成为系统的单点依赖。

### 9.4 在线链路和实验链路漂移

曾出现离线 Final100 正确、线上同题错误，原因包括 Jina 输入被截断、在线参数与实验参数不同、Memory 压缩占用 evidence budget、Prompt 不是冻结基线。后来通过独立 `finance_online_v2` 配置和 Trace 固定：

- Query Rewrite；
- Dense/BM25/RRF 深度；
- Jina 输入/输出和文本长度；
- Answer Prompt、temperature、thinking、max tokens；
- 第一轮 evidence budget；

避免“代码相同但配置不同”的隐性漂移。

## 10. 如何证明没有靠单题硬编码刷分

项目明确禁止以下做法：

- 根据 FinanceBench ID 分支；
- 针对 Adobe、Apple、Amazon 等公司写规则；
- 把参考答案写入 Prompt、Query Rewrite 或 Evidence Selection；
- 为单个 quick ratio、gross margin 案例写答案补丁；
- Best-of-run 后只保留表现最好的一次。

允许的增强必须是通用能力，例如：

- 会计术语的语义等价与不等价边界；
- 通用 ratio/margin/growth 计算结构；
- 实体、期间、币种、单位和符号一致性；
- 同文档/同页稳定关联；
- 统一缓存、Trace 和失败恢复。

此外，所有高风险方案先跑固定小集合并统计新增正确和回退，只有达到门槛才考虑接入。即便如此，也要承认 fixed100 已被反复观察，最终仍需外部 unseen set 验证。

## 11. 当前局限与下一步路线

### 11.1 当前局限

1. Jina 是主要延迟和外部依赖，本地 BGE 尚不能达到同等效果。
2. 表格抽取对多级表头、跨页表格和复杂布局仍不稳定。
3. Final100 的 22 个 Answer Failure 说明 Flash 对计算、指标边界、方向和拒答仍有不足。
4. 多轮会话已经工程化，但还需要更多真实用户场景验证历史复用和错误传播。
5. 固定 FinanceBench 反复使用，存在测试集熟悉风险。

### 11.2 建议的后续优先级

第一优先级：建立全新未见评测集。

- 选择未参与开发的新公司和新年份；
- 保留 lookup、calculation、comparison、selection、judgment 分层；
- 同时统计中文、英文和跨语言问题；
- 冻结一次性 holdout，避免继续在 FinanceBench 100 上调参。

第二优先级：可靠的表格与事实身份。

- 引入更强的表格解析器或视觉版面模型，但先做离线 A/B；
- 建立 row/header/unit/period 的置信度和 provenance；
- 结构化证据失败时永远回退 page text；
- 不在身份不明确时执行自动计算。

第三优先级：确定性金融计算与输出校验。

- 对 operation、operand、period、unit 全部明确且唯一的题使用 Decimal；
- 校验负号、百分比/小数转换、rounding 和趋势方向；
- 只在高置信时修正答案，不扩大猜测范围；
- 低置信时记录缺失字段而不是生成不存在的值。

第四优先级：降低重排成本。

- 缓存复用和批量请求；
- 研究更强的本地/蒸馏 reranker；
- 用 unseen set 验证 Input K=60/80 的成本—效果边界；
- 监控 P50/P95，而不只记录平均延迟。

第五优先级：受限 Agentic 深度模式。

只有在静态链路可靠后，才为跨公司、跨年份、多操作数和证据冲突问题开放有限的 `search/find/open_page/calculate`，设置最多轮次、工具次数和无新证据停止条件。简单题仍走静态路径，避免 Agent 增加不必要成本与回退。

## 12. 简历写法示例

### 12.1 精简版

**EvidenceRAG｜金融财报可追溯问答系统**

- 将通用 RAG 重构为金融财报问答链路，基于 BGE-M3、Milvus Dense/BM25、RRF、Jina Reranker 和 DeepSeek，实现文件/页码级证据引用与完整 Trace。
- 建立 Candidate → Rerank → Context → Answer 四级评测漏斗；FinanceBench 100 题由干净基线 33% 提升至 68% Strict Judge，RRF Candidate hit 94%，Context hit 81%。
- 通过 Jina 深度消融将重排输入由 120 降至 80，报告 token 下降约 33.6%；实现缓存、断点恢复、分阶段 Answer/Judge 和本地 Markdown/JSON 报告。
- 修复 PDF/TableStore 页码与关联契约，引入稳定 `document_id/page_id/table_id`，随机 100 个 page-table 关联错误率由旧契约下 100% 降为 0%。
- 实现 Redis + PostgreSQL 多轮记忆、Conversation Policy、RAG Trace、Vue 证据检查器和管理员文档管理，并为 DOCX/XLSX/PPTX/CSV 与扫描文档提供 Adapter/OCR fallback。

### 12.2 面试中避免夸大的表述

不要说：

- “系统准确率已经达到 78%。”——78% 是 Oracle Gold Evidence，不是生产链路。
- “Evidence Focus 后达到 74%。”——74/90 是冻结子集上的 Gated Shadow，不是正式 100 题。
- “已经解决金融表格解析。”——实际仍有多级表头、行列对齐和跨页问题。
- “Agent 能自动完成复杂金融分析。”——当前 Agent 不是默认生产路径。

建议说：

- “当前正式 Final100 为 68%，Oracle Evidence 为 78%，二者差值用于衡量证据选择和回答推理的剩余空间。”
- “Shadow Prompt Router 在 90 题上净提升 6、回退 2，已具备候选价值，但仍需新的 unseen set 验证后才能成为正式默认。”
- “我不仅实现了成功方案，也为失败方案设置停止条件，避免为了 benchmark 单题继续堆规则。”

## 13. 高频面试问题与回答要点

### Q1：为什么不用纯向量检索？

财报包含大量年份、金额、百分比、缩写和固定会计词项。Dense 擅长语义召回，BM25 擅长精确词项；RRF 不要求两路分数同尺度，可以稳定融合排序。实验中也观察到 BM25 有补充作用，但不能让弱 BM25 排名覆盖 Dense 主召回，因此后期采用更宽的融合候选，再交给 reranker。

### Q2：为什么 Jina 比本地 BGE 好这么多？

固定 30 题中，BGE raw 和优化输入后的 Context hit 都只有 23.33%，metadata-aware 提升到 33.33%，Jina 达到 63.33%。消除 tokenizer truncation 后 BGE 没有继续提升，说明根因不是单纯输入被截断，而是模型对长金融证据的语义和细粒度匹配能力不足。

### Q3：为什么召回提高后准确率没有提高？

Candidate hit 从 85% 到 94%、Context hit 从 73% 到 81%，Strict Judge 仍为 68%。这说明正确页面进入上下文后，模型仍可能选错实体/期间/指标、算错、写反方向或拒答。Oracle Gold Evidence 只有 78%，进一步证明 Answer Reasoning 是独立瓶颈。

### Q4：为什么不直接把 Top120 全部发给 LLM？

原始 Top120 平均约 35 万字符，单题可能接近 10 万输入 token。这样会显著增加成本、超出部分模型上下文限制，并产生 lost-in-the-middle 和竞争事实干扰。一次 15 题测试已经消耗约 147.9 万 answer token，而且 Judge 因额度不足未执行。因此生产链路必须重排和控制上下文。

### Q5：如何保证引用可信？

引用来自检索证据 metadata，不由模型凭空生成；系统保存 document/page/chunk 并在 Trace 中记录最终 evidence。表格使用稳定 ID 和质量门控，关联失败回退 page text。但当前还不能声称完成形式化 faithfulness 验证，后续仍需逐 claim 对齐和引用覆盖评测。

### Q6：为什么当前不默认上 Agent？

Agent 不能修复错误索引和错误证据身份，反而会增加调用次数、延迟与不可控分支。当前先用确定性链路解决 80% 的普通题，再考虑只对跨期、跨公司、多个操作数或证据冲突题开放受限工具循环。

### Q7：如何处理服务失败？

Jina、Answer 和 Judge 分阶段保存；每题写入结果和错误，成功缓存不重复调用，失败可以续跑。线上 Trace 区分 rerank success/degraded/failed，不将异常伪装为“没有内容”。Docker 启停使用 compose stop 而不是 down，保留数据库和向量数据。

### Q8：最有价值的一次 Bug 修复是什么？

页码契约。Evaluator 曾对已经是内部页码的 `evidence_page_num` 再减一，Table parser 和 PDF loader 也使用不同基准。这会让算法看起来检索失败，实际是评测和关联错位。统一 0-based、引入稳定 ID 和回归测试后，才建立了可信的后续实验基础。

## 14. Demo 建议流程

一个 8 分钟的秋招 Demo 可以这样安排：

1. 30 秒介绍业务问题：真实财报不只是关键词问答，而是跨表格、跨年份和计算。
2. 上传一份 PDF 或展示已有 Adobe 财报索引。
3. 提问一个直接查值问题，展示答案和文件/页码引用。
4. 继续追问另一财年，展示 Conversation Understanding 和独立查询。
5. 提问一个比率或趋势问题，展示计算、单位和方向。
6. 打开“高级信息”，展示 Query Rewrite、Dense/BM25/RRF、Jina、证据和 token/latency Trace。
7. 展示 Final100 漏斗，而不是只说 68%。
8. 主动展示一个失败案例，说明 Oracle 78% 和下一步为什么是证据身份/确定性计算，而不是盲目加 Agent。

## 15. 项目总结

EvidenceRAG 最终形成了三个层面的能力：

1. **算法层**：Hybrid Retrieval、RRF、Jina、证据预算、金融 Prompt 与严格消融。
2. **工程层**：GPU 构建、稳定 ID、缓存/断点恢复、Redis/PostgreSQL Memory、Trace、SSE、前端证据检查和多格式 ingestion。
3. **实验层**：固定数据切分、Answer/Judge 分离、候选到答案的漏斗、Shadow A/B、回退门槛和失败归因。

项目当前正式成绩不是终点。更重要的是，它已经从“复杂但难以解释的通用 RAG”变成一套能够回答下面三个问题的工程系统：

- 正确证据有没有被召回？
- 正确证据在哪一步丢失？
- 证据已经到达时，模型为什么仍然回答错误？

这套可诊断、可回退、可复现的方法，才是项目对真实 RAG 落地最有价值的部分。

---

## 附录：报告数据来源

本文关键数字来自仓库内冻结产物：

- `reports/evidencerag-clean-baseline-v1-summary.json`
- `reports/evidencerag-rag-core-v2-skills-all100-final-summary.json`
- `reports/evidencerag-rag-core-v3-skills-all100-final-summary.json`
- `reports/evidencerag-finance-v7-all100-summary.json`
- `reports/evidencerag-finance-v14-general-all100-resolved-summary.json`
- `reports/final100/final_summary.json`
- `reports/financebench_answer_only_oracle_shadow_v1/summary.json`
- `reports/answer_model_shadow_v1/summary.json`
- `reports/evidence_focus_semantic_gating_shadow_v2/summary.json`
- `reports/financial_operation_planner_shadow_v2/judge/summary.json`
- `docs/rag_structure_audit.md`
- `docs/evidence_assembly_v1.md`
- `docs/retrieval_core_v4.md`
- `docs/retrieval_core_v4_phase3_report.md`

若某个 Shadow 目录在后续清理中被移动，应以 Git 历史和对应实验 JSON 为准。正式对外展示前，建议重新核对当前 commit、配置文件和报告哈希。
