# EvidenceRAG

EvidenceRAG 是一个面向专业知识库的 RAG 工作台，强调答案可检索、可引用、可验证和可审计。当前重点场景是 FinanceBench 金融报告问答，同时保留 `general` profile。

## 核心能力

- 页面优先的 Dense + Milvus BM25 混合检索与 RRF 融合
- 可选 rerank；失败时保留融合排序，不把服务异常伪装成空结果
- `static`、受限 `agentic` 与自适应 `auto` 三种执行模式
- 文件/页码引用、证据状态、路由原因、trace ID、使用量和延迟信息
- PostgreSQL 会话与页面数据、Redis 缓存、Milvus 向量和稀疏索引
- Vue 3 三栏工作台、证据检查器、批量文档上传与索引管理

## 架构

后端职责分为：

- `conversation_service.py`：会话持久化与缓存
- `rag_orchestrator.py`：profile、执行模式、受限检索循环与 trace
- `answer_generator.py`：只依据证据生成回答
- `prompts.py`：版本化回答、路由、Agent、计算和摘要提示词
- `agent_tools.py`：受限 `find`、`open_page` 与 Decimal 算术能力

金融默认使用 `auto`。单一事实问题走静态通道；跨年份、比较、排名、计算或低证据覆盖问题进入受限深度模式。深度模式默认最多 3 轮检索和 5 次工具调用，连续两轮没有新证据后停止，不使用互联网或模型记忆补全。

## 环境与启动

推荐使用项目约定的 conda 环境：

```powershell
conda activate rag
docker compose up -d
python backend/app.py
```

浏览器访问 `http://127.0.0.1:8000`，OpenAPI 文档位于 `http://127.0.0.1:8000/docs`。

复制 `.env.example` 为 `.env`，至少配置模型、embedding 与数据库连接信息。关键运行参数：

```dotenv
RAG_PROFILE=finance
RAG_EXECUTION_MODE=auto
FINANCE_POLICY_ENABLED=false
RAG_AGENT_MAX_ROUNDS=3
RAG_AGENT_MAX_TOOL_CALLS=5
MILVUS_SPARSE_MODE=milvus_bm25
FINANCE_RAG_CANDIDATE_K=40
FINANCE_RAG_FINAL_TOP_K=5
ANSWER_TEMPERATURE=0.1
```

项目固定 `transformers>=4.49,<5`，以避免本地 BGE-M3 与 Transformers 5 的兼容问题。新金融集合使用 Milvus analyzer/BM25，不再依赖应用侧可变词表文件。

### Financial Task Policy Layer

金融任务策略层可在回答生成前，根据已有的 `task_type` 加载 `calculation`、`comparison`、`lookup`、`selection` 或 `judgment` 通用处理规范。策略来自 `configs/finance_policies/`，只描述证据处理步骤，不包含具体公司、指标、数据集答案，也不作为事实来源。

- 默认 `FINANCE_POLICY_ENABLED=false`，关闭时继续使用原有 v14 回答模板。当前 dev20 A/B 未证明 Policy 能提高准确率，因此正式 holdout 仍建议保持关闭；该功能保留为实验开关。
- 开启后仅增加一次带缓存的本地配置读取；不会新增 LLM 调用、检索、重排或 Agent 循环。
- trace 会记录 `task_type`、`policy`、策略字符数、估算 token、缓存命中和本地加载耗时。
- 本地 FinanceBench 评测使用 `--finance-policy` 开启，使用 `--no-finance-policy` 或省略参数关闭，便于执行同样本 A/B。

示例：

```powershell
conda run --no-capture-output -n rag python -u scripts/run_financebench_local_experiment.py --split dev --limit 10 --finance-policy --experiment-prefix evidencerag-finance-policy-smoke
```

FinanceBench 评测默认使用本地 CSV、JSONL 和独立 Judge，不访问 LangSmith。`.env` 中的 `FINANCEBENCH_EVALUATION_BACKEND=local`、`LANGSMITH_TRACING=false` 会同时关闭实验上传和应用 tracing。本地入口为 `scripts/run_financebench_local_experiment.py`，完成后会自动调用 `scripts/judge_financebench_local_answers.py`。旧 LangSmith 入口仅作为以后恢复服务时的兼容代码保留。

### 显式公式求解 Skill

`clean_baseline_formula_skill` 是建立在冻结 `clean_baseline` 上的独立实验 profile。它只在问题明确出现 `defined as`、`define ... as`、`calculated as` 或 `formula is` 且表达式可安全解析时触发；不内置 quick ratio、ROA 等标准指标公式。技能最多进行 4 次确定性操作数检索，严格校验公司、期间、报表类型、币种、scale、scope 和唯一性，再使用受限 Decimal AST 计算。成功时直接返回带引用的确定性答案；失败时原 Evidence、clean prompt 和普通回答路径不变。

自动识别并运行显式公式固定回归集：

```powershell
conda run --no-capture-output -n rag python -u scripts/run_financebench_explicit_formula_skill.py
```

显式指定 `--split all` 可运行完整 100 题。该 profile 不启用 Formula Advisory、Query Planner、Agent、EvidenceFrame 或标准金融公式库。实现与实验结论见 [`docs/explicit_formula_skill_v1.md`](docs/explicit_formula_skill_v1.md)。

## 重建 40 份金融文档索引

测试集为 [`data/financebench_top40_100_langsmith_with_evidence.csv`](data/financebench_top40_100_langsmith_with_evidence.csv)，其中恰好引用 40 份 PDF。先执行只读校验：

```powershell
conda run -n rag python scripts/rebuild_financebench_index.py
```

确认输出的集合与 40 个文件正确后执行重建：

```powershell
conda run -n rag python scripts/rebuild_financebench_index.py --execute
```

`--execute` 会替换当前 `MILVUS_COLLECTION`，并清理 PostgreSQL 中的派生页面、父块与表格记录；不会删除 `data/documents` 中的 PDF，也不会改写历史会话。单文件导入失败时会补偿清理该文件已经写入的索引。

## API 兼容

原有 `/chat`、`/chat/stream`、会话与文档管理路径保持不变。`ChatRequest` 可选传入：

```json
{
  "message": "比较 3M 2021 与 2022 年净销售额，并计算变化率。",
  "session_id": "finance-review",
  "profile": "finance",
  "execution_mode": "auto"
}
```

同步响应新增 `execution_mode`、`route_reason`、`citations`、`evidence_status`、`calculation`、`trace_id` 和 `usage`。流式接口统一输出 `status`、`content`、`citation`、`trace`、`error`、`done`；前端仍兼容旧 `rag_step` 事件。

管理员可调用 `/debug/retrieval` 查看文档、页面、chunk 命中、RRF/rerank 信息、延迟与失败原因。普通界面只展示运行阶段摘要，不展示模型思维链。

## 测试与评测

运行仓库测试：

```powershell
conda run -n rag python -m pytest tests -q
```

当前 FinanceBench 100 题已经被多次查看，统一作为 `fixed_seen_regression`，历史 20/80 划分只用于与 v14/v7 按 ID 对齐，不再称为未见 holdout，也不以分数上涨直接证明泛化能力。新功能应先通过结构化单元测试和 Oracle 诊断，再运行完整 100 题。

结构化金融链路默认关闭，可分别回退：

```dotenv
EVIDENCE_FRAME_ENABLED=false
STRUCTURED_EXECUTOR_ENABLED=false
STRUCTURED_COVERAGE_ENABLED=false
FRAME_ALIGNMENT_ENABLED=false
STRUCTURED_COVERAGE_ADVISORY_ENABLED=true
STRUCTURED_TASK_EXECUTOR_ENABLED=false
ANSWER_CONSISTENCY_VALIDATOR_ENABLED=false
RAG_PROTECTED_EVIDENCE_SLOTS_ENABLED=false
STAGE_AWARE_COVERAGE_ENABLED=false
PROTECTED_PAGE_SLOTS_ENABLED=false
NUMERIC_DISPLAY_VALIDATOR_ENABLED=false
ANSWER_REQUIRED_FACETS_ENABLED=false
EXPLICIT_FORMULA_ADVISORY_ENABLED=false
SUPPLEMENTAL_FIND_ENABLED=false
```

启用后流程仍保留现有 Dense/BM25、RRF、Jina 和页面选择。结构化 coverage 默认只作 advisory；高置信 executor 结果可由本地一致性校验器验证，protected slots 只重分配现有页面/压缩预算。显式公式 advisory 只读取问题明确给出的公式和操作数，不改变检索改写或 executor；一次性补搜仅在目标文档已确定且真实 QuerySpec requirement 缺失时触发。正式运行及 v14/Oracle 对比命令见 [`docs/financebench_fixed_regression_protocol.md`](docs/financebench_fixed_regression_protocol.md)。

## 数据与迁移说明

- Docker volume 路径保持 `volumes/postgres`、`volumes/redis`、`volumes/milvus` 不变。
- 容器名和默认 Redis key 前缀已改为 `evidencerag-*` / `evidencerag`；Redis 仅发生缓存冷启动。
- 不自动删除或改写用户历史消息。
- 不默认启用多 Agent、GraphRAG、RL 或 SFT；这些方向应在检索基线和高质量工具轨迹稳定后再单独评估。
