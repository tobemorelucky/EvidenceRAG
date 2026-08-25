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
- LangSmith 评测脚本使用 `--finance-policy` 开启，使用 `--no-finance-policy` 或省略参数关闭，便于执行同样本 A/B。

示例：

```powershell
conda run --no-capture-output -n rag python -u scripts/run_financebench_langsmith_experiment.py --split dev --limit 10 --finance-policy --experiment-prefix evidencerag-finance-policy-smoke
```

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

评测顺序建议固定为：flat hybrid、页面优先 hierarchical hybrid、hierarchical + rerank、static 与自适应 agentic。先固定 20 题开发集和 80 题 holdout，再运行完整 100 题并记录 LangSmith。第一阶段验收目标：空检索率为 0、page hit@10 ≥ 65%、100 题正确率 ≥ 50%，且每个引用都可解析到真实文件和页码。

## 数据与迁移说明

- Docker volume 路径保持 `volumes/postgres`、`volumes/redis`、`volumes/milvus` 不变。
- 容器名和默认 Redis key 前缀已改为 `evidencerag-*` / `evidencerag`；Redis 仅发生缓存冷启动。
- 不自动删除或改写用户历史消息。
- 不默认启用多 Agent、GraphRAG、RL 或 SFT；这些方向应在检索基线和高质量工具轨迹稳定后再单独评估。
