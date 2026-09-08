# EvidenceRAG

EvidenceRAG 是一个面向金融财报的可追溯 RAG 系统。它通过混合检索、重排序和证据引用回答问题，并提供会话管理、文档索引和检索轨迹查看能力。

## 当前能力

- BGE-M3 Dense 检索 + Milvus 原生 BM25 + RRF 融合
- Jina Reranker 重排序，回答引用到真实文件和页码
- DeepSeek-V4-Flash 生成答案，支持金融计算和跨期比较
- Vue 3 单页工作台，支持流式回答、历史会话和证据检查器
- 管理员可上传、索引、查看和删除知识库文档
- PostgreSQL 持久化，Redis 缓存，Milvus 向量与稀疏索引
- 完整本地测试、FinanceBench 离线评测及独立 shadow 实验

## 默认金融链路

`finance` profile 使用 [`configs/production/finance_online_v2.json`](configs/production/finance_online_v2.json)：

```text
问题
  → Query Rewrite（保留原问题，最多 2 个改写）
  → Dense Top 240 + BM25 Top 240
  → RRF Top 120
  → Jina 输入 Top 80、输出 Top 12
  → 28,000 字符证据上下文
  → DeepSeek-V4-Flash 回答
  → 文件与页码引用、运行 Trace
```

`auto` 模式只负责多轮对话理解与是否需要重新检索，不会覆盖金融 profile 的检索参数。Agent、Planner、结构化执行器和多数研究性模块默认关闭。

## 环境要求

- Windows / PowerShell
- Python 3.12
- Conda 环境 `rag`
- Docker Desktop
- 支持 CUDA 的 NVIDIA GPU（推荐，用于 BGE-M3 embedding）

安装依赖：

```powershell
conda activate rag
python -m pip install -e .
```

## 配置

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

至少配置以下项目，不要把真实密钥提交到 Git：

```dotenv
ARK_API_KEY=your_ark_key
BASE_URL=https://ark.cn-beijing.volces.com/api/v3
MODEL=deepseek-v4-flash-ga-260731

RERANK_API_KEY=your_jina_key
RERANK_BINDING_HOST=https://api.jina.ai/v1/rerank
RERANK_MODEL=jina-reranker-v3

EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cuda
RAG_PROFILE=finance
RAG_EXECUTION_MODE=auto
```

可选功能：

```dotenv
# 多轮 Conversation API；两个开关必须同时开启
ENABLE_CONVERSATION_MEMORY=true
USE_CONVERSATION_API=true

# 对计算、比例、比较和趋势问题选择 Evidence Focus prompt
FINANCE_EVIDENCE_FOCUS_ROUTER_ENABLED=true

# 生产环境必须修改
JWT_SECRET_KEY=replace-with-a-random-secret
ADMIN_INVITE_CODE=replace-with-an-admin-invite-code
```

项目没有硬编码默认账号。首次打开页面时注册普通用户；注册管理员需要 `.env` 中的 `ADMIN_INVITE_CODE`。

## 启动

启动 PostgreSQL、Redis、Milvus、MinIO 和 Attu：

```powershell
docker compose up -d
docker compose ps
```

启动后端和前端静态页面：

```powershell
conda activate rag
python backend/app.py
```

访问：

- 工作台：<http://127.0.0.1:8000>
- OpenAPI：<http://127.0.0.1:8000/docs>
- Attu：<http://127.0.0.1:8084>

如果端口 `8000` 被占用，请先关闭旧的 Python/Uvicorn 进程，或在 `.env` 中设置其他 `PORT`。

## 金融数据索引

FinanceBench 数据位于 [`data/financebench_top40_100_langsmith_with_evidence.csv`](data/financebench_top40_100_langsmith_with_evidence.csv)，对应 PDF 位于 `data/documents/`。

先只读检查待重建内容：

```powershell
conda run -n rag python scripts/rebuild_financebench_index.py
```

确认后重建 40 份金融文档索引：

```powershell
conda run --no-capture-output -n rag python scripts/rebuild_financebench_index.py --execute
```

`--execute` 会替换金融索引和相关派生数据，但不会删除原始 PDF 或历史会话。

## 常用接口

| 功能 | 接口 |
| --- | --- |
| 注册 / 登录 | `POST /auth/register`、`POST /auth/login` |
| 兼容问答 | `POST /chat`、`POST /chat/stream` |
| 多轮会话 | `POST /conversation/create`、`POST /conversation/chat/stream` |
| 会话历史 | `GET /conversation`、`GET /conversation/{id}/messages` |
| 检索轨迹 | `GET /conversation/{id}/trace` |
| 文档管理 | `GET /documents`、`POST /documents/upload/async`、`DELETE /documents/delete/async/{filename}` |
| 检索诊断 | `POST /debug/retrieval` |

## 测试与评测

运行测试：

```powershell
conda run --no-capture-output -n rag python -m pytest tests -q
node --test tests/frontend_api_adapter.test.js
```

最终 FinanceBench 100 题脚本按阶段运行，支持断点恢复：

```powershell
conda run --no-capture-output -n rag python scripts/run_final_financebench100.py --stage recall
conda run --no-capture-output -n rag python scripts/run_final_financebench100.py --stage rerank
conda run --no-capture-output -n rag python scripts/run_final_financebench100.py --stage answer
conda run --no-capture-output -n rag python scripts/run_final_financebench100.py --stage judge
conda run --no-capture-output -n rag python scripts/run_final_financebench100.py --stage report
```

最终结果和逐题报告位于 `reports/final100/`。LangSmith 当前默认关闭，评测结果保存在本地。

## 目录

```text
backend/                 API、检索、回答、会话与存储
frontend/                Vue 3 工作台
configs/production/      当前线上金融配置
configs/experiments/     可复现实验配置
data/                    FinanceBench 数据和原始文档
scripts/                 索引、评测、审计和报告工具
tests/                   后端与前端兼容测试
docs/                    架构、实验和审计文档
reports/                 本地实验结果（通常不提交）
```

## 安全说明

- 回答仅供信息检索与分析，不构成投资建议。
- 重要结论应通过回答中的文件和页码回查原始财报。
- `.env`、API Key、数据库密码和本地模型缓存不应提交到仓库。
