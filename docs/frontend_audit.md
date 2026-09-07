# EvidenceRAG 前端代码审计

审计日期：2026-09-07  
审计提交：`e15159a`  
审计范围：`frontend/`、FastAPI 静态文件入口、前端实际调用的后端接口，以及 Conversation-aware RAG v5/v6 的三个新接口。

## 结论

当前前端可以复用，但应区分两层：

- **界面和交互层可高度复用**：登录/注册、三栏聊天工作台、历史抽屉、引用与 trace 检查器、知识库管理、上传/删除进度、桌面与移动端响应式布局均已存在。
- **现有数据访问层继续适配旧接口**：聊天使用 `/chat/stream`，历史使用 `/sessions`，与当前后端仍保持一致，因此按现状可以运行。
- **Conversation-aware RAG v5/v6 不能直接替换旧接口**：现有前端没有调用 `/conversation/create`、`/conversation/chat`、`/conversation/{id}/trace`；新聊天接口是一次性 JSON 响应而非 SSE，且新接口没有提供 conversation 列表、消息读取和删除能力。
- 当前仓库没有正式标记为 deprecated 的前端接口。`/chat`、`/chat/stream`、`/sessions` 和文档管理接口仍然存在；它们只能称为“旧会话链路”或“未来待迁移接口”，不能称为已废弃。
- `ENABLE_CONVERSATION_MEMORY` 在代码和 `.env.example` 中默认都是 `false`。默认启动时，新 conversation 三接口会返回 HTTP 404，现有前端则继续走旧链路。

因此，当前前端适合保留并增设一个会话 API 适配层，不需要为了 Conversation-aware RAG 重写整套界面。

## 1. 技术栈和构建方式

### 1.1 当前真实技术栈

| 项目 | 当前实现 |
| --- | --- |
| UI 框架 | Vue 3.4.31，全局 production build |
| 开发方式 | 单个 `index.html` + `script.js` + `style.css` |
| Vue API | Options API：`createApp({ data, computed, methods, watch })` |
| Markdown | marked 12.0.2 |
| 代码高亮 | highlight.js 11.7.0 |
| 图标 | Font Awesome 6.4.0 |
| CSS | 原生 CSS，自定义变量和响应式 media query |
| 路由 | 无前端路由库，通过 `activeNav` 和条件渲染切换视图 |
| 状态管理 | Vue 组件本地状态 + `localStorage` |
| HTTP | 原生 `fetch`、`ReadableStream`、`XMLHttpRequest` |
| 构建工具 | 无 Vite、Webpack、Rollup 或 npm build |
| `package.json` | `frontend/` 中不存在，仓库根目录也没有前端 `package.json` |
| 前端测试 | 未发现专用前端测试、lint 或类型检查配置 |
| 托管方式 | FastAPI `StaticFiles` 挂载 `frontend/` 到 `/` |

相关文件：

- `frontend/index.html`：页面结构和 Vue 模板。
- `frontend/script.js`：全部状态、业务交互和 API 请求。
- `frontend/style.css`：完整视觉和响应式样式。
- `backend/app.py`：将 `frontend/` 作为根路径静态站点托管。

### 1.2 构建与运行含义

当前前端**没有编译步骤**，不需要执行 `npm install` 或 `npm run build`。浏览器直接加载三个本地静态文件，并从公共 CDN 加载 Vue、marked、highlight.js 和 Font Awesome。

优点是启动简单、与 FastAPI 同源、部署文件少。限制是：

- 浏览器必须能够访问外部 CDN，否则 Vue、Markdown 渲染、代码高亮或图标会失效。
- 没有 lockfile、模块边界、TypeScript、前端单元测试和打包期校验。
- `script.js` 已超过 44 KB，继续增加多轮 memory/trace 功能时维护成本会快速上升。
- 外部资源虽然固定了版本，但没有本地 vendoring 或 SRI 完整性校验。

这不妨碍当前复用；如果后续长期迭代前端，才值得将相同 Vue 页面逐步组件化，而不是在本次会话接口迁移中同步重写。

## 2. 页面与功能清单

### 2.1 登录和注册

已有完整认证入口：

- 登录/注册模式切换。
- 用户名、密码输入。
- 注册时选择普通用户或管理员，管理员可填写注册码。
- Bearer token 保存到 `localStorage.accessToken`。
- 页面加载时通过 `/auth/me` 恢复用户身份。
- 401 响应统一退出登录并清理本地状态。
- 管理员角色控制知识库管理入口和检索诊断展示。

### 2.2 聊天工作台

已有专业 RAG 聊天界面：

- 左侧导航、中间对话、右侧证据检查器的三栏布局。
- `finance/general` profile 切换。
- `auto/static/agentic` execution mode 切换。
- 空状态专业示例问题。
- Enter 发送、Shift+Enter 换行、中文输入法 composition 保护。
- SSE 流式内容增量展示。
- 检索阶段状态展示。
- `AbortController` 停止生成。
- Markdown 和代码高亮渲染。
- 服务异常、停止回答、登录过期处理。

### 2.3 历史记录

已有旧会话体系的历史功能：

- 获取当前用户的 session 列表。
- 展示更新时间和消息数。
- 加载指定 session 的全部消息。
- 从历史消息 `rag_trace` 恢复引用。
- 删除 session。
- 新建会话时由浏览器生成 `session_<timestamp>`。

这部分可继续用于 `/sessions` 体系，但不能直接用于新 `/conversation` 体系，因为两套 ID 的创建方式和持久化表不同。

### 2.4 引用和 trace 展示

已有可复用的证据检查器：

- 回答下方显示 `filename · page_number` 引用按钮。
- 点击引用后在右侧定位证据片段。
- 显示执行模式、证据状态、引用数量和 trace ID。
- 显示 `status`/`rag_step` 阶段摘要。
- 显示计算过程。
- 管理员可查看 candidate 数、final top-k、RRF 数量、rerank、工具调用、路由原因和延迟。
- 窄屏下右侧栏变成抽屉，并提供悬浮入口。

现有 UI 不展示模型思维链，适合直接承载 v6 可观测 trace；主要工作是字段映射，而不是重新设计组件。

### 2.5 知识库管理

管理员页面已包含：

- PDF、Word、Excel、Text、Markdown、CSV 多文件选择。
- 顺序批量上传。
- 上传、清理、解析、父块入库、向量入库的步骤进度。
- 文档列表、文件类型、页数、chunk 数和索引状态。
- 单文件删除和批量删除。
- 异步删除步骤及完成后延迟移除。
- 加载、空状态、失败提示和刷新。

### 2.6 响应式与可访问性

- 1180px 以下证据栏变为右侧抽屉。
- 760px 以下隐藏桌面左栏并压缩聊天布局。
- 存在全局 `:focus-visible` 样式。
- 使用系统中文字体栈，不依赖 Google 字体。
- 支持 `prefers-reduced-motion: reduce`。
- 登录区使用基本 label、required 和 autocomplete。

仍有一个安全复用风险：助手回答通过 `marked.parse()` 后交给 `v-html`，当前没有看到 HTML sanitizer。若模型输出或知识库内容中含恶意 HTML，可能形成前端 XSS；后续上线前应使用 DOMPurify 等方式消毒，但本次审计未修改代码。

## 3. 前端实际 API 调用清单

所有请求都使用同源相对 URL，没有单独的 `API_BASE_URL`。除登录和注册外，请求通过 `authFetch()` 或带 Bearer header 的 XHR 发送。`authFetch()` 在 HTTP 401 时清理登录状态。

### 3.1 认证

| URL | 方法 | Request body | Response 处理 |
| --- | --- | --- | --- |
| `/auth/register` | POST | JSON：`username`、`password`、`role`、`admin_code` | 读取 `access_token`、`username`、`role`；token 写入 localStorage |
| `/auth/login` | POST | JSON：`username`、`password` | 同注册响应；失败读取 `detail` |
| `/auth/me` | GET | 无；Bearer token | JSON 直接赋给 `currentUser`，字段为 `username`、`role` |

### 3.2 聊天和旧会话

| URL | 方法 | Request body | Response 处理 |
| --- | --- | --- | --- |
| `/chat/stream` | POST | JSON：`message`、`session_id`、`profile`、`execution_mode` | 按 SSE `data:` 帧读取；处理 `content`、`trace`、`citation`、`status`、兼容 `rag_step`、`error`；`done` 和 `[DONE]` 不产生 UI 内容 |
| `/sessions` | GET | 无 | 读取 `{sessions: [{session_id, updated_at, message_count}]}` |
| `/sessions/{session_id}` | GET | path 中 URL-encode session ID | 读取 `{messages: [{type, content, timestamp, rag_trace}]}`；映射为用户/助手消息并从 trace 派生引用 |
| `/sessions/{session_id}` | DELETE | 无 | 读取 `{session_id, message}`；从本地列表移除并在需要时创建新的本地 session ID |

前端没有调用同步 `/chat` 接口。

### 3.3 文档管理

| URL | 方法 | Request body | Response 处理 |
| --- | --- | --- | --- |
| `/documents` | GET | 无；管理员 Bearer token | 读取 `{documents: [...]}`，使用 `filename`、`file_type`、`page_count`、`chunk_count`、`index_status` |
| `/documents/upload/async` | POST | `multipart/form-data`，单个字段 `file`；使用 XHR | 读取 `{job_id, filename, message}`；XHR upload progress 更新上传百分比 |
| `/documents/upload/jobs/{job_id}` | GET | 无 | 每秒轮询；读取 job 的 `status`、`message`、`error` 和 `steps`，直到 completed/failed |
| `/documents/delete/async/batch` | POST | JSON：`{filenames: string[]}` | 读取 `{jobs: [{job_id, filename, message}], message}`；分别启动删除轮询 |
| `/documents/delete/async/{filename}` | DELETE | 无；path 中 URL-encode filename | 读取 `{job_id, filename, message}`；启动删除轮询 |
| `/documents/delete/jobs/{job_id}` | GET | 无 | 每秒轮询统一 job/steps 响应，完成后保留 3 秒摘要并刷新文档列表 |

前端代码还保留了两套上传触发封装：`uploadSelectedFiles()` 是当前页面使用的批量顺序上传；`uploadDocument()` 和 `startUploadJobPolling()` 是单文件封装，但当前模板未绑定它们。它们调用的仍是同一个异步上传接口，不属于独立 API。

### 3.4 后端存在但当前前端未调用

- `POST /chat`：同步旧聊天响应。
- `POST /debug/retrieval`：管理员检索诊断。
- `GET /documents/upload/jobs`：列出全部上传任务。
- `POST /documents/upload`：旧同步上传。
- `DELETE /documents/{filename}`：旧同步删除。
- 新增的三个 `/conversation/*` 接口。

## 4. 与 Conversation-aware RAG v5/v6 接口对比

### 4.1 `POST /conversation/create`

真实契约：

```json
// request
{
  "metadata": {}
}

// response
{
  "conversation_id": "...",
  "created_at": "...",
  "memory_trace": {}
}
```

判断：**后端可用，前端需要修改后才能复用。**

当前“新建会话”只在浏览器端执行 `sessionId = "session_" + Date.now()`，不会请求后端。迁移后必须在新建会话或首次发送前调用该接口，并保存后端返回的 `conversation_id`。现有按钮、清空消息和检查器复位逻辑可直接保留。

### 4.2 `POST /conversation/chat`

真实契约：

```json
// request
{
  "conversation_id": "...",
  "message": "...",
  "profile": "finance",
  "execution_mode": "auto"
}

// response
{
  "conversation_id": "...",
  "response": "...",
  "citations": [],
  "usage": {},
  "trace": {}
}
```

判断：**聊天 UI 可复用，请求和响应处理必须修改。**

差异包括：

- 当前前端请求字段是 `session_id`，新接口要求 `conversation_id`。
- 当前 `/chat/stream` 返回 SSE；新 `/conversation/chat` 返回一次性 JSON。
- 当前前端逐 token 更新 `message.text`，新接口只能在请求完成后一次赋值。
- 当前 trace SSE 事件字段是 `rag_trace`，新接口的字段是 `trace`。
- 前端停止按钮可以取消浏览器 fetch，但非流式接口下服务端计算不一定随客户端断开而立即停止。

如果产品需要保留当前流式体验，后端还缺少 `/conversation/chat/stream`；如果接受非流式多轮聊天，则可只修改前端适配器。

### 4.3 `GET /conversation/{id}/trace`

真实契约：

```json
{
  "conversation_id": "...",
  "traces": []
}
```

支持查询参数 `limit`，后端限制到 1 至 500。

判断：**证据检查器可直接复用，取数和字段映射需要修改。**

当前前端从聊天 SSE 或历史消息中的 `rag_trace` 读取 trace，不会单独请求该接口。v6 trace 内容包括 conversation understanding、policy decision、standalone/retrieval query、是否检索、Dense/BM25/RRF 数量、rerank 参数、evidence、answer model、token 和 latency。现有检查器能承载其中大部分信息，但需要把 v6 trace 结构映射到当前 `activeTrace` 和 `debugTrace` 所期待的扁平字段。

### 4.4 功能兼容矩阵

| 前端能力 | 旧 `/chat` + `/sessions` | 新 `/conversation/*` | 结论 |
| --- | --- | --- | --- |
| 登录/权限 | 使用现有 auth | 同样依赖 Bearer auth | 直接复用 |
| 新建会话 | 前端生成 session ID | 必须 POST create | 需修改 |
| 单轮聊天 | 支持 | 支持 | 需请求适配 |
| 流式回答 | `/chat/stream` 支持 | 当前没有流式接口 | 不能直接复用 |
| 停止生成 | AbortController + SSE | 仅能中断客户端等待 | 行为有差异 |
| 引用展示 | 已支持 | response 含 citations | UI 直接复用，字段适配 |
| trace 展示 | SSE/历史 rag_trace | inline trace + trace list | UI 复用，加载与映射需改 |
| 历史会话列表 | GET `/sessions` | 没有 list conversations | 新链路缺接口 |
| 读取历史消息 | GET `/sessions/{id}` | 没有 conversation messages API | 新链路缺接口 |
| 删除会话 | DELETE `/sessions/{id}` | 没有 conversation delete API | 新链路缺接口 |
| 文档管理 | 完整 | 与 conversation 无关 | 直接复用 |

### 4.5 可以直接复用、需要修改、已废弃

**可以直接复用：**

- 整体三栏布局和响应式样式。
- 登录/注册和角色控制。
- 用户/助手消息渲染。
- 引用按钮和证据检查器 UI。
- profile/mode 控件。
- 知识库列表、上传、删除及进度组件。
- 旧 `/chat/stream` + `/sessions` 工作流，因为对应后端路由仍存在。

**需要修改：**

- 新建会话改为从 `/conversation/create` 获取 ID。
- 多轮聊天请求改用 `conversation_id` 并解析 JSON，或先新增后端流式 conversation 接口。
- trace 改为可按 conversation 加载并映射 v6 数据模型。
- 历史页若迁移到新 memory 表，需要新增并消费 conversation list/messages/delete 接口。
- 启用新链路时需要显式设置 `ENABLE_CONVERSATION_MEMORY=true`，并保证 PostgreSQL/Redis 可用。

**已废弃：**

- 当前未发现通过装饰器、文档或代码明确标记 deprecated 的上述接口。
- `/documents/upload`、`DELETE /documents/{filename}` 是仍存在但前端未使用的同步旧实现，适合后续单独退役；本次不能将它们写成已经废弃。

## 5. 建议的最小复用路径

在不重写界面的前提下，后续迁移可按以下顺序进行：

1. 保留现有 HTML、CSS、认证和文档管理代码。
2. 在 `script.js` 内先抽出一个小型 conversation API adapter，让旧 `/chat/stream` 与新 `/conversation/chat` 可由 feature flag 切换。
3. 新链路首次进入聊天时调用 `/conversation/create`；不要再由浏览器制造 conversation ID。
4. 将 `response`、`citations`、`trace` 映射为当前 message 对象的 `text`、`citations`、`ragTrace`。
5. 决定是否必须保留流式体验：必须保留则先补后端 conversation SSE；可以暂时非流式则直接消费 JSON。
6. 在迁移历史抽屉前，先补齐 conversation list、messages 和 delete 后端接口。不要把新 conversation ID 交给旧 `/sessions` 接口混用。
7. 对 v6 trace 建立显式 view model，避免 UI 直接依赖后端深层 trace 字段。
8. 上线前对 `marked` 结果进行 HTML 消毒，并增加最少的前端 API 契约测试。

## 6. 如何启动当前项目

### 6.1 默认启动方式

在项目根目录执行：

```powershell
conda activate rag
docker compose up -d
python backend/app.py
```

也可以直接用 uvicorn：

```powershell
conda activate rag
docker compose up -d
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

访问：

- 前端：`http://127.0.0.1:8000`
- OpenAPI：`http://127.0.0.1:8000/docs`
- Milvus Attu：`http://127.0.0.1:8084`

无需启动独立前端服务器，也无需 npm 构建。Docker Compose 只启动 PostgreSQL、Redis、Milvus、etcd、MinIO 和 Attu；FastAPI 仍由 conda `rag` 环境在宿主机启动。

### 6.2 当前默认行为

`.env.example` 中：

```dotenv
ENABLE_CONVERSATION_MEMORY=false
```

因此默认前端会正常使用 `/chat/stream` 和 `/sessions`。若只是验证现有前后端，不需要开启 Conversation-aware Memory。开启该 flag 后只是让三个新接口可访问，**不会自动让现有前端改用它们**。

## 7. 最终判断

现有前端不是需要推倒重做的原型，而是一套已经覆盖主要产品功能的轻量 Vue 工作台。当前最值得保留的是页面结构、专业视觉、流式聊天交互、引用检查器和知识库管理。真正不兼容的是会话 API 编排层：新 conversation 服务还缺列表、消息读取、删除和流式聊天接口，现有前端也尚未建立对应适配器。

建议后续目标定义为“复用现有 UI，替换会话数据层”，而不是“重做前端”。
