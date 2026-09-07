# Frontend Conversation Migration v1

日期：2026-09-07

## 实现范围

现有 Vue 页面结构、CSS 和组件布局保持不变。新增 `frontend/api-adapter.js`，负责在旧 session API 与 Conversation API v1.1 之间切换并统一前端数据结构。

Conversation 模式使用：

- `POST /conversation/create`
- `POST /conversation/chat/stream`
- `GET /conversation`
- `GET /conversation/{conversation_id}/messages`
- `GET /conversation/{conversation_id}/trace`
- `DELETE /conversation/{conversation_id}`

旧模式继续使用：

- `POST /chat/stream`
- `GET /sessions`
- `GET /sessions/{session_id}`
- `DELETE /sessions/{session_id}`

## Feature flag

默认配置：

```dotenv
ENABLE_CONVERSATION_MEMORY=false
USE_CONVERSATION_API=false
```

启用完整 Conversation 前端链路时，两项都必须设置为 `true`：

```dotenv
ENABLE_CONVERSATION_MEMORY=true
USE_CONVERSATION_API=true
```

浏览器通过 `GET /config/frontend` 获取最终有效配置。只有两个开关都开启时，`use_conversation_api` 才会返回 `true`。配置接口请求失败时，adapter 会安全回退到旧 session API。

## 状态迁移

- Conversation 模式不再生成 `session_<timestamp>` 作为服务端标识。
- “新建会话”只进入空白草稿；首次发送时才通过 `/conversation/create` 获取数据库 `conversation_id`，避免空会话进入历史记录。
- 同一页面内后续追问复用该 `conversation_id`。
- Conversation 列表被适配为原页面使用的 `session_id/updated_at/message_count` view model，因此历史抽屉模板无需修改。
- 历史 message 的 `role/created_at/trace` 被适配为原页面的 `type/timestamp/rag_trace`。
- 加载历史时额外读取 v6 trace，将 evidence items 映射回引用检查器。
- 删除当前 conversation 后清空当前消息和证据状态；下一次发送会重新向数据库创建 conversation。

## 自动化测试

### JavaScript adapter

```powershell
node --test tests\frontend_api_adapter.test.js
```

结果：`4 passed`。

覆盖：

- Conversation v1.1 六个端点选择和 body 映射。
- 旧 `/chat/stream` 和 `/sessions` fallback。
- 配置加载失败时回退旧链路。
- v6 observability trace 到现有证据检查器模型的映射。

### 后端 Conversation API

```powershell
conda run --no-capture-output -n rag python -m pytest tests\test_conversation_api_v1_1.py -q
```

结果：`6 passed`。

### 全仓回归

```powershell
conda run --no-capture-output -n rag python -m pytest tests -q
```

结果：`926 passed, 12 warnings`。12 个 warning 均为原有 TableStore `datetime.utcnow()` 弃用提示。

## 浏览器手工验证

使用真实 Chromium 页面和隔离的 Conversation API mock 完成，不调用 Retrieval、Jina、Answer 模型，不写真实数据库。

| 场景 | 验证内容 | 结果 |
| --- | --- | --- |
| 新建会话 | 点击后保持未持久化草稿；首次发送时创建 `db-1` | 通过 |
| 多轮聊天 | 连续发送两轮，均复用 `db-1`；渲染 content、citation、trace | 通过 |
| Trace 展示 | 第一轮显示 `t1` 和 page 1；第二轮显示 `t2` 和 page 2 | 通过 |
| 历史列表 | `GET /conversation` 显示 `db-1`、4 条消息 | 通过 |
| 历史恢复 | 加载 messages 与 trace 后恢复两轮问答、引用和 trace ID | 通过 |
| 删除会话 | `DELETE /conversation/db-1` 后历史为空，当前消息和证据清空 | 通过 |

## 已知边界

Conversation API v1.1 的流式端点保持旧 SSE 事件协议，但当前后端会先完成一次同步 Conversation Answer，再发送 `content/citation/trace/done`。因此前端兼容成立，但暂时不是模型生成过程中的 token 级实时流式。实现真正 token 流需要给 Memory/Answer 核心增加流式写入接口，不属于本阶段范围。

## 真实环境复测

```powershell
conda activate rag
docker compose up -d
python backend/app.py
```

确认 `.env` 中两个开关均为 `true`，然后访问 `http://127.0.0.1:8000`，依次验证登录、新建会话、连续追问、刷新后从历史恢复、查看引用与 trace、删除会话。
