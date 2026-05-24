# API 后端说明

这个目录是 Echo 的 FastAPI 后端入口。

主链路：

```text
FastAPI Route -> ChatService -> Session / Messages -> WorkflowService -> LangGraph -> Model / Tools
```

更细的 workflow 设计见：

- [echo/workflow/README.md](../../echo/workflow/README.md)

## 后端负责什么

- 暴露 HTTP 与 SSE 接口
- 管理 session / message 持久化
- 管理 `models.json`
- 管理 `settings.json`
- 管理 `databases.json`
- 管理 skill settings 和 editable skill files
- 管理 database 配置、active database、document summaries 和 upload jobs
- 运行 LangGraph workflow
- 调用 OpenAI-compatible chat providers
- 调用外部 OpenAI-compatible embedding providers
- 流式推送 `Thoughts` 过程和最终答案
- 持久化 `web_fetch` screenshot artifacts，并通过 `/api/artifacts` 提供静态访问

后端不负责：

- 托管 embedding 推理服务
- 提供内置 `/v1/embeddings`
- 在项目内运行本地 embedding model
- 执行前端 workflow 决策

所有 embedding model 都被视为外部 OpenAI 兼容 provider，只通过 `models.json` 配置。

## 入口

- [main.py](./app/main.py)

`create_app(...)` 会：

- 确保 database settings 存在
- 创建 `ChatService`
- 注册 session、message、database、model-settings、runtime-settings、skill-settings 路由
- 挂载 `/api/artifacts`
- 在 `apps/web/dist` 存在时挂载 `/ui`

## 当前 Workflow 语义

固定节点：

- `plan`
- `retrieve`
- `tool`
- `think`
- `answer`

关键规则：

- `plan` / `think` 是唯一模型决策节点
- `retrieve` / `answer` 是内部控制节点
- `tool` 执行 MCP 工具：`load_skill`、`date`、`database_search`、`web_search`、`web_fetch` 和 `workspace_*`
- 检索只能通过 provider-native tool calls
- 一个决策可以发出一个 tool call，也可以发出多个相互独立的 tool calls
- 同一批 tool calls 会在 `tool` 节点并行执行
- 每个 provider tool call 都会产生一条匹配 `tool_call_id` 的 `role == "tool"` 消息
- 运行时 memory 使用 flat transcript，完整保留当前回合的 `plan / tool / think / answer`
- workflow 结束后把真实内部记录写入 session history

最终会落盘：

- `system`
- `user`
- assistant `plan`
- zero or more `tool`
- zero or more assistant `think`
- assistant `answer`

## SSE 事件

聊天流式接口会发出这些事件：

- `workflow`
  - live workflow snapshot
- `record`
  - 一条内部 workflow 记录，例如 assistant `plan`、tool `tool`、assistant `think`、assistant `answer`
- `chunk`
  - 最终 `answer` 的增量文本
- `done`
  - 已持久化完成后的最终 session state
- `error`
  - 终止性错误

Web UI 用法：

- `record` 驱动 `Thoughts`
- `chunk` 驱动最终 answer 正文
- `workflow` 驱动右侧 live workflow panel
- `done` 用真实持久化状态替换 pending UI

## 共享结构

### `MessageRecord`

常见字段：

- `id`
- `role`
- `content`
- `message_type`
- `workflow_turn_id`
- `tool_name`
- `tool_call_id`
- `tool_calls`
- `attachments`
- `token_usage`

`role` 允许：

- `system`
- `user`
- `assistant`
- `tool`

`message_type` 允许：

- `system`
- `user`
- `plan`
- `think`
- `tool`
- `answer`

只读内部消息：

- `plan`
- `think`
- `tool`

这些消息不允许：

- edit
- delete
- rollback
- regenerate

### `WorkflowSnapshot`

包含：

- `workflow_turn_id`
- `query`
- `answer`
- `status`
- `active_node`
- `retrieve_round`
- `tool_name`
- `node_statuses`
- `logs`
- `errors`

约束：

- `node_statuses` 必须存在
- `node_statuses.length` 必须等于 `meta.workflow_steps.length`
- `active_node` 必须存在，结束时可为 `null`
- `logs` 必须存在
- `errors` 必须存在
- `tool_name` 在并行 tool-call 批次中可能是逗号分隔名称

### `DatabaseRecord`

包含：

- `id`
- `name`
- `collection_name`
- `backend`
- `embedding_model_name`
- `document_count`
- `created_at`
- `updated_at`

`backend` 允许：

- `chroma`
- `faiss`

### `DatabaseDocumentRecord`

包含：

- `id`
- `source_name`
- `source_type`
- `file_path`
- `chunk_count`

### `UploadJobRecord`

包含：

- `job_id`
- `database_id`
- `status`
- `message`
- `progress`
- `total_files`
- `completed_files`
- `total_chunks`
- `embedded_chunks`
- `current_file_name`
- `files`
- `error`
- `error_stage`
- `created_at`
- `updated_at`

## 持久化与上下文

长期上下文来源只有 session history。

下一轮 `build_context()` 规则：

- 保留唯一 system prompt
- 排除 `tool`
- 排除 tool result bodies
- 排除 provider `tool_calls`
- 同一 `workflow_turn_id` 会压缩成一条 assistant workflow context
- 保留可见的 `plan`、`think`、`answer` sections

运行中的恢复依赖：

- `memory/workflow_live/`

每个 session 只保留一个 live workflow draft，用来处理中断恢复。

## RAG 与 Database

database 和 embedding model 是一对一配对：

- 一个 database 只绑定一个 embedding model
- 一个 database 选择一个 vector backend：`chroma` 或 `faiss`
- 创建 database 时如果没有传 `backend`，后端使用 `settings.json` 的 `default_database_backend`
- 该库的入库和检索都必须使用这个配对模型
- 检索时 query embedding 也由这个模型生成

支持入库文件：

- `.md`
- `.txt`
- `.pdf`

上传路径：

- 同步入库：`POST /api/databases/{database_id}/documents`
- 异步 job：`POST /api/databases/{database_id}/documents/jobs`

异步 job 会写入 `data/upload_jobs.json`。如果服务重启，未完成 job 会恢复为 interrupted/failed 状态，避免 UI 一直等待。

## Message 可变更边界

这些内部记录是只读的：

- `message_type == "plan"`
- `message_type == "think"`
- `message_type == "tool"`

不能：

- edit
- delete
- rollback
- regenerate

普通 `user`、`system` 和最终 `answer` 仍然允许按路由规则操作。清空 system prompt 会被拒绝，因为 session 必须始终保留一个 system prompt。

## 主要接口

系统与配置：

- `GET /api/health`
- `GET /api/meta`
- `GET /api/model-settings`
- `PUT /api/model-settings`
- `POST /api/model-settings/test`
- `GET /api/app-settings`
- `PUT /api/app-settings`
- `GET /api/skills`
- `PUT /api/skills`

Session：

- `GET /api/sessions`
- `POST /api/sessions`
- `GET /api/sessions/{session_id}`
- `PATCH /api/sessions/{session_id}`
- `PATCH /api/sessions/{session_id}/system-prompt`
- `DELETE /api/sessions/{session_id}`

聊天与消息：

- `POST /api/sessions/{session_id}/messages/stream`
- `PATCH /api/sessions/{session_id}/messages/{message_id}`
- `DELETE /api/sessions/{session_id}/messages/{message_id}`
- `POST /api/sessions/{session_id}/messages/{message_id}/rollback`
- `POST /api/sessions/{session_id}/messages/{message_id}/regenerate/stream`

数据库：

- `GET /api/databases`
- `POST /api/databases`
- `PATCH /api/databases/{database_id}`
- `POST /api/databases/{database_id}/select`
- `DELETE /api/databases/{database_id}`
- `POST /api/databases/{database_id}/documents`
- `POST /api/databases/{database_id}/documents/jobs`
- `GET /api/databases/{database_id}/documents/jobs/current`
- `GET /api/databases/{database_id}/documents/jobs/{job_id}`
- `GET /api/databases/{database_id}/documents`
- `PATCH /api/databases/{database_id}/documents/{document_id}`
- `DELETE /api/databases/{database_id}/documents/{document_id}`

Artifacts：

- `GET /api/artifacts/{path}`

## Runtime Settings

`GET /api/app-settings` / `PUT /api/app-settings` 管理：

- `chunk_size`
- `chunk_overlap`
- `max_retrieve_rounds`
- `use_marker_pdf_loader`
- `default_database_backend`
- `web_search_backend`
- `web_fetch_screenshot_mode`

`enabled_skills` 和 `default_skills` 由 skill settings 保存到同一个 `settings.json`，但通过 `/api/skills` 管理。

## Model Settings

`models.json` 包含：

- `active_chat_model`
- `active_embedding_model`
- `chat_models`
- `embedding_models`

chat model 支持：

- `wire_api == "chat_completions"`
- `wire_api == "responses"`

`custom_request_params` 会透传到 provider 请求的 extra body。Responses API 还支持 app-level alias：

- `model_reasoning_effort`
- `disable_response_storage`

## 开发

启动后端：

```bash
python -m uvicorn apps.api.app.main:app --reload
```

或：

```bash
python run.py
```

运行后端相关测试：

```bash
python -m unittest discover tests/unit -p "test_api_chat.py"
python -m unittest discover tests/unit -p "test_chat_service.py"
python -m unittest discover tests/unit -p "test_workflow.py"
```
