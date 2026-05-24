# Echo Web UI

这个目录是 Echo 的 React 前端。

前端不自己决定 workflow，只消费后端提供的 session history、SSE 事件、database state、model settings、runtime settings 和 skill settings。

## 技术栈

- React 19
- TypeScript
- Vite
- 原生 CSS
- REST + SSE

## 前端负责什么

- 渲染聊天工作台
- 管理 session 列表和当前会话
- 发送消息、重生成、编辑、删除、回滚
- 渲染 live workflow panel
- 在最终 answer 里重建 `Thoughts`
- 管理 database 列表、active database、上传进度和文档列表
- 创建、选择、重命名、删除 database
- 上传、轮询、重命名、删除 indexed documents
- 管理 model settings、runtime settings 和 skill settings UI
- 在移动端和桌面端使用同一套 workspace 状态

前端不负责：

- workflow 决策
- tool 执行
- embedding 推理
- 向量检索逻辑
- session 持久化
- 文件入库和 chunk/index 生成

## 当前流式交互

发送消息后，前端消费这些 SSE 事件：

- `workflow`
  - 更新右侧 live workflow panel
- `record`
  - 把当前回合的 `plan / tool / think / answer` 追加进 pending answer 的 `Thoughts`
- `chunk`
  - 增量更新最终 answer 文本
- `done`
  - 返回已持久化的最终 session state、reply、token usage 和 workflow snapshot
- `error`
  - 显示终止性错误并结束 pending 状态

完成时，本地 pending UI 会被真实持久化消息替换。

## 聊天区渲染规则

主聊天流只显示：

- `user`
- 最终 assistant `answer`

不会单独显示：

- `system`
- 原始 `plan`
- 原始 `think`
- 原始 `tool`

这些内部消息会按 `workflow_turn_id` 归组，然后显示到最终 answer card 的 `Thoughts` 中。

`Thoughts` 会显示：

- `plan` 的 reasoning block
- `think` 的 reasoning block
- tool 结果
- tool artifact 附件，例如 `web_fetch` screenshot

`Thoughts` 不会显示：

- 内嵌的 `<echo_answer>...</echo_answer>`
- provider `tool_calls` 原始 JSON
- routine workflow log spam

## Workflow 面板

右侧 `Workflow` 面板是 live 状态面板：

- 显示 `plan -> retrieve -> tool -> think -> answer` graph
- `tool` 节点会显示当前工具名
- 并行 tool-call 批次会显示逗号分隔的工具名
- 默认标签为 `</>`
- 下方 `Logs` 展示 workflow step detail 和高信号错误日志

历史流程回放不依赖这个面板，而是来自 answer card 里的 `Thoughts`。

## Database 面板

Session 列表下方有 database 面板：

- 选择 active database
- 显示数据库文档数
- 显示绑定的 embedding model
- 显示 vector backend
- 点击配置图标打开 database settings
- 展示当前 active database 的 indexed documents
- 上传文件并显示 async indexing progress
- 重命名或删除已入库文档

当前 database settings 支持：

- 创建 database
- 选择 database
- 重命名 database
- 删除 database
- 选择 embedding model
- 选择 backend：`chroma` 或 `faiss`

支持上传格式：

- `.md`
- `.txt`
- `.pdf`

## Settings

前端 settings 入口管理三类配置：

- Model settings
  - chat models
  - embedding models
  - provider test
- Runtime settings
  - chunk 参数
  - max retrieve rounds
  - Marker PDF loader
  - default database backend
  - web search backend
  - web fetch screenshot mode
- Skill settings
  - enabled/default skills
  - editable non-protected skill content

## 网络层

主要 API 适配在：

- `src/lib/api.ts`
- `src/lib/sse.ts`
- `src/lib/workflow.ts`
- `src/lib/model-settings.ts`
- `src/lib/skill-settings.ts`

主要类型在：

- `src/types/chat.ts`

后端协议以这些类型和 [apps/api/README.md](../api/README.md) 为准。

## 关键目录

应用骨架：

- `src/components/app/`

聊天组件：

- `src/components/chat/`

面板组件：

- `src/components/panels/`

设置组件：

- `src/components/settings/`

状态与副作用：

- `src/hooks/useChatWorkspace.ts`
- `src/hooks/useSettingsManagement.ts`

通用组件：

- `src/components/common/`

## 开发

安装依赖：

```bash
npm install
```

本地开发：

```bash
npm run dev
```

生产构建：

```bash
npm run build
```

预览构建：

```bash
npm run preview
```

生产构建产物位于 `dist/`。当 `apps/web/dist` 存在时，FastAPI 会把它挂载到 `/ui`。
