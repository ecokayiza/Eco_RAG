# Workflow 说明

这份文档描述 Echo 当前真实生效的 LangGraph workflow。

核心目标：

- 保留 LangGraph 作为控制流编排层
- 让 `plan` 和 `think` 成为唯一模型决策节点
- 让 `retrieve` 和 `answer` 成为内部控制节点
- 让 `tool` 统一执行本地 MCP 工具
- 只接受 provider-native tool calls，不再支持文本检索协议
- 允许一次决策发出多个相互独立的 tool call，并在 `tool` 节点并行执行
- 当前回合使用一条 flat transcript memory
- state、snapshot、live draft 都只保留恢复与 UI 所需字段

## 当前节点

固定节点：

1. `plan`
2. `retrieve`
3. `tool`
4. `think`
5. `answer`

固定路由：

- `START -> plan | retrieve | tool | think | answer`
- `plan -> retrieve | answer`
- `retrieve -> tool`
- `tool -> think`
- `think -> retrieve | answer`
- `answer -> END`

`START` 支持 fresh run，也支持从 live draft 的 `next_step` 恢复。

## 节点职责

### `plan`

- 调用模型
- 读取当前 flat workflow memory
- 直接决定进入 `answer` 或 `retrieve`
- 持久化一条 assistant 记录；如果同次决策包含 `<echo_answer>`，类型为 `answer`，否则为 `plan`

允许的最终答案格式：

```text
<echo_plan>
...
</echo_plan>
<echo_answer>
...
</echo_answer>
```

允许的检索格式：

```text
<echo_plan>
...
</echo_plan>
provider-native tool call batch:
- web_search(...)
- web_fetch(...)
```

模型输出里不应该把 tool call 写成文本。真正的检索动作必须走 provider-native tool calling channel。

### `retrieve`

- 不调用模型
- 验证上一跳准备好的 `pending_retrieve`
- `pending_retrieve` 是当前检索批次，可以包含一个或多个 native tool call
- 让 workflow panel 能显示当前进入检索准备阶段

不会写入聊天历史。

### `tool`

- 并行执行当前 `pending_retrieve` 批次
- 支持 `load_skill(...)`
- 支持 `date(...)`
- 支持 `database_search(...)`
- 支持 `web_search(...)`
- 支持 `web_fetch(...)`
- 支持 `workspace_*` 文件工具
- 每个 provider tool call 都会得到一条匹配 `tool_call_id` 的 `role == "tool"` 记录
- 每条 tool 记录都会以完整结果写入本地记录和存储
- 同一批 tool 结果会临时追加进 flat workflow memory，让下一跳 `think` 可以读取完整结果
- 如果 `web_fetch` 返回 screenshot，workflow 会持久化 artifact，并把临时视觉 memory 加入当前回合
- 下一跳 `think` 会直接读取本批次全部 tool 结果

`retrieve_round` 统计的是检索批次轮数，不是单个 tool call 数量。一次并行批次仍然只增加一轮。

### `think`

- 调用模型
- 读取包含上一批完整 tool 结果的 flat workflow memory
- 必须在 `<echo_think>` JSON 中把可复用证据提取到 `valid_information`
- 决定继续进入 `retrieve`，或进入 `answer`
- 持久化一条 assistant 记录；如果同次决策包含 `<echo_answer>`，类型为 `answer`，否则为 `think`

格式与 `plan` 对称，只是它能看到前面的 tool 结果。

`think` 完成后，上一批 tool 结果会只在 model context 中被隐藏，后续模型调用依赖 `<echo_think>` 中的 `valid_information`。本地持久化的 tool 记录仍保留完整结果，供 UI、调试和历史查看使用。

### `answer`

- 不调用模型
- 只发布前一跳已经准备好的 `prepared_answer`
- 以 `chunk` 形式增量流式输出最终答案
- 不额外生成内部聊天记录

最终用户可见的 assistant `answer` 来自上一条包含 `<echo_answer>` 的 workflow 记录，不再额外生成一条重复消息。

## Prompt 结构

模板入口：

- [prompts.py](./prompts.py)

模板文件：

- [prompt_templates/system.yaml](./prompt_templates/system.yaml)

规则：

- session 中始终只保留一个 system prompt
- 默认 skills 会直接内联到 system prompt
- 非默认 skills 通过 `load_skill(skill_name)` native tool call 按需加载
- 检索必须走 provider-native tool calls
- 一次决策可以发出多个相互独立的 native tool calls
- 不兼容旧的 `<retrieve>` / `<echo_retrieve>` 文本协议

## Flat Workflow Memory

当前回合里，模型看到的是一条 flat transcript：

- session `system`
- 历史长上下文
- 当前 `user`
- 当前回合的 `plan`
- 当前回合的 `tool`
- 当前回合的 `think`
- 当前回合的 `answer`
- 后续重复的 `tool / think`

也就是说：

- `think` 会完整看到之前的 `plan`
- 一批并行 tool calls 会作为多条连续 `tool` message 进入 transcript
- 一批 tool calls 后面的第一条 `think` 能看到该批完整 tool 结果
- `think` 之后，该批 tool 结果会在 model context 中替换为隐藏占位符，并移除临时视觉 memory
- 多跳检索时，后续 `think` 通过之前 `<echo_think>` 的 `valid_information` 读取已提取证据，同时只看到最新未处理 tool 批次的完整结果

## State

状态定义见：

- [state.py](./state.py)

当前必要字段：

- `workflow_turn_id`
- `query`
- `requested_skill`
- `next_step`
- `retrieve_round`
- `pending_retrieve`
- `prepared_answer`
- `streamed_answer`
- `workflow_memory`

`pending_retrieve` 当前是一个 tool-call 列表。为了兼容旧 live draft，恢复逻辑仍能读取旧的单个 dict 形态。

## Snapshot 与 Logs

tracker 定义见：

- [tracker.py](./tracker.py)

live snapshot 包含：

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

设计原则：

- snapshot 用于 live UI，不重复存整个节点输出
- `tool_name` 在并行批次中会显示逗号分隔的工具名
- 真实的 `plan / tool / think / answer` 内容由持久化 message 负责
- logs 保持最小，只留高信号信息和错误诊断

## 记录与 Streaming

workflow 对外发这些核心流：

- `state`
  - live workflow snapshot
- `record`
  - 一条内部 workflow 记录
  - 例如 `plan`、`tool`、`think`、`answer`
- `chunk`
  - 最终 `answer` 的增量文本

chat 层会把它们适配成 API SSE：

- `workflow`
- `record`
- `chunk`
- `done`
- `error`

## Live Draft 与恢复

live draft 定义见：

- [drafts.py](./drafts.py)

规则：

- 每个 session 只保留一个 live workflow draft
- 在 material event 后更新 draft
- 通过 `session_id + user_message_id` 恢复
- 命中同一轮用户消息时，从保存的 `next_step` 继续
- 命中不同用户消息时，旧 draft 会被清掉
- draft 会保存当前 state、snapshot 和已经持久化的 workflow records

## 持久化与下一轮 Context

长期记忆由：

- [echo/chat/context_manager.py](../chat/context_manager.py)

负责。

当前规则：

- 只有落盘的 session history 会进入下一轮长期上下文
- `tool` 不会进入下一轮 context
- provider `tool_calls` 和 tool result bodies 不进入长期 context
- 同一 `workflow_turn_id` 只保留一条 assistant workflow context
- 优先最后一条 `think`
- 没有 `think` 时回退到 `plan`
- 同轮 `answer` 不再重复灌回下一轮 context
- 保留的 `plan / think` 会被裁剪成纯 reasoning block，不带 action block

## 当前会落盘哪些消息

一轮带检索的聊天最终会保存：

1. `user`
2. assistant `plan`
3. zero or more `tool`
4. zero or more assistant `think`
5. assistant `answer`

其中：

- `plan / think / tool` 是只读内部记录
- 并行 tool-call 批次会落盘为多条连续 `tool` 记录
- `answer` 是正常 assistant 回复

## 配置

运行时配置：

- [settings.json](../../settings.json)

当前运行时字段：

- `chunk_size`
- `chunk_overlap`
- `max_retrieve_rounds`
- `use_marker_pdf_loader`
- `default_database_backend`
- `web_search_backend`
- `web_fetch_screenshot_mode`
- `enabled_skills`
- `default_skills`

模型配置：

- [models.json](../../models.json)

其中 embedding model 一律视为外部 OpenAI 兼容 provider。
