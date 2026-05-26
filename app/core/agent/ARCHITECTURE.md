# Agent 运行时架构

本目录负责**模型调用、多轮历史、Tool Agent 与降级**。Prompt 文案组装见 [`prompt/ARCHITECTURE.md`](prompt/ARCHITECTURE.md)。

## 两条消费线

| 路径 | API | 实现 | 输出 |
| ------ | ----- | ------ | ------ |
| **普通对话** | `POST /chat`、`/chat/stream` | `RunnableWithHistory`（`prompt \| model`） | Markdown 文本，`ChatReplyData` |
| **Tool Agent** | `POST /chat/agent` | `ToolAgentRunner`（`bind_tools` + 循环） | 文本 + `tool_trace` + 降级元数据 |

原则（与 Prompt 层分工）：

- **给人看的排版**：对话线默认 Markdown；不在 prompt 里写 JSON schema。
- **给程序用的参数**：工具 / MCP 的 Pydantic `args_schema` + `bind_tools`。
- **策略与安全**：仍写在 `core/*.md`、`modules/*.md`（如 `smart_home.md`）。

## 模块入口

| 模块 | 作用 |
| ------ | ------ |
| `runnable.py` | `RunnableWithHistory`：LCEL + 滚动窗口历史 + 可选摘要 |
| `tool_agent.py` | `ToolAgentRunner`：工具循环、降级、与 Tool 路径一致的历史写入 |
| `tools/` | 工具注册（如 `light.control_light`，可替换为 MCP） |
| `default_model.py` | `BasicAdapterModel`（OpenAI 兼容） |
| `history_store.py` | 历史文件查询 API 用的扫描逻辑 |

## Tool Agent 消息顺序（单轮）

与 `/chat` 不同，Tool 路径**手拼**消息列表（不用 `ChatPromptTemplate` 的 history 占位符）：

```text
system (policy = Core + Module + Knobs)
→ system (context = RAG 参考材料)
→ …session 历史（含以往 human / ai / tool）…
→ human（本轮用户输入）
→ [循环] ai (tool_calls?) → tool × N → ai (最终自然语言)
```

成功时写入磁盘历史（`DefaultMessageHistory` / `RollingWindowChatHistory`）：

```text
human → ai? → tool* → ai_final
```

与 `RunnableWithMessageHistory` 写入的「仅 human + ai」兼容：历史反序列化已支持 `tool` 类型（见 `runnable._transform_messages`）。

## 工具循环（`ToolAgentRunner._run_tool_loop`）

1. `bound = model.bind_tools(tools)`，失败 → `BindToolsError` → 降级。
2. 每轮 `bound.invoke(messages)`：
   - 无 `tool_calls` → 正常结束。
   - 有 `tool_calls` → `_execute_tool`（单工具失败只写错误 `ToolMessage`，**不**整链降级）。
3. 达到 `max_tool_rounds`（默认 5）仍只有 tool：
   - 再调用一次 **未 bind_tools** 的 `model.invoke(messages)` 做收尾总结；
   - 收尾失败 → 模板文案；`degradation_reason=max_iterations`，`degraded=true`。

## 降级与历史（重点）

降级**不再**调用 `RunnableWithHistory.invoke()`（避免 LCEL 与手拼消息的 system/history 不一致）。

统一走 `_invoke_plain_chat_turn`：

1. 与 Tool 路径相同：`messages = _build_turn_messages(...)`。
2. 若有半途失败产生的 `partial_turn`（已执行的 `AIMessage` + `ToolMessage`），`messages.extend(partial_turn)` 后再 `model.invoke`（无工具）。
3. 一次性 `_persist_turn`：`[human, *partial_turn, final_ai]`。

### 降级原因（`degradation_reason`）

| 值 | 触发条件 | `partial_turn` |
| ---- | ---------- | ---------------- |
| `none` | 未降级 | — |
| `tools_disabled` | `enable_tools=false` | 无 |
| `bind_tools_failed` | `bind_tools` 异常或工具列表为空 | 无 |
| `model_invoke_failed` | 循环内 `bound.invoke` 抛错（见 `ToolAgentModelError`）或其它未预期错误 | 可能有 |
| `max_iterations` | 工具轮次用尽 | 有（含 tool 结果 + 收尾 ai） |

### `retryable`

`_classify_model_exception` 根据异常文案/类型判断（429、timeout、502/503 等）。响应字段 `retryable=true` 时，**业务层**可考虑重试；Agent 内部不自动重试。

### 异常分层（`invoke`）

```text
BindToolsError          → plain chat，无 partial
ToolAgentModelError     → plain chat + partial_turn + tool_trace
Exception               → plain chat，retryable 按分类
```

## 与 `/chat` 共享 session

`ToolAgentRunner` 通过构造参数注入 `chat_fallback: RunnableWithHistory`，**仅复用**其 `_get_session_history`（同一 `session_id` 对应同一 `RollingWindowChatHistory` 实例）。

因此：

- `/chat` 与 `/chat/agent` 可交替使用同一 `session_id`；
- 在 agent 场景写入的 `tool` 消息会被下一轮 `_build_turn_messages` 读入。

注意：`/chat` 的 policy 由 LCEL 模板注入；`/chat/agent` 的 policy 由 `_build_turn_messages` 注入——两者内容均来自 `compose_system_prompt(scene, knobs)`，**需使用相同 `scene` / knobs** 才一致。

## 新增工具 / 场景

1. 在 `tools/` 实现 `StructuredTool`（Pydantic `args_schema`）。
2. 在 `tools/__init__.py` 的 `get_smart_home_tools()`（或按场景拆 registry）注册。
3. 在 `prompt/modules/` 增加场景说明（策略，不写 JSON 格式）。
4. `manifest.yaml` + `Scene` 枚举增加场景（如已有 `smart_home`）。
5. 为 Tool Agent 补充 `tests/test_tool_agent.py` 用例。

## 观测与调试

`POST /chat/agent` 响应体（`AgentChatReplyData`）：

| 字段 | 含义 |
| ------ | ------ |
| `mode` | `tool_agent` / `chat_fallback` |
| `degraded` | 是否发生降级 |
| `degradation_reason` | 见上表 |
| `retryable` | 是否建议客户端重试 |
| `tool_trace` | 每次工具调用的 name、args、result、ok、error |

## 相关文档

- OpenAPI：`doc/openapi.yaml`（`POST /chat/agent`，由 `doc/generate_openapi.py` 生成）
- Prompt 资产：`app/core/agent/prompt/ARCHITECTURE.md`
- 评测问题集：`doc/prompt-eval-set.md`（对话线）；Tool 场景可另建固定用例
