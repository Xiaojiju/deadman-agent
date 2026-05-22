# Prompt 模块化架构

本目录采用 **Core + Module + Knobs + Context** 四层分离，由 `manifest.yaml` 声明拼接顺序，`SystemPromptComposer` 一次组装后交给 LangChain。

## 四层职责

| 层 | 目录/文件 | 职责 | 变更频率 |
|----|-----------|------|----------|
| **Core** | `core/*.md` | 身份、输出格式、审过不变的安全与原则 | 低，需评审 |
| **Module** | `modules/*.md` | 场景插件（通用对话、RAG、客服等） | 中，按产品迭代 |
| **Knobs** | `knobs.py` + 模板占位符 | 语言、语气、受众等运行时变量 | 每请求 |
| **Context** | `context.py` | 检索/工具结果，**单独一条 system**，不写入 Core |
| **Few-shot** | `few_shot/*.yaml` + `few_shot_loader.py` | 1～3 组「用户样例 → 助手样例」，仅 `default` 场景，**不写入 history** | 每请求 |

## 组装顺序（manifest）

```
core/base.md → core/safety.md → modules/<scene>.md → format(knobs)
```

片段之间用 `\n\n---\n\n` 分隔，便于日志阅读与 diff。

## 代码入口

| 模块 | 作用 |
|------|------|
| `prompt_loader.py` | 读磁盘片段（带 mtime 缓存） |
| `knobs.py` | `Scene`、`PromptKnobs`、`Tone` |
| `composer.py` | `SystemPromptComposer.compose(scene, knobs)` |
| `context.py` | RAG 等场景的参考材料 system 模板 |
| `manifest.yaml` | 场景 → 片段路径列表 |

## Few-shot（default 练习）

消息顺序：

```
system (policy) → system (context) → [human/ai 样例对 × N] → history → 本轮 human
```

- 样例文件：`few_shot/default.yaml`（4 组：概念 / 对比表 / 边界 / 短答）
- API：`include_few_shot`（默认 `true`）；仅 `scene=default` 时插入
- 对比实验：同一问题分别请求 `include_few_shot: true/false`，使用**新 session_id**

## 与 Runnable 的关系

- **Policy（Core+Module+Knobs）**：`RunnableWithHistory` 的第一条 `SystemMessage`，默认 `DEFAULT_COMPOSED_SYSTEM_PROMPT`。
- **Context**：调用 `build_context_system_message(text)` 得到第二条 system，在 invoke 时传入 `{context}`（见 `context.py` 常量）。

## 性能

- 片段按 `(路径, mtime)` 缓存，改文件自动失效。
- 组装结果按 `(scene, knobs, manifest mtime)` 缓存。
- 进程内默认场景在导入时预组装一次（`DEFAULT_COMPOSED_SYSTEM_PROMPT`）。

## 新增场景

1. 在 `modules/` 新增 `xxx.md`。
2. 在 `manifest.yaml` 的 `scenes` 下增加 `xxx` 条目。
3. 在 `knobs.py` 的 `Scene` 枚举增加成员。
4. API/路由传入对应 `Scene` 与 `PromptKnobs`。
