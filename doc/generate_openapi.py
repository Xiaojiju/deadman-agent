#!/usr/bin/env python3
"""从 FastAPI 应用导出 ``doc/openapi.yaml``，并补充 SSE 说明与中文概要。"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _trim_schema_description(desc: str | None, max_len: int = 400) -> str | None:
    if not desc:
        return desc
    for cut in ("\n\nExample:", "\n\nAttributes:", "\n\nMethods:"):
        if cut in desc:
            desc = desc.split(cut)[0].strip()
    if len(desc) > max_len:
        desc = desc[: max_len - 3].rstrip() + "..."
    return desc


def _sse_response(description: str) -> dict:
    return {
        "description": description,
        "headers": {
            "Cache-Control": {
                "schema": {"type": "string"},
                "description": "``no-cache``",
            },
            "Connection": {
                "schema": {"type": "string"},
                "description": "``keep-alive``",
            },
            "X-Accel-Buffering": {
                "schema": {"type": "string"},
                "description": "``no``（减轻 Nginx 等对响应的缓冲）",
            },
        },
        "content": {
            "text/event-stream": {
                "schema": {
                    "type": "string",
                    "description": (
                        "SSE 文本流。每行 ``data:`` 后为 JSON，类型见 "
                        "``ChatStreamTokenEvent`` / ``ChatStreamDoneEvent`` / "
                        "``ChatStreamErrorEvent``；流结束前追加 ``data: [DONE]``。"
                    ),
                },
                "example": (
                    'data: {"type":"token","delta":"你"}\n\n'
                    'data: {"type":"token","delta":"好"}\n\n'
                    'data: {"type":"done"}\n\n'
                    "data: [DONE]\n\n"
                ),
            }
        },
    }


def build_spec() -> dict:
    import sys

    sys.path.insert(0, str(ROOT))
    from main import app  # noqa: WPS433

    spec = app.openapi()

    spec["info"]["description"] = (
        "Deadman Agent HTTP API。\n\n"
        "聊天接口使用模块化 system prompt：**Core + Module（scene）+ Knobs（语言/语气/受众）"
        "+ Context（RAG 参考材料）**。详见 ``app/core/agent/prompt/ARCHITECTURE.md``。\n\n"
        "Tool Agent（``POST /chat/agent``）见 ``app/core/agent/ARCHITECTURE.md``。"
    )

    if "tags" not in spec:
        spec["tags"] = []
    tag_names = {t["name"] for t in spec["tags"]}
    for name, desc in (
        ("health", "健康检查"),
        ("chat", "多轮对话（模块化 prompt + 可选流式 SSE）"),
        ("chat-agent", "Tool Agent（工具调用 + 降级，非流式）"),
        ("chat-history", "聊天历史查询（分页）"),
        ("prompt", "Prompt 资产包版本与契约"),
    ):
        if name not in tag_names:
            spec["tags"].append({"name": name, "description": desc})

    for schema in spec.get("components", {}).get("schemas", {}).values():
        if isinstance(schema, dict) and "description" in schema:
            schema["description"] = _trim_schema_description(schema.get("description"))

    paths = spec.setdefault("paths", {})

    _chat_examples = {
        "general": {
            "summary": "通用对话",
            "value": {
                "session_id": "sess-1",
                "user_input": "你好",
                "scene": "default",
                "output_language": "简体中文",
                "tone": "正式",
                "audience": "普通用户",
            },
        },
        "rag": {
            "summary": "RAG 问答",
            "value": {
                "session_id": "sess-1",
                "user_input": "如何退款？",
                "scene": "rag_qa",
                "output_language": "简体中文",
                "tone": "简洁",
                "context": "文档：7 日内可申请退款。",
            },
        },
    }

    if "/chat/prompt/meta" in paths and "get" in paths["/chat/prompt/meta"]:
        paths["/chat/prompt/meta"]["get"]["description"] = (
            "返回当前加载的 Prompt 资产包版本（prompt_version、pack_format、scene 列表），"
            "用于 A/B 记录与回滚对照。与 pyproject 应用版本分离。"
        )

    if "/chat/sessions" in paths and "get" in paths["/chat/sessions"]:
        paths["/chat/sessions"]["get"]["description"] = (
            "分页列出已有聊天会话（扫描 ``data/history/*_messages.json``），"
            "按文件修改时间倒序。"
        )
    messages_path = paths.get("/chat/sessions/{session_id}/messages", {})
    if "get" in messages_path:
        messages_path["get"]["description"] = (
            "分页查询指定 ``session_id`` 的聊天消息（磁盘全量 JSON，"
            "非模型上下文截断视图）。"
        )

    if "/chat" in paths and "post" in paths["/chat"]:
        body = paths["/chat"]["post"].setdefault("requestBody", {}).setdefault("content", {})
        json_body = body.setdefault("application/json", {})
        json_body["examples"] = _chat_examples
        paths["/chat"]["post"]["description"] = (
            "发送用户消息并返回**整段**模型回复（JSON）。\n\n"
            "请求体为 ``ChatRequest``：``scene`` 选择 manifest 场景（``default`` / "
            "``rag_qa`` / ``customer_support``）；``output_language``、``tone``、"
            "``audience`` 控制旋钮；``context`` 传入 RAG 参考材料（单独 system 层，"
            "非流式与流式共用）。\n\n"
            "使用 ``session_id`` 区分多轮会话历史。流式输出请用 ``/chat/stream``。"
        )

    stream = paths.get("/chat/stream", {})
    if "get" in stream:
        stream["get"]["summary"] = "Chat stream (SSE, GET)"
        stream["get"]["description"] = (
            "使用 **Server-Sent Events**（``text/event-stream``）流式返回增量，"
            "适合浏览器 ``EventSource``（仅支持 GET）。\n\n"
            "Query 参数与 ``ChatRequest`` 字段一致；超长 ``context`` 建议用 POST。\n\n"
            "**帧格式**：``data: {\"type\":\"token\",\"delta\":\"...\"}``；"
            "结束 ``{\"type\":\"done\"}``；随后 ``data: [DONE]``；"
            "异常 ``{\"type\":\"error\",\"message\":\"...\"}``。"
        )
        stream["get"]["responses"] = {
            "200": _sse_response("SSE 字节流"),
            "422": stream["get"].get("responses", {}).get("422", {"description": "Validation Error"}),
        }
    if "post" in stream:
        post_body = stream["post"].setdefault("requestBody", {}).setdefault("content", {})
        post_body.setdefault("application/json", {})["examples"] = _chat_examples
        stream["post"]["summary"] = "Chat stream (SSE, POST)"
        stream["post"]["description"] = (
            "与 GET 语义相同，**请求体为 ``ChatRequest``**（推荐携带 ``context``）。\n\n"
            "``EventSource`` 仅支持 GET；POST 请用 ``fetch`` + ``ReadableStream`` 解析 SSE。"
        )
        stream["post"]["responses"] = {
            "200": _sse_response("同 GET /chat/stream"),
            "422": stream["post"].get("responses", {}).get("422", {"description": "Validation Error"}),
        }

    _agent_examples = {
        "smart_home": {
            "summary": "智能家居关灯",
            "value": {
                "session_id": "sess-agent-1",
                "user_input": "帮我把客厅灯关了",
                "scene": "smart_home",
                "output_language": "简体中文",
                "tone": "正式",
            },
        },
        "tools_off": {
            "summary": "禁用工具（plain chat 降级）",
            "value": {
                "session_id": "sess-agent-2",
                "user_input": "你好",
                "scene": "smart_home",
                "enable_tools": False,
            },
        },
    }

    if "/chat/agent" in paths and "post" in paths["/chat/agent"]:
        agent_post = paths["/chat/agent"]["post"]
        agent_post["description"] = (
            "Tool Agent：**bind_tools** 驱动设备操作（如 ``control_light``），"
            "参数由工具 Pydantic schema 约束，不在 prompt 里写 JSON。\n\n"
            "与 ``POST /chat`` 共用 ``session_id`` 历史存储；消息拼装与降级见 "
            "``app/core/agent/ARCHITECTURE.md``。\n\n"
            "**降级**：``enable_tools=false``、bind 失败、模型循环失败或达到工具轮次上限时 "
            "``degraded=true``、``mode=chat_fallback``。半途失败会保留 ``partial_turn`` "
            "写入历史后再做无工具总结。\n\n"
            "响应：``tool_trace``、``degradation_reason``、``retryable``。"
        )
        agent_body = agent_post.setdefault("requestBody", {}).setdefault("content", {})
        agent_body.setdefault("application/json", {})["examples"] = _agent_examples

    # 手写 SSE 事件 schema（FastAPI 流式响应无结构化 body）
    components = spec.setdefault("components", {}).setdefault("schemas", {})
    components.setdefault(
        "ChatStreamTokenEvent",
        {
            "type": "object",
            "description": "SSE 正文增量事件。",
            "required": ["type", "delta"],
            "properties": {
                "type": {"type": "string", "const": "token"},
                "delta": {"type": "string", "description": "本帧 UTF-8 文本片段"},
            },
        },
    )
    components.setdefault(
        "ChatStreamDoneEvent",
        {
            "type": "object",
            "required": ["type"],
            "properties": {"type": {"type": "string", "const": "done"}},
        },
    )
    components.setdefault(
        "ChatStreamErrorEvent",
        {
            "type": "object",
            "required": ["type", "message"],
            "properties": {
                "type": {"type": "string", "const": "error"},
                "message": {"type": "string"},
            },
        },
    )

    return spec


def main() -> None:
    out = Path(__file__).with_name("openapi.yaml")
    spec = build_spec()
    header = (
        "# 由 doc/generate_openapi.py 根据 FastAPI 路由生成，可重复执行。\n"
        "# 生成: python doc/generate_openapi.py\n"
    )
    body = yaml.dump(
        spec,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=100,
    )
    out.write_text(header + body, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
