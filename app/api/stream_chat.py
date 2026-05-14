"""聊天 SSE：将 LangChain 流式 chunk 编码为 ``text/event-stream`` 帧。"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

log = logging.getLogger(__name__)


def extract_stream_delta(chunk: Any) -> str:
    """从单次 stream chunk 中取出可拼接的文本增量（用于打字机效果）。"""
    if isinstance(chunk, AIMessageChunk):
        c = chunk.content
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts: list[str] = []
            for part in c:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    if part.get("type") == "text" and "text" in part:
                        parts.append(str(part["text"]))
                    elif isinstance(part.get("content"), str):
                        parts.append(part["content"])
            return "".join(parts)
        return ""
    if isinstance(chunk, AIMessage):
        c = chunk.content
        if isinstance(c, str):
            return c
        return str(c) if c is not None else ""
    if isinstance(chunk, BaseMessage):
        c = chunk.content
        if isinstance(c, str):
            return c
        return str(c) if c is not None else ""
    if isinstance(chunk, dict):
        if chunk.get("output") is not None:
            return extract_stream_delta(chunk["output"])
        for key in ("content", "text"):
            val = chunk.get(key)
            if isinstance(val, str):
                return val
    if isinstance(chunk, str):
        return chunk
    return ""


def _encode_sse(payload: dict[str, Any]) -> bytes:
    line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    return line.encode("utf-8")


def iter_chat_sse(stream_iter: Iterator[Any]) -> Iterator[bytes]:
    """将 LangChain ``stream()`` 迭代器转为 SSE 字节流。

    事件格式（每行一条 JSON，以空行结束一帧）：

    - ``{"type":"token","delta":"..."}``：正文增量
    - ``{"type":"done"}``：正常结束
    - ``{"type":"error","message":"..."}``：异常（随后仍会发 ``done``）

    末尾追加一行 ``data: [DONE]``，便于部分客户端识别结束。
    """
    try:
        for chunk in stream_iter:
            delta = extract_stream_delta(chunk)
            if delta:
                yield _encode_sse({"type": "token", "delta": delta})
        yield _encode_sse({"type": "done"})
        yield b"data: [DONE]\n\n"
    except Exception as e:  # noqa: BLE001 — 将任意链路错误转为 SSE error 帧
        log.exception("聊天流式输出失败")
        yield _encode_sse({"type": "error", "message": str(e)})
        yield _encode_sse({"type": "done"})
        yield b"data: [DONE]\n\n"
