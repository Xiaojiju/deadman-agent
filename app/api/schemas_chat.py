"""聊天 API 数据模型"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field


class ChatReplyData(BaseModel):
    """单轮模型回复（从 LangChain 消息抽取的精简 JSON 结构）"""

    role: str = Field(default="assistant", description="消息角色")
    content: str = Field(description="助手回复正文")
    model: str | None = Field(default=None, description="模型名称")
    finish_reason: str | None = Field(default=None, description="结束原因")


def chat_reply_from_invoke(result: Any) -> ChatReplyData:
    """将链路的原始输出转为 ``ChatReplyData``（支持 ``AIMessage``、其它 ``BaseMessage``、``str``、``dict``）。"""
    if isinstance(result, AIMessage):
        text = result.content
        if not isinstance(text, str):
            text = str(text)
        meta = result.response_metadata or {}
        return ChatReplyData(
            content=text,
            model=meta.get("model_name"),
            finish_reason=meta.get("finish_reason"),
        )
    if isinstance(result, BaseMessage):
        text = result.content
        if not isinstance(text, str):
            text = str(text)
        role = getattr(result, "type", None) or "assistant"
        return ChatReplyData(role=str(role), content=text)
    if isinstance(result, str):
        return ChatReplyData(content=result)
    if isinstance(result, dict):
        out = result.get("output")
        if isinstance(out, BaseMessage):
            return chat_reply_from_invoke(out)
        if isinstance(out, str):
            return ChatReplyData(content=out)
        if out is not None:
            return ChatReplyData(content=str(out))
        if "content" in result:
            return ChatReplyData(content=str(result["content"]))
        return ChatReplyData(content=str(result))
    return ChatReplyData(content=str(result))
