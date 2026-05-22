"""聊天历史查询 API 模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PaginationMeta(BaseModel):
    """分页元数据。"""

    page: int = Field(description="当前页码，从 1 开始")
    page_size: int = Field(description="每页条数")
    total: int = Field(description="总条数")
    total_pages: int = Field(description="总页数")


class ChatSessionSummary(BaseModel):
    """会话摘要。"""

    session_id: str = Field(description="会话 ID")
    message_count: int = Field(description="该会话消息条数")
    updated_at_ms: int = Field(description="历史文件最后修改时间（毫秒时间戳）")


class ChatSessionsPageData(BaseModel):
    """会话列表分页数据。"""

    items: list[ChatSessionSummary]
    pagination: PaginationMeta


class ChatMessageRecord(BaseModel):
    """单条历史消息（来自磁盘 JSON）。"""

    index: int = Field(description="在会话中的序号，从 0 开始")
    id: str | None = Field(default=None, description="LangChain 消息 id（若有）")
    role: str = Field(description="角色：human / assistant / system / tool 等")
    type: str = Field(description="LangChain 消息 type 字段")
    content: str = Field(description="消息正文")


class ChatMessagesPageData(BaseModel):
    """会话消息分页数据。"""

    session_id: str
    items: list[ChatMessageRecord]
    pagination: PaginationMeta
