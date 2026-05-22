"""聊天历史查询路由（分页）。"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.api_typing import ApiResponse
from app.api.schemas_history import (
    ChatMessageRecord,
    ChatMessagesPageData,
    ChatSessionSummary,
    ChatSessionsPageData,
    PaginationMeta,
)
from app.core.agent.history_store import (
    DEFAULT_PAGE_SIZE,
    HistoryStoreError,
    get_session_messages,
    list_sessions,
)

router = APIRouter()


def _pagination_meta(result) -> PaginationMeta:
    return PaginationMeta(
        page=result.page,
        page_size=result.page_size,
        total=result.total,
        total_pages=result.total_pages,
    )


@router.get("/chat/sessions", response_model=ApiResponse[ChatSessionsPageData])
def list_chat_sessions(
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(
        DEFAULT_PAGE_SIZE,
        ge=1,
        le=100,
        description="每页条数，最大 100",
    ),
) -> ApiResponse[ChatSessionsPageData]:
    """分页列出已有聊天记录的会话（按历史文件更新时间倒序）。

    数据来源于 ``data/history/{session_id}_messages.json``。
    """
    try:
        result = list_sessions(page=page, page_size=page_size)
    except HistoryStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ApiResponse(
        data=ChatSessionsPageData(
            items=[ChatSessionSummary.model_validate(item) for item in result.items],
            pagination=_pagination_meta(result),
        ),
    )


@router.get(
    "/chat/sessions/{session_id}/messages",
    response_model=ApiResponse[ChatMessagesPageData],
)
def list_chat_messages(
    session_id: str,
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(
        DEFAULT_PAGE_SIZE,
        ge=1,
        le=100,
        description="每页条数，最大 100",
    ),
) -> ApiResponse[ChatMessagesPageData]:
    """分页查询指定会话的聊天历史（磁盘全量存档，按消息顺序）。

    返回的是持久化 JSON 中的全部 human/assistant 消息，与模型上下文中的
    摘要截断视图可能不一致。
    """
    try:
        result = get_session_messages(session_id, page=page, page_size=page_size)
    except HistoryStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}") from exc

    return ApiResponse(
        data=ChatMessagesPageData(
            session_id=session_id.strip(),
            items=[ChatMessageRecord.model_validate(item) for item in result.items],
            pagination=_pagination_meta(result),
        ),
    )
