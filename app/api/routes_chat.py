"""聊天路由模块
主要负责定义聊天路由，返回应用的聊天
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.api_typing import ApiResponse
from app.api.schemas_chat import ChatReplyData, chat_reply_from_invoke
from app.api.stream_chat import iter_chat_sse
from app.core.agent.runnable import (
    runnable as chat_runnable,
    stream_runnable as chat_stream_runnable
)

router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

def _chat_sse_response(session_id: str, user_input: str) -> StreamingResponse:
    gen = iter_chat_sse(chat_stream_runnable.invoke_stream(session_id, user_input))
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/chat", response_model=ApiResponse[ChatReplyData])
def chat(session_id: str, user_input: str) -> ApiResponse[ChatReplyData]:
    """聊天

    Args:
        session_id: 会话ID
        user_input: 用户输入文本
    """
    raw = chat_runnable.invoke(session_id, user_input)
    return ApiResponse(data=chat_reply_from_invoke(raw))


@router.post("/chat/stream")
def chat_stream_post(session_id: str, user_input: str) -> StreamingResponse:
    """流式聊天
    Args:
        session_id: 会话ID
        user_input: 用户输入文本
    Returns:
        StreamingResponse: 流式响应
    """
    return _chat_sse_response(session_id, user_input)

@router.get("/chat/stream")
def chat_stream_get(session_id: str, user_input: str) -> StreamingResponse:
    """流式聊天
    Args:
        session_id: 会话ID
        user_input: 用户输入文本
    Returns:
        StreamingResponse: 流式响应
    """
    return _chat_sse_response(session_id, user_input)
