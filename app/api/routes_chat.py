"""聊天路由：接入模块化 prompt（Scene + Knobs + Context）。"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.api.api_typing import ApiResponse
from app.api.schemas_chat import ChatReplyData, ChatRequest, chat_reply_from_invoke
from app.api.stream_chat import iter_chat_sse
from app.core.agent.prompt.knobs import Scene, Tone
from app.core.agent.runnable import (
    runnable as chat_runnable,
    stream_runnable as chat_stream_runnable,
)

router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _run_chat(request: ChatRequest):
    """执行非流式聊天并返回原始链路输出。"""
    knobs = request.to_prompt_knobs()
    return chat_runnable.invoke(
        request.session_id,
        request.user_input,
        scene=request.scene,
        knobs=knobs,
        context=request.normalized_context() or None,
        include_few_shot=request.include_few_shot,
    )


def _chat_sse_response(request: ChatRequest) -> StreamingResponse:
    """流式 SSE 响应。"""
    knobs = request.to_prompt_knobs()
    stream_iter = chat_stream_runnable.invoke_stream(
        request.session_id,
        request.user_input,
        scene=request.scene,
        knobs=knobs,
        context=request.normalized_context() or None,
        include_few_shot=request.include_few_shot,
    )
    gen = iter_chat_sse(stream_iter)
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


@router.post("/chat", response_model=ApiResponse[ChatReplyData])
def chat(request: ChatRequest) -> ApiResponse[ChatReplyData]:
    """聊天（JSON 请求体，推荐）。

    通过 ``scene`` 选择 manifest 场景，通过 ``output_language`` / ``tone`` /
    ``audience`` 控制旋钮；``context`` 传入 RAG 参考材料（单独 system 层）。
    """
    knobs = request.to_prompt_knobs()
    raw = _run_chat(request)
    return ApiResponse(
        data=chat_reply_from_invoke(raw, scene=request.scene, knobs=knobs),
    )


@router.post("/chat/stream")
def chat_stream_post(request: ChatRequest) -> StreamingResponse:
    """流式聊天（JSON 请求体，推荐）。"""
    return _chat_sse_response(request)


@router.get("/chat/stream")
def chat_stream_get(
    session_id: str = Query(..., min_length=1, description="会话 ID"),
    user_input: str = Query(..., min_length=1, description="用户输入"),
    scene: Scene = Query(default=Scene.DEFAULT, description="prompt 场景"),
    output_language: Literal["简体中文", "English"] = Query(
        default="简体中文",
        description="输出语言",
    ),
    tone: Tone = Query(default=Tone.FORMAL, description="语气"),
    audience: str = Query(default="普通用户", max_length=32, description="受众"),
    context: str | None = Query(default=None, max_length=32_000, description="RAG 参考材料"),
    include_few_shot: bool = Query(
        default=True,
        description="是否插入 Few-shot（仅 default 场景）",
    ),
) -> StreamingResponse:
    """流式聊天（Query 参数，便于 EventSource / 简单调试）。

    与 POST 行为一致；复杂或超长 ``context`` 请使用 POST ``/chat/stream``。

    Args:
        session_id: 会话 ID
        user_input: 用户输入
        scene: prompt 场景
        output_language: 输出语言
        tone: 语气
        audience: 受众
        context: RAG 参考材料

    Returns:
        StreamingResponse: 流式响应
    """
    request = ChatRequest(
        session_id=session_id.strip(),
        user_input=user_input.strip(),
        scene=scene,
        output_language=output_language,
        tone=tone,
        audience=audience,
        context=context,
        include_few_shot=include_few_shot,
    )
    return _chat_sse_response(request)
