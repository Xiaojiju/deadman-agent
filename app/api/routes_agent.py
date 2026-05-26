"""Tool Agent 路由：工具调用 + 降级到普通对话。"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.api_typing import ApiResponse
from app.api.schemas_agent import AgentChatReplyData, AgentChatRequest, agent_reply_from_result
from app.core.agent.tool_agent import tool_agent_runner

router = APIRouter()


@router.post("/chat/agent", response_model=ApiResponse[AgentChatReplyData])
def chat_agent(request: AgentChatRequest) -> ApiResponse[AgentChatReplyData]:
    """Tool Agent 聊天（非流式）。

    - 默认 ``scene=smart_home``，工具参数由 schema 约束（如 ``control_light``）；
    - 失败时走 **plain chat 降级**（与 Tool 路径共用 ``_build_turn_messages``，不经 ``/chat`` LCEL）；
    - 响应含 ``mode``、``degraded``、``degradation_reason``、``retryable``、``tool_trace``。

    设计说明见 ``app/core/agent/ARCHITECTURE.md``。
    """
    knobs = request.to_prompt_knobs()
    raw = tool_agent_runner.invoke(
        request.session_id,
        request.user_input,
        scene=request.scene,
        knobs=knobs,
        context=request.normalized_context() or None,
        enable_tools=request.enable_tools,
    )
    return ApiResponse(
        data=agent_reply_from_result(raw, scene=request.scene, knobs=knobs),
    )
