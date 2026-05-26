"""Tool Agent API 数据模型。"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, field_validator

from app.core.agent.prompt.knobs import PromptKnobs, Scene, Tone
from app.core.agent.tool_agent import (
    AgentMode,
    DegradationReason,
    ToolAgentResult,
    ToolTraceItem,
)


class AgentChatRequest(BaseModel):
    """Tool Agent 聊天请求（默认 ``smart_home`` 场景）。"""

    session_id: str = Field(..., min_length=1, description="会话 ID")
    user_input: str = Field(..., min_length=1, description="用户输入")
    scene: Scene = Field(
        default=Scene.SMART_HOME,
        description="prompt 场景，工具 Agent 默认 smart_home",
    )
    output_language: Literal["简体中文", "English"] = Field(
        default="简体中文",
        description="输出语言",
    )
    tone: Tone = Field(default=Tone.FORMAL, description="回复语气")
    audience: str = Field(default="普通用户", max_length=32, description="目标受众")
    context: str | None = Field(
        default=None,
        max_length=32_000,
        description="可选参考材料（第二条 system）",
    )
    enable_tools: bool = Field(
        default=True,
        description="是否启用工具；False 时直接走普通对话降级路径",
    )

    @field_validator("user_input", "session_id", mode="before")
    @classmethod
    def strip_strings(cls, value: str) -> str:
        if isinstance(value, str):
            return value.strip()
        return value

    def to_prompt_knobs(self) -> PromptKnobs:
        return PromptKnobs(
            output_language=self.output_language,
            tone=self.tone,
            audience=self.audience,
        )

    def normalized_context(self) -> str:
        return (self.context or "").strip()


class ToolTraceRecord(BaseModel):
    """API 暴露的工具调用轨迹。"""

    tool: str
    args: dict[str, Any]
    result: dict[str, Any] | str
    ok: bool
    error: str | None = None


class AgentChatReplyData(BaseModel):
    """Tool Agent 回复（含降级与工具轨迹）。"""

    role: str = Field(default="assistant", description="消息角色")
    content: str = Field(description="面向用户的最终回复")
    model: str | None = Field(default=None, description="模型名称")
    finish_reason: str | None = Field(default=None, description="结束原因")
    scene: str | None = Field(default=None, description="prompt 场景")
    output_language: str | None = None
    tone: str | None = None
    mode: AgentMode = Field(description="实际消费路径：tool_agent 或 chat_fallback")
    degraded: bool = Field(description="是否发生降级")
    degradation_reason: DegradationReason = Field(description="降级原因")
    retryable: bool = Field(
        default=False,
        description="是否建议在业务层重试（如限流、超时、网关错误）",
    )
    tool_trace: list[ToolTraceRecord] = Field(default_factory=list, description="工具调用轨迹")


def _trace_to_record(item: ToolTraceItem) -> ToolTraceRecord:
    return ToolTraceRecord(
        tool=item.tool,
        args=item.args,
        result=item.result,
        ok=item.ok,
        error=item.error,
    )


def agent_reply_from_result(
    result: ToolAgentResult,
    *,
    scene: Scene | None = None,
    knobs: PromptKnobs | None = None,
) -> AgentChatReplyData:
    """将 ``ToolAgentResult`` 转为 API 响应体。"""
    msg = result.final_message
    text = msg.content if isinstance(msg.content, str) else str(msg.content)
    resp_meta = msg.response_metadata or {} if isinstance(msg, AIMessage) else {}

    return AgentChatReplyData(
        content=text,
        model=resp_meta.get("model_name") if isinstance(msg, AIMessage) else None,
        finish_reason=resp_meta.get("finish_reason") if isinstance(msg, AIMessage) else None,
        scene=scene.value if scene else None,
        output_language=knobs.output_language if knobs else None,
        tone=knobs.tone.value if knobs else None,
        mode=result.mode,
        degraded=result.degraded,
        degradation_reason=result.degradation_reason,
        retryable=result.retryable,
        tool_trace=[_trace_to_record(t) for t in result.tool_trace],
    )
