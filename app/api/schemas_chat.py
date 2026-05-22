"""聊天 API 数据模型"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field, field_validator

from app.core.agent.prompt.knobs import PromptKnobs, Scene, Tone


class ChatRequest(BaseModel):
    """聊天请求体（模块化 prompt：场景 + 旋钮 + 可选 RAG 上下文）。

    Attributes:
        session_id: 会话 ID
        user_input: 用户输入
        scene: prompt 场景，对应 manifest.yaml
        output_language: 输出语言
        tone: 回复语气
        audience: 目标受众
        context: RAG 参考材料；为空时注入占位文案

    Methods:
        to_prompt_knobs(): 转为 ``PromptKnobs``，供 ``compose_system_prompt`` 使用。
        normalized_context(): 供模板 ``{context}`` 使用的正文；空白时由 Runnable 填占位符。

    Example:
        POST /chat ::

            {
              "session_id": "sess-1",
              "user_input": "你好",
              "scene": "default",
              "output_language": "简体中文",
              "tone": "正式",
              "audience": "普通用户"
            }

        RAG 场景带参考材料 ::

            {
              "session_id": "sess-1",
              "user_input": "如何退款？",
              "scene": "rag_qa",
              "context": "文档：7 日内可申请退款……"
            }
    """

    session_id: str = Field(..., min_length=1, description="会话 ID")
    user_input: str = Field(..., min_length=1, description="用户输入")
    scene: Scene = Field(default=Scene.DEFAULT, description="prompt 场景，对应 manifest.yaml")
    output_language: Literal["简体中文", "English"] = Field(
        default="简体中文",
        description="输出语言",
    )
    tone: Tone = Field(default=Tone.FORMAL, description="回复语气")
    audience: str = Field(default="普通用户", max_length=32, description="目标受众")
    context: str | None = Field(
        default=None,
        max_length=32_000,
        description="RAG 参考材料；为空时注入占位文案",
    )
    include_few_shot: bool = Field(
        default=True,
        description="是否插入 Few-shot 样例（仅 scene=default 时生效，用于对比格式与边界）",
    )

    @field_validator("user_input", "session_id", mode="before")
    @classmethod
    def strip_strings(cls, value: str) -> str:
        """将字符串两端空白字符去掉。
        
        Args:
            value: 字符串

        Returns:
            去掉两端空白字符的字符串
        """
        if isinstance(value, str):
            return value.strip()
        return value

    def to_prompt_knobs(self) -> PromptKnobs:
        """转为 ``PromptKnobs``，供 ``compose_system_prompt`` 使用。"""
        return PromptKnobs(
            output_language=self.output_language,
            tone=self.tone,
            audience=self.audience,
        )

    def normalized_context(self) -> str:
        """供模板 ``{context}`` 使用的正文；空白时由 Runnable 填占位符。"""
        return (self.context or "").strip()


class ChatReplyData(BaseModel):
    """单轮模型回复（从 LangChain 消息抽取的精简 JSON 结构）
    
    Attributes:
        role: 消息角色
        content: 助手回复正文
        model: 模型名称
        finish_reason: 结束原因
        scene: 本次使用的 prompt 场景
        output_language: 本次输出语言设置
        tone: 本次语气设置
    """

    role: str = Field(default="assistant", description="消息角色")
    content: str = Field(description="助手回复正文")
    model: str | None = Field(default=None, description="模型名称")
    finish_reason: str | None = Field(default=None, description="结束原因")
    scene: str | None = Field(default=None, description="本次使用的 prompt 场景")
    output_language: str | None = Field(default=None, description="本次输出语言设置")
    tone: str | None = Field(default=None, description="本次语气设置")


def chat_reply_from_invoke(
    result: Any,
    *,
    scene: Scene | None = None,
    knobs: PromptKnobs | None = None,
) -> ChatReplyData:
    """将链路的原始输出转为 ``ChatReplyData``（支持 ``AIMessage``、其它 ``BaseMessage``、``str``、``dict``）。
    
    Args:
        result: 链路的原始输出
        scene: 本次使用的 prompt 场景
        knobs: 本次输出语言设置和语气设置

    Returns:
        ChatReplyData: 单轮模型回复（从 LangChain 消息抽取的精简 JSON 结构）
    """
    meta_scene = scene.value if scene else None
    meta_lang = knobs.output_language if knobs else None
    meta_tone = knobs.tone.value if knobs else None

    if isinstance(result, AIMessage):
        text = result.content
        if not isinstance(text, str):
            text = str(text)
        resp_meta = result.response_metadata or {}
        return ChatReplyData(
            content=text,
            model=resp_meta.get("model_name"),
            finish_reason=resp_meta.get("finish_reason"),
            scene=meta_scene,
            output_language=meta_lang,
            tone=meta_tone,
        )
    if isinstance(result, BaseMessage):
        text = result.content
        if not isinstance(text, str):
            text = str(text)
        role = getattr(result, "type", None) or "assistant"
        return ChatReplyData(
            role=str(role),
            content=text,
            scene=meta_scene,
            output_language=meta_lang,
            tone=meta_tone,
        )
    if isinstance(result, str):
        return ChatReplyData(
            content=result,
            scene=meta_scene,
            output_language=meta_lang,
            tone=meta_tone,
        )
    if isinstance(result, dict):
        out = result.get("output")
        if isinstance(out, BaseMessage):
            return chat_reply_from_invoke(out, scene=scene, knobs=knobs)
        if isinstance(out, str):
            return ChatReplyData(
                content=out,
                scene=meta_scene,
                output_language=meta_lang,
                tone=meta_tone,
            )
        if out is not None:
            return ChatReplyData(
                content=str(out),
                scene=meta_scene,
                output_language=meta_lang,
                tone=meta_tone,
            )
        if "content" in result:
            return ChatReplyData(
                content=str(result["content"]),
                scene=meta_scene,
                output_language=meta_lang,
                tone=meta_tone,
            )
        return ChatReplyData(
            content=str(result),
            scene=meta_scene,
            output_language=meta_lang,
            tone=meta_tone,
        )
    return ChatReplyData(
        content=str(result),
        scene=meta_scene,
        output_language=meta_lang,
        tone=meta_tone,
    )
