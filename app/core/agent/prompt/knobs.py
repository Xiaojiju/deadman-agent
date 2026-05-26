"""提示词旋钮：每请求可变的模板变量（语言、语气、受众）。

旋钮对应 ``core/base.md`` 中的占位符 ``{output_language}``、``{tone}``、
``{audience}``，由 ``SystemPromptComposer`` 在组装时注入。

场景枚举 ``Scene`` 对应 ``manifest.yaml`` 的 ``scenes`` 键。

示例::

    from app.core.agent.prompt import Scene, PromptKnobs, Tone, compose_system_prompt

    knobs = PromptKnobs(output_language="English", tone=Tone.CONCISE)
    policy = compose_system_prompt(Scene.RAG_QA, knobs)
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Scene(str, Enum):
    """业务场景，对应 ``manifest.yaml`` 中的 ``scenes`` 键。

    每个成员决定在组装时追加哪些 ``modules/*.md`` 片段。

    Example:
        >>> from app.core.agent.prompt.knobs import Scene
        >>> Scene.RAG_QA.value
        'rag_qa'
        >>> Scene.DEFAULT.value
        'default'
    """

    DEFAULT = "default"
    RAG_QA = "rag_qa"
    CUSTOMER_SUPPORT = "customer_support"
    SMART_HOME = "smart_home"


class Tone(str, Enum):
    """语气白名单。

    限制为枚举值，避免将用户任意输入直接拼进 system，降低 prompt 注入风险。

    Example:
        >>> from app.core.agent.prompt.knobs import Tone
        >>> Tone.FORMAL.value
        '正式'
    """

    FORMAL = "正式"
    CONCISE = "简洁"
    FRIENDLY = "友好"
    PROFESSIONAL = "专业"


class PromptKnobs(BaseModel):
    """注入 ``core/base.md`` 末尾「运行时偏好」占位符的参数。

    Attributes:
        output_language: 模型输出语言。
        tone: 回复语气（枚举）。
        audience: 目标受众描述，最长 32 字符。

    Example:
        >>> from app.core.agent.prompt.knobs import PromptKnobs, Tone
        >>> knobs = PromptKnobs(
        ...     output_language="English",
        ...     tone=Tone.CONCISE,
        ...     audience="开发者",
        ... )
        >>> knobs.template_vars()
        {'output_language': 'English', 'tone': '简洁', 'audience': '开发者'}
    """

    output_language: Literal["简体中文", "English"] = "简体中文"
    tone: Tone = Tone.FORMAL
    audience: str = Field(default="普通用户", max_length=32)

    def template_vars(self) -> dict[str, str]:
        """生成供 ``str.format`` 使用的扁平字典。

        键名须与 ``core/base.md`` 中占位符一致：
        ``{output_language}``、``{tone}``、``{audience}``。

        Returns:
            占位符名到字符串值的映射。

        Example:
            >>> PromptKnobs().template_vars()["tone"]
            '正式'
        """
        return {
            "output_language": self.output_language,
            "tone": self.tone.value,
            "audience": self.audience,
        }
