"""模块化 system prompt：Core + Module + Knobs + Context。

四层职责
--------
- **Core**（``core/*.md``）：审过不变的身份、格式、安全原则。
- **Module**（``modules/*.md``）：按 ``Scene`` 拼接的场景插件。
- **Knobs**（``knobs.py``）：语言、语气、受众等运行时变量。
- **Context**（``context.py``）：检索结果，单独 ``SystemMessage``，不进 manifest。

快速上手
--------
::

    from app.core.agent.prompt import (
        DEFAULT_COMPOSED_SYSTEM_PROMPT,
        Scene,
        PromptKnobs,
        Tone,
        compose_system_prompt,
        build_context_system_message,
    )

    # 默认 policy（runnable 已使用）
    policy = DEFAULT_COMPOSED_SYSTEM_PROMPT

    # 指定场景与旋钮
    policy = compose_system_prompt(
        Scene.RAG_QA,
        PromptKnobs(output_language="简体中文", tone=Tone.CONCISE),
    )

    # RAG 参考材料（单独一条 system）
    ctx_msg = build_context_system_message("片段1\\n片段2")

详见同目录 ``ARCHITECTURE.md``。
"""

from app.core.agent.prompt.composer import (
    DEFAULT_COMPOSED_SYSTEM_PROMPT,
    SystemPromptComposer,
    compose_system_prompt,
)
from app.core.agent.prompt.context import (
    CONTEXT_SYSTEM_TEMPLATE,
    CONTEXT_TEMPLATE_KEY,
    EMPTY_CONTEXT_PLACEHOLDER,
    build_context_system_message,
)
from app.core.agent.prompt.knobs import PromptKnobs, Scene, Tone

__all__ = [
    "CONTEXT_SYSTEM_TEMPLATE",
    "CONTEXT_TEMPLATE_KEY",
    "EMPTY_CONTEXT_PLACEHOLDER",
    "DEFAULT_COMPOSED_SYSTEM_PROMPT",
    "PromptKnobs",
    "Scene",
    "SystemPromptComposer",
    "Tone",
    "build_context_system_message",
    "compose_system_prompt",
]
