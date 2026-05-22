"""Context 层：检索/工具结果，与 Core+Module 分离，单独一条 system message。

Policy（身份、安全、场景规则）由 ``composer`` 组装；本模块只负责**每请求变化**
的参考材料，避免写入 ``manifest.yaml`` 或 git 中的 md 文件。

在 LangChain 链中的典型用法（示意）::

    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from app.core.agent.prompt import (
        DEFAULT_COMPOSED_SYSTEM_PROMPT,
        CONTEXT_TEMPLATE_KEY,
        build_context_system_message,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", DEFAULT_COMPOSED_SYSTEM_PROMPT),
        ("system", CONTEXT_SYSTEM_TEMPLATE),  # 含 {context} 占位符
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ])
    # invoke 时传入 context=检索结果文本，而非写进 composer
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage

CONTEXT_TEMPLATE_KEY = "context"
"""``ChatPromptTemplate`` 中参考材料占位符的变量名。

Example:
    >>> CONTEXT_TEMPLATE_KEY
    'context'
"""

CONTEXT_SYSTEM_TEMPLATE = """\
# 参考材料

以下为本次检索或工具返回的内容，可能为空。回答须遵守系统提示中的 RAG 规则。

{context}
"""
"""第二条 system 消息的模板字符串；``{context}`` 由每次 invoke 注入。"""

EMPTY_CONTEXT_PLACEHOLDER = "（暂无参考材料）"
"""未提供检索结果时写入模板的默认文案（供 API / Runnable 注入 ``{context}``）。"""

# 兼容旧名称
_EMPTY_CONTEXT = EMPTY_CONTEXT_PLACEHOLDER


def build_context_system_message(
    context: str | None,
    *,
    empty_placeholder: str = _EMPTY_CONTEXT,
) -> SystemMessage:
    """构建「参考材料」system 消息，供每请求注入。

    与 ``compose_system_prompt`` 分离：检索内容不进入 manifest 片段，
    避免污染可审阅的 Core/Module，也便于按请求清空或替换。

    Args:
        context: 检索或工具返回的原文；``None`` 或空白时使用 ``empty_placeholder``。
        empty_placeholder: 无内容时的展示文案。

    Returns:
        可直接加入 ``ChatPromptTemplate`` 或消息列表的 ``SystemMessage``。

    Example:
        >>> from app.core.agent.prompt.context import build_context_system_message
        >>> msg = build_context_system_message("文档A：退款流程……")
        >>> msg.type
        'system'
        >>> "文档A" in msg.content
        True

        无检索结果时::

            >>> msg2 = build_context_system_message(None)
            >>> "暂无参考材料" in msg2.content
            True
    """
    body = (context or "").strip() or empty_placeholder
    return SystemMessage(content=CONTEXT_SYSTEM_TEMPLATE.format(context=body))
