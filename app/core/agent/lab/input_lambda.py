from typing import Any, Mapping

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda


# 与 RunnableWithHistory 默认一致，便于后续无缝接到 prompt | model
DEFAULT_INPUT_MESSAGE_KEY = "input"
_ALTERNATE_TEXT_KEYS = ("user_input", "raw", "text", "message")

def normalize_user_input(
    value: str | Mapping[str, Any],
    *,
    input_message_key: str = "input"
) -> dict[str, Any]:
    """把 ``str`` 或多种形态的 ``dict`` 规范为含 ``HumanMessage`` 的链输入 dict。
    
    支持：
      - ``"你好"`` → ``{"input": HumanMessage(content="你好")}``
      - ``{"raw": "你好"}`` / ``{"user_input": "..."}`` → 写入 ``input_message_key``
      - ``{"input": "..."}`` 或已有 ``HumanMessage`` → 转为 / 保留 ``HumanMessage``
      - dict 中已有的其它键（如预置的 ``context``）原样保留
    
    Raises:
        TypeError: 输入类型不支持
        ValueError: dict 中找不到可用的用户文本
    """
    if isinstance(value, str):
        return {input_message_key: HumanMessage(content=value.strip())}

    out: dict[str, Any] = dict(value)

    if input_message_key in out:
        normalized = _coerce_to_human_message(out[input_message_key])
        if normalized is not None:
            out[input_message_key] = normalized
            return out

    for alt_key in _ALTERNATE_TEXT_KEYS:
        if alt_key not in out:
            continue
        candidate = out.pop(alt_key)
        normalized = _coerce_to_human_message(candidate)
        if normalized is None:
            msg = f"键 {alt_key!r} 的值无法转为 HumanMessage: {type(candidate)!r}"
            raise ValueError(msg)
        out[input_message_key] = normalized
        return out

    msg = (
    f"dict 中需要 {input_message_key!r}、"
    f"{', '.join(repr(k) for k in _ALTERNATE_TEXT_KEYS)} 之一，且为可解析的用户文本")
    raise ValueError(msg)

def _coerce_to_human_message(value: str | BaseMessage) -> HumanMessage | None:
    """尝试将字符串或 BaseMessage 转为 HumanMessage。
    
    Args:
        value: 字符串或 BaseMessage

    Returns:
        HumanMessage 或 None

    Note:
        - 与 ``RunnableWithHistory`` 一致，便于无缝接到 prompt | model
        - 可再接 ``assign`` / ``prompt | model``
    """
    if isinstance(value, HumanMessage):
        return value
    if isinstance(value, str):
        return HumanMessage(content=value.strip())
    if isinstance(value, BaseMessage) and value.type == "human":
        return HumanMessage(content=value.content)
    return None

def build_normalize_chain(
    *,
    input_message_key: str = DEFAULT_INPUT_MESSAGE_KEY
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """``RunnableLambda(normalize)`` 链；可再接 ``assign`` / ``prompt | model``。
    
    Args:
        input_message_key: 输入消息键；默认与 ``RunnableWithHistory`` 一致

    Returns:
        ``RunnableLambda(normalize)`` 链；可再接 ``assign`` / ``prompt | model``

    Note:
        - 与 ``RunnableWithHistory`` 一致，便于无缝接到 prompt | model
        - 可再接 ``assign`` / ``prompt | model``
    """

    def _normalize(value: str | Mapping[str, Any]) -> dict[str, Any]:
        return normalize_user_input(value, input_message_key=input_message_key)

    return RunnableLambda(_normalize)
