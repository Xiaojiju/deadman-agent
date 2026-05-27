"""测试 normalize_user_input 函数。"""
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from app.core.agent.lab.input_lambda import (
    DEFAULT_INPUT_MESSAGE_KEY,
    build_normalize_chain,
    normalize_user_input
)
from app.core.agent.prompt.context import CONTEXT_TEMPLATE_KEY


def test_normalize_from_plain_string():
    """从纯字符串构建。"""
    out = normalize_user_input(" 你好 ")
    assert DEFAULT_INPUT_MESSAGE_KEY in out
    msg = out[DEFAULT_INPUT_MESSAGE_KEY]
    assert isinstance(msg, HumanMessage)
    assert msg.content == "你好"

def test_normalize_from_raw_dict_key():
    """从 raw 键构建。"""
    out = normalize_user_input({"raw": " 你好 "})
    assert out[DEFAULT_INPUT_MESSAGE_KEY].content == "你好"

def test_normalize_from_input_string_key():
    """从 input 字符串键构建。"""
    out = normalize_user_input({DEFAULT_INPUT_MESSAGE_KEY: "查天气"})
    assert out[DEFAULT_INPUT_MESSAGE_KEY].content == "查天气"

def test_normalize_preserves_extra_keys():
    """保留额外键。"""
    out = normalize_user_input({
        "raw": " 你好 ",
        CONTEXT_TEMPLATE_KEY: " 上下文 ",
    })
    assert out[DEFAULT_INPUT_MESSAGE_KEY].content == "你好"
    assert out[CONTEXT_TEMPLATE_KEY] == " 上下文 "

def test_normalize_chain_invoke_string():
    """链内调用字符串。"""
    chain = build_normalize_chain()
    out = chain.invoke("链内调用")
    assert out[DEFAULT_INPUT_MESSAGE_KEY].content == "链内调用"

def test_normalize_chain_compose_with_peek_lambda():
    """演示 RunnableLambda 用 ``|`` 串联：前一步 enrich dict，后一步只读断言。"""
    recorded: list[dict] = []

    def _record(state: dict) -> dict:
        recorded.append(state)
        return state

    chain = build_normalize_chain() | RunnableLambda(_record)
    result = chain.invoke({"user_input": "串联测试"})
    assert len(recorded) == 1
    assert recorded[0][DEFAULT_INPUT_MESSAGE_KEY].content == "串联测试"
    assert result[DEFAULT_INPUT_MESSAGE_KEY].content == "串联测试"

def test_normalize_invalid_dict_raises():
    """无效 dict 抛 ValueError。"""
    try:
        normalize_user_input({"scene": "default"})
    except ValueError as exc:
        assert "dict" in str(exc)
    else:
        raise AssertionError("expected ValueError")
