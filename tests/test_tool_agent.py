"""Tool Agent 与降级逻辑测试。"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from app.core.agent.prompt.knobs import PromptKnobs, Scene
from app.core.agent.runnable import RunnableWithHistory
from app.core.agent.tool_agent import (
    DegradationReason,
    ToolAgentModelError,
    ToolAgentRunner,
    _execute_tool,
)
from app.core.agent.tools.light import control_light, get_control_light_tool, reset_light_state


def _tool_call_message() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "control_light",
                "args": {"room": "living_room", "action": "off"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )


def _make_bound_model(*responses: AIMessage) -> MagicMock:
    bound = MagicMock()
    bound.invoke.side_effect = list(responses)
    model = MagicMock()
    model.bind_tools.return_value = bound
    return model


@pytest.fixture(autouse=True)
def _clear_lights():
    reset_light_state()
    yield
    reset_light_state()


def test_control_light_tool():
    """测试控制灯光工具。"""
    out = control_light("living_room", "off")
    assert out["ok"] is True
    assert out["state"] == "off"


def test_execute_tool_unknown():
    """测试执行未知工具。"""
    msg, trace = _execute_tool({}, "missing_tool", {"x": 1}, "id-1")
    assert trace.ok is False
    payload = json.loads(msg.content)
    assert payload["ok"] is False


def test_execute_tool_success():
    """测试执行成功工具。"""
    tool = get_control_light_tool()
    msg, trace = _execute_tool(
        {tool.name: tool},
        "control_light",
        {"room": "bedroom", "action": "on"},
        "id-2",
    )
    assert trace.ok is True
    assert json.loads(msg.content)["state"] == "on"


def test_tool_agent_happy_path():
    """测试工具 Agent 正常路径。"""
    chat_fallback = MagicMock(spec=RunnableWithHistory)
    chat_fallback._get_session_history = MagicMock( # pylint: disable=protected-access
        return_value=MagicMock(messages=[], add_messages=MagicMock())
    )
    runner = ToolAgentRunner(
        model=_make_bound_model(
            _tool_call_message(),
            AIMessage(content="好的，已关闭客厅灯。"),
        ),
        chat_fallback=chat_fallback,
        max_tool_rounds=5,
    )
    result = runner.invoke(
        "sess-tool-1",
        "帮我把客厅灯关了",
        scene=Scene.SMART_HOME,
        knobs=PromptKnobs(),
    )
    assert result.degraded is False
    assert result.mode.value == "tool_agent"
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0].ok is True
    assert "关闭" in result.final_message.content


def test_tool_agent_tools_disabled_falls_back():
    """测试工具禁用时降级（走 plain chat，不经 RunnableWithHistory）。"""
    history_store = MagicMock(messages=[], add_messages=MagicMock())
    chat_fallback = MagicMock(spec=RunnableWithHistory)
    chat_fallback._get_session_history = MagicMock(return_value=history_store)  # pylint: disable=protected-access
    model = _make_bound_model()
    model.invoke.return_value = AIMessage(content="降级回复：普通对话。")
    runner = ToolAgentRunner(model=model, chat_fallback=chat_fallback)
    result = runner.invoke(
        "sess-fallback-1",
        "关灯",
        enable_tools=False,
    )
    assert result.degraded is True
    assert result.degradation_reason == DegradationReason.TOOLS_DISABLED
    assert result.mode.value == "chat_fallback"
    model.invoke.assert_called_once()
    history_store.add_messages.assert_called_once()
    chat_fallback.invoke.assert_not_called()


def test_tool_agent_bind_tools_failure_falls_back():
    """测试工具绑定失败时降级（plain chat）。"""
    model = MagicMock(spec=BaseChatModel)
    model.bind_tools = MagicMock(side_effect=RuntimeError("unsupported"))
    model.invoke.return_value = AIMessage(content="bind 降级。")
    history_store = MagicMock(messages=[], add_messages=MagicMock())
    chat_fallback = MagicMock(spec=RunnableWithHistory)
    chat_fallback._get_session_history = MagicMock(return_value=history_store)  # pylint: disable=protected-access
    runner = ToolAgentRunner(model=model, chat_fallback=chat_fallback)
    result = runner.invoke("sess-bind-fail", "关灯")
    assert result.degraded is True
    assert result.degradation_reason == DegradationReason.BIND_TOOLS_FAILED
    model.invoke.assert_called_once()
    chat_fallback.invoke.assert_not_called()


def test_tool_agent_max_iterations_degraded():
    """测试工具迭代上限时降级。"""
    chat_fallback = MagicMock(spec=RunnableWithHistory)
    chat_fallback._get_session_history = MagicMock(  # pylint: disable=protected-access
        return_value=MagicMock(messages=[], add_messages=MagicMock())
    )
    model = _make_bound_model(
        _tool_call_message(),
        _tool_call_message(),
    )
    runner = ToolAgentRunner(
        model=model,
        chat_fallback=chat_fallback,
        max_tool_rounds=2,
    )
    result = runner.invoke("sess-max", "关灯")
    assert result.degraded is True
    assert result.degradation_reason == DegradationReason.MAX_ITERATIONS
    assert len(result.tool_trace) == 2
    # 达上限后应再 invoke 一次无工具收尾（3 次 bound + 1 次 base model）
    assert model.bind_tools.return_value.invoke.call_count == 2
    assert model.invoke.call_count == 1


def test_tool_agent_max_iterations_uses_model_summary_when_available():
    """测试工具迭代上限时使用模型总结。"""
    chat_fallback = MagicMock(spec=RunnableWithHistory)
    chat_fallback._get_session_history = MagicMock( # pylint: disable=protected-access
        return_value=MagicMock(messages=[], add_messages=MagicMock())
    )
    model = _make_bound_model(_tool_call_message(), _tool_call_message())
    model.invoke.return_value = AIMessage(content="已尽力关闭客厅灯，请确认设备状态。")
    runner = ToolAgentRunner(
        model=model,
        chat_fallback=chat_fallback,
        max_tool_rounds=1,
    )
    result = runner.invoke("sess-cap-summary", "关灯")
    assert result.degradation_reason == DegradationReason.MAX_ITERATIONS
    assert "客厅灯" in result.final_message.content
    model.invoke.assert_called_once()


def test_tool_agent_model_error_preserves_partial_and_plain_chat():
    """工具循环中途模型失败：保留 partial_turn + tool_trace，再 plain chat 总结。"""
    history_store = MagicMock(messages=[], add_messages=MagicMock())
    chat_fallback = MagicMock(spec=RunnableWithHistory)
    chat_fallback._get_session_history = MagicMock(return_value=history_store)  # pylint: disable=protected-access

    bound = MagicMock()
    bound.invoke.side_effect = [
        _tool_call_message(),
        RuntimeError("503 upstream"),
    ]
    model = MagicMock()
    model.bind_tools.return_value = bound
    model.invoke.return_value = AIMessage(content="灯控服务异常，请稍后在 App 内重试。")

    runner = ToolAgentRunner(model=model, chat_fallback=chat_fallback, max_tool_rounds=5)
    result = runner.invoke("sess-partial", "关客厅灯")

    assert result.degraded is True
    assert result.degradation_reason == DegradationReason.MODEL_INVOKE_FAILED
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0].ok is True
    assert result.retryable is True
    model.invoke.assert_called_once()
    stored = history_store.add_messages.call_args[0][0]
    assert len(stored) >= 3  # human + ai(tool_calls) + tool + final ai
    assert stored[0].type == "human"
    chat_fallback.invoke.assert_not_called()


def test_classify_rate_limit_retryable():
    """限流类异常应标记 retryable。"""
    from app.core.agent.tool_agent import _classify_model_exception

    assert _classify_model_exception(RuntimeError("Error 429 rate limit exceeded")) is True
    assert _classify_model_exception(ValueError("bad request")) is False
