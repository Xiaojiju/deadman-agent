"""Tool Agent：工具调用循环 + 降级到普通对话。

两条消费线：
- **tool_agent**：``bind_tools`` + 工具 schema；历史写入 ``[human, ai?, tool*, ai_final]``。
- **chat_fallback**：与 Tool 共用 ``_build_turn_messages`` 的无工具 ``model.invoke``（**不**再走
  ``RunnableWithHistory``），避免 system/history 拼装不一致；Tool 半途失败时合并 ``partial_turn``。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_core.chat_history import BaseChatMessageHistory

from app.core.agent.default_model import BasicAdapterModel
from app.core.agent.prompt import (
    EMPTY_CONTEXT_PLACEHOLDER,
    compose_system_prompt,
)
from app.core.agent.prompt.context import build_context_system_message
from app.core.agent.prompt.knobs import PromptKnobs, Scene
from app.core.agent.runnable import RunnableWithHistory, completation_runnable
from app.core.agent.tools import get_smart_home_tools
from app.core.config import get_settings
from app.core.log_config import logger as log

DEFAULT_MAX_TOOL_ROUNDS = 5


class BindToolsError(Exception):
    """``bind_tools`` 失败，用于触发降级。"""


class ToolAgentModelError(Exception):
    """工具循环内 ``bound.invoke`` / 收尾调用失败。

    可携带已产生的 ``partial_turn``（AIMessage + ToolMessage），
    供降级路径合并进同一条消息链后再做**无工具**总结，避免历史断层。
    """

    def __init__(
        self,
        message: str,
        *,
        partial_turn: list[BaseMessage] | None = None,
        tool_trace: list[ToolTraceItem] | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_turn = partial_turn or []
        self.tool_trace = tool_trace or []


class AgentMode(str, Enum):
    """本次请求实际使用的消费路径。 """

    TOOL_AGENT = "tool_agent"
    CHAT_FALLBACK = "chat_fallback"


class DegradationReason(str, Enum):
    """降级原因（``degraded=False`` 时为 ``none``）。"""

    NONE = "none"
    TOOLS_DISABLED = "tools_disabled"
    BIND_TOOLS_FAILED = "bind_tools_failed"
    MODEL_INVOKE_FAILED = "model_invoke_failed"
    MAX_ITERATIONS = "max_iterations"


@dataclass
class ToolTraceItem:
    """单次工具调用记录（供 API / 审计）。

    Attributes:
        tool: 工具名称。
        args: 工具参数。
        result: 工具结果。
        ok: 是否成功。
        error: 错误信息。
    """

    tool: str
    args: dict[str, Any]
    result: dict[str, Any] | str
    ok: bool
    error: str | None = None


@dataclass
class ToolAgentResult:
    """Tool Agent 单次 invoke 结果。

    Attributes:
        final_message: 最终消息。
        tool_trace: 工具调用轨迹。
        mode: 模式。
        degraded: 是否降级。
        degradation_reason: 降级原因。
        retryable: 是否重试。
    """

    final_message: AIMessage
    tool_trace: list[ToolTraceItem] = field(default_factory=list)
    mode: AgentMode = AgentMode.TOOL_AGENT
    degraded: bool = False
    degradation_reason: DegradationReason = DegradationReason.NONE
    retryable: bool = False


def _tool_error_payload(tool_name: str, error: str) -> dict[str, object]:
    """工具错误 payload。

    Args:
        tool_name: 工具名称。
        error: 错误信息。

    Returns:
        payload: 工具错误 payload。
    """
    return {"ok": False, "tool": tool_name, "error": error}


def _execute_tool(
    tools_by_name: dict[str, BaseTool],
    tool_name: str,
    tool_args: dict[str, Any],
    tool_call_id: str,
) -> tuple[ToolMessage, ToolTraceItem]:
    """执行单个 tool_call；失败时返回错误 ToolMessage，不抛异常。
    
    Args:
        tools_by_name: 工具名称到工具实例的映射。
        tool_name: 工具名称。
        tool_args: 工具参数。
        tool_call_id: 工具调用 ID。

    Returns:

        tool_msg: 工具消息。
        trace: 工具调用轨迹。
        
    Raises:
        Exception: 工具执行失败时抛出。
    """
    tool = tools_by_name.get(tool_name)
    if tool is None:
        err = f"未知工具: {tool_name}"
        payload = _tool_error_payload(tool_name, err)
        trace = ToolTraceItem(
            tool=tool_name,
            args=tool_args,
            result=payload,
            ok=False,
            error=err,
        )
        return (
            ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id=tool_call_id),
            trace
        )

    try:
        raw = tool.invoke(tool_args)
        if isinstance(raw, dict):
            result: dict[str, Any] | str = raw
            ok = bool(raw.get("ok", True))
            err = None if ok else str(raw.get("error", "工具返回失败"))
        else:
            result = str(raw)
            ok = True
            err = None
        trace = ToolTraceItem(tool=tool_name, args=tool_args, result=result, ok=ok, error=err)
        content = json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else result
        return ToolMessage(content=content, tool_call_id=tool_call_id), trace
    except Exception as exc: # pylint: disable=broad-exception-caught
        log.exception("工具执行失败 tool=%s args=%s", tool_name, tool_args)
        err = str(exc)
        payload = _tool_error_payload(tool_name, err)
        trace = ToolTraceItem(
            tool=tool_name,
            args=tool_args,
            result=payload,
            ok=False,
            error=err,
        )
        return (
            ToolMessage(
                content=json.dumps(payload, ensure_ascii=False),
                tool_call_id=tool_call_id,
            ),
            trace,
        )


def _parse_tool_call(
    call: Any,
    *,
    round_idx: int,
    call_idx: int,
) -> tuple[str, dict[str, Any], str]:
    """从 LangChain tool_call（dict 或对象）解析 name / args / id。
    
    Args:
        call: LangChain tool_call。
        round_idx: 工具轮次索引。
        call_idx: 工具调用索引。
    Returns:
        name: 工具名称。
    """
    if isinstance(call, dict):
        name = str(call.get("name") or "")
        args = call.get("args") or {}
        tool_call_id = str(call.get("id") or "")
    else:
        name = str(getattr(call, "name", "") or "")
        args = getattr(call, "args", None) or {}
        tool_call_id = str(getattr(call, "id", "") or "")
    if not isinstance(args, dict):
        args = dict(args) if args else {}
    if not tool_call_id:
        tool_call_id = f"call_{round_idx}_{call_idx}_{uuid.uuid4().hex[:8]}"
    return name, args, tool_call_id


def _ai_text_content(message: AIMessage) -> str:
    """将 AIMessage 转换为文本内容。"""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return str(content).strip()


def _max_iterations_fallback_text(tool_trace: list[ToolTraceItem]) -> str:
    """工具轮次用尽时的模板文案。"""
    if not tool_trace:
        return (
            "抱歉，设备操作步骤较多，未能在一轮内全部完成。"
            "请补充更具体的信息，或稍后再试。"
        )
    ok_count = sum(1 for t in tool_trace if t.ok)
    return (
        f"已执行 {len(tool_trace)} 次工具调用（成功 {ok_count} 次），"
        "但未得到最终确认回复。请查看操作结果或重试。"
    )


def _plain_chat_unavailable_text() -> str:
    """普通对话不可用时的模板文案。"""
    return "服务暂时不可用，请稍后再试。"


def _to_ai_message(raw: Any) -> AIMessage:
    """将模型响应转换为 AIMessage。"""
    if isinstance(raw, AIMessage):
        return raw
    if isinstance(raw, BaseMessage):
        return AIMessage(content=str(raw.content))
    return AIMessage(content=str(raw))


def _classify_model_exception(exc: BaseException) -> bool:
    """根据异常类型/文案判断是否建议业务层重试（限流、超时、网关错误等）。"""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if "ratelimit" in name or "rate limit" in msg or "429" in msg:
        return True
    if "timeout" in name or "timed out" in msg:
        return True
    if "connection" in msg or "503" in msg or "502" in msg or "504" in msg:
        return True
    return False


class ToolAgentRunner:
    """带历史记录的工具 Agent；失败时降级为 ``RunnableWithHistory`` 普通对话。"""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        chat_fallback: RunnableWithHistory,
        tools: Sequence[BaseTool] | None = None,
        default_scene: Scene = Scene.SMART_HOME,
        default_knobs: PromptKnobs | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    ) -> None:
        """初始化 Tool Agent Runner。
        
        Args:
            model: 模型实例。
            chat_fallback: 普通对话降级路径的 RunnableWithHistory 实例。
            tools: 工具列表。
            default_scene: 默认场景。
            default_knobs: 默认 knobs，默认为空。
            max_tool_rounds: 最大工具轮次，默认为 5。
        """
        self.model = model
        self.chat_fallback = chat_fallback
        self.tools = list(tools or get_smart_home_tools())
        self.default_scene = default_scene
        self.default_knobs = default_knobs or PromptKnobs()
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds 至少为 1")
        self.max_tool_rounds = max_tool_rounds
        self._get_session_history = chat_fallback._get_session_history

    def _build_turn_messages(
        self,
        session_id: str,
        user_input: str,
        *,
        scene: Scene,
        knobs: PromptKnobs,
        context: str | None,
    ) -> list[BaseMessage]:
        """构建消息列表。"""
        history: BaseChatMessageHistory = self._get_session_history(session_id)
        policy = compose_system_prompt(scene, knobs)
        ctx_body = (context or "").strip() or EMPTY_CONTEXT_PLACEHOLDER
        context_msg = build_context_system_message(ctx_body)
        return [
            SystemMessage(content=policy),
            context_msg,
            *history.messages,
            HumanMessage(content=user_input),
        ]

    def _persist_turn(self, session_id: str, new_messages: Sequence[BaseMessage]) -> None:
        """持久化消息到历史记录。"""
        if not new_messages:
            return
        history = self._get_session_history(session_id)
        history.add_messages(list(new_messages))

    def _invoke_plain_chat_turn(
        self,
        session_id: str,
        user_input: str,
        *,
        scene: Scene,
        knobs: PromptKnobs,
        context: str | None,
        reason: DegradationReason,
        partial_turn: Sequence[BaseMessage] | None = None,
        tool_trace: list[ToolTraceItem] | None = None,
        retryable: bool = False,
    ) -> ToolAgentResult:
        """普通对话降级：与 Tool 路径共用 ``_build_turn_messages``，只调无工具的 ``model.invoke``。

        与 ``chat_fallback.invoke``（LCEL + RunnableWithHistory）分离，避免：
        - 同一 session 里 system/history 拼装方式不一致；
        - 降级时重复写入 human（Tool 半途失败时已产生 partial_turn）。

        ``partial_turn`` 为 Tool 循环已产生的 AIMessage / ToolMessage，会拼进本次
        ``messages`` 再让模型总结，并一次性写入历史。
        """
        human = HumanMessage(content=user_input)
        messages = self._build_turn_messages(
            session_id,
            user_input,
            scene=scene,
            knobs=knobs,
            context=context,
        )
        if partial_turn:
            messages.extend(partial_turn)

        try:
            final = _to_ai_message(self.model.invoke(messages))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log.exception(
                "普通对话降级模型调用失败 session=%s reason=%s",
                session_id,
                reason.value,
            )
            final = AIMessage(content=_plain_chat_unavailable_text())
            retryable = retryable or _classify_model_exception(exc)

        to_store: list[BaseMessage] = [human]
        if partial_turn:
            to_store.extend(partial_turn)
        to_store.append(final)
        self._persist_turn(session_id, to_store)

        return ToolAgentResult(
            final_message=final,
            tool_trace=tool_trace or [],
            mode=AgentMode.CHAT_FALLBACK,
            degraded=True,
            degradation_reason=reason,
            retryable=retryable,
        )

    def _invoke_chat_fallback(
        self,
        session_id: str,
        user_input: str,
        *,
        scene: Scene,
        knobs: PromptKnobs,
        context: str | None,
        reason: DegradationReason,
        partial_turn: Sequence[BaseMessage] | None = None,
        tool_trace: list[ToolTraceItem] | None = None,
        retryable: bool = False,
    ) -> ToolAgentResult:
        """Tool Agent 降级为普通对话。"""
        log.warning(
            "Tool Agent 降级为普通对话 session=%s reason=%s partial_msgs=%s",
            session_id,
            reason.value,
            len(partial_turn or []),
        )
        return self._invoke_plain_chat_turn(
            session_id,
            user_input,
            scene=scene,
            knobs=knobs,
            context=context,
            reason=reason,
            partial_turn=partial_turn,
            tool_trace=tool_trace,
            retryable=retryable,
        )

    def _finalize_after_tool_cap(
        self,
        messages: list[BaseMessage],
        tool_trace: list[ToolTraceItem],
    ) -> AIMessage:
        """工具轮次用尽后，再调用一次**无工具**模型生成面向用户的总结；失败则用模板文案。"""
        try:
            raw = self.model.invoke(messages)
            if isinstance(raw, AIMessage):
                if raw.tool_calls:
                    text = _ai_text_content(raw)
                    if text:
                        return AIMessage(content=text)
                else:
                    text = _ai_text_content(raw)
                    if text:
                        return raw
            elif raw is not None:
                text = str(raw).strip()
                if text:
                    return AIMessage(content=text)
        except Exception: # pylint: disable=broad-exception-caught
            log.exception("达工具轮次上限后的收尾模型调用失败")
        return AIMessage(content=_max_iterations_fallback_text(tool_trace))

    def _run_tool_loop(
        self,
        messages: list[BaseMessage],
        tools_by_name: dict[str, BaseTool],
    ) -> tuple[AIMessage, list[BaseMessage], list[ToolTraceItem], DegradationReason]:
        """执行 model ↔ tool 循环。

        每一轮：模型 →（若有 tool_calls）执行工具 → 进入下一轮。
        正常结束：模型返回无 tool_calls 的 AIMessage。
        达到 ``max_tool_rounds``：在已有 ToolMessage 基础上再调用一次**无 bind_tools** 的模型收尾；
        若仍失败则使用模板降级文案（``degraded=True``）。
        """
        try:
            bound = self.model.bind_tools(self.tools)
        except Exception as exc:
            raise BindToolsError(str(exc)) from exc

        turn_messages: list[BaseMessage] = []
        tool_trace: list[ToolTraceItem] = []

        for round_idx in range(self.max_tool_rounds):
            try:
                ai_msg = bound.invoke(messages)
            except Exception as exc:
                raise ToolAgentModelError(
                    f"工具循环第 {round_idx + 1} 轮模型调用失败: {exc}",
                    partial_turn=list(turn_messages),
                    tool_trace=list(tool_trace),
                ) from exc
            if not isinstance(ai_msg, AIMessage):
                ai_msg = AIMessage(content=str(ai_msg))
            messages.append(ai_msg)
            turn_messages.append(ai_msg)

            # 模型响应不需要调用工具
            tool_calls = ai_msg.tool_calls or []
            if not tool_calls:
                return ai_msg, turn_messages, tool_trace, DegradationReason.NONE

            # 需要调用工具时，执行工具
            for call_idx, call in enumerate(tool_calls):
                tool_name, tool_args, tool_call_id = _parse_tool_call(
                    call,
                    round_idx=round_idx,
                    call_idx=call_idx,
                )
                tool_msg, trace = _execute_tool(
                    tools_by_name,
                    tool_name,
                    tool_args,
                    tool_call_id,
                )
                messages.append(tool_msg)
                turn_messages.append(tool_msg)
                tool_trace.append(trace)

        degradation = DegradationReason.MAX_ITERATIONS
        final = self._finalize_after_tool_cap(messages, tool_trace)
        messages.append(final)
        turn_messages.append(final)
        return final, turn_messages, tool_trace, degradation

    def invoke(
        self,
        session_id: str,
        user_input: str,
        *,
        scene: Scene | None = None,
        knobs: PromptKnobs | None = None,
        context: str | None = None,
        enable_tools: bool = True,
    ) -> ToolAgentResult:
        """执行 Tool Agent；``enable_tools=False`` 或异常时降级为普通对话。"""
        resolved_scene = scene or self.default_scene
        resolved_knobs = knobs or self.default_knobs

        if not enable_tools:
            return self._invoke_chat_fallback(
                session_id,
                user_input,
                scene=resolved_scene,
                knobs=resolved_knobs,
                context=context,
                reason=DegradationReason.TOOLS_DISABLED,
            )

        if not self.tools:
            return self._invoke_chat_fallback(
                session_id,
                user_input,
                scene=resolved_scene,
                knobs=resolved_knobs,
                context=context,
                reason=DegradationReason.BIND_TOOLS_FAILED,
            )

        tools_by_name = {t.name: t for t in self.tools}
        human = HumanMessage(content=user_input)

        try:
            messages = self._build_turn_messages(
                session_id,
                user_input,
                scene=resolved_scene,
                knobs=resolved_knobs,
                context=context,
            )
            final, turn_messages, tool_trace, deg_reason = self._run_tool_loop(
                messages,
                tools_by_name,
            )
        except BindToolsError:
            log.warning("bind_tools 失败，降级为普通对话 session=%s", session_id)
            return self._invoke_chat_fallback(
                session_id,
                user_input,
                scene=resolved_scene,
                knobs=resolved_knobs,
                context=context,
                reason=DegradationReason.BIND_TOOLS_FAILED,
            )
        except ToolAgentModelError as exc:
            log.warning(
                "工具循环模型失败，带 partial 降级 session=%s partial=%s",
                session_id,
                len(exc.partial_turn),
            )
            return self._invoke_chat_fallback(
                session_id,
                user_input,
                scene=resolved_scene,
                knobs=resolved_knobs,
                context=context,
                reason=DegradationReason.MODEL_INVOKE_FAILED,
                partial_turn=exc.partial_turn,
                tool_trace=exc.tool_trace,
                retryable=_classify_model_exception(exc.__cause__ or exc),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log.exception("Tool Agent 未预期错误，降级为普通对话 session=%s", session_id)
            return self._invoke_chat_fallback(
                session_id,
                user_input,
                scene=resolved_scene,
                knobs=resolved_knobs,
                context=context,
                reason=DegradationReason.MODEL_INVOKE_FAILED,
                retryable=_classify_model_exception(exc),
            )

        self._persist_turn(session_id, [human, *turn_messages])
        return ToolAgentResult(
            final_message=final,
            tool_trace=tool_trace,
            mode=AgentMode.TOOL_AGENT,
            degraded=deg_reason != DegradationReason.NONE,
            degradation_reason=deg_reason,
        )


settings = get_settings()

tool_agent_runner = ToolAgentRunner(
    model=BasicAdapterModel.from_settings(settings),
    chat_fallback=completation_runnable,
)
