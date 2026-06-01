"""Assign 任务增强：上下文增强、系统策略增强。

在 assign 任务中，我们希望增强任务的上下文和系统策略，以提高任务的完成质量。

上下文增强：
- 在 assign 任务中，我们希望增强任务的上下文，以提高任务的完成质量。
- 上下文增强的目的是为了提高任务的完成质量。
- 上下文增强的目的是为了提高任务的完成质量。
"""
from __future__ import annotations
from typing import Any, Mapping

from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from app.core.agent.prompt import (
    CONTEXT_TEMPLATE_KEY,
    PromptKnobs,
    Scene,
    compose_system_prompt,
    EMPTY_CONTEXT_PLACEHOLDER
)
from app.core.agent.runnable import SYSTEM_POLICY_KEY


def resolve_context_body(context: str | None) -> str:
    """解析 context 字符串：
     - 非空白字符串 → 原样返回
     - None、""、" " → 都应得到 EMPTY_CONTEXT_PLACEHOLDER。

    Args:
        context: 上下文字符串

    Returns:
        context 字符串
    """
    return (context or "").strip() or EMPTY_CONTEXT_PLACEHOLDER

def resolve_system_policy(
    scene: Scene | None,
    knobs: PromptKnobs | None,
    *,
    fixed_system_prompt: str | None = None,
) -> str:
    """解析 system 策略：
     - 若 fixed_system_prompt 为 str 则固定使用。
     - 否则使用 ``compose_system_prompt`` 组装。

    Args:
        scene: 业务场景
        knobs: 运行时旋钮
        fixed_system_prompt: 固定 system 策略

    Returns:
        system 策略字符串
    """
    if isinstance(fixed_system_prompt, str):
        return fixed_system_prompt

    return compose_system_prompt(scene, knobs)

def _coerce_scene(value: Any, *, default: Scene) -> Scene:
    """dict 里 scene 可能是 Scene、str 或缺失。"""
    if value is None:
        return default
    if isinstance(value, Scene):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return Scene(text)  # 例如 "rag_qa" -> Scene.RAG_QA
        except ValueError as exc:
            msg = f"无效的 scene: {value!r}，可选: {[s.value for s in Scene]}"
            raise ValueError(msg) from exc
    raise TypeError(f"scene 需要 Scene、str 或 None，收到 {type(value)!r}")

def _coerce_knobs(value: Any, *, default: PromptKnobs) -> PromptKnobs:
    """dict 里 knobs 可能是 PromptKnobs、dict 或缺失。"""
    if value is None:
        return default
    if isinstance(value, PromptKnobs):
        return value
    if isinstance(value, dict):
        return PromptKnobs.model_validate(value)
    raise TypeError(f"knobs 需要 PromptKnobs、dict 或 None，收到 {type(value)!r}")

def _read_context_raw(state: Mapping[str, Any]) -> str | None:
    """读取 context 原始字符串。"""
    for key in ("context", CONTEXT_TEMPLATE_KEY):
        if key not in state:
            continue
        raw = state[key]
        if raw is None or isinstance(raw, str):
            return raw
        raise TypeError(f"{key!r} 需要 str 或 None，收到 {type(raw)!r}")
    return None

def resolve_chain_metadata(
    state: dict[str, Any],
    *,
    default_scene: Scene = Scene.DEFAULT,
    default_knobs: PromptKnobs | None = None
) -> tuple[Scene, PromptKnobs, str | None]:
    """解析链路元数据"""
    knobs_default = default_knobs or PromptKnobs()

    scene = _coerce_scene(state.get("scene"), default=default_scene)
    knobs = _coerce_knobs(state.get("knobs"), default=knobs_default)
    context = _read_context_raw(state)
    return scene, knobs, context

def build_assign_chain(
    *,
    default_scene: Scene = Scene.DEFAULT,
    default_knobs: PromptKnobs | None = None,
    fixed_system_prompt: str | None = None,
) -> Runnable:
    """构建 assign 链路"""
    def _assign_policy(state: dict[str, Any]) -> str:
        scene, knobs, _ = resolve_chain_metadata(
            state,
            default_scene=default_scene,
            default_knobs=default_knobs
        )
        return resolve_system_policy(scene, knobs, fixed_system_prompt=fixed_system_prompt)

    def _assign_context(state: dict[str, Any]) -> str:
        _, _, context = resolve_chain_metadata(
            state,
            default_scene=default_scene,
            default_knobs=default_knobs
        )
        return resolve_context_body(context)

    return RunnablePassthrough.assign(
        **{
            SYSTEM_POLICY_KEY: RunnableLambda(_assign_policy),
            CONTEXT_TEMPLATE_KEY: RunnableLambda(_assign_context),
        }
    )
