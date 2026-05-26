"""Agent 工具注册。"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.core.agent.tools.light import get_control_light_tool


def get_smart_home_tools() -> list[BaseTool]:
    """smart_home 场景可用工具列表。"""
    return [get_control_light_tool()]
