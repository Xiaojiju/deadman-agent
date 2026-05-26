"""智能家居灯光控制工具（演示 / 本地假实现，可替换为 MCP 调用）。"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

# 进程内假状态，便于演示与测试；生产环境应改为 MCP / IoT API
_LIGHT_STATE: dict[str, bool] = {}


class ControlLightInput(BaseModel):
    """``control_light`` 工具的参数 schema（由模型读取，不写入 prompt）。"""

    room: str = Field(
        description="房间标识，如 living_room（客厅）、bedroom（卧室）",
        min_length=1,
        max_length=64,
    )
    action: Literal["on", "off"] = Field(description="开灯 on 或关灯 off")


def control_light(room: str, action: str) -> dict[str, object]:
    """控制指定房间灯光（假实现）。"""
    on = action == "on"
    _LIGHT_STATE[room] = on
    return {
        "ok": True,
        "room": room,
        "action": action,
        "state": "on" if on else "off",
    }


def get_control_light_tool() -> StructuredTool:
    """返回 LangChain 结构化工具实例。"""
    return StructuredTool.from_function(
        func=control_light,
        name="control_light",
        description="控制指定房间的灯光开关。仅在用户明确要求且房间可识别时调用。",
        args_schema=ControlLightInput,
    )


def reset_light_state() -> None:
    """测试用：清空假设备状态。"""
    _LIGHT_STATE.clear()
