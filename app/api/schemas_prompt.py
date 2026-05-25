"""Prompt 版本查询 API 模型。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PromptPackMetaData(BaseModel):
    """当前加载的 Prompt 资产包元数据。"""

    prompt_version: str = Field(description="Prompt 文案版本，见 prompt/CHANGELOG.md")
    prompt_pack_format: int = Field(description="Prompt 契约格式版本")
    supported_pack_format: int = Field(description="当前应用代码支持的 format")
    app_compat_min: str = Field(description="manifest 声明的建议最低应用版本")
    scene_ids: list[str] = Field(description="manifest 中已配置的场景 ID")
    app_version: str = Field(description="当前运行的应用版本（pyproject）")
    changelog_path: str = Field(
        default="app/core/agent/prompt/CHANGELOG.md",
        description="变更日志相对仓库路径",
    )
