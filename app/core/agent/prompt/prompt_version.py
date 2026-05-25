"""Prompt 资产包版本元数据（与应用程序版本分离）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from app.core.agent.prompt.prompt_loader import MANIFEST_BASENAME, PROMPT_DIR

# 当前应用代码支持的 manifest / 片段契约；不兼容时需升应用 semver
SUPPORTED_PROMPT_PACK_FORMAT = 1

DEFAULT_PROMPT_VERSION = "0.0.0"
DEFAULT_PACK_FORMAT = 1


@dataclass(frozen=True, slots=True)
class PromptPackMeta:
    """``manifest.yaml`` 中的版本与场景映射。"""

    prompt_version: str
    prompt_pack_format: int
    app_compat_min: str | None
    scenes: dict[str, list[str]]

    def as_dict(self) -> dict[str, str | int | list[str]]:
        """供 API 返回的扁平结构（不含 scenes 路径列表的嵌套展开）。"""
        return {
            "prompt_version": self.prompt_version,
            "prompt_pack_format": self.prompt_pack_format,
            "supported_pack_format": SUPPORTED_PROMPT_PACK_FORMAT,
            "app_compat_min": self.app_compat_min or "",
            "scene_ids": sorted(self.scenes.keys()),
        }


class UnsupportedPromptPackFormatError(ValueError):
    """manifest 中 ``prompt_pack_format`` 高于应用支持能力。"""


def parse_manifest_file(path: Path) -> PromptPackMeta:
    """解析 manifest 文件。"""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"manifest 根节点须为映射: {path}")

    scenes = data.get("scenes")
    if not scenes or not isinstance(scenes, dict):
        raise ValueError(f"manifest 缺少 scenes: {path}")

    version = str(data.get("prompt_version") or DEFAULT_PROMPT_VERSION)
    pack_format = int(data.get("prompt_pack_format", DEFAULT_PACK_FORMAT))
    app_compat = data.get("app_compat_min")
    app_compat_min = str(app_compat) if app_compat is not None else None

    return PromptPackMeta(
        prompt_version=version,
        prompt_pack_format=pack_format,
        app_compat_min=app_compat_min,
        scenes={str(k): list(v) for k, v in scenes.items()},
    )


def validate_pack_format(meta: PromptPackMeta) -> None:
    """校验契约版本是否被当前应用支持。"""
    if meta.prompt_pack_format != SUPPORTED_PROMPT_PACK_FORMAT:
        raise UnsupportedPromptPackFormatError(
            f"prompt_pack_format={meta.prompt_pack_format} 不受支持，"
            f"当前应用仅支持 format {SUPPORTED_PROMPT_PACK_FORMAT}。"
            "请升级应用或回滚 Prompt 包。"
        )


def load_prompt_pack_meta(root: Path | None = None) -> PromptPackMeta:
    """从 prompt 根目录加载并校验版本元数据。"""
    base = root or PROMPT_DIR
    meta = parse_manifest_file(base / MANIFEST_BASENAME)
    validate_pack_format(meta)
    return meta
