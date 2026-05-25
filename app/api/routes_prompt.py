"""Prompt 版本与契约查询（与聊天 API 分离）。"""

from __future__ import annotations

import importlib.metadata

from fastapi import APIRouter

from app.api.api_typing import ApiResponse
from app.api.schemas_prompt import PromptPackMetaData
from app.core.agent.prompt.composer import get_prompt_pack_meta

router = APIRouter()


def _app_version() -> str:
    try:
        return importlib.metadata.version("deadman-agent-py")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


@router.get("/chat/prompt/meta", response_model=ApiResponse[PromptPackMetaData])
def get_chat_prompt_meta() -> ApiResponse[PromptPackMetaData]:
    """返回当前运行的 Prompt 资产包版本（用于 A/B 记录、回滚对照、排障）。

    仅反映磁盘上 ``manifest.yaml`` 与代码加载结果；切换 Prompt 后重启服务生效。
    """
    meta = get_prompt_pack_meta()
    flat = meta.as_dict()
    return ApiResponse(
        data=PromptPackMetaData(
            prompt_version=str(flat["prompt_version"]),
            prompt_pack_format=int(flat["prompt_pack_format"]),
            supported_pack_format=int(flat["supported_pack_format"]),
            app_compat_min=str(flat["app_compat_min"]),
            scene_ids=list(flat["scene_ids"]),
            app_version=_app_version(),
        ),
    )
