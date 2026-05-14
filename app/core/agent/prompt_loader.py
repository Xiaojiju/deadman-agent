"""从 ``app/core/agent/prompt/`` 目录加载提示词文件（如 Markdown）。"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# 与当前模块同级的 ``prompt`` 目录
PROMPT_DIR: Path = Path(__file__).resolve().parent / "prompt"

DEFAULT_SYSTEM_PROMPT_BASENAME = "system_prompt.md"

_FALLBACK_SYSTEM_PROMPT = (
    "你是一名有帮助的助手。若用户未指定输出格式，请使用 Markdown 组织回复。"
)


def load_prompt_text(relative_name: str, *, encoding: str = "utf-8") -> str:
    """读取 ``prompt`` 目录下指定文件的文本内容。

    Args:
        relative_name: 相对 ``prompt`` 目录的路径，例如 ``system_prompt.md``。
        encoding: 文件编码。

    Returns:
        去除首尾空白后的文本；文件不存在或为空时返回空字符串。
    """
    path = PROMPT_DIR / relative_name
    if not path.is_file():
        log.warning("提示词文件不存在: %s", path)
        return ""
    text = path.read_text(encoding=encoding)
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    return text.strip()


def load_default_system_prompt() -> str:
    """加载默认系统提示词（``system_prompt.md``）。

    若文件缺失或读入为空，则返回内置简短中文回退文案。
    """
    content = load_prompt_text(DEFAULT_SYSTEM_PROMPT_BASENAME)
    if not content:
        log.warning("使用内置回退系统提示词（%s 为空或不存在）", DEFAULT_SYSTEM_PROMPT_BASENAME)
        return _FALLBACK_SYSTEM_PROMPT
    return content


# 模块导入时加载一次，供 ``runnable`` / ``stream_runnable`` 共用
DEFAULT_SYSTEM_PROMPT: str = load_default_system_prompt()
