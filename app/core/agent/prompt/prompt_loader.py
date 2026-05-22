"""从 ``app/core/agent/prompt/`` 目录加载 Markdown 片段（供 Composer 使用）。

本模块只做**磁盘 IO**，不负责场景拼接；拼接逻辑见 ``composer.py``。

示例::

    from app.core.agent.prompt.prompt_loader import load_prompt_text, PROMPT_DIR

    base = load_prompt_text("core/base.md")
    assert "{output_language}" in base
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

PROMPT_DIR: Path = Path(__file__).resolve().parent
"""prompt 包根目录（含 ``core/``、``modules/``、``manifest.yaml``）。"""

LEGACY_SYSTEM_PROMPT_BASENAME = "system_prompt.md"
"""遗留单文件提示词文件名；仅 ``load_legacy_system_prompt`` 使用。"""

_FALLBACK_SYSTEM_PROMPT = (
    "你是一名有帮助的助手。若用户未指定输出格式，请使用 Markdown 组织回复。"
)
"""遗留文件缺失时的最短内置文案。"""


def load_prompt_text(
    relative_name: str,
    *,
    encoding: str = "utf-8",
    root: Path | None = None,
) -> str:
    """读取 prompt 根目录下指定相对路径的文本。

    Args:
        relative_name: 相对路径，例如 ``core/base.md``、``modules/rag_qa.md``。
        encoding: 文件编码，默认 UTF-8。
        root: 覆盖默认 ``PROMPT_DIR``；单元测试可指向临时目录。

    Returns:
        去除首尾空白及 BOM 后的文本；**文件不存在时返回空字符串**（不抛异常）。

    Example:
        >>> from app.core.agent.prompt.prompt_loader import load_prompt_text
        >>> snippet = load_prompt_text("core/safety.md")
        >>> "安全" in snippet or snippet == ""
        True

        测试自定义根目录::

            >>> import tempfile
            >>> from pathlib import Path
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     p = Path(tmp) / "a.md"
            ...     p.write_text("hello", encoding="utf-8")
            ...     load_prompt_text("a.md", root=Path(tmp))
            'hello'
    """
    base = root or PROMPT_DIR
    path = base / relative_name
    if not path.is_file():
        log.warning("提示词文件不存在: %s", path)
        return ""
    text = path.read_text(encoding=encoding)
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    return text.strip()


def load_legacy_system_prompt() -> str:
    """加载旧版 ``system_prompt.md``（兼容 / 应急，非默认路径）。

    正常运行时应使用 ``composer.DEFAULT_COMPOSED_SYSTEM_PROMPT`` 或
    ``compose_system_prompt()``。本函数仅在排查或回退到单文件时使用。

    Returns:
        ``system_prompt.md`` 正文；缺失或为空时返回 ``_FALLBACK_SYSTEM_PROMPT``。

    Example:
        >>> from app.core.agent.prompt.prompt_loader import load_legacy_system_prompt
        >>> text = load_legacy_system_prompt()
        >>> isinstance(text, str) and len(text) > 0
        True
    """
    content = load_prompt_text(LEGACY_SYSTEM_PROMPT_BASENAME)
    if not content:
        log.warning(
            "使用内置回退系统提示词（%s 为空或不存在）",
            LEGACY_SYSTEM_PROMPT_BASENAME,
        )
        return _FALLBACK_SYSTEM_PROMPT
    return content
