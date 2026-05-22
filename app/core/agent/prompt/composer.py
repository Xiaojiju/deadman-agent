"""将 Core + Module + Knobs 按 manifest 组装为一条 system 文本。

本模块负责读取 ``manifest.yaml``、按场景拼接 Markdown 片段，并将
``PromptKnobs`` 注入 ``core/base.md`` 中的占位符。组装结果可交给
``ChatPromptTemplate`` 或 ``RunnableWithMessageHistory`` 的第一条 system 消息。

示例::

    from app.core.agent.prompt import Scene, PromptKnobs, compose_system_prompt

    text = compose_system_prompt(Scene.RAG_QA, PromptKnobs(tone=Tone.CONCISE))
    # -> 单条完整 system 字符串
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

from app.core.agent.prompt.knobs import PromptKnobs, Scene
from app.core.agent.prompt.prompt_loader import PROMPT_DIR, load_prompt_text

log = logging.getLogger(__name__)

MANIFEST_BASENAME = "manifest.yaml"
"""场景清单文件名，位于 prompt 根目录。"""

_FRAGMENT_SEPARATOR = "\n\n---\n\n"
"""各 Markdown 片段之间的分隔符，便于日志与 diff 阅读。"""

_FALLBACK_CORE = (
    "你是一名有帮助的助手。若用户未指定输出格式，请使用 Markdown 组织回复。\n\n"
    "输出语言：{output_language}\n语气：{tone}\n受众：{audience}\n"
)
"""manifest 或片段缺失时使用的最短回退模板（仍支持 Knobs 占位符）。"""


def _manifest_path(root: Path) -> Path:
    """返回 ``manifest.yaml`` 的绝对路径。

    Args:
        root: prompt 根目录（通常为 ``PROMPT_DIR``）。

    Returns:
        manifest 文件的 ``Path`` 对象。

    Example:
        >>> from app.core.agent.prompt.prompt_loader import PROMPT_DIR
        >>> _manifest_path(PROMPT_DIR).name
        'manifest.yaml'
    """
    return root / MANIFEST_BASENAME


@lru_cache(maxsize=128)
def _cached_fragment(root_str: str, relative: str, mtime_ns: int) -> str:
    """按文件修改时间缓存单个 Markdown 片段。

    ``mtime_ns`` 作为缓存键的一部分：文件保存后 mtime 变化，缓存自动失效。

    Args:
        root_str: prompt 根目录的字符串形式（``str(Path)``）。
        relative: 相对根目录的路径，如 ``core/base.md``。
        mtime_ns: 片段文件的 ``st_mtime_ns``，仅用于缓存键。

    Returns:
        片段正文（已 strip）。

    Raises:
        ValueError: 文件不存在或读入为空。

    Note:
        一般通过 ``_cached_compose`` 间接调用，勿在业务代码中直接使用。
    """
    text = load_prompt_text(relative, root=Path(root_str))
    if not text:
        raise ValueError(f"提示词片段为空或不存在: {relative}")
    return text


@lru_cache(maxsize=1)
def _cached_manifest(root_str: str, mtime_ns: int) -> dict[str, list[str]]:
    """解析并缓存 ``manifest.yaml`` 中的 ``scenes`` 映射。

    Args:
        root_str: prompt 根目录的字符串形式。
        mtime_ns: manifest 文件的 ``st_mtime_ns``，修改 manifest 后缓存失效。

    Returns:
        场景名 -> 片段相对路径列表。例如 ``{"default": ["core/base.md", ...]}``。

    Raises:
        ValueError: YAML 中缺少 ``scenes`` 字段。

    Example:
        若 ``manifest.yaml`` 含 ``scenes.default: [core/base.md, ...]``，
        则返回值 ``["default"]`` 对应键下为路径字符串列表。
    """
    path = _manifest_path(Path(root_str))
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    scenes = data.get("scenes") if isinstance(data, dict) else None
    if not scenes:
        raise ValueError(f"manifest 缺少 scenes: {path}")
    return {str(k): list(v) for k, v in scenes.items()}


@lru_cache(maxsize=64)
def _cached_compose(
    root_str: str,
    scene_value: str,
    knobs_json: str,
    manifest_mtime_ns: int,
) -> str:
    """执行实际的拼接与 ``str.format``（带 LRU 缓存）。

    Args:
        root_str: prompt 根目录。
        scene_value: 场景枚举值，如 ``"rag_qa"``。
        knobs_json: ``PromptKnobs.model_dump_json()``，作为缓存键之一。
        manifest_mtime_ns: manifest 的 mtime，变更时整表重组失效。

    Returns:
        组装后的完整 system 文本。

    Raises:
        KeyError: manifest 中未定义 ``scene_value``。
        FileNotFoundError: 某片段路径在磁盘上不存在。
        KeyError: 片段中的占位符与 ``PromptKnobs.template_vars()`` 不一致。

    Example:
        内部等价逻辑::

            parts = [read("core/base.md"), read("core/safety.md"), read("modules/rag_qa.md")]
            return "\\n\\n---\\n\\n".join(parts).format(
                output_language="简体中文", tone="正式", audience="普通用户"
            )
    """
    root = Path(root_str)
    manifest = _cached_manifest(root_str, manifest_mtime_ns)
    if scene_value not in manifest:
        raise KeyError(f"manifest 未定义场景: {scene_value}")

    knobs = PromptKnobs.model_validate_json(knobs_json)
    parts: list[str] = []
    for relative in manifest[scene_value]:
        fragment_path = root / relative
        if not fragment_path.is_file():
            raise FileNotFoundError(fragment_path)
        mtime_ns = fragment_path.stat().st_mtime_ns
        parts.append(_cached_fragment(root_str, relative, mtime_ns))

    template = _FRAGMENT_SEPARATOR.join(parts)
    try:
        return template.format(**knobs.template_vars())
    except KeyError as exc:
        log.error("片段占位符与 PromptKnobs 不一致: %s", exc)
        raise


class SystemPromptComposer:
    """Core + Module + Knobs 组装器。

    根据 ``manifest.yaml`` 声明的顺序加载片段，注入旋钮变量，返回单条
    system 字符串。片段与 manifest 均带 mtime 缓存；失败时回退到
    ``_FALLBACK_CORE``。

    Example:
        >>> from app.core.agent.prompt.composer import SystemPromptComposer
        >>> from app.core.agent.prompt.knobs import Scene, PromptKnobs, Tone
        >>> composer = SystemPromptComposer()
        >>> text = composer.compose(Scene.DEFAULT, PromptKnobs(tone=Tone.FORMAL))
        >>> "输出语言" in text
        True
        >>> composer.list_scenes()
        ['default', 'rag_qa', 'customer_support']
    """

    def __init__(self, root: Path | None = None) -> None:
        """初始化组装器。

        Args:
            root: prompt 根目录；默认 ``PROMPT_DIR``（本包目录）。
                测试时可传入临时目录。

        Example:
            >>> composer = SystemPromptComposer()  # 使用项目内 prompt/
            >>> composer.root.name  # 目录名
            'prompt'
        """
        self.root = root or PROMPT_DIR

    def _manifest_mtime_ns(self) -> int:
        """读取 manifest 的纳秒级修改时间，供缓存键使用。

        Returns:
            ``manifest.yaml`` 的 ``st_mtime_ns``。
        """
        return _manifest_path(self.root).stat().st_mtime_ns

    def compose(
        self,
        scene: Scene = Scene.DEFAULT,
        knobs: PromptKnobs | None = None,
    ) -> str:
        """组装完整 system 文本。

        Args:
            scene: 业务场景，决定加载哪些 ``modules/*.md``。
            knobs: 运行时旋钮；``None`` 时使用 ``PromptKnobs()`` 默认值。

        Returns:
            可传入 ``SystemMessage(content=...)`` 或 ``RunnableWithHistory`` 的字符串。

        Example:
            >>> from app.core.agent.prompt.knobs import Scene, PromptKnobs, Tone
            >>> c = SystemPromptComposer()
            >>> policy = c.compose(
            ...     Scene.RAG_QA,
            ...     PromptKnobs(output_language="English", tone=Tone.CONCISE),
            ... )
            >>> "参考材料" in policy  # rag_qa 模块中的规则
            True
        """
        resolved_knobs = knobs or PromptKnobs()
        try:
            return _cached_compose(
                str(self.root),
                scene.value,
                resolved_knobs.model_dump_json(),
                self._manifest_mtime_ns(),
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            log.warning("模块化 prompt 组装失败，使用回退文案: %s", exc)
            return _FALLBACK_CORE.format(**resolved_knobs.template_vars())

    def list_scenes(self) -> list[str]:
        """列出 manifest 中已配置的场景 ID。

        Returns:
            场景名字符串列表，与 ``Scene`` 枚举的 ``value`` 对应。

        Example:
            >>> SystemPromptComposer().list_scenes()
            ['default', 'rag_qa', 'customer_support']
        """
        return list(_cached_manifest(str(self.root), self._manifest_mtime_ns()).keys())


_default_composer = SystemPromptComposer()
"""进程内默认组装器实例，供 ``compose_system_prompt`` 复用。"""


def compose_system_prompt(
    scene: Scene = Scene.DEFAULT,
    knobs: PromptKnobs | None = None,
) -> str:
    """使用默认组装器拼接 system 文本（模块级便捷入口）。

    Args:
        scene: 业务场景。
        knobs: 运行时旋钮；``None`` 时为默认简体中文 + 正式语气。

    Returns:
        组装后的 system 字符串。

    Example:
        >>> from app.core.agent.prompt import compose_system_prompt, Scene, PromptKnobs
        >>> text = compose_system_prompt(Scene.DEFAULT)
        >>> len(text) > 100
        True
    """
    return _default_composer.compose(scene, knobs)


# 导入时预组装默认场景，避免每次请求重复 IO（供 runnable 模块级引用）
DEFAULT_COMPOSED_SYSTEM_PROMPT: str = compose_system_prompt(Scene.DEFAULT, PromptKnobs())
"""默认场景 + 默认旋钮的 system 全文。

Example:
    >>> from app.core.agent.prompt import DEFAULT_COMPOSED_SYSTEM_PROMPT
    >>> # runnable.py 中:
    >>> # RunnableWithHistory(..., system_prompt=DEFAULT_COMPOSED_SYSTEM_PROMPT)
"""
