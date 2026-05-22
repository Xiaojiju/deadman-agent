"""Few-shot 样例加载：用户样例 → 助手样例，供 ChatPromptTemplate 插入。"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from app.core.agent.prompt.knobs import Scene
from app.core.agent.prompt.prompt_loader import PROMPT_DIR

log = logging.getLogger(__name__)

FEW_SHOT_DIR = PROMPT_DIR / "few_shot"

# (scene, path) 相对 few_shot 目录
_SCENE_FILES: dict[Scene, str] = {
    Scene.DEFAULT: "default.yaml",
}


def load_few_shot_pairs(
    scene: Scene,
    *,
    root: Path | None = None,
) -> list[tuple[str, str]]:
    """加载某场景的 Few-shot 对列表 ``[(user, assistant), ...]``。

    Args:
        scene: 业务场景；当前仅 ``Scene.DEFAULT`` 有样例文件。
        root: 覆盖 ``few_shot`` 目录（测试用）。

    Returns:
        样例对；文件缺失、解析失败或 ``examples`` 为空时返回 ``[]``。

    Example:
        >>> pairs = load_few_shot_pairs(Scene.DEFAULT)
        >>> pairs[0][0]
        '什么是 REST？请简要说明。'
    """
    rel = _SCENE_FILES.get(scene)
    if not rel:
        return []

    base = root or FEW_SHOT_DIR
    path = base / rel
    if not path.is_file():
        log.warning("Few-shot 文件不存在: %s", path)
        return []

    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        log.warning("Few-shot 解析失败 %s: %s", path, exc)
        return []

    if not isinstance(data, dict):
        return []

    raw_examples = data.get("examples")
    if not isinstance(raw_examples, list):
        return []

    pairs: list[tuple[str, str]] = []
    for item in raw_examples:
        if not isinstance(item, dict):
            continue
        user = item.get("user")
        assistant = item.get("assistant")
        if isinstance(user, str) and isinstance(assistant, str):
            u, a = user.strip(), assistant.strip()
            if u and a:
                pairs.append((u, a))
    return pairs


def few_shot_message_tuples(
    scene: Scene,
    *,
    enabled: bool = True,
) -> list[tuple[str, str]]:
    """转为 ``ChatPromptTemplate.from_messages`` 可用的 ``(role, content)`` 列表。

    Args:
        scene: 场景；仅 ``default`` 且 ``enabled=True`` 时返回非空列表。
        enabled: 是否启用 Few-shot（API ``include_few_shot``）。

    Returns:
        形如 ``[("human", "..."), ("ai", "..."), ...]``。
    """
    if not enabled or scene is not Scene.DEFAULT:
        return []
    pairs = load_few_shot_pairs(scene)
    messages: list[tuple[str, str]] = []
    for user, assistant in pairs:
        messages.append(("human", user))
        messages.append(("ai", assistant))
    return messages
