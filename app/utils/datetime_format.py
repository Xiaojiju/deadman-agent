"""时间戳格式化"""

from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["get_timestamp"]


def get_timestamp(dt: datetime | None = None, *, milliseconds: bool = False) -> int:
    """将datetime格式的时间转换为Unix时间戳

    Args:
        dt: 时间
        milliseconds: 是否返回毫秒时间戳

    Returns:
        Unix时间戳
    """
    value = dt or datetime.now(timezone.utc)
    ts = value.timestamp()
    return int(ts * 1000) if milliseconds else int(ts)
