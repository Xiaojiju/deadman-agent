"""聊天历史磁盘存储的只读查询（分页）。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.agent.runnable import DEFAULT_HISTORY_DIR, DEFAULT_HISTORY_FILE

SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")
MESSAGES_SUFFIX = "_messages.json"
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


class HistoryStoreError(ValueError):
    """历史查询参数或数据错误。"""


def validate_session_id(session_id: str) -> str:
    """校验 session_id，防止路径穿越。"""
    sid = session_id.strip()
    if not sid or not SESSION_ID_PATTERN.match(sid):
        raise HistoryStoreError(
            "session_id 仅允许字母、数字、点、下划线、连字符，且不能以特殊符号开头"
        )
    return sid


def _history_dir() -> Path:
    path = Path(DEFAULT_HISTORY_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _message_file(session_id: str) -> Path:
    return _history_dir() / DEFAULT_HISTORY_FILE.format(session_id=session_id)


def _load_raw_message_dicts(session_id: str) -> list[dict[str, Any]]:
    """读取会话在磁盘上的原始消息列表（LangChain ``messages_to_dict`` 格式）。"""
    path = _message_file(session_id)
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise HistoryStoreError(f"历史文件 JSON 损坏: {path}") from exc
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def message_dict_to_record(index: int, item: dict[str, Any]) -> dict[str, Any]:
    """将单条 LangChain 序列化消息转为 API 友好结构。"""
    msg_type = str(item.get("type", "unknown"))
    data = item.get("data") if isinstance(item.get("data"), dict) else {}
    content = data.get("content", "")
    if not isinstance(content, str):
        content = str(content) if content is not None else ""
    role = msg_type
    if role == "ai":
        role = "assistant"
    return {
        "index": index,
        "id": data.get("id"),
        "role": role,
        "type": msg_type,
        "content": content,
    }


@dataclass(frozen=True)
class PageResult[T]:
    """分页结果。"""

    items: list[T]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        if self.total == 0:
            return 0
        return math.ceil(self.total / self.page_size)


def normalize_page(page: int, page_size: int) -> tuple[int, int]:
    """规范化页码与每页条数（page 从 1 开始）。"""
    if page < 1:
        raise HistoryStoreError("page 必须 >= 1")
    if page_size < 1:
        raise HistoryStoreError("page_size 必须 >= 1")
    if page_size > MAX_PAGE_SIZE:
        raise HistoryStoreError(f"page_size 不能超过 {MAX_PAGE_SIZE}")
    return page, page_size


def paginate[T](items: list[T], page: int, page_size: int) -> PageResult[T]:
    """对内存列表分页。"""
    page, page_size = normalize_page(page, page_size)
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    return PageResult(items=items[start:end], total=total, page=page, page_size=page_size)


def list_sessions(*, page: int = 1, page_size: int = DEFAULT_PAGE_SIZE) -> PageResult[dict[str, Any]]:
    """扫描历史目录，分页返回会话摘要。"""
    page, page_size = normalize_page(page, page_size)
    root = _history_dir()
    sessions: list[dict[str, Any]] = []
    for path in root.glob(f"*{MESSAGES_SUFFIX}"):
        session_id = path.name[: -len(MESSAGES_SUFFIX)]
        if not session_id:
            continue
        try:
            validate_session_id(session_id)
        except HistoryStoreError:
            continue
        raw = _load_raw_message_dicts(session_id)
        stat = path.stat()
        sessions.append(
            {
                "session_id": session_id,
                "message_count": len(raw),
                "updated_at_ms": int(stat.st_mtime * 1000),
            }
        )
    sessions.sort(key=lambda s: s["updated_at_ms"], reverse=True)
    return paginate(sessions, page, page_size)


def get_session_messages(
    session_id: str,
    *,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> PageResult[dict[str, Any]]:
    """分页返回指定会话的聊天消息（磁盘全量，非模型上下文截断）。"""
    sid = validate_session_id(session_id)
    path = _message_file(sid)
    if not path.is_file():
        raise FileNotFoundError(sid)

    raw_dicts = _load_raw_message_dicts(sid)
    records = [message_dict_to_record(i, item) for i, item in enumerate(raw_dicts)]
    return paginate(records, page, page_size)
