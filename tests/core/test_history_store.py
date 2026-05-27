"""聊天历史分页查询测试。"""

import json

import pytest
from fastapi.testclient import TestClient

from app.core.agent.history_store import (
    get_session_messages,
    list_sessions,
    message_dict_to_record,
)
from app.core.agent.runnable import DEFAULT_HISTORY_DIR, DEFAULT_HISTORY_FILE
from main import app


@pytest.fixture
def sample_session(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    monkeypatch.setattr(
        "app.core.agent.history_store.DEFAULT_HISTORY_DIR",
        str(history_dir),
    )
    monkeypatch.setattr(
        "app.core.agent.runnable.DEFAULT_HISTORY_DIR",
        str(history_dir),
    )
    sid = "test-session-1"
    path = history_dir / DEFAULT_HISTORY_FILE.format(session_id=sid)
    messages = [
        {
            "type": "human",
            "data": {"content": f"msg-{i}", "type": "human", "id": None},
        }
        if i % 2 == 0
        else {
            "type": "ai",
            "data": {"content": f"reply-{i}", "type": "ai", "id": None},
        }
        for i in range(5)
    ]
    path.write_text(json.dumps(messages, ensure_ascii=False), encoding="utf-8")
    return sid


def test_paginate_messages(sample_session):
    page1 = get_session_messages(sample_session, page=1, page_size=2)
    assert page1.total == 5
    assert page1.total_pages == 3
    assert len(page1.items) == 2
    assert page1.items[0]["content"] == "msg-0"

    page3 = get_session_messages(sample_session, page=3, page_size=2)
    assert len(page3.items) == 1


def test_list_sessions(sample_session):
    result = list_sessions(page=1, page_size=10)
    assert result.total >= 1
    assert any(s["session_id"] == sample_session for s in result.items)


def test_api_list_messages(sample_session):
    client = TestClient(app)
    r = client.get(
        f"/chat/sessions/{sample_session}/messages",
        params={"page": 1, "page_size": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["code"] == 0
    assert body["data"]["pagination"]["total"] == 5
    assert len(body["data"]["items"]) == 2


def test_api_session_not_found():
    client = TestClient(app)
    r = client.get("/chat/sessions/nonexistent-id-xyz/messages")
    assert r.status_code == 404


def test_message_dict_to_record():
    item = {"type": "human", "data": {"content": "hi", "id": "abc"}}
    rec = message_dict_to_record(0, item)
    assert rec["role"] == "human"
    assert rec["content"] == "hi"
    assert rec["id"] == "abc"
