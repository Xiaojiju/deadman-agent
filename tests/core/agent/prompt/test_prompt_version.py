"""Prompt 版本与 manifest 契约测试。"""

import textwrap

import pytest
from fastapi.testclient import TestClient

from app.core.agent.prompt.prompt_version import (
    SUPPORTED_PROMPT_PACK_FORMAT,
    UnsupportedPromptPackFormatError,
    load_prompt_pack_meta,
    parse_manifest_file,
)
from main import app


def test_parse_manifest_with_version(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text(
        textwrap.dedent(
            """
            prompt_version: "2.0.0"
            prompt_pack_format: 1
            app_compat_min: "0.1.0"
            scenes:
              default:
                - core/base.md
            """
        ),
        encoding="utf-8",
    )
    meta = parse_manifest_file(path)
    assert meta.prompt_version == "2.0.0"
    assert meta.prompt_pack_format == 1
    assert meta.scenes["default"] == ["core/base.md"]


def test_unsupported_pack_format(tmp_path):
    path = tmp_path / "manifest.yaml"
    path.write_text(
        "prompt_pack_format: 99\nscenes:\n  default: []\n",
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedPromptPackFormatError):
        load_prompt_pack_meta(tmp_path)


def test_load_project_manifest():
    meta = load_prompt_pack_meta()
    assert meta.prompt_version == "1.1.0"
    assert meta.prompt_pack_format == SUPPORTED_PROMPT_PACK_FORMAT
    assert "default" in meta.scenes


def test_api_prompt_meta():
    client = TestClient(app)
    r = client.get("/chat/prompt/meta")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["prompt_version"] == "1.1.0"
    assert data["supported_pack_format"] == 1
