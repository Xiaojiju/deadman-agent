"""Few-shot 加载与模板插入测试。"""

from langchain_core.prompts import ChatPromptTemplate

from app.core.agent.prompt.few_shot_loader import (
    few_shot_message_tuples,
    load_few_shot_pairs,
)
from app.core.agent.prompt.knobs import Scene
from app.core.agent.runnable import RunnableWithHistory
from app.core.agent.default_model import BasicAdapterModel
from app.core.config import get_settings


def test_load_default_pairs():
    """测试加载 default 场景的 Few-shot 样例对。"""
    pairs = load_few_shot_pairs(Scene.DEFAULT)
    assert len(pairs) == 4
    assert "缓存" in pairs[0][0]
    assert "DNS" in pairs[3][0]


def test_disabled_or_non_default_scene():
    """测试禁用或非 default 场景的 Few-shot 样例对。"""
    assert not few_shot_message_tuples(Scene.DEFAULT, enabled=False)
    assert not few_shot_message_tuples(Scene.RAG_QA, enabled=True)


def test_message_tuples_shape():
    """测试 Few-shot 样例对转换为消息列表的形状。"""
    msgs = few_shot_message_tuples(Scene.DEFAULT, enabled=True)
    assert len(msgs) == 8  # 4 组 × (human + ai)
    assert msgs[0] == ("human", msgs[0][1])
    assert msgs[1][0] == "ai"


def test_build_chat_prompt_includes_few_shot():
    """测试构建包含 Few-shot 样例对的 ChatPromptTemplate。"""
    settings = get_settings()
    runner = RunnableWithHistory(
        model=BasicAdapterModel.from_settings(settings),
        default_scene=Scene.DEFAULT,
    )
    prompt = runner._build_chat_prompt(Scene.DEFAULT, include_few_shot=True)
    assert isinstance(prompt, ChatPromptTemplate)
    # system, system, human, ai, human, ai, history placeholder, human
    assert len(prompt.messages) >= 7
