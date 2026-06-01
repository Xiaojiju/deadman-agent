"""测试 assign 链路增强：解析链路元数据。"""
from app.core.agent.lab.enrich_assign import resolve_chain_metadata
from app.core.agent.prompt import Scene


def test_metadata_scene_str():
    """测试场景字符串解析。"""
    scene, _, _ = resolve_chain_metadata({"scene": "rag_qa"})
    assert scene == Scene.RAG_QA

def test_metadata_defaults():
    """测试默认值解析。"""
    scene, knobs, _ = resolve_chain_metadata({})
    assert scene == Scene.DEFAULT
    assert knobs.output_language == "简体中文"

def test_metadata_context_key():
    """测试 context 键解析。"""
    _, _, ctx = resolve_chain_metadata({"context": "test context"})
    assert ctx == "test context"
