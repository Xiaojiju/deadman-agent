from pathlib import Path

from app.utils.file.file_loader import FileLoadResult, XlsxFileLoader


TEST_MOCK_FILE_PATH = (
    Path(__file__).resolve().parent / "mock_file" / "0e1ad1d844a74472a3ae52d5db1a7eda.xlsx"
)

loader = XlsxFileLoader()
test_file = TEST_MOCK_FILE_PATH

def test_load_returns_result():
    """测试 load_file 返回结果"""
    res = loader.load_file(test_file)
    assert isinstance(res, FileLoadResult)

def test_load_file():
    """测试 load_file 返回结果"""
    res = loader.load_file(test_file)
    assert res is not None
    assert len(res.chunks) > 0

def test_load_full_text_not_empty():
    """测试 full_text 是否正常生成"""
    res = loader.load_file(test_file)
    assert res.text is not None
    assert len(res.text) > 0

def test_supported_suffixes():
    """测试支持的文件后缀"""
    suffixes = XlsxFileLoader.supported_suffixes()
    assert ".xlsx" in suffixes
    assert ".xls" in suffixes