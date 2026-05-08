"""Markdown分割器"""

from langchain_text_splitters import MarkdownHeaderTextSplitter

from app.core.documents.base import DocumentChunk
from app.core.documents.base_helper import transform_document

class MarkdownSplitter:
    """Markdown分割器
    TODO: 暂时使用langchain的MarkdownHeaderTextSplitter进行分割，后续根据实际场景进行优化或更深入的分割器实现


    Attributes:
        headers_split_on: 分割头
    """

    def __init__(self, headers_split_on: list[tuple[str, str]]) -> None:
        """初始化Markdown分割器
        
        Args:
            headers_split_on: 分割头
        """
        self.headers_split_on = headers_split_on

    def _delagate_splitter(self) -> MarkdownHeaderTextSplitter:
        """委托分割器
        
        Returns:
            MarkdownHeaderTextSplitter: Markdown头文本分割器
        """
        return MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_split_on)

    def split_document(self, text: str) -> list[DocumentChunk]:
        """分割文档
        
        Args:
            text: 文本
        
        Returns:
            list[DocumentChunk]: 文档块列表
        """
        delegate_splitter = self._delagate_splitter()
        documents = delegate_splitter.split_text(text)
        return transform_document(documents)
