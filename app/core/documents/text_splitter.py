"""文本文档分割器"""

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.documents.base import BaseDocumentSplitter, DocumentResult
from app.core.documents.base_helper import transform_document


class TextDocumentSplitter(BaseDocumentSplitter):
    """文本文档分割器"""

    def __init__(self,
        separators: list[str] = None,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """初始化文本文档分割器
        
        Args:
            separator: 分隔符
            is_separator_regex: 是否使用正则表达式
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex

    def _delagate_splitter(self) -> RecursiveCharacterTextSplitter:
        """委托分割器
        默认使用langchain的RecursiveCharacterTextSplitter进行分割

        Returns:
            RecursiveCharacterTextSplitter: 递归字符文本分割器
        """
        return RecursiveCharacterTextSplitter(
            separators=self._separators,
            is_separator_regex=self._is_separator_regex,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split_document(self, text: str) -> DocumentResult:
        """分割文档
        
        Args:
            text: 文本
        
        Returns:
            DocumentResult: 文档结果
        """
        splitter = self._delagate_splitter()
        documents = splitter.split_documents(text)
        return transform_document(documents)
