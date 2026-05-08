"""分割器管理器"""

from pathlib import Path
from app.core.log_config import logger as log

from app.core.documents.base import BaseDocumentSplitter, DocumentResult
from app.core.documents.text_splitter import TextDocumentSplitter

class SplitterManager:
    """分割器管理器"""

    def __init__(self, splitters: list[BaseDocumentSplitter]) -> None:
        """初始化分割器管理器
        
        Args:
            splitters: 分割器列表
        """
        self._splitters: dict[str, BaseDocumentSplitter] = {}
        for splitter in splitters:
            for extension in splitter.supported_extensions():
                self._splitters[extension] = splitter

    def _get_splitter(self, extension: str) -> BaseDocumentSplitter:
        """获取分割器
        
        Args:
            extension: 文件扩展名
        """
        return self._splitters.get(extension)

    def register_splitter(self, splitter: BaseDocumentSplitter) -> None:
        """注册分割器
        
        Args:
            splitter: 分割器
        """
        extensions = splitter.supported_extensions()
        len_extensions = len(extensions)
        if len_extensions == 0:
            raise ValueError(f"Splitter {splitter} has no supported extensions")

        for extension in extensions:
            if extension in self._splitters:
                log.warning("Splitter for extension %s already registered", extension)
                continue
            self._splitters[extension] = splitter

    def cleanup(self) -> None:
        """清理分割器
        
        """
        self._splitters.clear()

    def remove_splitter(self, supported_extensions: tuple[str, ...]) -> None:
        """删除分割器
        
        Args:
            supported_extensions: 支持的文件扩展名
        """
        for extension in supported_extensions:
            if extension in self._splitters:
                del self._splitters[extension]

    def split_document(self, path: str) -> DocumentResult:
        """分割文档
        
        Args:
            path: 文件路径
        """
        extension = Path(path).suffix.lower()
        splitter = self._get_splitter(extension)
        if splitter is None:
            raise ValueError(f"No splitter found for extension {extension}")
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()
        chunks = splitter.split_document(text)
        return DocumentResult(chunks=tuple(chunks), source_path=str(path))


default_splitter_manager = SplitterManager(
    splitters=[
        TextDocumentSplitter
    ]
)
