"""文件加载器注册表"""

from pathlib import Path
from typing import Iterable

from app.utils.file.file_errors import MissingFileError, UnsupportedFormatError
from app.utils.file.file_loader import FileLoadResult, FileLoader, TxtFileLoader, XlsxFileLoader


__all__ = ["default_registry"]

class FileLoaderRegistry:
    """文件加载器注册表
    用于注册和管理文件加载器

    Attributes:
        _by_suffix: 后缀到加载器的映射
    """

    def __init__(self, loaders: Iterable[FileLoader] | None = None) -> None:
        """初始化文件加载器注册表
        
        Args:
            loaders: 文件加载器列表
        """
        self._by_suffix: dict[str, FileLoader] = {}
        if loaders:
            for loader in loaders:
                self.register(loader)


    def register(self, loader: type[FileLoader]) -> None:
        """注册文件加载器
        
        Args:
            loader: 文件加载器类型
        """
        instance = loader()
        for suffix in instance.supported_suffixes():
            suf_lower = suffix.lower()
            if suf_lower in self._by_suffix:
                raise ValueError(f"Loader for suffix {suf_lower} already registered")
            self._by_suffix[suf_lower] = instance


    def resolve(self, path: str | Path) -> FileLoader:
        """解析文件加载器
        
        Args:
            path: 文件路径
        
        Returns:
            FileLoader: 文件加载器
        """
        path = Path(path)
        if not path.is_file():
            raise MissingFileError(f"File {path} not found")
        suf_lower = path.suffix.lower()
        loader = self._by_suffix.get(suf_lower)
        if loader is None:
            raise UnsupportedFormatError(f"no loader for suffix {suf_lower!r}: {[path]}")
        return loader


    def load(self, path: str | Path) -> FileLoadResult:
        """加载文件
        
        Args:
            path: 文件路径
        
        Returns:
            FileLoadResult: 文件加载结果
        """
        loader: FileLoader = self.resolve(path)
        return loader.load_file(path)


default_registry = FileLoaderRegistry(
    loaders=[
        TxtFileLoader,
        XlsxFileLoader,
    ]
)
