"""文档块和文档结果的基类"""
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """文档块
    
    Attributes:
        text: 文本
        metadata: 元数据，用于存储文档块的元数据
    """
    id: str | Field(default=None, coerce_numbers_to_str=True)
    """对于每个切片的唯一标识，用于区分不同的切片，默认使用uuid"""
    text: str
    """文本内容"""
    metadata: dict = Field(default_factory=dict)
    """元数据，用于存储文档块的元数据
    主要用于存储被切分后的文件或文本的摘要，方便后续的检索和分析
    """

class DocumentResult(BaseModel):
    """文档结果
    
    Attributes:
        chunks: 文档块
        source_path: 源文件路径
    """
    chunks: tuple[DocumentChunk, ...] = Field(default_factory=tuple)
    """文档块"""
    source_path: str | Field(default=None, coerce_numbers_to_str=True)
    """源文件路径
    在下游处理或其他操作的时候，尽可能可以溯源到源文件
    """

class BaseDocumentSplitter(ABC):
    """抽象文档分割器

    Attributes:
        supported_extensions: 支持的文件扩展名
        split_document: 分割文档
    """

    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200) -> None:
        """初始化文档分割器
        
        Args:
            chunk_size: 块大小，默认4000
            chunk_overlap: 块重叠大小，默认200
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
        if chunk_overlap > chunk_size:
            raise ValueError(f"chunk_overlap must be <= chunk_size, got {chunk_overlap}")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @classmethod
    @abstractmethod
    def supported_extensions(cls) -> tuple[str, ...]:
        """支持的文件扩展名
        
        Returns:
            tuple[str, ...]: 支持的文件扩展名
        """
    @abstractmethod
    def split_document(self, text: str) -> list[DocumentChunk]:
        """分割文档
        
        Args:
            text: 文本
        
        Returns:
            list[DocumentChunk]: 文档块列表
        """
