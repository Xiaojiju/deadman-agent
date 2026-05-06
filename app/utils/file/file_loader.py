"""文件加载：统一结果模型、抽象加载器与 TXT / XLSX 等具体实现。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping
import pandas as pd


@dataclass(frozen=True, slots=True)
class TextChunk:
    """文本块
    便于 RAG 切片处理：一页、一段、一行表头或数据等等

    Attributes:
        text: 文本内容
        chunk_type: 文本块类型
        index: 文本块索引
        metadata: 文本块元数据

    Note:
        chunk_type 用于区分文本块类型，page 表示一页，paragraph 表示一段，row 表示一行，whole 表示整个文本
        index 用于区分文本块索引，用于后续排序
        metadata 用于存储文本块元数据，用于后续检索
    """

    text: str
    chunk_type: Literal["page", "paragraph", "row", "table_block", "whole"] = "whole"
    index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FileLoadResult:
    """文件加载结果
    各格式解析后的统一载体，Agent / 向量句可致依赖着一层

    Attributes:
        source_path: 源文件路径
        format: 逻辑格式名，如 "txt" | "xlsx" | "pdf"
        mime_type: 文件 MIME 类型
        text: 文本内容
        chunks: 全量可检索文本（可由 chunks 拼接或单独维护）
        structured: 结构化数据（可选：表格、大纲等原始结构）
        metadata: 元数据
    """

    source_path: str
    format: str
    text: str
    mime_type: str | None = None
    chunks: tuple[TextChunk, ...] = ()
    structured: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class FileLoader(ABC):
    """文件加载器抽象基类
    实现类需要实现 load_file 和 supported_suffixes 方法，分别用于加载文件和获取支持的文件后缀
    load_file 方法需要返回文件内容，supported_suffixes 方法需要返回支持的文件后缀
    """

    @abstractmethod
    def load_file(self, path: Path) -> FileLoadResult:
        """加载文件
        
        Args:
            path: 文件路径
        
        Returns:
            文件加载结果，具体实现类需要实现
        """

    @classmethod
    @abstractmethod
    def supported_suffixes(cls) -> frozenset[str]:
        """支持的文件后缀
        
        Returns:
            支持的文件后缀
        """


class TxtFileLoader(FileLoader):
    """文本文件加载器
    """

    @classmethod
    def supported_suffixes(cls) -> frozenset[str]:
        """支持的文件后缀
        
        Returns:
            支持的文件后缀
        """
        return frozenset([".txt", ".md", ".markdown"])

    def load_file(self, path: Path) -> FileLoadResult:
        """加载文本文件
        
        Args:
            path: 文件路径
        
        Returns:
            文件加载结果
        """
        raw: bytes = path.read_bytes()
        text: str = raw.decode("utf-8")
        chunk: TextChunk = TextChunk(text=text, chunk_type="whole", index=0, metadata={})
        return FileLoadResult(
            source_path=str(path),
            format="txt",
            mime_type="text/plain",
            text=text,
            chunks=(chunk,),
            structured=None,
            metadata={})


class XlsxFileLoader(FileLoader):
    """Excel 文件加载器

    不强行解析表格结构，而是把 Excel 变成 “对向量友好的文本”，再统一切块向量化。
    执行load_file，将 Excel 文件转换为文本内容和文本块列表。
    - 每行 = 一行
    - 每列用 | 分隔
    - 合并单元格尽量展开
    - 空行保留但标记
    - 不做任何业务判断（不判断合计、不判断表头）

    Attributes:
        chunk_rows: 每个文本块的行数，默认5行切分
    """

    def __init__(self, chunk_rows: int = 5) -> None:
        """初始化 Excel 文件加载器

        Args:
            chunk_rows: 每个文本块的行数，默认5行切分
        """
        self.chunk_rows = chunk_rows

    @classmethod
    def supported_suffixes(cls) -> frozenset[str]:
        """支持的文件后缀
        
        Returns:
            支持的文件后缀
        """
        return frozenset([".xlsx", ".xls"])

    def load_file(self, path: Path) -> FileLoadResult:
        """加载 Excel 文件
        仅仅可负载处理网格数据 + 首行表头，尽量不处理其他复杂数据
        
        Args:
            path: 文件路径
        
        Returns:
            文件加载结果
        """
        full_text_parts = []
        chunks = []
        chunk_index = 0

        excel_file = pd.ExcelFile(path)
        for sheet_name in excel_file.sheet_names:
            sheet_df =pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            lines = []
            for _, row in sheet_df.iterrows():
                cells = [str(cell).strip() if pd.notna(cell) else "" for cell in row]
                line = " | ".join(cells)
                if line.strip():
                    lines.append(line)
            sheet_header = f"sheet table: {sheet_name}\n"
            full_text_parts.append(sheet_header)
            full_text_parts.extend(lines)

            for i in range(0, len(lines), self.chunk_rows):
                block_lines = lines[i:i+self.chunk_rows]
                block_text = "\n".join([sheet_header] + block_lines)
                chunk = TextChunk(
                    text=block_text,
                    chunk_type="table_block",
                    index=chunk_index,
                    metadata={
                        "sheet": sheet_name,
                        "row_start": i + 1,
                        "row_end": i + len(block_lines),
                        "source": str(path)
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
        full_text = "\n".join(full_text_parts)

        return FileLoadResult(
            source_path=str(path),
            format="xlsx",
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            text=full_text,
            chunks=tuple(chunks),
            structured=None,
            metadata={
                "sheet_count": len(excel_file.sheet_names),
                "total_chunks": len(chunks)
            }
        )
