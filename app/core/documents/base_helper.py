"""文档转换辅助函数"""

from langchain_core.documents.base import Document

from app.core.documents.base import DocumentChunk


def transform_document(chunks: list[Document]) -> list[DocumentChunk]:
    """转换langchain的Document为DocumentResult
    
    Args:
        chunks: langchain的Document
    
    Returns:
        list[DocumentChunk]: 文档块列表
    """
    if not chunks:
        return []
    return [DocumentChunk(id=chunk.id, text=chunk.page_content) for chunk in chunks]
