"""文件加载错误"""

class FileLoadError(Exception):
    """文件加载错误"""

class UnsupportedFormatError(FileLoadError):
    """不支持的文件格式"""

class MissingFileError(FileLoadError):
    """路径存在但指向的不是可读文件（未找到或不可访问）。"""
