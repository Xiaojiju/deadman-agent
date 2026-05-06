class FileLoadError(Exception):
    """文件加载错误"""


class UnsupportedFormatError(FileLoadError):
    """不支持的文件格式"""


class FileNotFoundError(FileLoadError):
    """文件未找到"""