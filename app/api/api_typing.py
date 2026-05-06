from __future__ import annotations

from typing import Generic, TypeVar
from pydantic import BaseModel, Field
from app.utils.datetime_format import get_timestamp

T = TypeVar("T", bound=BaseModel | dict | list | str | int | None)

DEFAULT_SUCCESS_CODE = 0
DEFAULT_SUCCESS_MESSAGE = "Success"


class ApiResponse(BaseModel, Generic[T]):
    """API 统一响应模型
    
    Attributes:
        code: 业务响应码（0 表示成功）
        message: 响应描述信息
        timestamp: 响应时间戳（毫秒级）
        data: 响应数据体
    """
    code: int = Field(description="The code of the response.", default=DEFAULT_SUCCESS_CODE)
    message: str = Field(description="The message of the response.", default=DEFAULT_SUCCESS_MESSAGE)
    timestamp: int = Field(
        description="The timestamp of the response.",
        default_factory=lambda: get_timestamp(milliseconds=True)
    )
    data: T | None = Field(default=None, description="The data of the response.")


def create_api_response(
    code: int,
    message: str,
    data: T | None = None,
    timestamp: int | None = None,
) -> ApiResponse[T]:
    """创建自定义 API 响应
    
    Args:
        code: 业务响应码
        message: 响应描述信息
        data: 响应数据体
        timestamp: 自定义时间戳（不传则自动生成当前时间）

    Returns:
        结构化的 API 响应对象
    """
    return ApiResponse(
        code=code,
        message=message,
        timestamp=timestamp or get_timestamp(milliseconds=True),
        data=data,
    )


def create_error_api_response(code: int, message: str, timestamp: int | None = None) -> ApiResponse[None]:
    """创建错误类型的 API 响应（data 固定为 None）
    
    Args:
        code: 业务错误码
        message: 错误描述信息
        timestamp: 自定义时间戳（不传则自动生成当前时间）

    Returns:
        结构化的错误响应对象
    """
    return create_api_response(code=code, message=message, data=None, timestamp=timestamp)


def create_success_api_response(data: T | None = None, timestamp: int | None = None) -> ApiResponse[T]:
    """创建成功类型的 API 响应（使用默认成功码和消息）
    
    Args:
        data: 响应数据体
        timestamp: 自定义时间戳（不传则自动生成当前时间）

    Returns:
        结构化的成功响应对象
    """
    return create_api_response(
        code=DEFAULT_SUCCESS_CODE,
        message=DEFAULT_SUCCESS_MESSAGE,
        data=data,
        timestamp=timestamp
    )