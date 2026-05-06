"""认证依赖模块
负责提供认证相关的依赖，暂时没有实现认证功能，返回匿名主体
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Principal:
    """主体
    
    Attributes:
        subject: 主体
        is_anonymous: 是否匿名
    """
    subject: str
    is_anonymous: bool = True


def get_principal() -> Principal:
    """获取主体
    
    Returns:
        主体
    """
    return Principal(subject="anonymous", is_anonymous=True)