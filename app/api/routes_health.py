"""健康检查路由模块
主要负责定义健康检查路由，返回应用的健康状态
"""
from typing import Literal
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.api_typing import ApiResponse, create_success_api_response

router = APIRouter()


class HealthData(BaseModel):
    """健康检查数据
    
    Attributes:
        status: 状态
        version: 版本
    """

    status: Literal["ok", "error", "warning"] = Field(description="状态")
    version: str = Field(description="版本")


@router.get("/health", response_model=ApiResponse[HealthData])
async def health() -> ApiResponse[HealthData]:
    """检查应用的健康状态
    
    Returns:
        健康检查数据
    """
    return create_success_api_response(data=HealthData(status="ok", version="0.1.0"))