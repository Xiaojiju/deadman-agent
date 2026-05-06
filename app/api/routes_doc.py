"""文档路由模块
主要负责定义文档路由，返回应用的文档
"""
from fastapi import APIRouter, File, UploadFile

from app.api.api_typing import ApiResponse

router = APIRouter()

@router.post("/upload_file")
async def upload_file(file: UploadFile = File(...)) -> ApiResponse[str]:
    """上传文件
    
    Args:
        file: 文件
    """
    return {"file": file}