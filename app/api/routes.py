"""路由配置模块
主要负责将定义的路由添加到FastAPI应用中
"""

from fastapi import APIRouter
from app.api.routes_agent import router as routes_agent
from app.api.routes_chat import router as routes_chat
from app.api.routes_health import router as routes_health
from app.api.routes_history import router as routes_history
from app.api.routes_prompt import router as routes_prompt
from app.core.config import get_settings

settings = get_settings()

router = APIRouter()

router.include_router(routes_health, tags=["health"])
router.include_router(routes_chat, tags=["chat"])
router.include_router(routes_agent, tags=["chat-agent"])
router.include_router(routes_history, tags=["chat-history"])
router.include_router(routes_prompt, tags=["prompt"])
