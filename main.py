"""主模块，用于创建FastAPI应用
"""
from fastapi import FastAPI

from app.api.routes import router as api_router
from app.core.log_config import configure_logging


def create_app() -> FastAPI:
    """创建FastAPI应用
    
    Returns:
        FastAPI应用
    """
    configure_logging()
    application = FastAPI(
        title="Deadman Agent", version="0.1.0", docs_url=None, redoc_url=None)
    application.include_router(api_router)
    return application


app = create_app()
