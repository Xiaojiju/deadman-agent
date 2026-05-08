"""日志配置模块"""

import logging

from app.core.config import get_settings


def configure_logging() -> None:
    """配置应用的日志记录

    Returns:
        None
    """
    settings = get_settings()
    level = settings.log_level.upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

logger: logging.Logger = logging.getLogger(__name__)
