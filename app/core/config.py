from dataclasses import dataclass
import os
from typing import TypeVar
import dotenv


__all__ = ["get_settings"]

T = TypeVar("T")

dotenv.load_dotenv()


def _get_env(name: str, cast: type[T], default: T | None = None) -> T:
    """获取环境变量并转换为指定类型
    
    Args:
        name: 环境变量名称
        cast: 环境变量类型
        default: 环境变量默认值

    Returns:
        转换后的环境变量

    Raises:
        KeyError: 环境变量未设置且没有默认值
        ValueError: 环境变量不是有效值
    """
    raw = os.getenv(name)

    if raw is None or raw == "":
        if default is not None:
            return default
        raise KeyError(f"Missing required environment variable: {name}")

    if cast is str:
        return raw

    if cast is bool:
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False 
        raise ValueError(f"Invalid boolean for env var {name}: {raw!r}")

    try:
        return cast(raw)
    except Exception as e:
        raise ValueError(f"Invalid {cast.__name__} for env var {name}: {raw!r}") from e


@dataclass(frozen=True, slots=True)
class Settings:
    """应用配置
    
    Attributes:
        embedding_db_url: 嵌入数据库URL
        base_url: 基础URL
        base_model: 基础模型
        api_key: API密钥
        log_level: 日志级别
        port: 端口
        api_prefix: API前缀
    """
    embedding_db_url: str
    base_url: str
    base_model: str
    api_key: str
    log_level: str
    # Server
    port: int
    api_prefix: str


def get_settings() -> Settings:
    """获取应用配置
    
    Returns:
        应用配置
    """
    return Settings(
        embedding_db_url=_get_env("EMBEDDING_DB_URL", str),
        base_url=_get_env("BASE_URL", str),
        base_model=_get_env("BASE_MODEL", str),
        api_key=_get_env("API_KEY", str),
        log_level=_get_env("LOG_LEVEL", str),
        port=_get_env("PORT", int),
        api_prefix=_get_env("API_PREFIX", str),
    )