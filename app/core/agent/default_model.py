"""默认模型

默认模型类，用于连接到 OpenAI 兼容的第三方模型。
"""

from __future__ import annotations

import os
from typing import Any

from langchain_openai import ChatOpenAI
from typing_extensions import Self

from app.core.config import Settings

BASIC_MODEL_ENV = "BASIC_MODEL"
BASIC_MODEL_API_KEY_ENV = "BASIC_MODEL_API_KEY"
BASIC_MODEL_BASE_URL_ENV = "BASIC_MODEL_BASE_URL"


class BasicAdapterModel(ChatOpenAI):
    """基础模型适配器

    继承 ``ChatOpenAI``（LangChain 的 ``BaseChatModel`` 实现），可直接参与
    ``prompt | model`` 等 LCEL 组合，无需再包一层委托对象。
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        stream: bool = False,
        default_query: dict[str, Any] | None = None,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        """初始化基础模型适配器

        Args:
            model: 模型名称
            api_key: API密钥
            base_url: 基础 URL；不提供则走 OpenAI 客户端默认
            temperature: 温度
            max_tokens: 最大生成 token
            top_p: top_p
            frequency_penalty: 频率惩罚
            presence_penalty: 存在惩罚
            stop: 停止序列
            stream: 是否流式（映射为父类的 ``streaming``）
            default_query: 默认查询参数
            default_headers: 默认请求头
        """
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            streaming=stream,
            default_query=default_query,
            default_headers=default_headers,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """从设置中构建模型

        Args:
            settings: 设置
        Returns:
            Self: 模型
        """
        return cls(
            model=settings.base_model,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )

    @classmethod
    def from_env(cls) -> Self:
        """从环境变量中构建模型

        Returns:
            Self: 模型
        """

        def _env(*names: str, default: str | None = None) -> str | None:
            """获取环境变量并转换为指定类型

            Args:
                names: 环境变量名称
                default: 环境变量默认值
            Returns:
                str | None: 环境变量值
            """
            for name in names:
                value = os.getenv(name)
                if value:
                    return value
            return default

        return cls(
            model=_env(BASIC_MODEL_ENV),
            api_key=_env(BASIC_MODEL_API_KEY_ENV),
            base_url=_env(BASIC_MODEL_BASE_URL_ENV),
        )
