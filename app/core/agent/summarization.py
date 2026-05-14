"""摘要中间件

该中间件主要用于在对话过程中，对对话历史进行摘要，并将其添加到对话历史中。
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from app.core.agent.default_model import BasicAdapterModel
from app.core.config import get_settings

settings = get_settings()

class ConversationSummarization:
    """对话摘要
    """
    def __init__(
        self,
        *,
        model: BaseChatModel | None = None,
        prompt: ChatPromptTemplate | str | list[tuple[str, str]] | None = None,
    ) -> None:
        """初始化对话摘要
        
        Args:
            model: 模型
            prompt: 提示模板
        """
        if model is None:
            model = BasicAdapterModel.from_settings(settings)
        self.model = model
        if prompt is not None:
            if isinstance(prompt, str):
                prompt = ChatPromptTemplate.from_template(prompt)
            elif isinstance(prompt, list):
                prompt = ChatPromptTemplate.from_messages(prompt)
            self.prompt = prompt
        else:
            self.prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a helpful assistant that summarizes conversations."),
            ])

    def summarize(self, existing_summary: str, messages: list[BaseMessage]) -> str:
        """摘要对话
        
        Args:
            existing_summary: 现有的摘要
            messages: 对话历史
        Returns:
            str: 摘要
        """
        ai_response = self.model.invoke(
            self.prompt.invoke({"existing_summary": existing_summary, "messages": messages}),
            stop=["\n\n"],
        )
        return ai_response.content.strip()

    async def asummarize(self, existing_summary: str, messages: list[BaseMessage]) -> str:
        """异步摘要对话
        
        Args:
            existing_summary: 现有的摘要
            messages: 对话历史
        Returns:
            str: 摘要
        """
        return await self.summarize(existing_summary, messages)
