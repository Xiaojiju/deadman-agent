"""可执行的轻量多轮对话

本文件主要构建一个轻量的多轮对话记忆可运行方式，文件中包含了主动摘要、自动历史上下文存储等功能。
 - 只需要多轮对话记忆，不需要工具调用（如客服聊天、RAG 问答）；
 - 已有普通 LCEL 链，想低成本加历史，不想重构为 Agent；
 - 追求轻量、低延迟、易调试。
"""
import json
import os
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Callable, Sequence
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from app.core.agent.default_model import BasicAdapterModel
from app.core.agent.prompt_loader import DEFAULT_SYSTEM_PROMPT
from app.core.agent.summarization import ConversationSummarization
from app.core.config import get_settings
from app.core.log_config import logger as log

# 项目根目录下的 data/history（不再使用绝对路径 /data/history）
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HISTORY_DIR = str(_REPO_ROOT / "data" / "history")
DEFAULT_HISTORY_FILE = "{session_id}_messages.json"  # 历史记录文件
DEFAULT_HISTORY_FILE_PATH = os.path.join(DEFAULT_HISTORY_DIR, DEFAULT_HISTORY_FILE)
DEFAULT_HISTORY_SUMMARY_FILE = "summary.json"
DEFAULT_HISTORY_SUMMARY_FILE_PATH = os.path.join(DEFAULT_HISTORY_DIR, DEFAULT_HISTORY_SUMMARY_FILE)
DEFAULT_SESSION_FILE = "sessions.json"
DEFAULT_SESSION_FILE_PATH = os.path.join(DEFAULT_HISTORY_DIR, DEFAULT_SESSION_FILE)

class SummaryContext:
    """摘要上下文
    
    Attributes:
        session_id: 会话ID
        summary: 摘要
        last_id: 最后一条消息的ID
    """
    session_id: str = ""
    summary: str = ""
    last_id: str = ""

class DefaultMessageHistory(BaseChatMessageHistory):
    """默认总结历史记录

    该类主要提供默认的历史记录保存方式，使用文件的json格式进行保存。在消息保存的时候，没有对格式进行整理，直接使用的
    langchain_core.messages.BaseMessage的原始格式进行保存。如果在使用的时候需要进行转换格式或进行其他处理，请
    使用其他方式进行保存或构建新的hooks去处理。
    
    主要方法：
        - messages（只读属性）: 获取历史记录
        - add_messages(): 将 LangChain 传入的本轮新消息与磁盘已有记录合并后写回
        - clear(): 清空历史记录
    该类仅仅适用于测试或进行大模型多轮对话比对结果等等场景，请不要使用该类作为历史记录保存的方式作为生产环境使用。
    
    Attributes:
        session_id: 会话ID

    """
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._target_message_path = DEFAULT_HISTORY_FILE_PATH.format(session_id=session_id)
        self._target_summary_path = DEFAULT_HISTORY_SUMMARY_FILE_PATH.format(session_id=session_id)

    def _load_stored_messages(self) -> list[BaseMessage]:
        """从磁盘读取已保存的消息（不创建空文件；供 ``messages`` 与增量 ``add_messages`` 使用）。"""
        try:
            with open(self._target_message_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not data:
                return []
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                try:
                    return messages_from_dict(data)
                except ValueError:
                    log.exception("Failed to parse history messages from JSON")
                    return []
            return []
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            log.error("Failed to decode history file: %s", self._target_message_path)
            return []

    @property
    def messages(self) -> list[BaseMessage]:
        """获取历史记录（LangChain 要求为属性，供 ``RunnableWithMessageHistory`` 等读取）。"""
        loaded = self._load_stored_messages()
        if loaded:
            return loaded
        if not Path(self._target_message_path).exists():
            log.debug("尚无历史文件，将创建空历史: %s", self._target_message_path)
            self._write_file(self._target_message_path, [])
        return []

    def _transform_messages(
        self, message: dict[str, Any] | BaseMessage
        ) -> AIMessage | HumanMessage | SystemMessage | ToolMessage:
        """转换历史记录
        
        Args:
            messages: 历史记录
        Returns:
            list[BaseMessage]: 转换后的历史记录
        """
        msg_type = message.get("type")
        if msg_type == "human":
            return HumanMessage.model_validate(message)
        if msg_type in ("ai", "assistant"):
            return AIMessage.model_validate(message)
        if msg_type == "system":
            return SystemMessage.model_validate(message)
        if msg_type == "tool":
            return ToolMessage.model_validate(message)
        # Fallback: try common chat roles
        return HumanMessage.model_validate(message)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """添加历史记录

        LangChain 的 ``RunnableWithMessageHistory`` 每次只传入**本轮新增**的消息；
        此处与磁盘已有记录合并后再整文件写入（JSON 仍是一次性重写，语义上是增量追加）。

        Args:
            messages: 本轮待追加的消息
        """
        prior = self._load_stored_messages()
        merged = prior + list(messages)
        self._write_file(self._target_message_path, messages_to_dict(merged))

    def clear(self) -> None:
        """清空历史记录
        
        """
        self._remove_file(self._target_message_path)

    def _read_file(self, path: Path) -> list[dict[str, Any]]:
        """读取文件
        
        Args:
            path: 文件路径
        Returns:
            dict[str, Any]: 文件内容
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not data:
                    return {}
                return data
        except FileNotFoundError:
            log.debug("文件不存在（将按需创建）: %s", path)
            return {}
        except json.JSONDecodeError:
            log.error("Failed to decode file: %s", path)
            return {}

    def _write_file(self, path: Path | str, data: Any) -> None:
        """写入文件
        
        Args:
            path: 文件路径
            data: 文件内容
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _remove_file(self, path: Path | str) -> None:
        """删除文件
        
        Args:
            path: 文件路径
        """
        path = Path(path)
        if path.exists():
            path.unlink()
        else:
            log.info("File not found: %s", path)

class RollingWindowChatHistory(DefaultMessageHistory):
    """滚动窗口会话历史记录"""
    def __init__(
        self,
        session_id: str,
        *,
        summarizer: ConversationSummarization,
        trigger_at: int = 20,
        keep_recent: int = 10,
    ) -> None:
        super().__init__(session_id)
        self.summarizer = summarizer
        self.trigger_at = trigger_at
        self.keep_recent = keep_recent
        self._summary: SummaryContext | dict[str, Any] | None = None
        self._recent: list[BaseMessage] = []

    def _read_summary(self) -> SummaryContext | None:
        """读取摘要
        
        Returns:
            str: 摘要
        """
        if not self._summary:
            summary_data = super()._read_file(self._target_summary_path)
            if not summary_data or not isinstance(summary_data, list):
                return None
            self._summary = next(
                (target for target in summary_data if target['session_id'] == self.session_id),
                None,
            )
        if not self._summary:
            return None
        return self._summary

    @property
    def messages(self) -> list[BaseMessage]:
        """获取会话历史记录（含摘要系统消息与按 last_id 截断后的消息）。"""
        base_messages = list(super().messages)
        summary = self._read_summary()
        if not summary:
            return base_messages
        messages = [m for m in base_messages if m.id is not None and m.id >= summary.last_id]
        messages.insert(0, SystemMessage(content=summary.summary))
        return messages

    async def _atrigger_summary(self) -> None:
        """异步触发摘要
        
        Returns:
            None
        """
        if len(self._recent) < self.trigger_at:
            return
        to_compress = self._recent[: -self.keep_recent]
        self._recent = self._recent[-self.keep_recent :]
        last_id = self._recent[-1].id
        self._summary = await self.summarizer.summarize(self._read_summary(), to_compress)
        summary_data = self._read_file(self._target_summary_path)
        summary_data.remove(lambda x: x['session_id'] == self.session_id)
        self._summary = SummaryContext(
            session_id=self.session_id,
            summary=self._summary,
            last_id=last_id,
        )
        summary_data.append(self._summary)
        self._write_file(self._target_summary_path, summary_data)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """添加会话历史记录
        
        Args:
            messages: 会话历史记录
        """
        self._recent.extend(messages)
        super().add_messages(messages)

        self._atrigger_summary()

    def clear(self) -> None:
        self._recent = []
        self._summary = ""
        super().clear()

def make_get_session_history(
    summarizer: ConversationSummarization,
    store: dict[str, RollingWindowChatHistory] | None = None,
) -> Callable[[str], BaseChatMessageHistory]:
    """构建获取会话历史记录的工厂函数。

    若传入 ``store``（通常为 ``RunnableWithHistory`` 实例上的共享字典），
    则多次构建 runnable 或同一包装器内复用工厂时，同一 ``session_id`` 仍指向同一历史对象。
    """
    _store = store if store is not None else {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in _store:
            _store[session_id] = RollingWindowChatHistory(
                session_id=session_id,
                summarizer=summarizer,
                trigger_at=20,
                keep_recent=10,
            )
        return _store[session_id]

    return get_session_history

class RunnableWithHistory:
    """可执行的轻量多轮对话
    
    Attributes:
        model: 模型
        window_size: 窗口大小
        trigger_summarized: 触发总结的次数
        callable_history: 可调用的历史记录函数
        prompt_template: 提示模板
        input_message_key: 输入消息键
        output_message_key: 输出消息键
        history_key: 历史键

    Note:
        - 只需要多轮对话记忆，不需要工具调用（如客服聊天、RAG 问答）；
        - 已有普通 LCEL 链，想低成本加历史，不想重构为 Agent；
        - 追求轻量、低延迟、易调试。
    """
    def __init__(
        self,
        model: BaseChatModel,
        *,
        callable_history: Callable[[], BaseChatMessageHistory] | None = None,
        conversation_summarizer: ConversationSummarization | None = None,
        system_prompt: str | list[str] | None = None,
        input_message_key: str = "input",
        output_message_key: str = "output",
        history_key: str = "history"
    ) -> None:
        self.model = model
        self.callable_history = callable_history
        self.conversation_summarizer = conversation_summarizer
        self.system_prompt = system_prompt
        self.input_message_key = input_message_key
        self.output_message_key = output_message_key
        self.history_key = history_key
        self._session_store: dict[str, RollingWindowChatHistory] = {}
        self.runnable = self._runnable()

    def _runnable(self) -> Runnable:
        """构建可执行的轻量多轮对话
        
        该方法主要是用于构建可执行的轻量多轮对话，构建的RunnableWithMessageHistory
        是langchain_core.runnables.RunnableWithMessageHistory的实例。
        Returns:
            Runnable: 可执行的轻量多轮对话
        """
        system_messages: list[BaseMessage] = []
        if self.system_prompt is not None:
            if isinstance(self.system_prompt, str):
                system_messages.append(SystemMessage(content=self.system_prompt))
            elif isinstance(self.system_prompt, list):
                system_messages.extend(SystemMessage.model_validate(self.system_prompt))
        else:
            system_messages.append(SystemMessage(content="You are a helpful assistant."))
        prompt = ChatPromptTemplate.from_messages(
            [
                *system_messages,
                MessagesPlaceholder(variable_name=self.history_key),
                ("human", f"{{{self.input_message_key}}}"),
            ]
        )
        prompt_model_chain = prompt | self.model

        session_history = make_get_session_history(
            self.conversation_summarizer or ConversationSummarization(),
            self._session_store,
        )
        return RunnableWithMessageHistory(
            prompt_model_chain,
            session_history,
            input_messages_key=self.input_message_key,
            output_messages_key=self.output_message_key,
            history_messages_key=self.history_key,
        )

    def invoke(
        self,
        session_id: str,
        user_input: str,
    ):
        """执行模型
        
        Args:
            session_id: 会话ID
            user_input: 用户输入文本 
        Returns:
            dict[str, Any]: 输出
        """
        user_message = HumanMessage(content=user_input)
        response = self.runnable.invoke(
            {self.input_message_key: user_message},
            config={"configurable": {"session_id": session_id}},
        )
        return response

    def invoke_stream(
        self,
        session_id: str,
        user_input: str,
    ) -> Iterator[Any]:
        """流式执行模型（供 SSE 等场景迭代 chunk）。

        Args:
            session_id: 会话ID
            user_input: 用户输入文本
        Yields:
            LangChain 链路产生的流式 chunk（多为 ``AIMessageChunk``）。
        """
        user_message = HumanMessage(content=user_input)
        yield from self.runnable.stream(
            {self.input_message_key: user_message},
            config={"configurable": {"session_id": session_id}},
        )

settings = get_settings()

# 流式执行模型
stream_runnable = RunnableWithHistory(
    model=BasicAdapterModel.from_settings(settings, stream=True),
    conversation_summarizer=ConversationSummarization(),
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    input_message_key="input",
    output_message_key="output",
    history_key="history",
)

# 非流式执行模型
runnable = RunnableWithHistory(
    model=BasicAdapterModel.from_settings(settings),
    conversation_summarizer=ConversationSummarization(),
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    input_message_key="input",
    output_message_key="output",
    history_key="history",
)
