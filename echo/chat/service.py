from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable
from uuid import uuid4

from ..skills import list_available_skills
from mcp_server.client import ToolClient
from ..workflow.prompts import default_system_prompt
from .chat_model import BaseChatModel, Message
from .context_manager import DEFAULT_MAX_CONTEXT_MESSAGES, Messages, Sessions, default_session_title
from .registry import ChatModelSettings, build_chat_model

if TYPE_CHECKING:
    from ..workflow.service import WorkflowService


def infer_session_title(message: str) -> str:
    """Turn the first user message into a short session title."""
    cleaned = " ".join(message.strip().split())
    if not cleaned:
        return default_session_title()
    return cleaned[:48]


@dataclass
class SessionState:
    """Return the current session summary plus message history."""

    session: dict[str, Any]
    messages: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChatResult(SessionState):
    """Return one chat mutation plus the newest reply payload."""

    reply: str
    token_usage: dict[str, Any] | None = None
    workflow: dict[str, Any] | None = None


class ChatService:
    """Coordinate sessions, messages, and workflow-backed chat generation."""

    def __init__(
        self,
        model_factory: Callable[[ChatModelSettings | None], BaseChatModel] = build_chat_model,
        *,
        storage: dict[str, dict[str, Any]] | None = None,
        base_dir: str | Path | None = None,
        tool_client_factory: Callable[[], AbstractAsyncContextManager[ToolClient]] | None = None,
        workflow_factory: Callable[[], WorkflowService] | None = None,
        max_context_messages: int | None = None,
        preserve_system_messages: bool = True,
    ):
        self.model_factory = model_factory
        self.storage = storage
        self.base_dir = base_dir
        self.tool_client_factory = tool_client_factory
        self.workflow_factory = workflow_factory
        self.max_context_messages = DEFAULT_MAX_CONTEXT_MESSAGES if max_context_messages is None else max_context_messages
        self.preserve_system_messages = preserve_system_messages

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions, _ = self._chat("placeholder")
        return sessions.list()

    def create_session(self, session_id: str | None = None, title: str | None = None) -> dict[str, Any]:
        cleaned_title = (title or "").strip()
        if session_id is None and (not cleaned_title or cleaned_title == default_session_title()):
            blank_sessions = [session for session in self.list_sessions() if self._is_blank_new_session(session)]
            if blank_sessions:
                keeper = blank_sessions[0]
                for duplicate in blank_sessions[1:]:
                    self._chat(duplicate["session_id"])[0].delete()

                sessions, messages = self._chat(keeper["session_id"])
                sessions.ensure(title=cleaned_title or default_session_title())
                self._ensure_default_system(sessions, messages)
                return sessions.summary()

        resolved_session_id = session_id or str(uuid4())
        sessions, messages = self._chat(resolved_session_id)
        sessions.ensure(title=cleaned_title or default_session_title())
        self._ensure_default_system(sessions, messages)
        return sessions.summary()

    def delete_session(self, session_id: str):
        sessions, _ = self._chat(session_id)
        sessions.delete()

    def get_session_state(self, session_id: str) -> SessionState:
        sessions, messages = self._chat(session_id)
        sessions.ensure()
        self._ensure_default_system(sessions, messages)
        return SessionState(session=sessions.summary(), messages=messages.history())

    async def stream_message(
        self,
        message: str,
        session_id: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream one assistant reply event by event."""
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValueError("Message cannot be empty.")

        sessions, messages = self._chat(session_id)
        sessions.ensure()
        self._ensure_default_system(sessions, messages)
        previous_first_user = self._first_user_message(messages)

        if system_prompt is not None:
            await messages.apply("system_prompt", content=system_prompt)

        user_message = messages.append("user", cleaned_message, message_type="user")

        workflow_result: dict[str, Any] | None = None
        workflow_snapshot: dict[str, Any] | None = None

        try:
            async for item in self._workflow().stream_chat(
                cleaned_message,
                context=messages.build_context(),
                workflow_turn_id=user_message.id,
            ):
                if item["event"] == "chunk":
                    yield item
                    continue
                if item["event"] == "state":
                    workflow_snapshot = item["data"]
                    yield {"event": "workflow", "data": workflow_snapshot}
                    continue
                if item["event"] == "record":
                    yield item
                    continue
                workflow_result = item["data"]
                workflow_snapshot = workflow_result["snapshot"]
        except Exception:
            await self.delete_message(session_id, user_message.id)
            raise

        if workflow_result is None or workflow_snapshot is None:
            await self.delete_message(session_id, user_message.id)
            raise ValueError("Workflow stream ended without a final result.")

        self._append_workflow_records(messages, workflow_result["records"], workflow_result.get("token_usage"))
        reply = self._reply(workflow_snapshot["answer"])
        self._sync_inferred_title(sessions, messages, previous_first_user)

        result = ChatResult(
            session=sessions.summary(),
            messages=messages.history(),
            reply=reply,
            token_usage=workflow_result.get("token_usage"),
            workflow=workflow_snapshot,
        )
        yield {"event": "done", "data": result.to_dict()}

    async def update_system_prompt(self, session_id: str, content: str | None) -> SessionState:
        sessions, messages = self._chat(session_id)
        sessions.ensure()
        self._ensure_default_system(sessions, messages)
        await messages.apply("system_prompt", content=content)
        return SessionState(session=sessions.summary(), messages=messages.history())

    async def update_message(self, session_id: str, message_id: str, content: str) -> SessionState:
        sessions, messages = self._chat(session_id)
        sessions.ensure()
        self._ensure_default_system(sessions, messages)
        previous_first_user = self._first_user_message(messages)
        await messages.apply("edit", message_id=message_id, content=content)
        self._sync_inferred_title(sessions, messages, previous_first_user)
        return SessionState(session=sessions.summary(), messages=messages.history())

    async def delete_message(self, session_id: str, message_id: str) -> SessionState:
        sessions, messages = self._chat(session_id)
        sessions.ensure()
        self._ensure_default_system(sessions, messages)
        previous_first_user = self._first_user_message(messages)
        await messages.apply("delete", message_id=message_id)
        self._sync_inferred_title(sessions, messages, previous_first_user)
        return SessionState(session=sessions.summary(), messages=messages.history())

    async def rollback_message(self, session_id: str, message_id: str) -> SessionState:
        sessions, messages = self._chat(session_id)
        sessions.ensure()
        self._ensure_default_system(sessions, messages)
        previous_first_user = self._first_user_message(messages)
        await messages.apply("rollback", message_id=message_id)
        self._sync_inferred_title(sessions, messages, previous_first_user)
        return SessionState(session=sessions.summary(), messages=messages.history())

    async def stream_regenerate_message(
        self,
        session_id: str,
        message_id: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a regenerated assistant reply."""
        sessions, messages = self._chat(session_id)
        sessions.ensure()
        self._ensure_default_system(sessions, messages)
        previous_first_user = self._first_user_message(messages)
        question, user_message_id = self._prepare_regeneration(messages, message_id)
        workflow_result: dict[str, Any] | None = None
        workflow_snapshot: dict[str, Any] | None = None

        async for item in self._workflow().stream_chat(
            question,
            context=messages.build_context(),
            workflow_turn_id=user_message_id,
        ):
            if item["event"] == "chunk":
                yield item
                continue
            if item["event"] == "state":
                workflow_snapshot = item["data"]
                yield {"event": "workflow", "data": workflow_snapshot}
                continue
            if item["event"] == "record":
                yield item
                continue
            workflow_result = item["data"]
            workflow_snapshot = workflow_result["snapshot"]

        if workflow_result is None or workflow_snapshot is None:
            raise ValueError("Workflow stream ended without a final result.")

        self._append_workflow_records(messages, workflow_result["records"], workflow_result.get("token_usage"))
        reply = self._reply(workflow_snapshot["answer"])
        self._sync_inferred_title(sessions, messages, previous_first_user)

        result = ChatResult(
            session=sessions.summary(),
            messages=messages.history(),
            reply=reply,
            token_usage=workflow_result.get("token_usage"),
            workflow=workflow_snapshot,
        )
        yield {"event": "done", "data": result.to_dict()}

    def update_session_title(self, session_id: str, title: str) -> dict[str, Any]:
        cleaned = title.strip()
        if not cleaned:
            raise ValueError("Session title cannot be empty.")
        sessions, messages = self._chat(session_id)
        sessions.ensure()
        self._ensure_default_system(sessions, messages)
        sessions.set_title(cleaned)
        return sessions.summary()

    def _chat(self, session_id: str) -> tuple[Sessions, Messages]:
        sessions = Sessions(session_id=session_id, storage=self.storage, base_dir=self.base_dir)
        messages = Messages(
            sessions=sessions,
            max_context_messages=self.max_context_messages,
            preserve_system_messages=self.preserve_system_messages,
            default_system_prompt=self.default_system_prompt(),
        )
        return sessions, messages

    def _workflow(self) -> WorkflowService:
        from ..workflow.service import WorkflowService

        if self.workflow_factory is not None:
            return self.workflow_factory()
        return WorkflowService(model_factory=self.model_factory, tool_client_factory=self.tool_client_factory)

    @staticmethod
    def _append_workflow_records(messages: Messages, records: list[dict[str, Any]], total_usage: dict[str, Any] | None = None):
        last_added_assistant_msg_id = None
        for record in records:
            msg = messages.append(
                str(record.get("role") or "assistant"),
                str(record.get("content") or ""),
                message_type=str(record.get("message_type") or "") or None,
                workflow_turn_id=str(record.get("workflow_turn_id") or "") or None,
                tool_name=str(record.get("tool_name") or "") or None,
                tool_call_id=str(record.get("tool_call_id") or "") or None,
                tool_calls=record.get("tool_calls") if isinstance(record.get("tool_calls"), list) else None,
                attachments=record.get("attachments") if isinstance(record.get("attachments"), list) else None,
                token_usage=record.get("token_usage") if isinstance(record.get("token_usage"), dict) else None,
            )
            if msg.role == "assistant":
                last_added_assistant_msg_id = msg.id

        if total_usage and last_added_assistant_msg_id:
            # We must apply the total token usage to the last assistant message (e.g. "plan" or "think").
            # The ContextManager only permits updating content/system_prompt, not token_usage directly.
            # So we will fetch the raw session dict, update it, and persist it.
            session = messages.sessions.get()
            for msg in reversed(session["messages"]):
                if getattr(msg, "id", msg.get("id") if isinstance(msg, dict) else None) == last_added_assistant_msg_id:
                    if isinstance(msg, dict):
                        msg["token_usage"] = total_usage
                    else:
                        msg.token_usage = total_usage
                    break
            messages.sessions.persist(session)

    def default_system_prompt(self) -> str:
        """Render the default session-level system prompt."""
        return default_system_prompt(
            available_skills=list_available_skills(),
        )

    @staticmethod
    def _ensure_default_system(sessions: Sessions, messages: Messages):
        """Persist the default system prompt when a session is missing one."""
        session = sessions.get()
        system_messages = [message for message in session["messages"] if message.role == "system"]
        if len(system_messages) != 1 or (session["messages"] and session["messages"][0].role != "system"):
            current = system_messages[0].content if system_messages else None
            messages.ensure_system_prompt(current)
        elif ChatService._is_blank_new_session(session):
            # Force update empty sessions to the latest system prompt
            messages.ensure_system_prompt(None)

    @staticmethod
    def _is_blank_new_session(session: dict[str, Any]) -> bool:
        return (
            session.get("title") == default_session_title()
            and session.get("message_count") == 0
            and session.get("total_tokens", 0) == 0
            and not session.get("preview")
        )

    @staticmethod
    def _first_user_message(messages: Messages) -> Message | None:
        return next((message for message in messages.get() if message.role == "user"), None)

    @staticmethod
    def _reply(content: str) -> str:
        reply = content.strip()
        if not reply:
            raise ValueError("Model reply cannot be empty.")
        return reply

    def _prepare_regeneration(self, messages: Messages, message_id: str) -> tuple[str, str]:
        """Trim one branch and return the user query plus user message id that should be rerun."""
        current_messages = messages.get()
        index = messages.find_index(message_id, current_messages)
        target = current_messages[index]
        if target.role == "system":
            raise ValueError("System messages cannot be regenerated.")
        if target.role == "assistant":
            index = next((i for i in range(index - 1, -1, -1) if current_messages[i].role == "user"), None)
            if index is None:
                raise ValueError("Assistant message does not have a preceding user message.")

        session = messages.sessions.get()
        session["messages"] = session["messages"][: index + 1]
        messages.sessions.persist(session)
        return current_messages[index].content, current_messages[index].id

    def _sync_inferred_title(
        self,
        sessions: Sessions,
        messages: Messages,
        previous_first_user: Message | None,
    ):
        current_title = sessions.summary()["title"]
        tracked_titles = {default_session_title()}
        if previous_first_user is not None:
            tracked_titles.add(infer_session_title(previous_first_user.content))

        if current_title not in tracked_titles:
            return

        current_first_user = self._first_user_message(messages)
        inferred_title = (
            infer_session_title(current_first_user.content)
            if current_first_user is not None
            else default_session_title()
        )
        if current_title != inferred_title:
            sessions.set_title(inferred_title)
