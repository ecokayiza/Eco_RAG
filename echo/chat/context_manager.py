from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from inspect import isawaitable
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from ..settings import Config
from ..workflow_sections import (
    parse_workflow_sections,
    render_workflow_sections,
    workflow_section_entries,
)
from .chat_model import Message

USAGE_KEYS = ("prompt_tokens", "prompt_cache_hit_tokens", "completion_tokens", "total_tokens")
READ_ONLY_MESSAGE_TYPES = {"plan", "think", "tool"}
WORKFLOW_ANSWER_PROXY_TYPES = {"plan", "think", "answer"}
DEFAULT_MAX_CONTEXT_MESSAGES = 100


def utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def default_session_title() -> str:
    """Return the default title for a new session."""
    return "New Session"


MessageResponseFactory = Callable[
    [list[dict[str, Any]]],
    Awaitable[tuple[str, dict[str, Any] | None]] | tuple[str, dict[str, Any] | None],
]


class Sessions:
    """Read, write, and summarize chat sessions."""

    def __init__(
        self,
        session_id: str,
        *,
        storage: dict[str, dict[str, Any]] | None = None,
        base_dir: str | Path | None = None,
    ):
        self.session_id = session_id
        self.storage = storage
        self.base_dir = Path(base_dir or Config.CHAT_MEMORY_DIR)
        if self.storage is None:
            self.base_dir.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.session_id in self.storage if self.storage is not None else self._file(self.session_id).exists()

    def get(self) -> dict[str, Any]:
        if self.storage is not None:
            return self._session(self.storage.get(self.session_id))

        path = self._file(self.session_id)
        return self._session(json.loads(path.read_text(encoding="utf-8")) if path.exists() else None)

    def ensure(self, title: str | None = None) -> dict[str, Any]:
        if not self.exists():
            return self.persist(self._empty(title))

        session = self.get()
        title = (title or "").strip()
        if title and session["title"] == default_session_title():
            session["title"] = title
            return self.persist(session)
        return session

    def summary(self) -> dict[str, Any]:
        return self._summary(self.get())

    def list(self) -> list[dict[str, Any]]:
        if self.storage is not None:
            sessions = [self._summary(item) for item in self.storage.values()]
        else:
            sessions = [
                self._summary(json.loads(path.read_text(encoding="utf-8")))
                for path in self.base_dir.glob("*.json")
                if not path.name.startswith(".")
            ]
        return sorted(sessions, key=lambda item: item["updated_at"], reverse=True)

    def set_title(self, title: str):
        session = self.get()
        session["title"] = title.strip() or default_session_title()
        self.persist(session)

    def persist(self, session: dict[str, Any]) -> dict[str, Any]:
        session = self._session(session)
        session["updated_at"] = utc_now()
        session["usage"] = self._usage(session["messages"])
        if self.storage is not None:
            self.storage[self.session_id] = session
        else:
            self._file(self.session_id).write_text(
                json.dumps(
                    {
                        "session_id": session["session_id"],
                        "title": session["title"],
                        "created_at": session["created_at"],
                        "updated_at": session["updated_at"],
                        "usage": session["usage"],
                        "messages": [self._dump_message(message) for message in session["messages"]],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        return self.get()

    def delete(self):
        if self.storage is not None:
            self.storage.pop(self.session_id, None)
            return

        path = self._file(self.session_id)
        if path.exists():
            path.unlink()

    def _empty(self, title: str | None = None) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": (title or "").strip() or default_session_title(),
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "usage": {},
            "messages": [],
        }

    def _session(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return self._empty()
        return {
            "session_id": payload.get("session_id", self.session_id),
            "title": payload.get("title") or default_session_title(),
            "created_at": payload.get("created_at") or utc_now(),
            "updated_at": payload.get("updated_at") or utc_now(),
            "usage": self._usage_dict(payload.get("usage")),
            "messages": [
                message.model_copy(deep=True)
                if isinstance(message, Message)
                else Message(**{key: value for key, value in message.items() if key != "workflow"})
                for message in payload.get("messages", [])
            ],
        }

    def _summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = self._session(payload)
        
        # Prefer showing the latest assistant's answer for the preview
        preview = ""
        for message in reversed(session["messages"]):
            if message.role == "assistant" and message.content.strip():
                content = message.content.strip()
                preview = (_workflow_sections(content).get("answer") or content)[:80]
                break
        
        # Fallback to the latest non-system message if no assistant answer is found
        if not preview:
             for message in reversed(session["messages"]):
                 if message.role != "system" and message.content.strip():
                     preview = message.content.strip()[:80]
                     break
                     
        usage = dict(session["usage"])
        total = usage.get("total_tokens", (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0))
        return {
            "session_id": session["session_id"],
            "title": session["title"],
            "created_at": session["created_at"],
            "updated_at": session["updated_at"],
            "message_count": len([message for message in session["messages"] if message.role != "system"]),
            "preview": preview,
            "token_usage": usage,
            "total_tokens": int(total),
        }

    def _file(self, session_id: str) -> Path:
        prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id).strip("._-") or "session"
        digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:10]
        return self.base_dir / f"{prefix}-{digest}.json"

    @staticmethod
    def _dump_message(message: Message) -> dict[str, Any]:
        payload = {"id": message.id, "role": message.role, "content": message.content}
        if message.message_type:
            payload["message_type"] = message.message_type
        if message.workflow_turn_id:
            payload["workflow_turn_id"] = message.workflow_turn_id
        if message.tool_name:
            payload["tool_name"] = message.tool_name
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = message.tool_calls
        if message.attachments:
            payload["attachments"] = message.attachments
        usage = Sessions._usage_dict(message.token_usage)
        if usage:
            payload["token_usage"] = usage
        return payload

    @staticmethod
    def _usage_dict(token_usage: dict[str, Any] | None) -> dict[str, int | float]:
        if not isinstance(token_usage, dict):
            return {}
        return {
            key: value
            for key in USAGE_KEYS
            if isinstance((value := token_usage.get(key)), (int, float)) and not isinstance(value, bool)
        }

    @staticmethod
    def _usage(messages: list[Message]) -> dict[str, int | float]:
        usage: dict[str, int | float] = {}
        for message in messages:
            for key, value in Sessions._usage_dict(message.token_usage).items():
                usage[key] = usage.get(key, 0) + value
        return usage


class Messages:
    """Manage message history and in-chat operations for one session."""

    def __init__(
        self,
        sessions: Sessions,
        *,
        max_context_messages: int | None = None,
        preserve_system_messages: bool = True,
        default_system_prompt: str | None = None,
    ):
        self.sessions = sessions
        self.max_context_messages = DEFAULT_MAX_CONTEXT_MESSAGES if max_context_messages is None else max_context_messages
        self.preserve_system_messages = preserve_system_messages
        self.default_system_prompt = (default_system_prompt or "").strip()

    def get(self) -> list[Message]:
        return self.sessions.get()["messages"]

    def history(self) -> list[dict[str, Any]]:
        return [message.model_dump(exclude_none=True) for message in self.get()]

    def ensure_system_prompt(self, content: str | None = None) -> Message:
        """Ensure the session has exactly one system prompt at index 0."""
        session = self.sessions.get()
        content = (content or self.default_system_prompt).strip()
        if not content:
            raise ValueError("System prompt cannot be empty.")
        session = self._set_system(session, content)
        session = self.sessions.persist(session)
        return session["messages"][0].model_copy(deep=True)

    def build_context(self, messages: list[Message] | None = None) -> list[dict[str, Any]]:
        """Build the next-turn context from persisted long-term memory only."""
        if self.max_context_messages < 0:
            raise ValueError("max_context_messages must be non-negative.")

        messages = messages or self.get()
        system = [message.model_copy(deep=True) for message in messages if message.role == "system"][:1]
        recent = self._compact_recent(messages)
        recent = [] if self.max_context_messages == 0 else recent[-self.max_context_messages :]
        return [message.to_llm_message() for message in ([*system, *recent] if self.preserve_system_messages else recent)]

    def append(
        self,
        role: str,
        content: str,
        *,
        message_type: str | None = None,
        workflow_turn_id: str | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        attachments: list[dict[str, Any]] | None = None,
        token_usage: dict[str, Any] | None = None,
    ) -> Message:
        session = self.sessions.get()
        message = Message(
            role=role,
            content=self._message_content(role, content, tool_calls),
            message_type=message_type,
            workflow_turn_id=workflow_turn_id,
            tool_name=tool_name,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls,
            attachments=attachments,
            token_usage=token_usage,
        )
        session["messages"].append(message)
        self.sessions.persist(session)
        return message.model_copy(deep=True)

    def replace(self, messages: Iterable[Message | dict[str, Any]]) -> dict[str, Any]:
        session = self.sessions.get()
        session["messages"] = [message.model_copy(deep=True) if isinstance(message, Message) else Message(**message) for message in messages]
        return self.sessions.persist(session)

    def clear(self) -> dict[str, Any]:
        session = self.sessions.get()
        session["messages"] = []
        return self.sessions.persist(session)

    async def apply(
        self,
        operation: str,
        *,
        content: str | None = None,
        message_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        response_factory: MessageResponseFactory | None = None,
    ) -> dict[str, Any]:
        operation = operation.strip().lower()
        if operation not in {"send", "edit", "delete", "rollback", "regenerate", "system_prompt"}:
            raise ValueError(f"Unsupported message operation: {operation}")

        session = self.sessions.ensure()

        if operation == "system_prompt":
            session = self._set_system(session, content or self.default_system_prompt)
            session = self.sessions.persist(session)
            return self._result(session, affected_message_id=self.system_message_id(session))

        if operation == "send":
            session["messages"].append(
                Message(role="user", content=self._text(content, "Message content cannot be empty."), tool_calls=tool_calls)
            )
            return await self._generate(session, response_factory, operation)

        if message_id is None:
            raise ValueError(f"{operation} requires a message_id.")

        index = self.find_index(message_id, session["messages"])
        target = session["messages"][index]
        workflow_answer_proxy = _is_workflow_answer_proxy(target)
        if not workflow_answer_proxy or operation not in {"edit", "delete", "rollback"}:
            self._ensure_mutable(target, operation)

        if operation == "edit":
            if target.role == "system":
                session = self._set_system(session, content or self.default_system_prompt)
                session = self.sessions.persist(session)
                return self._result(session, affected_message_id=self.system_message_id(session))
            if workflow_answer_proxy:
                target.content = _replace_workflow_answer(target.content, self._text(content, "Message content cannot be empty."))
                return self._result(self.sessions.persist(session), affected_message_id=message_id)
            target.content = self._text(content, "Message content cannot be empty.")
            return self._result(self.sessions.persist(session), affected_message_id=message_id)

        if operation == "delete":
            if target.role == "system":
                return self._result(self.sessions.persist(self._set_system(session, self.default_system_prompt)))
            if workflow_answer_proxy and target.workflow_turn_id:
                session["messages"] = [item for item in session["messages"] if item.workflow_turn_id != target.workflow_turn_id]
                return self._result(self.sessions.persist(session), affected_message_id=message_id)
            session["messages"] = [*session["messages"][:index], *session["messages"][index + 1 :]]
            return self._result(self.sessions.persist(session), affected_message_id=message_id)

        if operation == "rollback":
            if workflow_answer_proxy and target.workflow_turn_id:
                session["messages"] = [item for item in session["messages"] if item.workflow_turn_id != target.workflow_turn_id]
                return self._result(self.sessions.persist(session), affected_message_id=message_id)
            session["messages"] = session["messages"][: index + 1]
            return self._result(self.sessions.persist(session), affected_message_id=message_id)

        if target.role == "system":
            raise ValueError("System messages cannot be regenerated.")
        if target.role == "assistant":
            index = next((i for i in range(index - 1, -1, -1) if session["messages"][i].role == "user"), None)
            if index is None:
                raise ValueError("Assistant message does not have a preceding user message.")

        session["messages"] = session["messages"][: index + 1]
        return await self._generate(session, response_factory, operation)

    def find_index(self, message_id: str, messages: list[Message] | None = None) -> int:
        for index, message in enumerate(messages or self.get()):
            if message.id == message_id:
                return index
        raise ValueError(f"Message '{message_id}' was not found in this session.")

    def system_message_id(self, session: dict[str, Any] | None = None) -> str | None:
        message = next((item for item in (session or self.sessions.get())["messages"] if item.role == "system"), None)
        return message.id if message else None

    async def _generate(
        self,
        session: dict[str, Any],
        response_factory: MessageResponseFactory | None,
        operation: str,
    ) -> dict[str, Any]:
        session = self.sessions.persist(session)
        context = self.build_context(session["messages"])
        if response_factory is None:
            raise ValueError(f"{operation} requires a response_factory.")
        result = response_factory(context)
        if isawaitable(result):
            result = await result
        reply, token_usage = result
        if not reply.strip():
            raise ValueError("Model reply cannot be empty.")

        session = self.sessions.get()
        assistant = Message(role="assistant", content=reply, token_usage=token_usage, message_type="answer")
        session["messages"].append(assistant)
        session = self.sessions.persist(session)
        return self._result(
            session,
            context=context,
            reply=reply,
            token_usage=token_usage,
            affected_message_id=assistant.id,
        )

    def _compact_recent(self, messages: list[Message]) -> list[Message]:
        """Drop tool records and collapse each workflow turn into one assistant message."""
        recent = [message.model_copy(deep=True) for message in messages if message.role != "system"]
        compacted: list[Message] = []
        index = 0
        while index < len(recent):
            message = recent[index]
            if message.role == "tool":
                index += 1
                continue
            if message.role == "assistant" and message.workflow_turn_id:
                turn_id = message.workflow_turn_id
                turn_messages: list[Message] = []
                while index < len(recent):
                    candidate = recent[index]
                    if candidate.workflow_turn_id != turn_id or candidate.role not in {"assistant", "tool"}:
                        break
                    turn_messages.append(candidate)
                    index += 1
                selected = self._workflow_context_message(turn_id, turn_messages)
                if selected is not None:
                    compacted.append(selected)
                continue
            compacted.append(message)
            index += 1
        return compacted

    def _workflow_context_message(self, turn_id: str, messages: list[Message]) -> Message | None:
        """Collapse one workflow turn into one assistant message for next-turn context."""
        entries: list[tuple[str, str]] = []
        has_answer = False
        for message in messages:
            if message.role != "assistant":
                continue
            message_entries = [
                (name, block)
                for name, block in workflow_section_entries(message.content)
                if name in {"plan", "think", "answer"} and block
            ]
            if message_entries:
                entries.extend(message_entries)
                has_answer = has_answer or any(name == "answer" for name, _block in message_entries)
                continue
            if message.message_type == "answer" and message.content.strip() and not has_answer:
                entries.append(("answer", message.content.strip()))
                has_answer = True

        if not entries:
            return None
        return Message(
            role="assistant",
            content=render_workflow_sections(entries),
            message_type="workflow",
            workflow_turn_id=turn_id,
        )

    @staticmethod
    def _ensure_mutable(message: Message, operation: str):
        if message.message_type in READ_ONLY_MESSAGE_TYPES:
            raise ValueError(f"Workflow {message.message_type} messages are read-only and cannot be {operation}d.")

    @staticmethod
    def _result(
        session: dict[str, Any],
        *,
        context: list[dict[str, Any]] | None = None,
        reply: str | None = None,
        token_usage: dict[str, Any] | None = None,
        affected_message_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "session": session,
            "context": context or [],
            "reply": reply,
            "token_usage": token_usage,
            "affected_message_id": affected_message_id,
        }

    @staticmethod
    def _text(content: str | None, error: str) -> str:
        content = (content or "").strip()
        if not content:
            raise ValueError(error)
        return content

    @staticmethod
    def _message_content(role: str, content: str | None, tool_calls: list[dict[str, Any]] | None) -> str:
        content = (content or "").strip()
        if content:
            return content
        if str(role or "").strip() == "assistant" and isinstance(tool_calls, list) and tool_calls:
            return ""
        raise ValueError("Message content cannot be empty.")

    @staticmethod
    def _set_system(session: dict[str, Any], content: str | None) -> dict[str, Any]:
        content = (content or "").strip()
        current = next((message for message in session["messages"] if message.role == "system"), None)
        others = [message for message in session["messages"] if message.role != "system"]
        if not content:
            raise ValueError("System prompt cannot be empty.")
        system = current.model_copy(deep=True) if current else Message(role="system", content=content, message_type="system")
        system.content = content
        session["messages"] = [system, *others]
        return session


def _is_workflow_answer_proxy(message: Message) -> bool:
    """Return whether one workflow record carries the user-facing answer block."""
    return (
        message.role == "assistant"
        and message.message_type in WORKFLOW_ANSWER_PROXY_TYPES
        and bool(_workflow_sections(message.content).get("answer"))
    )


def _replace_workflow_answer(content: str, answer: str) -> str:
    """Replace only the answer block inside one persisted workflow assistant message."""
    entries = workflow_section_entries(content)
    if not any(name == "answer" and block for name, block in entries):
        raise ValueError("Workflow message does not contain an answer block.")

    return render_workflow_sections(
        [(name, answer.strip() if name == "answer" else block) for name, block in entries]
    )


def _workflow_sections(content: str) -> dict[str, str]:
    """Parse workflow blocks into a normalized section map."""
    return parse_workflow_sections(content)
