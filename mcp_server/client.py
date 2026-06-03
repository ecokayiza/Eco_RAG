from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from echo.settings import load_app_settings


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


class ToolClient(Protocol):
    @property
    def tool_names(self) -> set[str]: ...

    @property
    def tool_schemas(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]: ...


class StdioMCPToolClient(AbstractAsyncContextManager["StdioMCPToolClient"]):
    """Local stdio MCP client for Echo workflow tools."""

    def __init__(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
    ):
        self.command = command or sys.executable
        self.args = args or ["-m", "mcp_server"]
        self.cwd = cwd
        self.env = env
        self._stdio_context: Any = None
        self._session_context: ClientSession | None = None
        self._session: ClientSession | None = None
        self._tools: list[ToolSpec] = []

    @property
    def tool_names(self) -> set[str]:
        return {tool.name for tool in self._tools}

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [_tool_schema(tool) for tool in self._tools]

    async def __aenter__(self) -> "StdioMCPToolClient":
        parameters = StdioServerParameters(
            command=self.command,
            args=self.args,
            cwd=self.cwd,
            env=self.env if self.env is not None else dict(os.environ),
        )
        self._stdio_context = stdio_client(parameters)
        read_stream, write_stream = await self._stdio_context.__aenter__()
        self._session_context = ClientSession(read_stream, write_stream)
        self._session = await self._session_context.__aenter__()
        await self._session.initialize()
        result = await self._session.list_tools()
        self._tools = [
            ToolSpec(
                name=str(tool.name),
                description=str(tool.description or ""),
                input_schema=dict(tool.inputSchema or {"type": "object", "properties": {}}),
            )
            for tool in result.tools
        ]
        return self

    async def __aexit__(self, exc_type, exc, tb):
        cleanup_error: BaseException | None = None
        if self._session_context is not None:
            try:
                await self._session_context.__aexit__(exc_type, exc, tb)
            except BaseException as session_exc:
                cleanup_error = session_exc
        if self._stdio_context is not None:
            try:
                await self._stdio_context.__aexit__(exc_type, exc, tb)
            except BaseException as stdio_exc:
                cleanup_error = stdio_exc if cleanup_error is None else cleanup_error
        if cleanup_error is not None and exc_type is None:
            raise cleanup_error
        return False

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("MCP tool client is not connected.")
        result = await self._session.call_tool(name, args)
        payload = _call_result_payload(name, result)
        return payload


class SharedMCPToolClient:
    """Process-wide warmed MCP client for workflow tool calls."""

    def __init__(self, client: StdioMCPToolClient | None = None):
        self._client = client or StdioMCPToolClient()
        self._connect_lock = asyncio.Lock()
        self._state_condition = asyncio.Condition()
        self._active_calls = 0
        self._closing = False
        self._connected = False

    @property
    def tool_names(self) -> set[str]:
        return self._client.tool_names

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return self._client.tool_schemas

    async def connect(self) -> "SharedMCPToolClient":
        async with self._connect_lock:
            if not self._connected:
                await self._client.__aenter__()
                async with self._state_condition:
                    self._closing = False
                self._connected = True
        return self

    async def aclose(self):
        async with self._connect_lock:
            if not self._connected:
                return
            async with self._state_condition:
                self._closing = True
                while self._active_calls:
                    await self._state_condition.wait()
            await self._client.__aexit__(None, None, None)
            self._connected = False
            async with self._state_condition:
                self._closing = False

    def context(self) -> AbstractAsyncContextManager[ToolClient]:
        return _SharedMCPToolClientContext(self)

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        async with self._connect_lock:
            if not self._connected:
                await self._client.__aenter__()
                self._connected = True
            async with self._state_condition:
                if self._closing:
                    raise RuntimeError("MCP tool client is closing.")
                self._active_calls += 1
        try:
            return await self._client.call_tool(name, args)
        finally:
            async with self._state_condition:
                self._active_calls -= 1
                if self._active_calls <= 0:
                    self._state_condition.notify_all()


class _SharedMCPToolClientContext(AbstractAsyncContextManager[ToolClient]):
    def __init__(self, owner: SharedMCPToolClient):
        self.owner = owner

    async def __aenter__(self) -> ToolClient:
        return await self.owner.connect()

    async def __aexit__(self, exc_type, exc, tb):
        return False


def local_mcp_tool_client() -> StdioMCPToolClient:
    return StdioMCPToolClient()


def tool_schemas_from_specs(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [_tool_schema(tool) for tool in tools]


def _tool_schema(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": _compact_input_schema(tool.input_schema, tool_name=tool.name),
        },
    }


def _compact_input_schema(schema: dict[str, Any], *, tool_name: str | None = None) -> dict[str, Any]:
    if not schema:
        return {"type": "object", "properties": {}}
    compacted = _strip_schema_titles(schema)
    if not isinstance(compacted, dict):
        return {"type": "object", "properties": {}}
    return _with_runtime_tool_limits(compacted, tool_name=tool_name)


def _with_runtime_tool_limits(schema: dict[str, Any], *, tool_name: str | None = None) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return schema

    settings = load_app_settings()
    next_properties = dict(properties)

    if tool_name == "database_search" and isinstance(next_properties.get("top_k"), dict):
        top_k = dict(next_properties["top_k"])
        limit = max(1, settings.max_database_search_top_k)
        top_k["minimum"] = 1
        top_k["maximum"] = limit
        top_k["description"] = str(top_k.get("description") or f"Number of passages to retrieve, capped at {limit}.")
        next_properties["top_k"] = top_k

    if tool_name == "web_search" and isinstance(next_properties.get("max_results"), dict):
        max_results = dict(next_properties["max_results"])
        limit = max(1, settings.max_web_search_results)
        max_results["minimum"] = 1
        max_results["maximum"] = limit
        max_results["description"] = str(
            max_results.get("description") or f"Maximum web search results to return, capped at {limit}."
        )
        next_properties["max_results"] = max_results

    if next_properties == properties:
        return schema

    return {**schema, "properties": next_properties}


def _strip_schema_titles(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_schema_titles(item) for key, item in value.items() if key != "title"}
    if isinstance(value, list):
        return [_strip_schema_titles(item) for item in value]
    return value


def _call_result_payload(name: str, result: Any) -> dict[str, Any]:
    if getattr(result, "isError", False):
        return _error_payload(name, _content_text(result))

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        wrapped_result = structured.get("result")
        if set(structured) == {"result"} and isinstance(wrapped_result, dict):
            return wrapped_result
        return structured

    text = _content_text(result)
    if text:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"type": "context", "skill_name": name, "items": [{"content": text}]}
        if isinstance(parsed, dict):
            return parsed
        return {"type": "context", "skill_name": name, "items": [{"content": str(parsed)}]}

    return {"type": "context", "skill_name": name, "items": []}


def _content_text(result: Any) -> str:
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def _error_payload(name: str, text: str) -> dict[str, Any]:
    return {
        "type": "context",
        "skill_name": name,
        "items": [],
        "error": text or f"MCP tool '{name}' failed.",
    }
