from __future__ import annotations

import asyncio
import base64
import binascii
import json
import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langgraph.config import get_stream_writer

from echo.settings import Config
from ..chat.chat_model import BaseChatModel, Response
from mcp_server.client import ToolClient
from ..workflow_sections import (
    parse_workflow_sections,
    render_workflow_section,
    render_workflow_sections,
    workflow_section_entries,
)
from .memory import hide_invalid_previous_tool_memory
from .state import WorkflowState, WorkflowStep

ANSWER_CHUNK_PATTERN = re.compile(r"\S+\s*|\s+")
TEXTUAL_RETRIEVE_PATTERN = re.compile(r"</?\s*(?:echo_)?retrieve\b", re.IGNORECASE)


@dataclass(frozen=True)
class WorkflowDependencies:
    """Bind runtime dependencies to one workflow instance."""

    model: BaseChatModel
    tool_client: ToolClient
    max_retrieve_rounds: int = 10


async def plan_node(state: WorkflowState, deps: WorkflowDependencies) -> WorkflowState:
    """Choose whether to answer now or enter retrieval."""
    response, streamed_answer = await _stream_decision_response(
        state,
        deps,
        node=WorkflowStep.PLAN.value,
    )
    decision = _decision_from_response(
        response,
        node=WorkflowStep.PLAN.value,
        allow_retrieve=bool(deps.tool_client.tool_names),
        allowed_tool_names=deps.tool_client.tool_names,
        requested_skill=state.get("requested_skill"),
    )
    content = (response.content or "").strip()
    pending_retrieve = _pending_retrieve_with_native_tool_calls(state, decision.get("pending_retrieve"))
    return {
        **state,
        "next_step": decision["next_step"],
        "pending_retrieve": pending_retrieve,
        "prepared_answer": decision.get("answer", ""),
        "streamed_answer": streamed_answer if decision["next_step"] == WorkflowStep.ANSWER.value else "",
        "workflow_memory": _append_memory(
            state["workflow_memory"],
            _assistant_memory_item(content, pending_retrieve),
        ),
    }


async def retrieve_node(state: WorkflowState) -> WorkflowState:
    """Validate the pending native tool calls before tool execution."""
    pending_retrieve = _pending_retrieve_calls(state.get("pending_retrieve"))
    if not pending_retrieve:
        raise ValueError("Retrieve node requires at least one pending native tool call.")
    return {
        **state,
        "next_step": WorkflowStep.TOOL.value,
    }


async def tool_node(state: WorkflowState, deps: WorkflowDependencies) -> WorkflowState:
    """Execute the pending MCP tool-call batch and store normalized results."""
    pending_retrieve = _pending_retrieve_calls(state.get("pending_retrieve"))
    if not pending_retrieve:
        raise ValueError("Tool node requires at least one pending native tool call.")

    round_number = state["retrieve_round"] + 1
    executions = await asyncio.gather(
        *[
            _execute_pending_tool_call(
                state,
                deps,
                pending_tool,
                round_number=round_number,
                call_index=index,
                call_count=len(pending_retrieve),
            )
            for index, pending_tool in enumerate(pending_retrieve, start=1)
        ]
    )

    tool_memory_items: list[dict[str, Any]] = []
    visual_memory_items: list[dict[str, Any]] = []
    for execution in executions:
        _emit_record(execution["record"])
        tool_memory_items.append(execution["memory"])
        visual_memory_items.extend(execution["visual_memory"])

    return {
        **state,
        "next_step": WorkflowStep.THINK.value,
        "retrieve_round": round_number,
        "pending_retrieve": None,
        "workflow_memory": _append_memory(
            state["workflow_memory"],
            *tool_memory_items,
            *visual_memory_items,
        ),
    }


async def think_node(state: WorkflowState, deps: WorkflowDependencies) -> WorkflowState:
    """Reflect on the accumulated transcript and decide whether to retrieve or answer."""
    limit_reached = bool(deps.tool_client.tool_names) and state["retrieve_round"] >= deps.max_retrieve_rounds
    allow_retrieve = bool(deps.tool_client.tool_names) and not limit_reached
    response, streamed_answer = await _stream_decision_response(
        state,
        deps,
        node=WorkflowStep.THINK.value,
        force_answer=limit_reached,
    )
    decision = _decision_from_response(
        response,
        node=WorkflowStep.THINK.value,
        allow_retrieve=allow_retrieve,
        allowed_tool_names=deps.tool_client.tool_names,
    )
    content = (response.content or "").strip()
    pending_retrieve = _pending_retrieve_with_native_tool_calls(state, decision.get("pending_retrieve"))
    next_memory = _append_memory(
        state["workflow_memory"],
        _assistant_memory_item(content, pending_retrieve),
    )
    return {
        **state,
        "next_step": decision["next_step"],
        "pending_retrieve": pending_retrieve,
        "prepared_answer": decision.get("answer", ""),
        "streamed_answer": streamed_answer if decision["next_step"] == WorkflowStep.ANSWER.value else "",
        "workflow_memory": hide_invalid_previous_tool_memory(next_memory, content),
    }


async def answer_node(state: WorkflowState) -> WorkflowState:
    """Emit the prepared answer without another model call."""
    answer = _required_block(state.get("prepared_answer"), "answer")
    already_streamed = str(state.get("streamed_answer") or "")
    remaining = answer[len(already_streamed) :] if answer.startswith(already_streamed) else answer
    writer = get_stream_writer()
    chunks = _answer_chunks(remaining)
    streamed = ""
    for index, chunk in enumerate(chunks):
        streamed = f"{streamed}{chunk}"
        writer(
            {
                "event": "chunk",
                "data": {
                    "delta": chunk,
                    "content": f"{already_streamed}{streamed}",
                },
            }
        )
        if index < len(chunks) - 1:
            await asyncio.sleep(0)
    return {
        **state,
        "next_step": None,
        "prepared_answer": answer,
        "streamed_answer": answer,
    }


def route_from_state(state: WorkflowState) -> str:
    """Start or resume the workflow from the saved next step."""
    return _next_step(
        state,
        {
            WorkflowStep.PLAN.value,
            WorkflowStep.RETRIEVE.value,
            WorkflowStep.TOOL.value,
            WorkflowStep.THINK.value,
            WorkflowStep.ANSWER.value,
        },
    )


def route_after_plan(state: WorkflowState) -> str:
    """Read the next node after plan."""
    return _next_step(state, {WorkflowStep.RETRIEVE.value, WorkflowStep.ANSWER.value})


def route_after_retrieve(_state: WorkflowState) -> str:
    """Retrieve always transitions into tool execution."""
    return WorkflowStep.TOOL.value


def route_after_tool(_state: WorkflowState) -> str:
    """Tool execution always transitions into think."""
    return WorkflowStep.THINK.value


def route_after_think(state: WorkflowState) -> str:
    """Read the next node after think."""
    return _next_step(state, {WorkflowStep.RETRIEVE.value, WorkflowStep.ANSWER.value})


def _workflow_messages(state: WorkflowState) -> list[dict[str, Any]]:
    """Build the provider payload from the flat workflow transcript."""
    payloads: list[dict[str, Any]] = []
    for item in state["workflow_memory"]:
        payload: dict[str, Any] = {"role": item["role"], "content": item["content"]}
        tool_calls = item.get("tool_calls")
        if item["role"] == "assistant" and isinstance(tool_calls, list) and tool_calls:
            payload["tool_calls"] = [dict(entry) for entry in tool_calls if isinstance(entry, dict)]
        tool_call_id = _optional_text(item.get("tool_call_id"))
        if item["role"] == "tool" and tool_call_id:
            payload["tool_call_id"] = tool_call_id
        payloads.append(payload)
    return payloads


def _append_memory(
    workflow_memory: list[dict[str, Any]],
    *items: dict[str, Any],
) -> list[dict[str, Any]]:
    """Append new flat-memory items while keeping provider transcript fields."""
    next_memory: list[dict[str, Any]] = []
    for item in [*workflow_memory, *items]:
        role = str(item.get("role") or "").strip()
        content = _message_content(item.get("content"))
        if role not in {"system", "user", "assistant", "tool"} or not content:
            continue
        payload: dict[str, Any] = {"role": role, "content": content}
        tool_calls = item.get("tool_calls")
        if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
            payload["tool_calls"] = [dict(entry) for entry in tool_calls if isinstance(entry, dict)]
        tool_call_id = _optional_text(item.get("tool_call_id"))
        if role == "tool" and tool_call_id:
            payload["tool_call_id"] = tool_call_id
        next_memory.append(payload)
    return next_memory


def _message_content(value: Any) -> Any:
    if isinstance(value, list):
        parts = [dict(item) for item in value if isinstance(item, dict)]
        return parts or None
    text = str(value or "").strip()
    return text or None


def _visual_memory_items(tool_name: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build transient user image messages for vision-capable models."""
    items = result.get("items")
    if tool_name != "web_fetch" or not isinstance(items, list):
        return []

    messages = []
    for item in items:
        if not isinstance(item, dict):
            continue
        image_url = str(item.get("image_url") or "").strip()
        if not image_url:
            continue
        title = _optional_text(item.get("title")) or "web_fetch screenshot"
        source_url = _optional_text(item.get("url"))
        caption = title if not source_url else f"{title}\nURL: {source_url}"
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": caption},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        )
    return messages


def _persist_tool_artifacts(
    tool_name: str,
    result: dict[str, Any],
    workflow_turn_id: str,
    *,
    round_number: int,
    item_offset: int = 1,
) -> dict[str, Any]:
    """Persist transient tool images as chat artifacts for later UI display."""
    if tool_name != "web_fetch" or not isinstance(result, dict):
        return result
    items = result.get("items")
    if not isinstance(items, list):
        return result

    next_items = []
    changed = False
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            next_items.append(item)
            continue
        next_item = dict(item)
        attachment = _persist_data_url_attachment(
            str(next_item.get("image_url") or ""),
            workflow_turn_id=workflow_turn_id,
            round_number=round_number,
            item_number=item_offset + index - 1,
            title=_optional_text(next_item.get("title")),
            source_url=_optional_text(next_item.get("url")),
        )
        if attachment:
            existing = next_item.get("attachments")
            attachments = [dict(entry) for entry in existing if isinstance(entry, dict)] if isinstance(existing, list) else []
            attachments.append(attachment)
            next_item["attachments"] = attachments
            changed = True
        next_items.append(next_item)

    if not changed:
        return result
    return {**result, "items": next_items}


def _persist_data_url_attachment(
    image_url: str,
    *,
    workflow_turn_id: str,
    round_number: int,
    item_number: int,
    title: str | None,
    source_url: str | None,
) -> dict[str, Any] | None:
    match = re.match(r"^data:(image/[A-Za-z0-9.+-]+);base64,(.+)$", image_url.strip(), flags=re.DOTALL)
    if not match:
        return None

    mime_type = match.group(1).lower()
    extension = {
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
    }.get(mime_type)
    if extension is None:
        return None

    try:
        data = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError):
        return None
    if not data:
        return None

    turn_dir_name = _artifact_path_segment(workflow_turn_id) or "workflow"
    artifact_dir = Config.CHAT_ARTIFACTS_DIR / turn_dir_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    filename = f"web_fetch-{round_number}-{item_number}-{uuid4().hex[:10]}.{extension}"
    path = artifact_dir / filename
    path.write_bytes(data)

    relative_path = path.relative_to(Config.CHAT_ARTIFACTS_DIR).as_posix()
    return {
        "id": uuid4().hex,
        "type": "image",
        "kind": "web_fetch_screenshot",
        "mime_type": mime_type,
        "url": f"/api/artifacts/{relative_path}",
        "path": relative_path,
        "title": title or "web_fetch screenshot",
        "source_url": source_url,
        "size_bytes": len(data),
    }


def _artifact_path_segment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._-")


def _tool_attachments(result: dict[str, Any]) -> list[dict[str, Any]]:
    items = result.get("items") if isinstance(result, dict) else None
    if not isinstance(items, list):
        return []
    attachments: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("attachments"), list):
            continue
        attachments.extend(dict(entry) for entry in item["attachments"] if isinstance(entry, dict))
    return attachments


def _decision_from_response(
    response: Response,
    *,
    node: str,
    allow_retrieve: bool,
    allowed_tool_names: set[str],
    requested_skill: str | None = None,
) -> dict[str, Any]:
    """Parse one native-tool-only decision-node response."""
    try:
        content = (response.content or "").strip()
        if TEXTUAL_RETRIEVE_PATTERN.search(content):
            raise ValueError("Textual retrieve blocks are not supported. Use a provider-native tool call.")

        sections = _sections(content, allow_unclosed=True)

        if requested_skill:
            if "load_skill" not in allowed_tool_names:
                raise ValueError("The load_skill tool is not configured.")
            return {
                "next_step": WorkflowStep.RETRIEVE.value,
                "pending_retrieve": [{"name": "load_skill", "args": {"skill_name": requested_skill}}],
            }

        if response.tool_calls:
            if _has_action_block(sections, "answer"):
                raise ValueError(f"{node.title()} node cannot include both <echo_answer> and a native tool call.")
            if not allow_retrieve:
                raise ValueError(f"{node.title()} node cannot request more retrieval.")
            return {
                "next_step": WorkflowStep.RETRIEVE.value,
                "pending_retrieve": _pending_retrieve_from_native_tool_calls(response.tool_calls, allowed_tool_names),
            }

        if _has_action_block(sections, "answer"):
            return {
                "next_step": WorkflowStep.ANSWER.value,
                "answer": _required_block(sections.get("answer"), "answer"),
            }

        raise ValueError(f"{node.title()} node must include <echo_answer> or at least one provider-native tool call.")
    except ValueError as exc:
        raise ValueError(_with_llm_raw_output(str(exc), response.content)) from exc


def _pending_retrieve_from_native_tool_calls(
    tool_calls: list[dict[str, Any]],
    allowed_tool_names: set[str],
) -> list[dict[str, Any]]:
    """Convert provider-native tool calls into Echo pending tool calls."""
    calls = [item for item in tool_calls if isinstance(item, dict) and str(item.get("name") or "").strip()]
    if not calls:
        raise ValueError("Workflow decisions must include at least one provider-native tool call.")

    pending_calls: list[dict[str, Any]] = []
    for tool_call in calls:
        name = str(tool_call.get("name") or "").strip()
        if name not in allowed_tool_names:
            allowed = ", ".join(sorted(allowed_tool_names))
            raise ValueError(f"Unknown tool '{name}'. Allowed tools: {allowed}.")

        args = tool_call.get("args")
        pending: dict[str, Any] = {"name": name, "args": dict(args) if isinstance(args, dict) else {}}
        tool_call_id = _optional_text(tool_call.get("id")) or _optional_text(tool_call.get("tool_call_id"))
        if tool_call_id:
            pending["tool_call_id"] = tool_call_id
        pending_calls.append(pending)
    return pending_calls


async def _execute_pending_tool_call(
    state: WorkflowState,
    deps: WorkflowDependencies,
    pending_tool: dict[str, Any],
    *,
    round_number: int,
    call_index: int,
    call_count: int,
) -> dict[str, Any]:
    """Run one pending tool call and build its stream record plus memory item."""
    tool_name = str(pending_tool.get("name") or "").strip()
    tool_args = dict(pending_tool.get("args") or {})
    result = await _run_tool(deps.tool_client, tool_name, tool_args)
    result = _persist_tool_artifacts(
        tool_name,
        result,
        state["workflow_turn_id"],
        round_number=round_number,
        item_offset=call_index,
    )
    tool_content = _format_tool_message(tool_name, tool_args, result)
    tool_call_id = _optional_text(pending_tool.get("tool_call_id"))

    record = {
        "id": _record_id(
            state,
            WorkflowStep.TOOL.value,
            suffix=_tool_record_suffix(round_number=round_number, call_index=call_index, call_count=call_count),
        ),
        "role": "tool",
        "content": tool_content,
        "message_type": WorkflowStep.TOOL.value,
        "workflow_turn_id": state["workflow_turn_id"],
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
    }
    attachments = _tool_attachments(result)
    if attachments:
        record["attachments"] = attachments

    return {
        "record": record,
        "memory": {
            "role": "tool",
            "content": tool_content,
            "tool_call_id": tool_call_id,
        },
        "visual_memory": _visual_memory_items(tool_name, result),
    }


async def _run_tool(
    tool_client: ToolClient,
    tool_name: str,
    tool_args: dict[str, Any],
) -> dict[str, Any]:
    """Execute one workflow MCP tool and normalize exceptions into a stable payload."""
    if tool_name not in tool_client.tool_names:
        raise ValueError(f"Workflow tool '{tool_name}' is not configured.")
    try:
        result = await tool_client.call_tool(tool_name, tool_args)
    except Exception as exc:
        result = {
            "type": "context",
            "skill_name": tool_name,
            "items": [],
            "error": str(exc),
        }
    if isinstance(result, dict):
        return result
    return {"type": "context", "skill_name": tool_name, "items": [{"content": str(result)}]}


def _format_tool_message(tool_name: str, tool_args: dict[str, Any], result: dict[str, Any]) -> str:
    """Render one readable tool message for persisted history."""
    heading = f"{tool_name}({', '.join(f'{key}={value!r}' for key, value in tool_args.items())})"
    error = _optional_text(result.get("error"))
    if error:
        return render_workflow_section("tool", f"{heading}\n\nError: {error}")

    if result.get("type") == "skill":
        content = str(result.get("content") or "").strip()
        skill_name = _optional_text(result.get("skill_name")) or tool_name
        return render_workflow_section("tool", f"{heading}\n\nLoaded skill: {skill_name}\n\n{content}")

    items = result.get("items")
    if not isinstance(items, list) or not items:
        return render_workflow_section("tool", f"{heading}\n\nNo results.")

    parts = []
    for index, item in enumerate(items, start=1):
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            item = {"content": str(item)}
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", item.get("document", "")) or "").strip()
        line = f"{index}. {title}" if title else f"{index}."
        if item.get("url"):
            line = f"{line}\nURL: {item['url']}"
        if content:
            line = f"{line}\n{content}"
        fetch_error = _optional_text(item.get("fetch_error"))
        if fetch_error:
            line = f"{line}\nHTML fetch failed: {fetch_error}"
        screenshot_error = _optional_text(item.get("screenshot_error"))
        if screenshot_error:
            line = f"{line}\nScreenshot unavailable: {screenshot_error}"
        parts.append(line.strip())
    return render_workflow_section("tool", f"{heading}\n\n" + "\n\n".join(parts))


def _emit_record(record: dict[str, Any]):
    """Emit one buffered persisted record into the LangGraph custom stream."""
    writer = get_stream_writer()
    writer({"event": "record", "data": record})


async def _stream_decision_response(
    state: WorkflowState,
    deps: WorkflowDependencies,
    *,
    node: str,
    force_answer: bool = False,
) -> tuple[Response, str]:
    """Stream one plan/think decision, emitting live record updates and answer chunks."""
    usage: dict[str, Any] = {}
    native_tool_calls: list[dict[str, Any]] = []
    content = ""
    streamed_answer = ""
    record_id = _decision_record_id(state, node)
    writer = get_stream_writer()

    def on_usage(payload: dict[str, Any] | None):
        if isinstance(payload, dict):
            usage.clear()
            usage.update(payload)

    def on_tool_calls(payload: list[dict[str, Any]] | None):
        if isinstance(payload, list):
            native_tool_calls.clear()
            native_tool_calls.extend(dict(item) for item in payload if isinstance(item, dict))

    workflow_messages = _workflow_messages(state)
    decision_tools = deps.tool_client.tool_schemas
    if force_answer:
        workflow_messages = _round_limit_answer_messages(
            workflow_messages,
            max_retrieve_rounds=deps.max_retrieve_rounds,
        )
        decision_tools = None
    async for chunk in deps.model.stream_response(
        workflow_messages,
        tools=decision_tools,
        callbacks={"on_usage": on_usage, "on_tool_calls": on_tool_calls},
    ):
        if not chunk:
            continue
        content = f"{content}{chunk}"
        stripped = content.strip()
        if stripped:
            _emit_record(
                {
                    "id": record_id,
                    "role": "assistant",
                    "content": stripped,
                    "message_type": node,
                    "workflow_turn_id": state["workflow_turn_id"],
                    "persist": False,
                }
            )
            streamed_answer = _emit_streaming_answer(
                writer,
                sections=_sections(stripped, allow_unclosed=True),
                streamed_answer=streamed_answer,
            )

    if not content.strip() and not native_tool_calls:
        response = await deps.model.generate_response(
            workflow_messages,
            tools=decision_tools,
        )
        content = (response.content or "").strip()
        native_tool_calls.clear()
        native_tool_calls.extend(response.tool_calls or [])
        if response.token_usage:
            usage.clear()
            usage.update(response.token_usage)

    if _needs_decision_repair(node, content, native_tool_calls):
        if force_answer:
            response = await deps.model.generate_response(
                _answer_repair_messages(workflow_messages),
                tools=None,
            )
            content = _coerce_answer_content(response.content, state)
            native_tool_calls.clear()
            if response.token_usage:
                usage.clear()
                usage.update(response.token_usage)
        else:
            response = await deps.model.generate_response(
                _decision_repair_messages(workflow_messages, node),
                tools=decision_tools,
            )
            content = (response.content or "").strip()
            native_tool_calls.clear()
            native_tool_calls.extend(response.tool_calls or [])
            if response.token_usage:
                usage.clear()
                usage.update(response.token_usage)

    if force_answer and (native_tool_calls or TEXTUAL_RETRIEVE_PATTERN.search(content)):
        response = await deps.model.generate_response(
            _answer_repair_messages(workflow_messages),
            tools=None,
        )
        content = _coerce_answer_content(response.content, state)
        native_tool_calls.clear()
        if response.token_usage:
            usage.clear()
            usage.update(response.token_usage)

    selected_tool_calls = _select_native_tool_calls(native_tool_calls, deps.tool_client.tool_names)
    if node == WorkflowStep.THINK.value and _needs_decision_repair(node, content, selected_tool_calls):
        response = await deps.model.generate_response(
            _answer_repair_messages(workflow_messages),
            tools=None,
        )
        content = _coerce_answer_content(response.content, state)
        selected_tool_calls = []
        if response.token_usage:
            usage.clear()
            usage.update(response.token_usage)

    final_content = _sanitize_decision_content(
        _with_native_tool_call_content(node, content.strip(), selected_tool_calls)
    )
    if not final_content:
        raise ValueError(_empty_decision_message(node))

    record = {
        "id": record_id,
        "role": "assistant",
        "content": final_content,
        "message_type": _record_message_type(node, final_content, selected_tool_calls),
        "workflow_turn_id": state["workflow_turn_id"],
        "token_usage": usage or None,
        "persist": True,
    }
    provider_tool_calls = _provider_tool_calls(selected_tool_calls)
    if provider_tool_calls:
        record["tool_calls"] = provider_tool_calls
    _emit_record(record)

    streamed_answer = _emit_streaming_answer(
        writer,
        sections=_sections(final_content),
        streamed_answer=streamed_answer,
    )
    return (
        Response(
            content=final_content,
            tool_calls=selected_tool_calls or None,
            token_usage=usage or None,
            raw_response=None,
        ),
        streamed_answer,
    )


def _emit_streaming_answer(
    writer,
    *,
    sections: dict[str, str],
    streamed_answer: str,
) -> str:
    """Emit live answer chunks once the response has entered the answer block."""
    answer = str(sections.get("answer") or "")
    if not answer:
        return streamed_answer

    if answer.startswith(streamed_answer):
        delta = answer[len(streamed_answer) :]
    else:
        delta = answer
    if not delta:
        return streamed_answer

    writer({"event": "chunk", "data": {"delta": delta, "content": answer}})
    return answer


def _next_step(state: WorkflowState, allowed: set[str]) -> str:
    """Read and validate the next step from state."""
    next_step = state.get("next_step")
    if next_step not in allowed:
        joined = ", ".join(sorted(allowed))
        raise ValueError(f"Workflow next step '{next_step}' is invalid. Allowed: {joined}.")
    return str(next_step)


def _sections(content: str | None, *, allow_unclosed: bool = False) -> dict[str, str]:
    """Parse current Echo workflow sections."""
    return parse_workflow_sections(content, allow_unclosed=allow_unclosed)


def _sanitize_decision_content(content: str) -> str:
    """Keep only current visible workflow sections from one model decision."""
    entries = [
        (name, block)
        for name, block in workflow_section_entries(content, allow_unclosed=True)
        if name in {"plan", "think", "answer"} and block
    ]
    return render_workflow_sections(entries) if entries else content.strip()


def _required_block(value: Any, label: str) -> str:
    """Read one required multi-line text block without flattening it."""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Workflow node is missing '{label}'.")
    return text


def _optional_text(value: Any) -> str | None:
    """Normalize optional string-like values."""
    text = " ".join(str(value or "").split())
    return text or None


def _with_llm_raw_output(detail: str, content: str | None, *, limit: int = 2000) -> str:
    """Append one bounded raw-model output block to workflow decision errors."""
    raw = (content or "").strip()
    if not raw:
        rendered = "<empty>"
    elif len(raw) > limit:
        rendered = f"{raw[:limit]}\n...(truncated {len(raw) - limit} chars)"
    else:
        rendered = raw
    return f"{detail}\nLLM raw output:\n{rendered}"


def _needs_decision_repair(node: str, content: str, tool_calls: list[dict[str, Any]]) -> bool:
    """Return whether a decision has no executable action and should be repaired."""
    if tool_calls:
        return False

    text = content.strip()
    if not text:
        return True

    sections = _sections(text, allow_unclosed=True)
    if _has_action_block(sections, "answer"):
        return False
    if TEXTUAL_RETRIEVE_PATTERN.search(text):
        return False

    entries = workflow_section_entries(text, allow_unclosed=True)
    if not entries:
        return True

    return all(name == node for name, _block in entries)


def _decision_repair_messages(messages: list[dict[str, Any]], node: str) -> list[dict[str, Any]]:
    """Append a corrective instruction for models that emitted thought without an action."""
    label = "think" if node == WorkflowStep.THINK.value else "plan"
    return [
        *messages,
        {
            "role": "user",
            "content": (
                "__echo_workflow_repair__\n"
                f"Your previous {label} decision had no executable next step. Continue the workflow now. "
                "Output exactly one of these: one or more tool calls, or an <echo_answer>...</echo_answer> block. "
                "Do not write prose-only intent such as 'I should search' or 'I should fetch'."
            ),
        },
    ]


def _round_limit_answer_messages(messages: list[dict[str, Any]], *, max_retrieve_rounds: int) -> list[dict[str, Any]]:
    """Append a hard instruction to answer when retrieval has reached its configured limit."""
    return [
        *messages,
        {
            "role": "user",
            "content": (
                "__echo_workflow_round_limit__\n"
                f"The workflow has reached the maximum retrieval limit of {max_retrieve_rounds} rounds. "
                "Do not call tools or request more retrieval. "
                "Use only the transcript and retrieved tool results above. "
                "Produce the final response now, and it must contain exactly one <echo_answer>...</echo_answer> block. "
                "If the available evidence is incomplete, state that uncertainty inside the answer block."
            ),
        },
    ]


def _answer_repair_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append a final-answer-only recovery instruction after repeated empty think decisions."""
    return [
        *_plain_answer_repair_messages(messages),
        {
            "role": "user",
            "content": (
                "__echo_workflow_answer_repair__\n"
                "The previous think decision was empty or incomplete twice. Tools are disabled for this recovery. "
                "Using only the transcript and retrieved tool results above, produce the final answer now in one "
                "<echo_answer>...</echo_answer> block."
            ),
        },
    ]


def _plain_answer_repair_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten native tool-call transcript details for a no-tools answer recovery call."""
    repaired: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user").strip()
        content = message.get("content")
        if role == "tool":
            repaired.append({"role": "user", "content": f"Tool result:\n{content}"})
            continue
        if role not in {"system", "developer", "user", "assistant"}:
            role = "user"
        repaired.append({"role": role, "content": content})
    return repaired


def _coerce_answer_content(content: str | None, state: WorkflowState) -> str:
    """Turn answer-recovery output into a valid answer block, with a deterministic last resort."""
    text = str(content or "").strip()
    entries = workflow_section_entries(text, allow_unclosed=True)
    if text and _has_action_block(_sections(text, allow_unclosed=True), "answer"):
        return text
    if text and not entries and not TEXTUAL_RETRIEVE_PATTERN.search(text):
        return render_workflow_section("answer", text)

    fallback = _fallback_answer_from_latest_tool(state)
    return render_workflow_section("answer", fallback)


def _fallback_answer_from_latest_tool(state: WorkflowState, *, limit: int = 4000) -> str:
    """Build a conservative final answer from the latest tool result if the model stays empty."""
    memory = state.get("workflow_memory")
    if isinstance(memory, list):
        for item in reversed(memory):
            if not isinstance(item, dict) or item.get("role") != "tool":
                continue
            content = str(item.get("content") or "").strip()
            tool_block = _sections(content, allow_unclosed=True).get("tool") or content
            tool_block = tool_block.strip()
            if tool_block:
                if len(tool_block) > limit:
                    tool_block = f"{tool_block[:limit].rstrip()}\n...(truncated)"
                return f"Retrieved context:\n\n{tool_block}"
    return "I retrieved context, but the model did not produce a final synthesis."


def _empty_decision_message(node: str) -> str:
    if node == WorkflowStep.THINK.value:
        return (
            "Think node returned no action. Empty <echo_think> is allowed only when the response "
            "also includes <echo_answer> or at least one provider-native tool call."
        )
    return f"{node.title()} node returned an empty response."


def _with_native_tool_call_content(node: str, content: str, tool_calls: list[dict[str, Any]]) -> str:
    """Ensure native tool-call decisions still have a visible node record."""
    if not tool_calls:
        return content

    sections = _sections(content, allow_unclosed=True)
    names = [str(item.get("name") or "").strip() for item in tool_calls if isinstance(item, dict) and item.get("name")]
    if not names:
        return content

    node_block = _optional_text(sections.get(node))
    if node_block:
        return render_workflow_section(node, node_block)

    label = "Native tool call" if len(names) == 1 else "Native tool calls"
    return render_workflow_section(node, f"{label}: {', '.join(names)}")


def _select_native_tool_calls(tool_calls: list[dict[str, Any]], allowed_tool_names: set[str]) -> list[dict[str, Any]]:
    """Keep complete named native tool calls and drop empty placeholders."""
    return [
        dict(item)
        for item in tool_calls
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]


def _record_message_type(node: str, content: str, tool_calls: list[dict[str, Any]]) -> str:
    """Classify persisted decision records by the visible action they carry."""
    if tool_calls:
        return node
    if _has_action_block(_sections(content), "answer"):
        return WorkflowStep.ANSWER.value
    return node


def _has_action_block(sections: dict[str, str], name: str) -> bool:
    """Return whether one non-empty action block exists."""
    return bool(_optional_text(sections.get(name)))


def _record_id(state: WorkflowState, node: str, *, suffix: str | None = None) -> str:
    """Build one stable live-record id for the current workflow turn."""
    parts = [state["workflow_turn_id"], node]
    if suffix:
        parts.append(suffix)
    return ":".join(parts)


def _decision_record_id(state: WorkflowState, node: str) -> str:
    """Build a stable live id for one plan/think decision pass."""
    if node == WorkflowStep.THINK.value:
        return _record_id(state, node, suffix=str(state["retrieve_round"]))
    return _record_id(state, node)


def _tool_record_suffix(*, round_number: int, call_index: int, call_count: int) -> str:
    """Build stable tool record suffixes while preserving old single-call ids."""
    if call_count <= 1:
        return str(round_number)
    return f"{round_number}.{call_index}"


def _pending_retrieve_calls(pending_retrieve: Any) -> list[dict[str, Any]]:
    """Normalize old single-call drafts and current call batches into a list."""
    if isinstance(pending_retrieve, dict):
        return [dict(pending_retrieve)]
    if not isinstance(pending_retrieve, list):
        return []
    return [dict(item) for item in pending_retrieve if isinstance(item, dict)]


def _tool_call_id(state: WorkflowState, *, round_number: int, call_index: int = 1) -> str:
    """Build one stable provider-native tool call id for a retrieval round."""
    suffix = str(round_number) if call_index == 1 else f"{round_number}.{call_index}"
    return f"{state['workflow_turn_id']}:tool_call:{suffix}"


def _pending_retrieve_with_native_tool_calls(
    state: WorkflowState,
    pending_retrieve: Any,
) -> list[dict[str, Any]] | None:
    """Attach stable tool_call_ids to the pending retrieval batch."""
    pending_calls = _pending_retrieve_calls(pending_retrieve)
    if not pending_calls:
        return None
    resolved: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, pending_tool in enumerate(pending_calls, start=1):
        payload = {
            "name": str(pending_tool.get("name") or "").strip(),
            "args": dict(pending_tool.get("args") or {}),
        }
        if not payload["name"]:
            continue
        tool_call_id = _optional_text(pending_tool.get("tool_call_id")) or _optional_text(pending_tool.get("id"))
        if not tool_call_id:
            tool_call_id = _tool_call_id(state, round_number=state["retrieve_round"] + 1, call_index=index)
        if tool_call_id in seen_ids:
            tool_call_id = _tool_call_id(state, round_number=state["retrieve_round"] + 1, call_index=index)
        payload["tool_call_id"] = tool_call_id
        seen_ids.add(tool_call_id)
        resolved.append(payload)
    return resolved or None


def _assistant_memory_item(content: str, pending_retrieve: Any) -> dict[str, Any]:
    """Build one assistant transcript item with native tool_calls when retrieval is pending."""
    payload: dict[str, Any] = {"role": "assistant", "content": content}
    provider_tool_calls = _provider_tool_calls(_pending_retrieve_calls(pending_retrieve))
    if provider_tool_calls:
        payload["tool_calls"] = provider_tool_calls
    return payload


def _provider_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    provider_calls = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_call_id = _optional_text(tool_call.get("tool_call_id")) or _optional_text(tool_call.get("id"))
        tool_name = str(tool_call.get("name") or "").strip()
        tool_args = tool_call.get("args")
        if not tool_call_id or not tool_name or not isinstance(tool_args, dict):
            continue
        provider_calls.append(
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(tool_args, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                },
            }
        )
    return provider_calls or None


def _answer_chunks(answer: str, target_size: int = 48) -> list[str]:
    """Split one prepared answer into readable stream chunks."""
    text = answer.strip()
    if not text or len(text) <= target_size:
        return [text] if text else []

    chunks: list[str] = []
    current = ""
    for piece in ANSWER_CHUNK_PATTERN.findall(text):
        if current and len(current) + len(piece) > target_size:
            chunks.append(current)
            current = piece
            continue
        current = f"{current}{piece}"
    if current:
        chunks.append(current)
    return chunks or [text]
