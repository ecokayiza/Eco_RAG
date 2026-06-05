from __future__ import annotations

from enum import Enum
from typing import Any, NotRequired, TypedDict
from uuid import uuid4

from ..skills.manager import extract_requested_skill


class WorkflowStatus(str, Enum):
    """Define the workflow lifecycle exposed to the app."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowStep(str, Enum):
    """Name the workflow nodes exposed to the app."""

    PLAN = "plan"
    RETRIEVE = "retrieve"
    TOOL = "tool"
    THINK = "think"
    ANSWER = "answer"


class WorkflowRetrieveCall(TypedDict):
    """Store one pending retrieval command."""

    name: str
    args: dict[str, Any]
    tool_call_id: NotRequired[str]


class WorkflowMemoryMessage(TypedDict):
    """Store one flat in-workflow memory message."""

    role: str
    content: Any
    tool_calls: NotRequired[list[dict[str, Any]]]
    tool_call_id: NotRequired[str]


class WorkflowState(TypedDict, total=False):
    """Store the mutable workflow state."""

    workflow_turn_id: str
    query: str
    requested_skill: str | None
    next_step: str | None
    retrieve_round: int
    pending_retrieve: list[WorkflowRetrieveCall] | None
    prepared_answer: str
    streamed_answer: str
    workflow_memory: list[WorkflowMemoryMessage]


def new_state(
    query: str,
    context: list[dict[str, Any]] | None = None,
    *,
    workflow_turn_id: str | None = None,
) -> WorkflowState:
    """Create the initial workflow state for one query."""
    requested_skill, cleaned_query = extract_requested_skill(query)
    base_context = _normalize_memory(context) if context is not None else [{"role": "user", "content": query}]
    return {
        "workflow_turn_id": (workflow_turn_id or str(uuid4())).strip(),
        "query": cleaned_query,
        "requested_skill": requested_skill,
        "next_step": WorkflowStep.PLAN.value,
        "retrieve_round": 0,
        "pending_retrieve": None,
        "prepared_answer": "",
        "streamed_answer": "",
        "workflow_memory": base_context,
    }


def _normalize_memory(messages: list[dict[str, Any]] | None) -> list[WorkflowMemoryMessage]:
    """Keep only provider-safe role/content pairs in workflow memory."""
    normalized: list[WorkflowMemoryMessage] = []
    for item in messages or []:
        role = str(item.get("role", "")).strip()
        content = _normalize_content(item.get("content"))
        payload: WorkflowMemoryMessage = {"role": role, "content": content}
        tool_calls = item.get("tool_calls")
        provider_tool_calls = [dict(entry) for entry in tool_calls if isinstance(entry, dict)] if isinstance(tool_calls, list) else []
        has_tool_calls = role == "assistant" and bool(provider_tool_calls)
        if role not in {"system", "user", "assistant", "tool"} or (content is None and not has_tool_calls):
            continue
        payload = {"role": role, "content": content if content is not None else ""}
        if has_tool_calls:
            payload["tool_calls"] = provider_tool_calls
        tool_call_id = str(item.get("tool_call_id", "")).strip()
        if role == "tool" and tool_call_id:
            payload["tool_call_id"] = tool_call_id
        normalized.append(payload)
    return normalized


def _normalize_content(value: Any) -> Any:
    if isinstance(value, list):
        parts = [dict(item) for item in value if isinstance(item, dict)]
        return parts or None
    text = str(value or "").strip()
    return text or None
