from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any, AsyncIterator, Callable

from ..chat.registry import ChatModelSettings, build_chat_model
from ..skills import list_available_skills
from ..settings import load_app_settings
from mcp_server.client import ToolClient, local_mcp_tool_client
from .drafts import WorkflowDraftStore
from .graph import build_workflow
from .nodes import WorkflowDependencies
from .prompts import default_system_prompt
from .state import WorkflowState, WorkflowStep, new_state
from .tracker import WorkflowTracker
from ..workflow_sections import workflow_section_entries


class WorkflowService:
    """Run the LangGraph workflow and adapt it to the app stream contract."""

    def __init__(
        self,
        model_factory: Callable[[ChatModelSettings | None], Any] = build_chat_model,
        *,
        tool_client_factory: Callable[[], AbstractAsyncContextManager[ToolClient]] | None = None,
        draft_storage: dict[str, dict[str, Any]] | None = None,
        enable_drafts: bool = False,
    ):
        self.model_factory = model_factory
        self.tool_client_factory = tool_client_factory
        self.draft_store = WorkflowDraftStore(storage=draft_storage) if enable_drafts or draft_storage is not None else None

    def load_draft(self, session_id: str) -> dict[str, Any] | None:
        """Load a saved live workflow draft when resumable workflows are enabled."""
        return self.draft_store.load(session_id) if self.draft_store is not None else None

    def clear_draft(self, session_id: str):
        """Clear a saved live workflow draft when resumable workflows are enabled."""
        if self.draft_store is not None:
            self.draft_store.clear(session_id)

    async def stream(
        self,
        question: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream workflow events for one standalone question."""
        query = self._query(question)
        async for item in self._stream_state(
            new_state(query, self._default_context(query)),
            tracker=None,
            records=[],
        ):
            yield item

    async def stream_chat(
        self,
        question: str,
        *,
        context: list[dict[str, Any]] | None = None,
        workflow_turn_id: str | None = None,
        session_id: str | None = None,
        user_message_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream workflow events for one chat turn from persisted session history."""
        query = self._query(question)
        state = new_state(
            query,
            self._default_context(query, context),
            workflow_turn_id=workflow_turn_id or user_message_id,
        )
        records: list[dict[str, Any]] = []
        if self.draft_store is not None and session_id and user_message_id:
            draft = self.draft_store.load(session_id)
            if draft and draft["user_message_id"] == user_message_id:
                state = dict(draft["state"])
                records = [dict(item) for item in draft["records"]]
            elif draft:
                self.draft_store.clear(session_id)

        async for item in self._stream_state(
            state,
            tracker=None,
            records=records,
            draft_session_id=session_id,
            draft_user_message_id=user_message_id,
        ):
            yield item

    def _deps(self, tool_client: ToolClient) -> WorkflowDependencies:
        """Build one dependency bundle for a workflow run."""
        settings = load_app_settings()
        filtered_tool_client = FilteredToolClient(tool_client, settings.mcp_enabled_tools)
        return WorkflowDependencies(
            model=self.model_factory(),
            tool_client=filtered_tool_client,
            max_retrieve_rounds=settings.max_retrieve_rounds,
        )

    def _default_context(self, query: str, context: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Ensure the workflow transcript starts with one rendered system prompt."""
        base_context = [dict(item) for item in (context or []) if isinstance(item, dict)]
        if not any(str(item.get("role", "")).strip() == "system" for item in base_context):
            settings = load_app_settings()
            base_context.insert(
                0,
                {
                    "role": "system",
                    "content": default_system_prompt(
                        available_skills=list_available_skills(),
                        allow_load_skill="load_skill" in settings.mcp_enabled_tools,
                    ),
                },
            )
        if not any(str(item.get("role", "")).strip() == "user" for item in base_context):
            base_context.append({"role": "user", "content": query})
        return base_context

    @staticmethod
    def _query(question: str) -> str:
        """Normalize one workflow question."""
        query = " ".join(question.strip().split())
        if not query:
            raise ValueError("Question cannot be empty.")
        return query

    async def _stream_state(
        self,
        state: WorkflowState,
        *,
        tracker: WorkflowTracker | None,
        records: list[dict[str, Any]],
        draft_session_id: str | None = None,
        draft_user_message_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the workflow and emit state updates, chunks, and the final result."""
        async with self._tool_client_context() as tool_client:
            deps = self._deps(tool_client)
            graph = build_workflow(deps)
            runtime_tracker = tracker or WorkflowTracker(state["workflow_turn_id"], state["query"])
            current_state = dict(state)
            buffered_records = [dict(item) for item in records]
            latest_record: dict[str, Any] | None = None
            self._persist_draft(draft_session_id, draft_user_message_id, current_state, runtime_tracker, buffered_records)
            yield {"event": "state", "data": runtime_tracker.snapshot(current_state)}

            try:
                async for mode, payload in graph.astream(current_state, stream_mode=["tasks", "custom"]):
                    if mode == "custom":
                        if not isinstance(payload, dict):
                            continue
                        if payload.get("event") == "chunk":
                            yield payload
                        elif payload.get("event") == "transition" and isinstance(payload.get("data"), dict):
                            self._apply_transition(runtime_tracker, current_state, payload["data"])
                            yield {"event": "state", "data": runtime_tracker.snapshot(current_state)}
                        elif payload.get("event") == "record" and isinstance(payload.get("data"), dict):
                            record = dict(payload["data"])
                            persist = bool(record.pop("persist", True))
                            latest_record = dict(record)
                            if persist:
                                buffered_records.append(dict(record))
                            yield {"event": "record", "data": record}
                        continue

                    if mode != "tasks" or not isinstance(payload, dict):
                        continue

                    node_name = payload.get("name")
                    if node_name not in {step.value for step in WorkflowStep}:
                        continue

                    step = WorkflowStep(node_name)
                    if "input" in payload:
                        self._start_step(runtime_tracker, step)
                        yield {"event": "state", "data": runtime_tracker.snapshot(current_state)}
                        continue

                    if payload.get("error") is not None:
                        continue

                    result = payload.get("result")
                    if isinstance(result, dict):
                        current_state = {**current_state, **result}
                    runtime_tracker.complete(step, self._detail_for_step(step, current_state))
                    self._log_route(runtime_tracker, step, current_state)
                    self._persist_draft(draft_session_id, draft_user_message_id, current_state, runtime_tracker, buffered_records)
                    yield {"event": "state", "data": runtime_tracker.snapshot(current_state)}

                runtime_tracker.finish()
                snapshot = runtime_tracker.snapshot(current_state)
                if self.draft_store is not None and draft_session_id:
                    self.draft_store.clear(draft_session_id)
                yield {"event": "state", "data": snapshot}
                yield {
                    "event": "done",
                    "data": {
                        "snapshot": snapshot,
                        "records": buffered_records,
                        "token_usage": _sum_token_usage(buffered_records),
                    },
                }
            except Exception as exc:
                thought_log = _latest_error_thought(
                    current_state,
                    buffered_records,
                    latest_record,
                    active_node=runtime_tracker.active_node,
                )
                if thought_log:
                    runtime_tracker.log(thought_log, node="thought", level="error")
                runtime_tracker.fail(str(exc), node=runtime_tracker.active_node)
                self._persist_draft(draft_session_id, draft_user_message_id, current_state, runtime_tracker, buffered_records)
                yield {"event": "state", "data": runtime_tracker.snapshot(current_state)}
                raise

    def _tool_client_context(self) -> AbstractAsyncContextManager[ToolClient]:
        return self.tool_client_factory() if self.tool_client_factory is not None else local_mcp_tool_client()

    def _persist_draft(
        self,
        session_id: str | None,
        user_message_id: str | None,
        state: WorkflowState,
        tracker: WorkflowTracker,
        records: list[dict[str, Any]],
    ):
        """Persist resumable workflow state when a draft store is configured."""
        if self.draft_store is None or not session_id or not user_message_id:
            return
        self.draft_store.persist(
            session_id,
            user_message_id=user_message_id,
            state=dict(state),
            snapshot=tracker.snapshot(state),
            records=[dict(item) for item in records],
        )

    @staticmethod
    def _detail_for_step(step: WorkflowStep, state: WorkflowState) -> str:
        """Build one compact completion detail for the workflow tracker."""
        if step == WorkflowStep.PLAN:
            return _route_detail(state.get("next_step"))
        if step == WorkflowStep.RETRIEVE:
            tool_names = _pending_tool_names(state.get("pending_retrieve"))
            if len(tool_names) == 1:
                return f"Accepted '{tool_names[0]}'."
            if tool_names:
                return f"Accepted {len(tool_names)} tool calls: {', '.join(tool_names)}."
            return "Accepted the pending retrieve command."
        if step == WorkflowStep.TOOL:
            return f"Completed round {state['retrieve_round']}."
        if step == WorkflowStep.THINK:
            return _route_detail(state.get("next_step"))
        return "Final answer emitted."

    @staticmethod
    def _log_route(tracker: WorkflowTracker, step: WorkflowStep, state: WorkflowState):
        """Keep live workflow logs minimal and high-signal."""
        return

    @staticmethod
    def _start_step(tracker: WorkflowTracker, step: WorkflowStep):
        """Mark one workflow step as running with simple UI-facing detail."""
        detail = None
        if step == WorkflowStep.PLAN:
            detail = "Planning the response."
        elif step == WorkflowStep.RETRIEVE:
            detail = "Preparing the retrieve step."
        elif step == WorkflowStep.TOOL:
            detail = "Running the tool."
        elif step == WorkflowStep.THINK:
            detail = "Reviewing the tool results."
        elif step == WorkflowStep.ANSWER:
            detail = "Publishing the final answer."
        tracker.start(step, detail)

    @staticmethod
    def _apply_transition(tracker: WorkflowTracker, state: WorkflowState, payload: dict[str, Any]):
        """Reflect one live routing hint before the current node fully completes."""
        target = str(payload.get("to_node") or "").strip()
        if target != WorkflowStep.ANSWER.value:
            return

        source = str(payload.get("from_node") or "").strip()
        if source in {WorkflowStep.PLAN.value, WorkflowStep.THINK.value}:
            tracker.complete(WorkflowStep(source), _route_detail(WorkflowStep.ANSWER.value))
        answer = str(payload.get("answer") or "").strip()
        if answer:
            state["prepared_answer"] = answer
            state["streamed_answer"] = answer
        tracker.start(WorkflowStep.ANSWER, "Publishing the final answer.")


def _sum_token_usage(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Aggregate token usage across buffered workflow records."""
    usage: dict[str, int | float] = {}
    for record in records:
        token_usage = record.get("token_usage")
        if not isinstance(token_usage, dict):
            continue
        for key, value in token_usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage[key] = usage.get(key, 0) + value
    return usage or None


class FilteredToolClient:
    """Expose only the MCP tools configured for this workflow run."""

    def __init__(self, tool_client: ToolClient, enabled_tool_names: list[str] | tuple[str, ...]):
        self._tool_client = tool_client
        self._enabled_tool_names = set(enabled_tool_names)

    @property
    def tool_names(self) -> set[str]:
        return self._tool_client.tool_names & self._enabled_tool_names

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        allowed = self.tool_names
        return [
            schema
            for schema in self._tool_client.tool_schemas
            if str(schema.get("function", {}).get("name") or "").strip() in allowed
        ]

    async def call_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name not in self.tool_names:
            raise ValueError(f"Workflow tool '{name}' is not enabled.")
        return await self._tool_client.call_tool(name, args)


def _route_detail(next_step: Any) -> str:
    """Render a short route detail for decision nodes."""
    step = str(next_step or "").strip()
    if step == WorkflowStep.RETRIEVE.value:
        return "Will retrieve more context."
    if step == WorkflowStep.ANSWER.value:
        return "Answer is ready."
    return "Decision completed."


def _pending_tool_names(pending_retrieve: Any) -> list[str]:
    """Read pending tool names from old single-call drafts or current batches."""
    if isinstance(pending_retrieve, dict):
        pending_items = [pending_retrieve]
    elif isinstance(pending_retrieve, list):
        pending_items = [item for item in pending_retrieve if isinstance(item, dict)]
    else:
        pending_items = []
    return [name for item in pending_items if (name := str(item.get("name") or "").strip())]


def _latest_error_thought(
    state: dict[str, Any],
    records: list[dict[str, Any]],
    latest_record: dict[str, Any] | None,
    *,
    active_node: str | None = None,
) -> str | None:
    """Find the latest assistant reasoning visible enough to explain a workflow error."""
    candidates: list[dict[str, Any]] = []
    if isinstance(latest_record, dict):
        candidates.append(latest_record)
    candidates.extend(dict(record) for record in reversed(records) if isinstance(record, dict))
    memory = state.get("workflow_memory")
    if isinstance(memory, list):
        candidates.extend(dict(item) for item in reversed(memory) if isinstance(item, dict))

    for item in candidates:
        rendered = _thought_log_from_record(item, active_node=active_node)
        if rendered:
            return rendered
    return None


def _thought_log_from_record(record: dict[str, Any], *, active_node: str | None = None) -> str | None:
    """Render one compact thought log entry from a workflow record."""
    if str(record.get("role") or "").strip() != "assistant":
        return None

    content = str(record.get("content") or "").strip()
    if not content:
        return None

    entries = workflow_section_entries(content, allow_unclosed=True)
    if entries and all(name in {"think", "plan"} and not block for name, block in entries):
        return None
    for name, block in reversed(entries):
        if name in {"think", "plan"} and block:
            if active_node == WorkflowStep.THINK.value and name == WorkflowStep.PLAN.value:
                return f"No think output was emitted before error. Latest plan: {_compact_log_text(block)}"
            return f"Latest {name} before error: {_compact_log_text(block)}"
    for name, block in reversed(entries):
        if name == "answer" and block:
            return f"Latest answer draft before error: {_compact_log_text(block)}"
    return f"Latest assistant output before error: {_compact_log_text(content)}"


def _compact_log_text(value: str, *, limit: int = 900) -> str:
    """Keep workflow log context readable inside the side panel."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."
