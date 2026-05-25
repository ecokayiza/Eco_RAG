from __future__ import annotations

from typing import Any

from ..workflow_sections import parse_workflow_sections, render_workflow_section


def hide_previous_tool_results_from_context(
    workflow_memory: list[dict[str, Any]],
    think_content: str | None,
) -> list[dict[str, Any]]:
    """Hide the previous tool-result batch from future in-turn model context.

    The immediately following think node can see full tool results and must distill
    any reusable evidence into its <echo_think> valid_information field. After that
    point, local storage can still keep the full tool record, but future model calls
    only receive a redacted placeholder for that tool batch.
    """
    if not _has_think_or_answer_content(think_content):
        return workflow_memory

    next_memory = [dict(item) for item in workflow_memory]
    previous = _previous_tool_memory_slice(next_memory)
    if previous is None:
        return next_memory

    tool_index, end_index = previous
    redacted_slice = []
    for item in next_memory[tool_index:end_index]:
        if _is_visual_memory_item(item):
            continue
        if _role(item) == "tool":
            item["content"] = _redacted_tool_content(item.get("content"))
        redacted_slice.append(item)
    next_memory[tool_index:end_index] = redacted_slice
    return next_memory


def _has_think_or_answer_content(content: str | None) -> bool:
    sections = parse_workflow_sections(content, allow_unclosed=True)
    return bool(_optional_text(sections.get("think")) or _optional_text(sections.get("answer")))


def _previous_tool_memory_slice(messages: list[dict[str, Any]]) -> tuple[int, int] | None:
    end_index = len(messages)
    if end_index > 0 and _role(messages[-1]) == "assistant":
        end_index -= 1

    index = end_index - 1
    first_tool_index: int | None = None
    while index >= 0:
        item = messages[index]
        if _role(item) == "tool":
            first_tool_index = index
            index -= 1
            continue
        if _is_visual_memory_item(item):
            index -= 1
            continue
        break
    return (first_tool_index, end_index) if first_tool_index is not None else None


def _redacted_tool_content(content: Any) -> str:
    tool_block = parse_workflow_sections(str(content or ""), allow_unclosed=True).get("tool")
    source = tool_block if tool_block is not None else str(content or "")
    heading = _first_nonempty_line(source)
    marker = "[tool result hidden from model context]"
    hidden = f"{heading}\n\n{marker}" if heading else marker
    return render_workflow_section("tool", hidden)


def _first_nonempty_line(value: str) -> str:
    for line in str(value or "").splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _is_visual_memory_item(item: dict[str, Any]) -> bool:
    content = item.get("content")
    if _role(item) != "user" or not isinstance(content, list) or not content:
        return False
    allowed_types = {"text", "image_url"}
    return (
        any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)
        and all(isinstance(part, dict) and part.get("type") in allowed_types for part in content)
    )


def _role(item: dict[str, Any]) -> str:
    return str(item.get("role") or "").strip()


def _optional_text(value: Any) -> str | None:
    text = " ".join(str(value or "").split())
    return text or None
