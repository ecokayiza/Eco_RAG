from __future__ import annotations

import json
import re
from typing import Any

from ..workflow_sections import parse_workflow_sections, render_workflow_section

THINK_VALIDATION_PATTERN = re.compile(
    r"""(?i)(?:"|')?validation(?:"|')?\s*:\s*(?:"|')?(valid|invalid)\b"""
)


def think_validation(content: str | None) -> str | None:
    """Return the validation value from an <echo_think> block when present."""
    block = parse_workflow_sections(content, allow_unclosed=True).get("think")
    if not block:
        return None

    payload = _json_object(block)
    if isinstance(payload, dict):
        validation = str(payload.get("validation") or "").strip().lower()
        if validation in {"valid", "invalid"}:
            return validation

    match = THINK_VALIDATION_PATTERN.search(block)
    return match.group(1).lower() if match else None


def hide_invalid_previous_tool_memory(
    workflow_memory: list[dict[str, Any]],
    think_content: str | None,
) -> list[dict[str, Any]]:
    """Redact the previous tool result in in-turn memory when think marks it invalid."""
    if think_validation(think_content) != "invalid":
        return workflow_memory

    next_memory = [dict(item) for item in workflow_memory]
    previous = _previous_tool_memory_slice(next_memory)
    if previous is None:
        return next_memory

    tool_index, end_index = previous
    next_memory[tool_index]["content"] = _redacted_tool_content(next_memory[tool_index].get("content"))
    del next_memory[tool_index + 1 : end_index]
    return next_memory


def _json_object(value: str) -> dict[str, Any] | None:
    text = _strip_code_fence(value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _strip_code_fence(value: str) -> str:
    text = value.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _previous_tool_memory_slice(messages: list[dict[str, Any]]) -> tuple[int, int] | None:
    end_index = len(messages)
    if end_index > 0 and _role(messages[-1]) == "assistant":
        end_index -= 1

    index = end_index - 1
    while index >= 0:
        item = messages[index]
        if _role(item) == "tool":
            return index, end_index
        if _is_visual_memory_item(item):
            index -= 1
            continue
        return None
    return None


def _redacted_tool_content(content: Any) -> str:
    tool_block = parse_workflow_sections(str(content or ""), allow_unclosed=True).get("tool")
    source = tool_block if tool_block is not None else str(content or "")
    heading = _first_nonempty_line(source)
    hidden = f"{heading}\n\n[information hidden]" if heading else "[information hidden]"
    return render_workflow_section("tool", hidden)


def _first_nonempty_line(value: str) -> str:
    for line in str(value or "").splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _is_visual_memory_item(item: dict[str, Any]) -> bool:
    content = item.get("content")
    return (
        _role(item) == "user"
        and isinstance(content, list)
        and bool(content)
        and all(isinstance(part, dict) and part.get("type") == "image_url" for part in content)
    )


def _role(item: dict[str, Any]) -> str:
    return str(item.get("role") or "").strip()


def _optional_text(value: Any) -> str | None:
    text = " ".join(str(value or "").split())
    return text or None
