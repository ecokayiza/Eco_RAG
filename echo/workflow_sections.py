from __future__ import annotations

import re
from typing import Pattern

WORKFLOW_SECTION_NAMES = {"plan", "think", "answer", "tool"}
TAG_PATTERN = re.compile(r"</?\s*([a-z_]+)\s*>", re.IGNORECASE)


def parse_workflow_sections(content: str | None, *, allow_unclosed: bool = False) -> dict[str, str]:
    """Parse current Echo workflow blocks."""
    return {name: block for name, block in workflow_section_entries(content, allow_unclosed=allow_unclosed)}


def workflow_section_entries(content: str | None, *, allow_unclosed: bool = False) -> list[tuple[str, str]]:
    """Return current Echo workflow blocks in source order."""
    text = (content or "").strip()
    entries: list[tuple[str, str]] = []
    current: str | None = None
    block_start = 0
    code_ranges = _markdown_code_ranges(text)

    for match in TAG_PATTERN.finditer(text):
        if _inside_ranges(match.start(), code_ranges):
            continue
        tag_name = _canonical_workflow_section_name(match.group(1))
        if tag_name is None:
            continue
        is_close = match.group(0).startswith("</")
        if current is None:
            if not is_close:
                current = tag_name
                block_start = match.end()
            continue
        if is_close and tag_name == current:
            entries.append((current, text[block_start:match.start()].strip()))
            current = None
            continue
        if not is_close:
            nested_content = text[block_start:match.start()].strip()
            entries.append((current, nested_content))
            current = tag_name
            block_start = match.end()

    if current is not None and allow_unclosed:
        entries.append((current, text[block_start:].strip()))

    return entries


def _canonical_workflow_section_name(name: str) -> str | None:
    cleaned = name.lower()
    if not cleaned.startswith("echo_"):
        return None
    canonical = cleaned[len("echo_") :]
    return canonical if canonical in WORKFLOW_SECTION_NAMES else None


def contains_pattern_outside_markdown_code(content: str | None, pattern: Pattern[str]) -> bool:
    """Return whether one regex match appears outside Markdown code spans/fences."""
    text = content or ""
    code_ranges = _markdown_code_ranges(text)
    return any(not _inside_ranges(match.start(), code_ranges) for match in pattern.finditer(text))


def render_workflow_section(name: str, content: str) -> str:
    """Render one current Echo workflow block."""
    cleaned = name.strip().lower()
    if cleaned not in WORKFLOW_SECTION_NAMES:
        raise ValueError(f"Unknown workflow section '{name}'.")
    return f"<echo_{cleaned}>\n{content.strip()}\n</echo_{cleaned}>".strip()


def render_workflow_sections(entries: list[tuple[str, str]]) -> str:
    """Render current Echo workflow blocks in source order."""
    return "\n\n".join(render_workflow_section(name, content) for name, content in entries).strip()


def _markdown_code_ranges(text: str) -> list[tuple[int, int]]:
    ranges = _fenced_code_ranges(text)
    ranges.extend(_inline_code_ranges(text, ranges))
    return _merge_ranges(ranges)


def _fenced_code_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        line_end = text.find("\n", index)
        if line_end < 0:
            line_end = len(text)
            next_index = len(text)
        else:
            next_index = line_end + 1

        line = text[index:line_end]
        opener = re.match(r"[ \t]{0,3}(```+)", line)
        if opener is None:
            index = next_index
            continue

        marker_length = len(opener.group(1))
        close_index = next_index
        end = len(text)
        closer_pattern = re.compile(rf"[ \t]{{0,3}}`{{{marker_length},}}")
        while close_index < len(text):
            close_line_end = text.find("\n", close_index)
            if close_line_end < 0:
                close_line_end = len(text)
                close_next_index = len(text)
            else:
                close_next_index = close_line_end + 1
            if closer_pattern.match(text[close_index:close_line_end]):
                end = close_next_index
                break
            close_index = close_next_index

        ranges.append((index, end))
        index = end
    return ranges


def _inline_code_ranges(text: str, excluded_ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        excluded_end = _containing_range_end(index, excluded_ranges)
        if excluded_end is not None:
            index = excluded_end
            continue
        if text[index] != "`":
            index += 1
            continue

        marker_end = index + 1
        while marker_end < len(text) and text[marker_end] == "`":
            marker_end += 1
        marker = text[index:marker_end]
        search_index = marker_end
        while True:
            close_index = text.find(marker, search_index)
            if close_index < 0:
                index = marker_end
                break
            excluded_end = _containing_range_end(close_index, excluded_ranges)
            if excluded_end is not None:
                search_index = excluded_end
                continue
            close_end = close_index + len(marker)
            ranges.append((index, close_end))
            index = close_end
            break
    return ranges


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _inside_ranges(index: int, ranges: list[tuple[int, int]]) -> bool:
    return _containing_range_end(index, ranges) is not None


def _containing_range_end(index: int, ranges: list[tuple[int, int]]) -> int | None:
    for start, end in ranges:
        if index < start:
            return None
        if start <= index < end:
            return end
    return None
