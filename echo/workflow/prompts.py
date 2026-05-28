from __future__ import annotations

from pathlib import Path

import yaml

from ..skills import list_available_skills, list_default_skills, list_skill_metadata, load_skill_document

PROMPT_DIR = Path(__file__).with_name("prompt_templates")


def default_system_prompt(
    *,
    available_skills: list[str] | tuple[str, ...] = (),
) -> str:
    """Render the shared session-level system prompt."""
    resolved_available_skills = list(available_skills) if available_skills else list_available_skills()
    default_skills = list_default_skills()
    loadable_skills = [skill for skill in resolved_available_skills if skill not in set(default_skills)]
    return _template("system").format(
        default_skills=_default_skill_documents(default_skills),
        available_skills=_available_skills_document(loadable_skills, default_skills=default_skills),
    ).strip()


def _template(name: str) -> str:
    """Load one workflow prompt template from YAML."""
    path = PROMPT_DIR / f"{name}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "content" not in payload:
        raise ValueError(f"Workflow prompt template '{path.name}' must contain a 'content' field.")
    return str(payload["content"])


def _default_skill_documents(default_skills: list[str] | None = None) -> str:
    """Render the bundled markdown for all default skills."""
    documents = []
    for skill_name in default_skills if default_skills is not None else list_default_skills():
        _, content = load_skill_document(skill_name)
        if content.strip():
            documents.append(content.strip())
    return "\n\n".join(documents) if documents else "(none)"


def _skills_catalog_document(skills: list[str], *, default_skills: list[str]) -> str:
    """Render the shared skills index from discovered skill metadata."""
    skill_names = set(skills)
    metadata = [skill for skill in list_skill_metadata() if skill.name in skill_names]
    if not metadata:
        loaded = ", ".join(f"`{skill}`" for skill in default_skills) or "none"
        return (
            "# Available Skills\n\n"
            f"No non-default skills are available. Default skills are already loaded and must not be loaded again: {loaded}."
        )
    lines = [
        "# Available Skills",
        "",
        "These non-default skills are available to the workflow retrieve agent. Load a full skill only when its guidance is needed for the current turn.",
        "",
        f"Default skills are already loaded and must not be loaded again: {', '.join(f'`{skill}`' for skill in default_skills) or 'none'}.",
        "",
        "Use a provider-native `load_skill(skill_name)` tool call to load one of the available skills.",
        "",
    ]
    lines.extend(f"- `{skill.name}`: {skill.description}" for skill in metadata)
    return "\n".join(lines).strip()


def _available_skills_document(skills: list[str], *, default_skills: list[str]) -> str:
    """Render the skills index, appending any discovered extras if needed."""
    catalog = _skills_catalog_document(skills, default_skills=default_skills)
    extras = [skill for skill in skills if f"`{skill}`" not in catalog]
    if not extras:
        return catalog
    extra_lines = "\n".join(f"- `{skill}`" for skill in extras)
    return f"{catalog}\n\nAdditional discovered skills:\n{extra_lines}".strip()
