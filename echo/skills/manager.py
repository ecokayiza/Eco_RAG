from __future__ import annotations

import shutil
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Final

import yaml

from ..settings import AppSettings, load_app_settings, save_app_settings

SKILLS_DIR = Path(__file__).resolve().parent
SKILL_COMMAND_PATTERN = re.compile(r"^/skill\s+([A-Za-z0-9_-]+)(?:\s+(.*))?$", re.IGNORECASE | re.DOTALL)
FRONTMATTER_PATTERN = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)(.*)\Z", re.DOTALL)
DEFAULT_SKILLS: Final[tuple[str, ...]] = ("search",)
BUILT_IN_SKILLS: Final[tuple[str, ...]] = ("search", "workspace-files")
_DEFAULT_SKILL_SET: Final[frozenset[str]] = frozenset(DEFAULT_SKILLS)
_BUILT_IN_SKILL_SET: Final[frozenset[str]] = frozenset(BUILT_IN_SKILLS)
_VALID_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*[a-z0-9]$|^[a-z0-9]$")


@dataclass(frozen=True)
class SkillMetadata:
    """Metadata read from one standard skill folder."""

    name: str
    description: str


@dataclass(frozen=True)
class SkillRecord:
    """One editable skill plus runtime selection flags."""

    name: str
    description: str
    content: str
    enabled: bool = True
    default: bool = False
    protected: bool = False


@dataclass(frozen=True)
class SkillSettingsDocument:
    """Full skill management document returned to the web UI."""

    skills: list[SkillRecord]


def list_available_skills() -> list[str]:
    """List all concrete standard skill names."""
    settings = load_app_settings()
    enabled = _enabled_skill_names(settings)
    discovered = {metadata.name for metadata in list_skill_metadata() if metadata.name in enabled}
    ordered = [skill for skill in BUILT_IN_SKILLS if skill in discovered]
    extras = sorted(discovered - _BUILT_IN_SKILL_SET)
    return [*ordered, *extras]


def list_skill_metadata() -> list[SkillMetadata]:
    """List standard skill metadata with defaults first."""
    discovered: dict[str, SkillMetadata] = {}
    for path in _skill_paths():
        metadata = _load_skill(path).metadata
        discovered[metadata.name] = metadata
    ordered = [discovered[name] for name in BUILT_IN_SKILLS if name in discovered]
    extras = [discovered[name] for name in sorted(discovered) if name not in _BUILT_IN_SKILL_SET]
    return [*ordered, *extras]


def load_skill_document(skill_name: str) -> tuple[str, str]:
    """Load one standard skill document and return the normalized name plus markdown."""
    normalized = _normalize_skill_name(skill_name)
    enabled = _enabled_skill_names(load_app_settings())
    candidates = [
        normalized,
        normalized.replace("_", "-"),
    ]
    for candidate in dict.fromkeys(candidates):
        path = SKILLS_DIR / candidate / "SKILL.md"
        if path.exists():
            skill = _load_skill(path)
            if skill.metadata.name not in enabled:
                break
            return skill.metadata.name, skill.content
    available = ", ".join(list_available_skills()) or "none"
    raise ValueError(f"Unknown skill '{skill_name}'. Available skills: {available}.")


def list_default_skills() -> list[str]:
    """List enabled skills whose full documents should be injected by default."""
    settings = load_app_settings()
    enabled = _enabled_skill_names(settings)
    configured = settings.default_skills if settings.default_skills is not None else list(DEFAULT_SKILLS)
    discovered = {metadata.name for metadata in list_skill_metadata()}
    defaults = [_normalize_skill_name(skill) for skill in configured]
    return [skill for skill in dict.fromkeys(defaults) if skill in discovered and skill in enabled]


def load_skill_settings_document() -> SkillSettingsDocument:
    """Load the editable skill management document."""
    settings = load_app_settings()
    enabled = _enabled_skill_names(settings)
    defaults = set(list_default_skills())
    records = [
        SkillRecord(
            name=loaded.metadata.name,
            description=loaded.metadata.description,
            content=loaded.content,
            enabled=loaded.metadata.name in enabled,
            default=loaded.metadata.name in defaults,
            protected=loaded.metadata.name in _BUILT_IN_SKILL_SET,
        )
        for loaded in _load_all_skills()
    ]
    return SkillSettingsDocument(skills=_order_skill_records(records))


def save_skill_settings_document(document: SkillSettingsDocument | dict) -> SkillSettingsDocument:
    """Persist skill files plus enabled/default runtime settings."""
    payload = document if isinstance(document, dict) else {"skills": [record.__dict__ for record in document.skills]}
    raw_skills = payload.get("skills") if isinstance(payload, dict) else None
    if not isinstance(raw_skills, list):
        raise ValueError("Skill settings document must include a skills list.")

    existing = {loaded.metadata.name: loaded for loaded in _load_all_skills()}
    submitted = [_normalize_skill_record(item) for item in raw_skills if isinstance(item, dict)]
    names = [record.name for record in submitted]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"Duplicate skill name(s): {', '.join(duplicate_names)}.")

    submitted_by_name = {record.name: record for record in submitted}
    missing_protected = sorted(skill for skill in _BUILT_IN_SKILL_SET if skill in existing and skill not in submitted_by_name)
    if missing_protected:
        raise ValueError(f"Protected skill(s) cannot be deleted: {', '.join(missing_protected)}.")

    for name in sorted(set(existing) - set(submitted_by_name)):
        if name in _BUILT_IN_SKILL_SET:
            raise ValueError(f"Protected skill '{name}' cannot be deleted.")
        _delete_skill(name)

    for record in submitted:
        _write_skill(record.name, record.description, record.content)

    enabled = [record.name for record in submitted if record.enabled]
    defaults = [record.name for record in submitted if record.enabled and record.default]
    current_settings = load_app_settings()
    save_app_settings(
        AppSettings(
            chunk_size=current_settings.chunk_size,
            chunk_overlap=current_settings.chunk_overlap,
            max_retrieve_rounds=current_settings.max_retrieve_rounds,
            use_marker_pdf_loader=current_settings.use_marker_pdf_loader,
            default_database_backend=current_settings.default_database_backend,
            web_search_backend=current_settings.web_search_backend,
            web_fetch_screenshot_mode=current_settings.web_fetch_screenshot_mode,
            mcp_enabled_tools=current_settings.mcp_enabled_tools,
            enabled_skills=enabled,
            default_skills=defaults,
        )
    )
    return load_skill_settings_document()


def extract_requested_skill(query: str) -> tuple[str | None, str]:
    """Extract `/skill name` from the user query and return the normalized skill plus remaining text."""
    cleaned = " ".join((query or "").strip().split())
    if not cleaned:
        return None, ""

    match = SKILL_COMMAND_PATTERN.match(cleaned)
    if match is None:
        return None, cleaned

    skill_name = _normalize_skill_name(match.group(1))
    remaining = " ".join((match.group(2) or "").strip().split())
    return skill_name or None, remaining or cleaned


def _normalize_skill_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_-]+", "-", (value or "").strip().lower())
    return text.strip("-_")


@dataclass(frozen=True)
class _LoadedSkill:
    metadata: SkillMetadata
    content: str


def _skill_paths() -> list[Path]:
    return sorted(path for path in SKILLS_DIR.glob("*/SKILL.md") if path.is_file())


def _load_all_skills() -> list[_LoadedSkill]:
    return [_load_skill(path) for path in _skill_paths()]


def _load_skill(path: Path) -> _LoadedSkill:
    raw = path.read_text(encoding="utf-8")
    match = FRONTMATTER_PATTERN.match(raw)
    if match is None:
        raise ValueError(f"Skill '{path}' must start with YAML frontmatter.")

    payload = yaml.safe_load(match.group(1)) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Skill '{path}' frontmatter must be a mapping.")

    name = _normalize_skill_name(str(payload.get("name") or ""))
    description = " ".join(str(payload.get("description") or "").split())
    folder_name = path.parent.name
    if not name:
        raise ValueError(f"Skill '{path}' frontmatter must include a name.")
    if not description:
        raise ValueError(f"Skill '{path}' frontmatter must include a description.")
    if name != folder_name:
        raise ValueError(f"Skill '{path}' frontmatter name must match folder name '{folder_name}'.")

    content = match.group(2).strip()
    if not content:
        raise ValueError(f"Skill '{path}' must include markdown instructions after frontmatter.")
    return _LoadedSkill(metadata=SkillMetadata(name=name, description=description), content=content)


def _enabled_skill_names(settings: AppSettings) -> set[str]:
    discovered = {metadata.name for metadata in list_skill_metadata()}
    if settings.enabled_skills is None:
        return discovered
    enabled = {_normalize_skill_name(skill) for skill in settings.enabled_skills}
    return discovered & enabled


def _normalize_skill_record(payload: dict) -> SkillRecord:
    raw_name = str(payload.get("name") or "")
    name = _normalize_skill_name(raw_name)
    if not name or name != raw_name.strip().lower():
        raise ValueError(f"Invalid skill name '{raw_name}'. Use lowercase letters, numbers, dashes, or underscores.")
    if not _VALID_SKILL_NAME_PATTERN.match(name):
        raise ValueError(f"Invalid skill name '{raw_name}'. Use lowercase letters, numbers, dashes, or underscores.")

    description = " ".join(str(payload.get("description") or "").split())
    content = str(payload.get("content") or "").strip()
    if not description:
        raise ValueError(f"Skill '{name}' must include a description.")
    if not content:
        raise ValueError(f"Skill '{name}' must include markdown content.")

    existing_protected = name in _BUILT_IN_SKILL_SET
    return SkillRecord(
        name=name,
        description=description,
        content=content,
        enabled=bool(payload.get("enabled", True)),
        default=bool(payload.get("default", False)),
        protected=existing_protected,
    )


def _write_skill(name: str, description: str, content: str):
    path = _skill_document_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.safe_dump(
        {"name": name, "description": description},
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).strip()
    path.write_text(f"---\n{frontmatter}\n---\n\n{content.strip()}\n", encoding="utf-8")


def _delete_skill(name: str):
    path = _skill_document_path(name)
    skill_dir = path.parent
    if name in _BUILT_IN_SKILL_SET:
        raise ValueError(f"Protected skill '{name}' cannot be deleted.")
    if skill_dir.exists():
        shutil.rmtree(skill_dir)


def _skill_document_path(name: str) -> Path:
    normalized = _normalize_skill_name(name)
    if not normalized or normalized != name:
        raise ValueError(f"Invalid skill name '{name}'.")
    path = (SKILLS_DIR / normalized / "SKILL.md").resolve()
    root = SKILLS_DIR.resolve()
    if path.parent.parent != root:
        raise ValueError(f"Invalid skill path for '{name}'.")
    return path


def _order_skill_records(records: list[SkillRecord]) -> list[SkillRecord]:
    by_name = {record.name: record for record in records}
    ordered = [by_name[name] for name in BUILT_IN_SKILLS if name in by_name]
    ordered.extend(by_name[name] for name in sorted(by_name) if name not in _BUILT_IN_SKILL_SET)
    return ordered
