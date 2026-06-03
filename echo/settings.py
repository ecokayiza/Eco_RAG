from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

KNOWN_MCP_TOOL_NAMES = (
    "load_skill",
    "date",
    "database_search",
    "web_search",
    "web_fetch",
    "workspace_list_files",
    "workspace_read_file",
    "workspace_write_file",
    "workspace_edit_file",
)
DEFAULT_MCP_ENABLED_TOOLS = ("database_search", "web_search", "web_fetch")


class Config:
    ROOT_DIR = ROOT_DIR
    MODELS_PATH = ROOT_DIR / "models.json"
    SETTINGS_PATH = ROOT_DIR / "settings.json"
    DATABASES_PATH = ROOT_DIR / "databases.json"
    DATA_DIR = ROOT_DIR / "data"
    WORKSPACE_DIR = DATA_DIR / "workspace"
    DB_PATH = ROOT_DIR / "db"
    MEMORY_DIR = ROOT_DIR / "memory"
    CHAT_MEMORY_DIR = MEMORY_DIR / "chat_sessions"
    WORKFLOW_DRAFT_DIR = MEMORY_DIR / "workflow_live"
    MEMORY_ARTIFACTS_DIR = MEMORY_DIR / "artifacts"
    CHAT_ARTIFACTS_DIR = MEMORY_ARTIFACTS_DIR / "chat"
    TEST_FILE_PATH = DATA_DIR / "C1" / "markdown" / "easy-rl-chapter1.md"

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
    LOCAL_EMBEDDING_MODEL_NAME = "Local Qwen3 Embedding"
    LOCAL_EMBEDDING_HOST = "127.0.0.1"
    LOCAL_EMBEDDING_PORT = 8091
    LOCAL_EMBEDDING_BASE_URL = f"http://{LOCAL_EMBEDDING_HOST}:{LOCAL_EMBEDDING_PORT}/v1"
    LOCAL_EMBEDDING_API_KEY = "local-embedding-service"

    @staticmethod
    def get_relative_path(file_path, data_dir=None):
        resolved_data_dir = Path(data_dir) if data_dir is not None else Config.DATA_DIR
        try:
            rel_path = Path(file_path).relative_to(resolved_data_dir)
            rel_path = str(rel_path).replace("\\", "/")
        except ValueError:
            rel_path = Path(file_path).name
        return rel_path


@dataclass(frozen=True)
class AppSettings:
    """Store the small set of runtime settings used by the app."""

    chunk_size: int = Config.CHUNK_SIZE
    chunk_overlap: int = Config.CHUNK_OVERLAP
    max_retrieve_rounds: int = 10
    max_database_search_top_k: int = 5
    max_parallel_tool_calls: int = 3
    use_marker_pdf_loader: bool = True
    default_database_backend: str = "chroma"
    web_search_backend: str = "auto"
    web_fetch_screenshot_mode: bool = False
    mcp_enabled_tools: list[str] = field(default_factory=lambda: list(DEFAULT_MCP_ENABLED_TOOLS))
    enabled_skills: list[str] | None = None
    default_skills: list[str] | None = None


def normalize_app_settings(payload: dict | None = None) -> AppSettings:
    """Normalize partially configured settings into the runtime defaults."""
    data = payload if isinstance(payload, dict) else {}
    chunk_size = _positive_int(data.get("chunk_size"), Config.CHUNK_SIZE)
    chunk_overlap = min(_non_negative_int(data.get("chunk_overlap"), Config.CHUNK_OVERLAP), max(chunk_size - 1, 0))
    return AppSettings(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_retrieve_rounds=_positive_int(data.get("max_retrieve_rounds"), 10),
        max_database_search_top_k=_positive_int(data.get("max_database_search_top_k"), 5),
        max_parallel_tool_calls=_positive_int(data.get("max_parallel_tool_calls"), 3),
        use_marker_pdf_loader=_bool(data.get("use_marker_pdf_loader"), True),
        default_database_backend=_database_backend(data.get("default_database_backend")),
        web_search_backend=_web_search_backend(data.get("web_search_backend")),
        web_fetch_screenshot_mode=_bool(data.get("web_fetch_screenshot_mode"), False),
        mcp_enabled_tools=_mcp_tool_list(data.get("mcp_enabled_tools")),
        enabled_skills=_optional_skill_list(data.get("enabled_skills")),
        default_skills=_optional_skill_list(data.get("default_skills")),
    )


def load_app_settings() -> AppSettings:
    """Load the root settings file and fall back to defaults when missing."""
    path = Config.SETTINGS_PATH
    if not path.exists():
        return AppSettings()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Settings file '{path.name}' must contain a JSON object.")
    return normalize_app_settings(payload)


def save_app_settings(settings: AppSettings | dict) -> AppSettings:
    """Persist the runtime settings document."""
    resolved = normalize_app_settings(asdict(settings) if isinstance(settings, AppSettings) else settings)
    Config.SETTINGS_PATH.write_text(json.dumps(asdict(resolved), ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved


def _positive_int(value: object, default: int) -> int:
    """Keep positive integer settings and fall back for everything else."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value if value > 0 else default


def _non_negative_int(value: object, default: int) -> int:
    """Keep zero or positive integer settings and fall back for everything else."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value if value >= 0 else default


def _bool(value: object, default: bool) -> bool:
    """Keep explicit boolean settings and fall back for everything else."""
    return value if isinstance(value, bool) else default


def _web_search_backend(value: object) -> str:
    """Normalize the selected free web-search backend."""
    if not isinstance(value, str):
        return "auto"
    cleaned = value.strip().lower()
    aliases = {"ddg": "duckduckgo", "duck": "duckduckgo", "baidu_search": "baidu"}
    resolved = aliases.get(cleaned, cleaned)
    return resolved if resolved in {"auto", "duckduckgo", "baidu"} else "auto"


def _database_backend(value: object) -> str:
    """Normalize the default vector database backend."""
    if not isinstance(value, str):
        return "chroma"
    cleaned = value.strip().lower()
    return cleaned if cleaned in {"chroma", "faiss"} else "chroma"


def _optional_skill_list(value: object) -> list[str] | None:
    """Keep explicit skill-name lists while preserving missing/null defaults."""
    if value is None:
        return None
    if not isinstance(value, list):
        return None

    skills: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        skills.append(cleaned)
    return skills


def _mcp_tool_list(value: object) -> list[str]:
    """Keep configured MCP tools in a stable known order."""
    if not isinstance(value, list):
        return list(DEFAULT_MCP_ENABLED_TOOLS)

    requested = {item.strip() for item in value if isinstance(item, str) and item.strip()}
    return [name for name in KNOWN_MCP_TOOL_NAMES if name in requested]
