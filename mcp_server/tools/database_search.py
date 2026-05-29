from __future__ import annotations

from typing import Any

# Keep these imports at module load time. Lazy importing the RAG stack inside
# the stdio MCP request path can hang under the Windows child process.
from mcp_server.rag import database_registry, embedder, schema, vector_database

DEFAULT_DATABASE_SEARCH_TOP_K = 4
MAX_DATABASE_SEARCH_TOP_K = 5


def _clamp_top_k(top_k: Any, default: int = DEFAULT_DATABASE_SEARCH_TOP_K) -> int:
    try:
        requested = int(top_k or default)
    except (TypeError, ValueError):
        requested = default
    return max(1, min(requested, MAX_DATABASE_SEARCH_TOP_K))


def database_search(query: str, top_k: int = 4) -> dict[str, Any]:
    """Search the local Echo vector database for semantically related chunks."""
    cleaned = " ".join((query or "").strip().split())
    limit = _clamp_top_k(top_k)
    if not cleaned:
        return {
            "type": "context",
            "skill_name": "database_search",
            "items": [],
            "error": "Query cannot be empty.",
        }

    try:
        database = database_registry.get_active_database_settings()
        embedding_settings = database_registry.resolve_database_embedding_settings(database)
        query_embedding = embedder.OpenAICompatibleEmbedder.embed_query(cleaned, settings=embedding_settings)
        vector_db = vector_database.VectorDatabase(collection_name=database.collection_name, backend=database.backend)
        results = vector_db.query_with_vector(
            query_embedding,
            n_results=limit,
        )
        records = schema.RAGRecord.get_records_from_results(results)
        items = [
            {
                "title": record.metadata.source_name,
                "content": record.document,
                "source_type": record.metadata.source_type,
                "file_path": record.metadata.attributes.file_path,
                "url": record.metadata.attributes.url,
                "distance": record.distance,
                "database_name": database.name,
            }
            for record in records
        ]
        return {
            "type": "context",
            "skill_name": "database_search",
            "items": items,
        }
    except Exception as exc:
        return {
            "type": "context",
            "skill_name": "database_search",
            "items": [],
            "error": str(exc),
        }
