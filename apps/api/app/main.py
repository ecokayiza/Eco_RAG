import asyncio
import copy
import json
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from echo.chat import ChatService
from echo.chat.registry import (
    ChatModelSettings,
    EmbeddingModelSettings,
    ModelSettingsDocument,
    build_chat_model,
    get_active_chat_model_settings,
    load_model_settings_document,
    normalize_chat_model_settings,
    normalize_embedding_model_settings,
    normalize_model_settings_document,
    save_model_settings_document,
)
from echo.settings import DEFAULT_MCP_ENABLED_TOOLS, Config
from mcp_server.rag import (
    Assembler,
    ChunkerFactory,
    DataLoaderFactory,
    OpenAICompatibleEmbedder,
    VectorDatabase,
    create_database_settings,
    delete_database_settings,
    ensure_database_settings_document,
    list_database_settings,
    rename_database_settings,
    resolve_database_embedding_settings,
    select_database_settings,
)
from mcp_server.client import SharedMCPToolClient
from mcp_server.rag.errors import IndexingError
from echo.settings import AppSettings, load_app_settings, save_app_settings
from echo.skills import SkillRecord, SkillSettingsDocument, load_skill_settings_document, save_skill_settings_document
from echo.workflow import WorkflowStatus, WorkflowStep
from echo.workflow.prompts import default_system_prompt

WEB_DIR = Config.ROOT_DIR / "apps" / "web"
WEB_DIST_DIR = WEB_DIR / "dist"
SUPPORTED_UPLOAD_EXTENSIONS = {".md", ".pdf", ".txt"}


class SessionSummaryResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    preview: str = ""
    token_usage: dict[str, Any] = Field(default_factory=dict)
    total_tokens: int = 0


class SessionStateResponse(BaseModel):
    session: SessionSummaryResponse
    messages: list[dict[str, Any]]
    workflow_draft: dict[str, Any] | None = None


class DatabaseSummaryResponse(BaseModel):
    id: str
    name: str
    collection_name: str
    backend: str
    embedding_model_name: str
    document_count: int
    created_at: str
    updated_at: str


class DatabaseStateResponse(BaseModel):
    active_database_id: str | None
    databases: list[DatabaseSummaryResponse]


class DatabaseDocumentSummaryResponse(BaseModel):
    id: str
    source_name: str
    source_type: str
    file_path: str | None = None
    chunk_count: int = 0


class UploadJobFileResponse(BaseModel):
    id: str
    source_name: str
    status: str
    chunk_count: int = 0
    embedded_chunks: int = 0
    progress: float = 0.0
    message: str = ""
    error: str | None = None


class UploadJobResponse(BaseModel):
    job_id: str
    database_id: str
    status: str
    message: str = ""
    progress: float = 0.0
    total_files: int = 0
    completed_files: int = 0
    total_chunks: int = 0
    embedded_chunks: int = 0
    current_file_name: str | None = None
    files: list[UploadJobFileResponse] = Field(default_factory=list)
    error: str | None = None
    error_stage: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class CreateSessionRequest(BaseModel):
    title: str | None = None
    session_id: str | None = None


class UpdateSessionRequest(BaseModel):
    title: str = Field(min_length=1)


class UpdateSystemPromptRequest(BaseModel):
    content: str | None = None


class CreateDatabaseRequest(BaseModel):
    name: str | None = None
    embedding_model_name: str | None = None
    backend: str | None = None


class UpdateDatabaseRequest(BaseModel):
    name: str = Field(min_length=1)


class UpdateDatabaseDocumentRequest(BaseModel):
    source_name: str = Field(min_length=1)


class ChatModelSettingsRequest(BaseModel):
    name: str = "Default Chat Model"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    wire_api: str = "chat_completions"
    temperature: float = 1.0
    top_p: float | None = None
    custom_request_params: dict[str, Any] | None = None

    def to_settings(self) -> ChatModelSettings:
        return normalize_chat_model_settings(self.model_dump())

    @classmethod
    def from_settings(cls, settings: ChatModelSettings):
        return cls(**settings.__dict__)


class EmbeddingModelSettingsRequest(BaseModel):
    name: str = "Default Embedding Model"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int | None = Field(default=None, ge=1)

    def to_settings(self) -> EmbeddingModelSettings:
        return normalize_embedding_model_settings(self.model_dump())

    @classmethod
    def from_settings(cls, settings: EmbeddingModelSettings):
        return cls(**settings.__dict__)


class ModelSettingsDocumentRequest(BaseModel):
    active_chat_model: str | None = None
    active_embedding_model: str | None = None
    chat_models: list[ChatModelSettingsRequest] = Field(default_factory=list)
    embedding_models: list[EmbeddingModelSettingsRequest] = Field(default_factory=list)

    def to_document(self) -> ModelSettingsDocument:
        return normalize_model_settings_document(
            {
                "active_chat_model": self.active_chat_model,
                "active_embedding_model": self.active_embedding_model,
                "chat_models": [item.model_dump() for item in self.chat_models],
                "embedding_models": [item.model_dump() for item in self.embedding_models],
            }
        )

    @classmethod
    def from_document(cls, document: ModelSettingsDocument):
        return cls(
            active_chat_model=document.active_chat_model,
            active_embedding_model=document.active_embedding_model,
            chat_models=[ChatModelSettingsRequest.from_settings(item) for item in document.chat_models],
            embedding_models=[EmbeddingModelSettingsRequest.from_settings(item) for item in document.embedding_models],
        )


class ModelApiTestRequest(BaseModel):
    kind: str
    chat_model: ChatModelSettingsRequest | None = None
    embedding_model: EmbeddingModelSettingsRequest | None = None


class ModelApiTestResponse(BaseModel):
    ok: bool
    message: str


class AppSettingsRequest(BaseModel):
    chunk_size: int = Field(default=Config.CHUNK_SIZE, ge=1)
    chunk_overlap: int = Field(default=Config.CHUNK_OVERLAP, ge=0)
    max_retrieve_rounds: int = Field(default=10, ge=1)
    use_marker_pdf_loader: bool = True
    default_database_backend: str = "chroma"
    web_search_backend: str = "auto"
    web_fetch_screenshot_mode: bool = False
    mcp_enabled_tools: list[str] = Field(default_factory=lambda: list(DEFAULT_MCP_ENABLED_TOOLS))

    def to_settings(self) -> AppSettings:
        return AppSettings(**self.model_dump())

    @classmethod
    def from_settings(cls, settings: AppSettings):
        return cls(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            max_retrieve_rounds=settings.max_retrieve_rounds,
            use_marker_pdf_loader=settings.use_marker_pdf_loader,
            default_database_backend=settings.default_database_backend,
            web_search_backend=settings.web_search_backend,
            web_fetch_screenshot_mode=settings.web_fetch_screenshot_mode,
            mcp_enabled_tools=settings.mcp_enabled_tools,
        )


class SkillRecordRequest(BaseModel):
    name: str
    description: str
    content: str
    enabled: bool = True
    default: bool = False
    protected: bool = False

    def to_record(self) -> SkillRecord:
        return SkillRecord(
            name=self.name,
            description=self.description,
            content=self.content,
            enabled=self.enabled,
            default=self.default,
            protected=self.protected,
        )

    @classmethod
    def from_record(cls, record: SkillRecord):
        return cls(**record.__dict__)


class SkillSettingsDocumentRequest(BaseModel):
    skills: list[SkillRecordRequest] = Field(default_factory=list)

    def to_document(self) -> SkillSettingsDocument:
        return SkillSettingsDocument(skills=[item.to_record() for item in self.skills])

    @classmethod
    def from_document(cls, document: SkillSettingsDocument):
        return cls(skills=[SkillRecordRequest.from_record(item) for item in document.skills])


class SendMessageRequest(BaseModel):
    message: str = Field(min_length=1)
    system_prompt: str | None = None


class UpdateMessageRequest(BaseModel):
    content: str = Field(min_length=1)


def to_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(chat_service: ChatService | None = None):
    ensure_database_settings_document()
    shared_tool_client = None if chat_service is not None else SharedMCPToolClient()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if shared_tool_client is not None:
            await shared_tool_client.connect()
            try:
                build_chat_model()
            except Exception:
                pass
        try:
            yield
        finally:
            if shared_tool_client is not None:
                await shared_tool_client.aclose()

    app = FastAPI(
        title="Echo API",
        version="0.1.0",
        description="Backend entrypoint for the Echo desktop and web clients.",
        lifespan=lifespan,
    )
    service = chat_service or ChatService(tool_client_factory=shared_tool_client.context if shared_tool_client else None)
    upload_jobs: dict[str, dict[str, Any]] = _load_upload_jobs()
    upload_jobs_lock = Lock()

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/ui/")

    @app.get("/api/health")
    def health():
        active_chat_model = get_active_chat_model_settings(required=False)
        return {"status": "ok", "model": active_chat_model.model if active_chat_model else None}

    @app.get("/api/meta")
    def meta():
        prompt = service.default_system_prompt() if hasattr(service, "default_system_prompt") else default_system_prompt()
        return {
            "workflow_statuses": [status.value for status in WorkflowStatus],
            "workflow_steps": [step.value for step in WorkflowStep],
            "default_system_prompt": prompt,
        }

    @app.get("/api/model-settings", response_model=ModelSettingsDocumentRequest)
    def get_model_settings():
        return ModelSettingsDocumentRequest.from_document(load_model_settings_document())

    @app.put("/api/model-settings", response_model=ModelSettingsDocumentRequest)
    def update_model_settings(payload: ModelSettingsDocumentRequest):
        return ModelSettingsDocumentRequest.from_document(save_model_settings_document(payload.to_document()))

    @app.post("/api/model-settings/test", response_model=ModelApiTestResponse)
    async def test_model_settings(payload: ModelApiTestRequest):
        kind = payload.kind.strip().lower()
        try:
            if kind == "chat":
                if payload.chat_model is None:
                    raise ValueError("Chat model settings are required.")
                model = normalize_chat_model_settings(payload.chat_model.model_dump())
                response = await build_chat_model(model).generate_response(
                    [{"role": "user", "content": "Reply with only OK."}]
                )
                preview = (response.content or "").strip()
                suffix = f": {preview[:80]}" if preview else "."
                return ModelApiTestResponse(
                    ok=True,
                    message=f"Chat API test succeeded{suffix}",
                )

            if kind == "embedding":
                if payload.embedding_model is None:
                    raise ValueError("Embedding model settings are required.")
                settings = normalize_embedding_model_settings(payload.embedding_model.model_dump())
                embeddings = OpenAICompatibleEmbedder.embed_documents("Echo API test", settings=settings)
                dimensions = len(embeddings[0]) if embeddings else 0
                if dimensions <= 0:
                    raise ValueError("Embedding provider returned an empty vector.")
                return ModelApiTestResponse(ok=True, message=f"Embedding API test succeeded. Vector dimensions: {dimensions}.")

            raise ValueError("Model test kind must be 'chat' or 'embedding'.")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/app-settings", response_model=AppSettingsRequest)
    def get_app_settings():
        return AppSettingsRequest.from_settings(load_app_settings())

    @app.put("/api/app-settings", response_model=AppSettingsRequest)
    def update_app_settings(payload: AppSettingsRequest):
        current = load_app_settings()
        next_settings = AppSettings(
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            max_retrieve_rounds=payload.max_retrieve_rounds,
            use_marker_pdf_loader=payload.use_marker_pdf_loader,
            default_database_backend=payload.default_database_backend,
            web_search_backend=payload.web_search_backend,
            web_fetch_screenshot_mode=payload.web_fetch_screenshot_mode,
            mcp_enabled_tools=payload.mcp_enabled_tools,
            enabled_skills=current.enabled_skills,
            default_skills=current.default_skills,
        )
        return AppSettingsRequest.from_settings(save_app_settings(next_settings))

    @app.get("/api/skills", response_model=SkillSettingsDocumentRequest)
    def get_skill_settings():
        try:
            return SkillSettingsDocumentRequest.from_document(load_skill_settings_document())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/skills", response_model=SkillSettingsDocumentRequest)
    def update_skill_settings(payload: SkillSettingsDocumentRequest):
        try:
            return SkillSettingsDocumentRequest.from_document(save_skill_settings_document(payload.to_document()))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/databases", response_model=DatabaseStateResponse)
    def list_databases():
        return DatabaseStateResponse(**_database_state_payload())

    @app.post("/api/databases", response_model=DatabaseStateResponse)
    def create_database(payload: CreateDatabaseRequest | None = None):
        request = payload or CreateDatabaseRequest()
        backend = request.backend or load_app_settings().default_database_backend
        try:
            create_database_settings(
                name=request.name,
                embedding_model_name=request.embedding_model_name,
                backend=backend,
                select=True,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return DatabaseStateResponse(**_database_state_payload())

    @app.patch("/api/databases/{database_id}", response_model=DatabaseStateResponse)
    def update_database(database_id: str, payload: UpdateDatabaseRequest):
        try:
            rename_database_settings(database_id, payload.name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return DatabaseStateResponse(**_database_state_payload())

    @app.post("/api/databases/{database_id}/select", response_model=DatabaseStateResponse)
    def select_database(database_id: str):
        try:
            select_database_settings(database_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return DatabaseStateResponse(**_database_state_payload())

    @app.delete("/api/databases/{database_id}", response_model=DatabaseStateResponse)
    def delete_database(database_id: str):
        document = list_database_settings()
        database = next((item for item in document.databases if item.id == database_id), None)
        try:
            delete_database_settings(database_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if database is not None:
            try:
                _vector_database_for(database).delete_collection()
            except Exception:
                pass
        return DatabaseStateResponse(**_database_state_payload())

    @app.post("/api/databases/{database_id}/documents", response_model=DatabaseStateResponse)
    async def upload_database_documents(
        database_id: str,
        files: list[UploadFile] = File(...),
        skip_existing: bool = Form(True),
    ):
        document = list_database_settings()
        database = next((item for item in document.databases if item.id == database_id), None)
        if database is None:
            raise HTTPException(status_code=404, detail="Database not found.")
        if not files:
            raise HTTPException(status_code=400, detail="Select at least one file to upload.")

        vector_db = _vector_database_for(database)
        assembler = Assembler(
            vector_db,
            DataLoaderFactory(),
            ChunkerFactory(),
            OpenAICompatibleEmbedder(),
            database=database,
        )

        try:
            for upload in files:
                saved_path = await _save_uploaded_document(upload, database.collection_name)
                if skip_existing and _database_has_uploaded_file(
                    vector_db,
                    str(saved_path),
                ):
                    continue
                assembler.delete_file(str(saved_path))
                assembler.store_file(str(saved_path))
        except IndexingError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return DatabaseStateResponse(**_database_state_payload())

    @app.post("/api/databases/{database_id}/documents/jobs", response_model=UploadJobResponse)
    async def create_database_upload_job(
        database_id: str,
        files: list[UploadFile] = File(...),
        skip_existing: bool = Form(True),
    ):
        document = list_database_settings()
        database = next((item for item in document.databases if item.id == database_id), None)
        if database is None:
            raise HTTPException(status_code=404, detail="Database not found.")
        if not files:
            raise HTTPException(status_code=400, detail="Select at least one file to upload.")

        saved_paths = [await _save_uploaded_document(upload, database.collection_name) for upload in files]
        job_id = str(uuid4())
        job = _create_upload_job_state(job_id, database_id, saved_paths)
        with upload_jobs_lock:
            upload_jobs[job_id] = job
            _persist_upload_jobs(upload_jobs)
        asyncio.create_task(
            asyncio.to_thread(
                _process_database_upload_job,
                upload_jobs,
                upload_jobs_lock,
                job_id,
                database,
                [str(path) for path in saved_paths],
                skip_existing,
            )
        )
        return UploadJobResponse(**_get_upload_job_or_404(upload_jobs, upload_jobs_lock, database_id, job_id))

    @app.get("/api/databases/{database_id}/documents/jobs/current", response_model=UploadJobResponse | None)
    def get_current_database_upload_job(database_id: str):
        job = _get_current_upload_job(upload_jobs, upload_jobs_lock, database_id)
        return UploadJobResponse(**job) if job else None

    @app.get("/api/databases/{database_id}/documents/jobs/{job_id}", response_model=UploadJobResponse)
    def get_database_upload_job(database_id: str, job_id: str):
        return UploadJobResponse(**_get_upload_job_or_404(upload_jobs, upload_jobs_lock, database_id, job_id))

    @app.get("/api/databases/{database_id}/documents", response_model=list[DatabaseDocumentSummaryResponse])
    def list_database_documents(database_id: str):
        document = list_database_settings()
        database = next((item for item in document.databases if item.id == database_id), None)
        if database is None:
            raise HTTPException(status_code=404, detail="Database not found.")
        try:
            return _vector_database_for(database).list_document_summaries()
        except Exception:
            return []

    @app.patch("/api/databases/{database_id}/documents/{document_id:path}", response_model=list[DatabaseDocumentSummaryResponse])
    def update_database_document(database_id: str, document_id: str, payload: UpdateDatabaseDocumentRequest):
        document = list_database_settings()
        database = next((item for item in document.databases if item.id == database_id), None)
        if database is None:
            raise HTTPException(status_code=404, detail="Database not found.")

        vector_db = _vector_database_for(database)
        summaries = vector_db.list_document_summaries()
        target = next((item for item in summaries if item.get("id") == document_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail="Document not found.")

        source_name = payload.source_name.strip()
        if not source_name:
            raise HTTPException(status_code=400, detail="Document name cannot be empty.")

        file_path = str(target.get("file_path") or "").strip()
        source_type = str(target.get("source_type") or "").strip()
        where = {"file_path": file_path} if file_path else {"source_name": str(target.get("source_name") or "").strip(), "source_type": source_type}
        updated = vector_db.update_document_metadata(where, {"source_name": source_name})
        if not updated:
            raise HTTPException(status_code=404, detail="Document not found.")

        return vector_db.list_document_summaries()

    @app.delete("/api/databases/{database_id}/documents/{document_id:path}", response_model=list[DatabaseDocumentSummaryResponse])
    def delete_database_document(database_id: str, document_id: str):
        document = list_database_settings()
        database = next((item for item in document.databases if item.id == database_id), None)
        if database is None:
            raise HTTPException(status_code=404, detail="Database not found.")

        vector_db = _vector_database_for(database)
        summaries = vector_db.list_document_summaries()
        target = next((item for item in summaries if item.get("id") == document_id), None)
        if target is None:
            raise HTTPException(status_code=404, detail="Document not found.")

        file_path = str(target.get("file_path") or "").strip()
        source_name = str(target.get("source_name") or "").strip()
        source_type = str(target.get("source_type") or "").strip()

        if file_path:
            vector_db.delete_documents({"file_path": file_path})
            _delete_uploaded_document_file(file_path)
        else:
            vector_db.delete_documents({"source_name": source_name, "source_type": source_type})

        return vector_db.list_document_summaries()

    @app.get("/api/sessions", response_model=list[SessionSummaryResponse])
    def list_sessions():
        return [SessionSummaryResponse(**session) for session in service.list_sessions()]

    @app.post("/api/sessions", response_model=SessionSummaryResponse)
    def create_session(payload: CreateSessionRequest | None = None):
        request = payload or CreateSessionRequest()
        session = service.create_session(
            session_id=request.session_id,
            title=request.title,
        )
        return SessionSummaryResponse(**session)

    @app.get("/api/sessions/{session_id}", response_model=SessionStateResponse)
    def get_session(session_id: str):
        state = service.get_session_state(session_id)
        return SessionStateResponse(**state.to_dict())

    @app.patch("/api/sessions/{session_id}", response_model=SessionSummaryResponse)
    def update_session(session_id: str, payload: UpdateSessionRequest):
        try:
            session = service.update_session_title(session_id, payload.title)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SessionSummaryResponse(**session)

    @app.patch("/api/sessions/{session_id}/system-prompt", response_model=SessionStateResponse)
    async def update_system_prompt(session_id: str, payload: UpdateSystemPromptRequest):
        try:
            state = await service.update_system_prompt(session_id, payload.content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SessionStateResponse(**state.to_dict())

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str):
        service.delete_session(session_id)
        return {"session_id": session_id, "deleted": True}

    @app.post("/api/sessions/{session_id}/messages/stream")
    async def stream_message(session_id: str, payload: SendMessageRequest):
        async def event_stream():
            try:
                async for item in service.stream_message(
                    message=payload.message,
                    session_id=session_id,
                    system_prompt=payload.system_prompt,
                ):
                    yield to_sse(item["event"], item["data"])
            except ValueError as exc:
                yield to_sse("error", {"detail": str(exc)})
            except Exception as exc:
                yield to_sse("error", {"detail": f"Chat request failed: {_exception_detail(exc)}"})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.patch("/api/sessions/{session_id}/messages/{message_id}", response_model=SessionStateResponse)
    async def update_message(session_id: str, message_id: str, payload: UpdateMessageRequest):
        try:
            state = await service.update_message(session_id, message_id, payload.content)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SessionStateResponse(**state.to_dict())

    @app.delete("/api/sessions/{session_id}/messages/{message_id}", response_model=SessionStateResponse)
    async def delete_message(session_id: str, message_id: str):
        try:
            state = await service.delete_message(session_id, message_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SessionStateResponse(**state.to_dict())

    @app.post("/api/sessions/{session_id}/messages/{message_id}/rollback", response_model=SessionStateResponse)
    async def rollback_message(session_id: str, message_id: str):
        try:
            state = await service.rollback_message(session_id, message_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SessionStateResponse(**state.to_dict())

    @app.post("/api/sessions/{session_id}/messages/{message_id}/regenerate/stream")
    async def stream_regenerate_message(session_id: str, message_id: str):
        async def event_stream():
            try:
                async for item in service.stream_regenerate_message(session_id, message_id):
                    yield to_sse(item["event"], item["data"])
            except ValueError as exc:
                yield to_sse("error", {"detail": str(exc)})
            except Exception as exc:
                yield to_sse("error", {"detail": f"Chat request failed: {_exception_detail(exc)}"})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/api/sessions/{session_id}/workflow/continue/stream")
    async def stream_continue_workflow(session_id: str):
        async def event_stream():
            try:
                async for item in service.stream_continue_workflow(session_id):
                    yield to_sse(item["event"], item["data"])
            except ValueError as exc:
                yield to_sse("error", {"detail": str(exc)})
            except Exception as exc:
                yield to_sse("error", {"detail": f"Chat request failed: {_exception_detail(exc)}"})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    Config.CHAT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/api/artifacts", StaticFiles(directory=str(Config.CHAT_ARTIFACTS_DIR)), name="chat_artifacts")

    if WEB_DIST_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(WEB_DIST_DIR), html=True), name="web")

    return app


def _database_state_payload() -> dict[str, Any]:
    document = list_database_settings()
    databases = []
    for database in document.databases:
        try:
            count = _vector_database_for(database).file_count()
        except Exception:
            count = 0
        try:
            embedding_model_name = resolve_database_embedding_settings(database).name
        except Exception:
            embedding_model_name = database.embedding_model_name
        databases.append(
            {
                "id": database.id,
                "name": database.name,
                "collection_name": database.collection_name,
                "backend": database.backend,
                "embedding_model_name": embedding_model_name,
                "document_count": count,
                "created_at": database.created_at,
                "updated_at": database.updated_at,
            }
        )
    return {"active_database_id": document.active_database_id, "databases": databases}


def _vector_database_for(database) -> VectorDatabase:
    return VectorDatabase(collection_name=database.collection_name, backend=database.backend)


def _sanitize_upload_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", (filename or "").strip())
    return cleaned.strip(" .") or f"upload-{uuid4().hex[:8]}.txt"


def _exception_detail(exc: BaseException) -> str:
    """Render nested async ExceptionGroups as the useful inner error text."""
    if isinstance(exc, BaseExceptionGroup):
        messages = [_exception_detail(item) for item in exc.exceptions]
        messages = [message for message in messages if message]
        return "; ".join(messages) or str(exc)
    message = str(exc).strip()
    return message or exc.__class__.__name__


async def _save_uploaded_document(upload: UploadFile, collection_name: str):
    filename = _sanitize_upload_filename((upload.filename or "").split("/")[-1].split("\\")[-1])
    extension = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise ValueError(f"Unsupported file type for '{filename}'. Supported types: {supported}.")

    target_dir = Config.DATA_DIR / "uploads" / collection_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename

    payload = await upload.read()
    if not payload:
        raise ValueError(f"Uploaded file '{filename}' is empty.")
    target_path.write_bytes(payload)
    await upload.close()
    return target_path


def _create_upload_job_state(job_id: str, database_id: str, saved_paths) -> dict[str, Any]:
    now = _utc_now()
    files = [
        {
            "id": str(path),
            "source_name": path.name,
            "status": "queued",
            "chunk_count": 0,
            "embedded_chunks": 0,
            "progress": 0.0,
            "message": "",
            "error": None,
        }
        for path in saved_paths
    ]
    return {
        "job_id": job_id,
        "database_id": database_id,
        "status": "queued",
        "message": f"Queued {len(files)} file(s) for embedding.",
        "progress": 0.0,
        "total_files": len(files),
        "completed_files": 0,
        "total_chunks": 0,
        "embedded_chunks": 0,
        "current_file_name": None,
        "files": files,
        "error": None,
        "error_stage": None,
        "created_at": now,
        "updated_at": now,
    }


def _upload_jobs_path():
    return Config.DATA_DIR / "upload_jobs.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_upload_jobs() -> dict[str, dict[str, Any]]:
    path = _upload_jobs_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    jobs = payload.get("jobs", payload)
    if not isinstance(jobs, dict):
        return {}
    restored: dict[str, dict[str, Any]] = {}
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        if job.get("status") in {"queued", "processing"}:
            job = {
                **job,
                "status": "failed",
                "error": "Upload job was interrupted by server restart.",
                "error_stage": "interrupted",
                "message": "Upload job was interrupted by server restart.",
                "updated_at": _utc_now(),
            }
        restored[str(job_id)] = job
    return restored


def _persist_upload_jobs(upload_jobs: dict[str, dict[str, Any]]):
    path = _upload_jobs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"jobs": upload_jobs}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_upload_job_or_404(
    upload_jobs: dict[str, dict[str, Any]],
    upload_jobs_lock: Lock,
    database_id: str,
    job_id: str,
) -> dict[str, Any]:
    with upload_jobs_lock:
        job = upload_jobs.get(job_id)
        if job is None or job.get("database_id") != database_id:
            raise HTTPException(status_code=404, detail="Upload job not found.")
        return copy.deepcopy(job)


def _get_current_upload_job(
    upload_jobs: dict[str, dict[str, Any]],
    upload_jobs_lock: Lock,
    database_id: str,
) -> dict[str, Any] | None:
    with upload_jobs_lock:
        candidates = [
            job
            for job in upload_jobs.values()
            if job.get("database_id") == database_id and job.get("status") in {"queued", "processing"}
        ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda item: str(item.get("updated_at") or item.get("created_at") or ""))
        return copy.deepcopy(latest)


def _process_database_upload_job(
    upload_jobs: dict[str, dict[str, Any]],
    upload_jobs_lock: Lock,
    job_id: str,
    database,
    saved_paths: list[str],
    skip_existing: bool,
):
    vector_db = _vector_database_for(database)
    assembler = Assembler(
        vector_db,
        DataLoaderFactory(),
        ChunkerFactory(),
        OpenAICompatibleEmbedder(),
        database=database,
    )

    current_path = None
    try:
        _update_upload_job(
            upload_jobs,
            upload_jobs_lock,
            job_id,
            status="processing",
            message="Preparing uploaded documents...",
        )
        for saved_path in saved_paths:
            current_path = saved_path
            source_name = _source_name_from_path(saved_path)
            if skip_existing and _database_has_uploaded_file(vector_db, saved_path):
                _update_upload_job_file(
                    upload_jobs,
                    upload_jobs_lock,
                    job_id,
                    saved_path,
                    status="skipped",
                    progress=1.0,
                    message=f"Skipped existing file {source_name}.",
                    current_file_name=source_name,
                )
                continue
            _update_upload_job(
                upload_jobs,
                upload_jobs_lock,
                job_id,
                current_file_name=source_name,
                message=f"Preparing {source_name}...",
            )
            _update_upload_job_file(
                upload_jobs,
                upload_jobs_lock,
                job_id,
                saved_path,
                status="processing",
                progress=0.01,
            )
            try:
                assembler.delete_file(str(saved_path))
                assembler.store_file(
                    str(saved_path),
                    progress_callback=lambda stage, payload, current_path=saved_path: _handle_upload_progress_event(
                        upload_jobs,
                        upload_jobs_lock,
                        job_id,
                        current_path,
                        stage,
                        payload,
                    ),
                )
            except Exception as exc:
                error_message = _format_upload_error(exc)
                _update_upload_job_file(
                    upload_jobs,
                    upload_jobs_lock,
                    job_id,
                    saved_path,
                    status="failed",
                    error=error_message,
                    message=error_message,
                )
                raise
            _update_upload_job_file(
                upload_jobs,
                upload_jobs_lock,
                job_id,
                saved_path,
                status="completed",
                progress=1.0,
                message=f"Embedded {source_name}.",
            )

        _update_upload_job(
            upload_jobs,
            upload_jobs_lock,
            job_id,
            status="completed",
            progress=100.0,
            current_file_name=None,
            message=_build_upload_completion_message(_get_upload_job_or_404(upload_jobs, upload_jobs_lock, database.id, job_id)),
        )
    except Exception as exc:
        error_message = _format_upload_error(exc)
        _update_upload_job(
            upload_jobs,
            upload_jobs_lock,
            job_id,
            status="failed",
            error=error_message,
            error_stage=_upload_error_stage(exc),
            message=error_message,
            current_file_name=_source_name_from_path(current_path) if current_path else None,
        )


def _handle_upload_progress_event(
    upload_jobs: dict[str, dict[str, Any]],
    upload_jobs_lock: Lock,
    job_id: str,
    saved_path: str,
    stage: str,
    payload: dict[str, Any],
):
    source_name = str(payload.get("source_name") or _source_name_from_path(saved_path))
    updates: dict[str, Any] = {"status": "processing"}
    message = ""

    if stage == "load_started":
        updates["progress"] = 0.02
        message = f"Loading {source_name}..."
    elif stage == "marker_started":
        updates["progress"] = 0.03
        message = str(payload.get("message") or f"Converting {source_name} with Marker...")
    elif stage == "marker_progress":
        percent = _optional_float(payload.get("percent"))
        if percent is None:
            updates["progress"] = 0.04
        else:
            updates["progress"] = 0.03 + (0.07 * min(max(percent, 0.0), 100.0) / 100)
        message = str(payload.get("message") or f"Converting {source_name} with Marker...")
    elif stage == "marker_complete":
        updates["progress"] = 0.1
        message = str(payload.get("message") or f"Converted {source_name} with Marker.")
    elif stage == "marker_fallback":
        updates["progress"] = 0.05
        message = str(payload.get("message") or f"Marker unavailable for {source_name}; using PyPDF2...")
    elif stage == "load_complete":
        updates["progress"] = 0.1
        message = f"Loaded {source_name}."
    elif stage == "chunk_started":
        updates["progress"] = 0.12
        message = f"Chunking {source_name}..."
    elif stage == "chunk_complete":
        chunk_count = int(payload.get("chunk_count") or 0)
        updates["chunk_count"] = chunk_count
        updates["progress"] = 0.18 if chunk_count else 0.15
        message = f"Created {chunk_count} chunk(s) for {source_name}."
    elif stage == "embedding_started":
        chunk_count = int(payload.get("total_chunks") or payload.get("chunk_count") or 0)
        updates["chunk_count"] = chunk_count
        updates["embedded_chunks"] = 0
        updates["progress"] = 0.2
        message = f"Embedding {source_name}..."
    elif stage == "embedding_progress":
        total_chunks = max(int(payload.get("total_chunks") or 0), 1)
        embedded_chunks = min(int(payload.get("embedded_chunks") or 0), total_chunks)
        updates["chunk_count"] = total_chunks
        updates["embedded_chunks"] = embedded_chunks
        updates["progress"] = 0.2 + (0.72 * embedded_chunks / total_chunks)
        message = f"Embedding {source_name} ({embedded_chunks}/{total_chunks} chunks)..."
    elif stage == "storing_started":
        total_chunks = int(payload.get("total_chunks") or 0)
        updates["chunk_count"] = total_chunks
        updates["embedded_chunks"] = total_chunks
        updates["progress"] = 0.94
        message = f"Saving vectors for {source_name}..."
    elif stage == "storing_complete":
        total_chunks = int(payload.get("total_chunks") or payload.get("record_count") or 0)
        updates["chunk_count"] = total_chunks
        updates["embedded_chunks"] = total_chunks
        updates["progress"] = 1.0
        updates["status"] = "completed"
        message = f"Embedded {source_name}."

    _update_upload_job_file(
        upload_jobs,
        upload_jobs_lock,
        job_id,
        saved_path,
        current_file_name=source_name,
        message=message,
        **updates,
    )


def _update_upload_job(
    upload_jobs: dict[str, dict[str, Any]],
    upload_jobs_lock: Lock,
    job_id: str,
    **updates,
):
    with upload_jobs_lock:
        job = upload_jobs.get(job_id)
        if job is None:
            return
        job["updated_at"] = _utc_now()
        job.update(updates)
        _recalculate_upload_job(job)
        _persist_upload_jobs(upload_jobs)


def _update_upload_job_file(
    upload_jobs: dict[str, dict[str, Any]],
    upload_jobs_lock: Lock,
    job_id: str,
    saved_path: str,
    **updates,
):
    with upload_jobs_lock:
        job = upload_jobs.get(job_id)
        if job is None:
            return
        job["updated_at"] = _utc_now()
        for file_entry in job.get("files", []):
            if file_entry.get("id") == saved_path:
                file_entry.update({key: value for key, value in updates.items() if key in file_entry})
                break
        job.update({key: value for key, value in updates.items() if key not in {"chunk_count", "embedded_chunks", "progress", "status"}})
        _recalculate_upload_job(job)
        _persist_upload_jobs(upload_jobs)


def _optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _recalculate_upload_job(job: dict[str, Any]):
    files = job.get("files", [])
    total_files = len(files)
    job["total_files"] = total_files
    job["completed_files"] = sum(1 for item in files if item.get("status") in {"completed", "skipped"})
    job["total_chunks"] = sum(int(item.get("chunk_count") or 0) for item in files)
    job["embedded_chunks"] = sum(int(item.get("embedded_chunks") or 0) for item in files)
    if job.get("status") == "completed":
        job["progress"] = 100.0
        return
    if total_files == 0:
        job["progress"] = 0.0
        return
    progress = sum(float(item.get("progress") or 0.0) for item in files) / total_files
    job["progress"] = round(progress * 100, 1)


def _source_name_from_path(saved_path: str) -> str:
    return saved_path.replace("\\", "/").rstrip("/").split("/")[-1]


def _format_upload_error(exc: Exception) -> str:
    if isinstance(exc, IndexingError):
        return str(exc)
    return f"Indexing failed: {exc}"


def _upload_error_stage(exc: Exception) -> str | None:
    return exc.stage if isinstance(exc, IndexingError) else None


def _database_has_uploaded_file(vector_db: VectorDatabase, saved_path: str) -> bool:
    rel_path = Config.get_relative_path(saved_path)
    results = vector_db.query_by_metadata({"file_path": rel_path}, n_results=1)
    return bool(results.get("documents"))


def _delete_uploaded_document_file(relative_path: str):
    if not relative_path:
        return
    target_path = Config.DATA_DIR / relative_path
    try:
        if target_path.exists() and target_path.is_file():
            target_path.unlink()
    except OSError:
        pass


def _build_upload_completion_message(job: dict[str, Any]) -> str:
    files = job.get("files", [])
    embedded_count = sum(1 for item in files if item.get("status") == "completed")
    skipped_count = sum(1 for item in files if item.get("status") == "skipped")
    if skipped_count and embedded_count:
        return f"Embedded {embedded_count} file(s) and skipped {skipped_count} existing file(s)."
    if skipped_count:
        return f"Skipped {skipped_count} existing file(s)."
    return f"Embedded {embedded_count} file(s) successfully."


app = create_app()
