import { startTransition, useEffect, useReducer, useRef, useState } from "react";

import { api } from "@/lib/api";
import { formatNumber, trimOrNull } from "@/lib/format";
import { createClientId } from "@/lib/id";
import {
  getDefaultPrompt,
  findPreviousUserIndex,
  getPreferredSessionId,
  getPromptFromMessages,
  mergeSessions,
  sortSessions,
} from "@/lib/session";
import { readEventStream } from "@/lib/sse";
import { storage } from "@/lib/storage";
import { setSessionIdInUrl } from "@/lib/url-state";
import { buildFailedWorkflow, buildPendingWorkflow, normalizeWorkflow } from "@/lib/workflow";
import { useSettingsManagement } from "@/hooks/useSettingsManagement";
import type {
  ChatResponse,
  ConfirmDialogState,
  DatabaseDocumentRecord,
  DatabaseRecord,
  DatabaseState,
  HealthResponse,
  MessageRecord,
  MetaResponse,
  SessionState,
  SessionSummary,
  UploadJobRecord,
  WorkflowSnapshot,
} from "@/types/chat";

type StatusTone = "neutral" | "success" | "error";

interface WorkspaceState {
  ready: boolean;
  busy: boolean;
  health: HealthResponse | null;
  meta: MetaResponse | null;
  sessions: SessionSummary[];
  databases: DatabaseRecord[];
  databaseDocuments: DatabaseDocumentRecord[];
  activeDatabaseId: string | null;
  sessionId: string | null;
  messages: MessageRecord[];
  workflow: WorkflowSnapshot | null;
  uploadJob: UploadJobRecord | null;
  chatOpenRequest: number;
  statusText: string;
  statusTone: StatusTone;
  liveLabel: string;
}

type Action =
  | { type: "bootstrap"; meta: MetaResponse; health: HealthResponse }
  | { type: "ready" }
  | { type: "busy"; busy: boolean; label?: string }
  | { type: "status"; text: string; tone?: StatusTone; liveLabel?: string }
  | { type: "databases:set"; payload: DatabaseState }
  | { type: "database-documents:set"; documents: DatabaseDocumentRecord[] }
  | { type: "sessions:set"; sessions: SessionSummary[] }
  | { type: "session:select"; sessionId: string | null }
  | { type: "session:summary"; session: SessionSummary }
  | { type: "session:apply"; payload: SessionState; workflow?: WorkflowSnapshot | null }
  | { type: "session:remove"; sessionId: string }
  | { type: "messages:set"; messages: MessageRecord[] }
  | { type: "workflow:set"; workflow: WorkflowSnapshot | null }
  | { type: "upload-job:set"; uploadJob: UploadJobRecord | null }
  | { type: "chat:open" };

const initialState: WorkspaceState = {
  ready: false,
  busy: false,
  health: null,
  meta: null,
  sessions: [],
  databases: [],
  databaseDocuments: [],
  activeDatabaseId: null,
  sessionId: null,
  messages: [],
  workflow: null,
  uploadJob: null,
  chatOpenRequest: 0,
  statusText: "Connecting...",
  statusTone: "neutral",
  liveLabel: "Booting",
};

function reducer(state: WorkspaceState, action: Action): WorkspaceState {
  switch (action.type) {
    case "bootstrap":
      return {
        ...state,
        health: action.health,
        meta: action.meta,
      };
    case "ready":
      return {
        ...state,
        ready: true,
      };
    case "busy":
      return {
        ...state,
        busy: action.busy,
        statusText: action.label ?? state.statusText,
        liveLabel: action.label ? (action.busy ? "Working" : state.liveLabel) : state.liveLabel,
      };
    case "status":
      return {
        ...state,
        statusText: action.text,
        statusTone: action.tone ?? "neutral",
        liveLabel: action.liveLabel ?? state.liveLabel,
      };
    case "sessions:set":
      return {
        ...state,
        sessions: sortSessions(action.sessions),
      };
    case "databases:set":
      return {
        ...state,
        databases: action.payload.databases,
        activeDatabaseId: action.payload.active_database_id,
        uploadJob:
          state.uploadJob && state.uploadJob.database_id !== action.payload.active_database_id ? null : state.uploadJob,
      };
    case "database-documents:set":
      return {
        ...state,
        databaseDocuments: action.documents,
      };
    case "session:select":
      return {
        ...state,
        sessionId: action.sessionId,
      };
    case "session:summary":
      return {
        ...state,
        sessions: mergeSessions(state.sessions, action.session),
      };
    case "session:apply":
      const nextSessions = mergeSessions(state.sessions, action.payload.session);
      return {
        ...state,
        sessionId: action.payload.session.session_id,
        messages: action.payload.messages,
        sessions: isBlankNewSession(action.payload.session)
          ? nextSessions.filter(
              (session) => session.session_id === action.payload.session.session_id || !isBlankNewSession(session)
            )
          : nextSessions,
        workflow: action.workflow ?? null,
      };
    case "session:remove":
      return {
        ...state,
        sessions: state.sessions.filter((session) => session.session_id !== action.sessionId),
        sessionId: state.sessionId === action.sessionId ? null : state.sessionId,
        messages: state.sessionId === action.sessionId ? [] : state.messages,
        workflow: state.sessionId === action.sessionId ? null : state.workflow,
      };
    case "messages:set":
      return {
        ...state,
        messages: action.messages,
      };
    case "workflow:set":
      return {
        ...state,
        workflow: action.workflow,
      };
    case "upload-job:set":
      if (action.uploadJob && state.activeDatabaseId && action.uploadJob.database_id !== state.activeDatabaseId) {
        return state;
      }
      return {
        ...state,
        uploadJob: action.uploadJob,
      };
    case "chat:open":
      return {
        ...state,
        chatOpenRequest: state.chatOpenRequest + 1,
      };
    default:
      return state;
  }
}

function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : "Unknown request error.";
}

function replaceAtIndex<T>(items: T[], index: number, nextItem: T) {
  return items.map((item, itemIndex) => (itemIndex === index ? nextItem : item));
}

function delay(ms: number) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

function isBlankMessageList(messages: MessageRecord[]) {
  return messages.length === 0 || (messages.length === 1 && messages[0].role === "system");
}

function isBlankNewSession(session: SessionSummary) {
  return session.title === "New Session" && session.message_count === 0 && session.total_tokens === 0 && !session.preview;
}

function normalizeLiveWorkflowRecord(
  payload: unknown,
  options: {
    index: number;
    fallbackTurnId?: string | null;
  }
): MessageRecord | null {
  if (!payload || typeof payload !== "object") {
    return null;
  }

  const record = payload as Record<string, unknown>;
  const role = typeof record.role === "string" ? record.role : null;
  const content = typeof record.content === "string" ? record.content : "";
  if (!role || !content.trim()) {
    return null;
  }

  const workflowTurnId =
    typeof record.workflow_turn_id === "string" && record.workflow_turn_id.trim()
      ? record.workflow_turn_id
      : options.fallbackTurnId ?? null;

  return {
    id:
      typeof record.id === "string" && record.id.trim()
        ? record.id
        : `pending-record-${options.index}`,
    role: role as MessageRecord["role"],
    content,
    message_type: typeof record.message_type === "string" ? record.message_type : null,
    workflow_turn_id: workflowTurnId,
    tool_name: typeof record.tool_name === "string" ? record.tool_name : null,
    token_usage:
      record.token_usage && typeof record.token_usage === "object"
        ? (record.token_usage as MessageRecord["token_usage"])
        : null,
    attachments: normalizeAttachments(record.attachments),
  };
}

function normalizeAttachments(value: unknown): MessageRecord["attachments"] {
  if (!Array.isArray(value)) {
    return null;
  }
  const attachments = value.flatMap((item) => {
    if (!item || typeof item !== "object") {
      return [];
    }
    const record = item as Record<string, unknown>;
    const url = typeof record.url === "string" ? record.url.trim() : "";
    const type = typeof record.type === "string" ? record.type.trim() : "";
    if (!url || !type) {
      return [];
    }
    return [
      {
        id: typeof record.id === "string" ? record.id : undefined,
        type,
        kind: typeof record.kind === "string" ? record.kind : null,
        mime_type: typeof record.mime_type === "string" ? record.mime_type : null,
        url,
        path: typeof record.path === "string" ? record.path : null,
        title: typeof record.title === "string" ? record.title : null,
        source_url: typeof record.source_url === "string" ? record.source_url : null,
        size_bytes: typeof record.size_bytes === "number" ? record.size_bytes : null,
      },
    ];
  });
  return attachments.length > 0 ? attachments : null;
}

function upsertLiveWorkflowRecord(records: MessageRecord[], nextRecord: MessageRecord) {
  const existingIndex = records.findIndex((item) => item.id === nextRecord.id);
  if (existingIndex < 0) {
    return [...records, nextRecord];
  }
  return replaceAtIndex(records, existingIndex, nextRecord);
}

function buildPendingSendMessages(
  stableMessages: MessageRecord[],
  outgoing: string,
  workflow: WorkflowSnapshot,
  liveRecords: MessageRecord[],
  content: string
) {
  const workflowTurnId =
    workflow.workflow_turn_id ??
    liveRecords.at(-1)?.workflow_turn_id ??
    null;

  return [
    ...stableMessages,
    { id: "pending-user", role: "user" as const, content: outgoing, pending: true },
    ...liveRecords,
    {
      id: "pending-assistant",
      role: "assistant" as const,
      content,
      pending: true,
      workflow,
      workflow_turn_id: workflowTurnId,
    },
  ];
}

function buildPendingRegenerateMessages(
  baseMessages: MessageRecord[],
  workflow: WorkflowSnapshot,
  liveRecords: MessageRecord[],
  content: string
) {
  const workflowTurnId =
    workflow.workflow_turn_id ??
    liveRecords.at(-1)?.workflow_turn_id ??
    null;

  return [
    ...baseMessages,
    ...liveRecords,
    {
      id: "pending-assistant",
      role: "assistant" as const,
      content,
      pending: true,
      workflow,
      workflow_turn_id: workflowTurnId,
    },
  ];
}

export function useChatWorkspace() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [messageDraft, setMessageDraft] = useState("");
  const [systemPromptDraft, setSystemPromptDraft] = useState("");
  const [databaseSettingsOpen, setDatabaseSettingsOpen] = useState(false);
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialogState | null>(null);
  const messageInputRef = useRef<HTMLTextAreaElement | null>(null);
  const uploadPollRef = useRef<string | null>(null);
  const settings = useSettingsManagement({
    onRuntimeSettingsRefresh: (health, meta) => {
      dispatch({ type: "bootstrap", meta, health });
    },
    onStatus: (text, tone, liveLabel) => {
      dispatch({ type: "status", text, tone, liveLabel });
    },
    withBusy,
  });

  const activeSession = state.sessions.find((session) => session.session_id === state.sessionId) ?? null;
  const activePrompt = getPromptFromMessages(state.messages, getDefaultPrompt(state.meta));
  const systemPromptDirty = systemPromptDraft !== activePrompt;
  const totalStoredTokens = state.sessions.reduce((sum, session) => sum + (session.total_tokens || 0), 0);

  useEffect(() => {
    let cancelled = false;

    async function boot() {
      try {
        const [health, meta, modelSettings, skillSettings, appSettings, databaseState] = await Promise.all([
          api.getHealth(),
          api.getMeta(),
          api.getModelSettings(),
          api.getSkills(),
          api.getAppSettings(),
          api.listDatabases(),
        ]);
        if (cancelled) {
          return;
        }

        dispatch({ type: "bootstrap", meta, health });
        settings.actions.applyPersistedSettings(modelSettings, skillSettings, appSettings);

        const nextPrompt = meta.default_system_prompt;
        setSystemPromptDraft(nextPrompt);

        const sessions = await api.listSessions();
        if (cancelled) {
          return;
        }

        await syncDatabaseState(databaseState);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          dispatch({ type: "sessions:set", sessions });
        });

        if (sessions.length === 0) {
          await createSession();
          if (cancelled) {
            return;
          }
          dispatch({ type: "ready" });
          return;
        }

        const preferredSessionId = getPreferredSessionId(sessions);
        if (!preferredSessionId) {
          dispatch({ type: "ready" });
          dispatch({ type: "status", text: "Ready", tone: "success", liveLabel: "Ready" });
          return;
        }

        storage.setSessionId(preferredSessionId);
        setSessionIdInUrl(preferredSessionId);
        dispatch({ type: "session:select", sessionId: preferredSessionId });

        const payload = await api.getSession(preferredSessionId);
        if (cancelled) {
          return;
        }

        startTransition(() => {
          dispatch({ type: "session:apply", payload });
          dispatch({
            type: "status",
            text: payload.messages.length ? "Loaded session." : "Ready",
            tone: "success",
            liveLabel: "Ready",
          });
        });
        setSystemPromptDraft(getPromptFromMessages(payload.messages, meta.default_system_prompt));
        dispatch({ type: "ready" });
      } catch (error) {
        if (cancelled) {
          return;
        }

        dispatch({ type: "ready" });
        dispatch({ type: "status", text: getErrorMessage(error), tone: "error", liveLabel: "Error" });
      }
    }

    void boot();

    return () => {
      cancelled = true;
    };
  }, []);

  function syncSelectedSession(sessionId: string) {
    setSessionIdInUrl(sessionId);
    storage.setSessionId(sessionId);
    dispatch({ type: "session:select", sessionId });
  }

  function applySessionPayload(payload: SessionState, workflow?: WorkflowSnapshot | null) {
    const rawWorkflow = normalizeWorkflow(state.meta, workflow ?? null);
    startTransition(() => {
      dispatch({
        type: "session:apply",
        payload,
        workflow: rawWorkflow,
      });
    });

    syncSelectedSession(payload.session.session_id);

    const nextPrompt = getPromptFromMessages(payload.messages, getDefaultPrompt(state.meta));
    setSystemPromptDraft(nextPrompt);
  }

  async function withBusy(
    label: string,
    action: () => Promise<void>,
    onError?: (message: string) => Promise<void> | void
  ) {
    dispatch({ type: "busy", busy: true, label });

    try {
      await action();
    } catch (error) {
      const detail = getErrorMessage(error);
      if (onError) {
        await onError(detail);
      } else {
        dispatch({ type: "status", text: detail, tone: "error", liveLabel: "Error" });
      }
    } finally {
      dispatch({ type: "busy", busy: false });
      window.requestAnimationFrame(() => {
        messageInputRef.current?.focus();
      });
    }
  }

  async function loadSessionSnapshot(sessionId: string) {
    const payload = await api.getSession(sessionId);
    applySessionPayload(payload);
    return payload;
  }

  async function syncDatabaseState(payload: DatabaseState) {
    const documents = payload.active_database_id ? await api.listDatabaseDocuments(payload.active_database_id) : [];
    startTransition(() => {
      dispatch({ type: "databases:set", payload });
      dispatch({ type: "database-documents:set", documents });
    });
    return payload;
  }

  async function refreshDatabaseState() {
    const payload = await api.listDatabases();
    return syncDatabaseState(payload);
  }

  async function pollDatabaseUploadJob(databaseId: string, jobId: string) {
    for (;;) {
      const uploadJob = await api.getDatabaseUploadJob(databaseId, jobId);
      startTransition(() => {
        dispatch({ type: "upload-job:set", uploadJob });
      });
      dispatch({
        type: "status",
        text: uploadJob.message || "Embedding ...",
        tone: uploadJob.status === "failed" ? "error" : "neutral",
        liveLabel: uploadJob.status === "failed" ? "Error" : "Working",
      });
      if (uploadJob.status === "completed" || uploadJob.status === "failed") {
        return uploadJob;
      }
      await delay(500);
    }
  }

  function startDatabaseUploadJobPolling(databaseId: string, jobId: string, completionLabel?: string) {
    const pollKey = `${databaseId}:${jobId}`;
    if (uploadPollRef.current === pollKey) {
      return;
    }

    uploadPollRef.current = pollKey;
    void (async () => {
      try {
        const finishedJob = await pollDatabaseUploadJob(databaseId, jobId);
        if (finishedJob.status === "failed") {
          dispatch({
            type: "status",
            text: finishedJob.error || finishedJob.message || "Embedding upload failed.",
            tone: "error",
            liveLabel: "Error",
          });
          return;
        }

        await refreshDatabaseState();
        startTransition(() => {
          dispatch({ type: "upload-job:set", uploadJob: null });
        });
        dispatch({
          type: "status",
          text: completionLabel || finishedJob.message || "Upload completed.",
          tone: "success",
          liveLabel: "Ready",
        });
      } catch (error) {
        dispatch({ type: "status", text: getErrorMessage(error), tone: "error", liveLabel: "Error" });
      } finally {
        if (uploadPollRef.current === pollKey) {
          uploadPollRef.current = null;
        }
      }
    })();
  }

  useEffect(() => {
    const databaseId = state.activeDatabaseId;
    if (!databaseId) {
      return;
    }
    const activeDatabaseId = databaseId;

    let cancelled = false;
    async function restoreUploadJob() {
      try {
        const uploadJob = await api.getCurrentDatabaseUploadJob(activeDatabaseId);
        if (cancelled || !uploadJob) {
          return;
        }
        startTransition(() => {
          dispatch({ type: "upload-job:set", uploadJob });
        });
        dispatch({
          type: "status",
          text: uploadJob.message || "Embedding ...",
          tone: "neutral",
          liveLabel: "Working",
        });
        startDatabaseUploadJobPolling(activeDatabaseId, uploadJob.job_id);
      } catch {
        // Missing or stale jobs should not block loading the workspace.
      }
    }

    void restoreUploadJob();
    return () => {
      cancelled = true;
    };
  }, [state.activeDatabaseId]);

  async function selectSession(sessionId: string) {
    syncSelectedSession(sessionId);
    dispatch({ type: "chat:open" });

    await withBusy("Loading session...", async () => {
      const payload = await loadSessionSnapshot(sessionId);
      dispatch({
        type: "status",
        text: payload.messages.length ? "Loaded session." : "Ready",
        tone: "success",
        liveLabel: "Ready",
      });
    });
  }

  async function createSession(title?: string | null) {
    const currentPrompt = state.meta?.default_system_prompt || "";
    const existingBlankSession = state.sessions.find(isBlankNewSession);

    if (existingBlankSession) {
      syncSelectedSession(existingBlankSession.session_id);
      startTransition(() => {
        dispatch({
          type: "session:apply",
          payload: {
            session: existingBlankSession,
            messages: currentPrompt
              ? [
                  {
                    id: createClientId(),
                    role: "system" as const,
                    content: currentPrompt,
                    message_type: "system",
                  },
                ]
              : [],
          },
        });
        dispatch({ type: "chat:open" });
      });
    } else {
      startTransition(() => {
        dispatch({ type: "chat:open" });
      });
    }

    await withBusy("Creating session...", async () => {
      const session = existingBlankSession ?? (await api.createSession({ title: title ?? "New Session" }));
      syncSelectedSession(session.session_id);

      const payload = currentPrompt
        ? await api.updateSystemPrompt(session.session_id, currentPrompt)
        : await api.getSession(session.session_id);

      applySessionPayload(payload);
      dispatch({ type: "chat:open" });

      dispatch({ type: "status", text: "Ready for your first message.", tone: "success", liveLabel: "Ready" });
    });
  }

  async function applySystemPrompt() {
    if (!state.sessionId) {
      return;
    }

    await withBusy(
      "Updating system prompt...",
      async () => {
        const payload = await api.updateSystemPrompt(state.sessionId!, trimOrNull(systemPromptDraft));
        applySessionPayload(payload);
        dispatch({ type: "status", text: "System prompt updated.", tone: "success", liveLabel: "Ready" });
      },
      async (detail) => {
        setSystemPromptDraft(activePrompt);
        dispatch({ type: "status", text: detail, tone: "error", liveLabel: "Error" });
      }
    );
  }

  function resetSystemPromptDraft() {
    setSystemPromptDraft(activePrompt || state.meta?.default_system_prompt || "");
  }

  function openDatabaseSettings() {
    setDatabaseSettingsOpen(true);
  }

  function closeDatabaseSettings() {
    setDatabaseSettingsOpen(false);
  }

  async function selectDatabase(databaseId: string) {
    if (!databaseId || state.activeDatabaseId === databaseId) {
      return;
    }

    await withBusy(
      "Selecting database...",
      async () => {
        const payload = await api.selectDatabase(databaseId);
        await syncDatabaseState(payload);
        dispatch({ type: "status", text: "Database selected.", tone: "success", liveLabel: "Ready" });
      },
      async (detail) => {
        dispatch({ type: "status", text: detail, tone: "error", liveLabel: "Error" });
      }
    );
  }

  async function createDatabase(name: string, embeddingModelName: string, backend?: DatabaseRecord["backend"]) {
    const cleanedName = name.trim();
    if (!cleanedName) {
      dispatch({ type: "status", text: "Database name cannot be empty.", tone: "error", liveLabel: "Error" });
      return;
    }

    await withBusy(
      "Creating database...",
      async () => {
        const payload = await api.createDatabase({ name: cleanedName, embedding_model_name: embeddingModelName, backend });
        await syncDatabaseState(payload);
        dispatch({ type: "status", text: "Database created.", tone: "success", liveLabel: "Ready" });
      },
      async (detail) => {
        dispatch({ type: "status", text: detail, tone: "error", liveLabel: "Error" });
      }
    );
  }

  async function renameDatabase(database: DatabaseRecord, nextName: string) {
    const cleanedName = nextName.trim();
    if (!cleanedName) {
      dispatch({ type: "status", text: "Database name cannot be empty.", tone: "error", liveLabel: "Error" });
      return;
    }

    await withBusy(
      "Renaming database...",
      async () => {
        const payload = await api.renameDatabase(database.id, cleanedName);
        await syncDatabaseState(payload);
        dispatch({ type: "status", text: "Database renamed.", tone: "success", liveLabel: "Ready" });
      },
      async (detail) => {
        dispatch({ type: "status", text: detail, tone: "error", liveLabel: "Error" });
      }
    );
  }

  async function uploadDatabaseDocuments(files: File[]) {
    const targetDatabaseId = state.activeDatabaseId;
    if (!targetDatabaseId) {
      dispatch({ type: "status", text: "Select a database before uploading files.", tone: "error", liveLabel: "Error" });
      return;
    }
    if (files.length === 0) {
      return;
    }

    dispatch({
      type: "busy",
      busy: true,
      label: files.length === 1 ? `Uploading "${files[0].name}"...` : `Uploading ${files.length} files...`,
    });
    startTransition(() => {
      dispatch({ type: "upload-job:set", uploadJob: null });
    });

    try {
      const createdJob = await api.createDatabaseUploadJob(targetDatabaseId, files, { skipExisting: true });
      startTransition(() => {
        dispatch({ type: "upload-job:set", uploadJob: createdJob });
      });
      dispatch({ type: "status", text: createdJob.message || "Embedding upload queued.", tone: "neutral", liveLabel: "Working" });
      startDatabaseUploadJobPolling(
        targetDatabaseId,
        createdJob.job_id,
        files.length === 1
          ? `Finished "${files[0].name}". Existing files were skipped automatically.`
          : `Finished ${files.length} documents. Existing files were skipped automatically.`
      );
    } catch (error) {
      dispatch({ type: "status", text: getErrorMessage(error), tone: "error", liveLabel: "Error" });
    } finally {
      dispatch({ type: "busy", busy: false });
      window.requestAnimationFrame(() => {
        messageInputRef.current?.focus();
      });
    }
  }

  async function renameDatabaseDocument(document: DatabaseDocumentRecord, nextName: string) {
    if (!state.activeDatabaseId || state.busy) {
      return;
    }

    const cleanedName = nextName.trim();
    if (!cleanedName) {
      dispatch({ type: "status", text: "Document name cannot be empty.", tone: "error", liveLabel: "Error" });
      return;
    }

    if (cleanedName === document.source_name.trim()) {
      return;
    }

    await withBusy("Renaming document...", async () => {
      const documents = await api.renameDatabaseDocument(state.activeDatabaseId!, document.id, cleanedName);
      startTransition(() => {
        dispatch({ type: "database-documents:set", documents });
      });
      dispatch({ type: "status", text: `Renamed to "${cleanedName}".`, tone: "success", liveLabel: "Ready" });
    });
  }

  function openDeleteDatabaseDocumentDialog(document: DatabaseDocumentRecord) {
    if (!state.activeDatabaseId || state.busy) {
      return;
    }

    setConfirmDialog({
      title: "Delete Document",
      description: `Remove "${document.source_name}" from the active database?`,
      confirmLabel: "Delete Document",
      tone: "danger",
      onConfirm: async () => {
        await withBusy("Deleting document...", async () => {
          const documents = await api.deleteDatabaseDocument(state.activeDatabaseId!, document.id);
          startTransition(() => {
            dispatch({ type: "database-documents:set", documents });
          });
          await refreshDatabaseState();
          setConfirmDialog(null);
          dispatch({ type: "status", text: `Deleted "${document.source_name}".`, tone: "success", liveLabel: "Ready" });
        });
      },
    });
  }

  function openDeleteDatabaseDialog(database: DatabaseRecord) {
    if (state.busy) {
      return;
    }

    setConfirmDialog({
      title: "Delete Database",
      description: `Delete "${database.name}" and remove its vector collection?`,
      confirmLabel: "Delete Database",
      tone: "danger",
      onConfirm: async () => {
        await withBusy("Deleting database...", async () => {
          const payload = await api.deleteDatabase(database.id);
          await syncDatabaseState(payload);
          setConfirmDialog(null);
          dispatch({ type: "status", text: "Database deleted.", tone: "success", liveLabel: "Ready" });
        });
      },
    });
  }

  async function renameSession(targetSession: SessionSummary, nextTitle: string) {
    if (state.busy) {
      return;
    }

    const cleanedTitle = nextTitle.trim();
    if (!cleanedTitle) {
      dispatch({ type: "status", text: "Session title cannot be empty.", tone: "error", liveLabel: "Error" });
      return;
    }

    await withBusy("Renaming session...", async () => {
      const session = await api.renameSession(targetSession.session_id, cleanedTitle);
      startTransition(() => {
        dispatch({ type: "session:summary", session });
      });
      dispatch({ type: "status", text: "Session renamed.", tone: "success", liveLabel: "Ready" });
    });
  }

  function openDeleteSessionDialog(targetSession?: SessionSummary) {
    const target = targetSession ?? activeSession;
    if (!target || state.busy) {
      return;
    }

    setConfirmDialog({
      title: "Delete Session",
      description: `Delete "${target.title}" and remove its persisted chat history?`,
      confirmLabel: "Delete Session",
      tone: "danger",
      onConfirm: async () => {
        await withBusy("Deleting session...", async () => {
          const deletedId = target.session_id;
          const remainingSessions = state.sessions.filter((session) => session.session_id !== deletedId);

          await api.deleteSession(deletedId);
          startTransition(() => {
            dispatch({ type: "session:remove", sessionId: deletedId });
          });

          setConfirmDialog(null);

          if (deletedId === state.sessionId && remainingSessions.length > 0) {
            const nextSessionId = remainingSessions[0].session_id;
            syncSelectedSession(nextSessionId);
            await loadSessionSnapshot(nextSessionId);
          } else if (deletedId === state.sessionId) {
            await createSession();
          }

          dispatch({ type: "status", text: "Session deleted.", tone: "success", liveLabel: "Ready" });
        });
      },
    });
  }

  async function updateMessageContent(message: MessageRecord, nextContent: string) {
    if (!state.sessionId || state.busy) {
      return;
    }

    const content = nextContent.trim();
    if (!content) {
      dispatch({ type: "status", text: "Message content cannot be empty.", tone: "error", liveLabel: "Error" });
      return;
    }

    await withBusy("Updating message...", async () => {
      const payload = await api.updateMessage(state.sessionId!, message.id, content);
      applySessionPayload(payload);
      dispatch({ type: "status", text: "Message updated.", tone: "success", liveLabel: "Ready" });
    });
  }

  function openDeleteMessageDialog(message: MessageRecord) {
    if (!state.sessionId || state.busy) {
      return;
    }

    setConfirmDialog({
      title: "Delete Message",
      description:
        message.role === "system"
          ? "Clear the system prompt while keeping the conversation?"
          : "Delete only this message and keep the rest of the conversation?",
      confirmLabel: "Delete Message",
      tone: "danger",
      onConfirm: async () => {
        await withBusy("Deleting message...", async () => {
          const payload = await api.deleteMessage(state.sessionId!, message.id);
          applySessionPayload(payload);
          setConfirmDialog(null);
          dispatch({
            type: "status",
            text: message.role === "system" ? "System prompt cleared." : "Message deleted.",
            tone: "success",
            liveLabel: "Ready",
          });
        });
      },
    });
  }

  function openRollbackDialog(message: MessageRecord) {
    if (!state.sessionId || state.busy) {
      return;
    }

    setConfirmDialog({
      title: "Rollback Session",
      description: "Keep this message and remove everything that comes after it?",
      confirmLabel: "Rollback",
      onConfirm: async () => {
        await withBusy("Rolling back session...", async () => {
          const payload = await api.rollbackMessage(state.sessionId!, message.id);
          applySessionPayload(payload);
          setConfirmDialog(null);
          dispatch({ type: "status", text: "Rolled back to the selected message.", tone: "success", liveLabel: "Ready" });
        });
      },
    });
  }

  async function sendMessage() {
    const outgoing = messageDraft.trim();
    if (!outgoing || !state.sessionId || state.busy) {
      return;
    }

    const stableMessages = [...state.messages];
    const pendingWorkflow = buildPendingWorkflow(state.meta, outgoing);
    let currentWorkflow = pendingWorkflow;
    let currentContent = "Thinking...";
    let liveRecords: MessageRecord[] = [];

    // Ensure session exists on the backend before sending the first user message
    let activeSessionId = state.sessionId;
    if (stableMessages.length === 0 || (stableMessages.length === 1 && stableMessages[0].role === "system")) {
       const session = await api.createSession({ session_id: activeSessionId });
       activeSessionId = session.session_id;
       syncSelectedSession(activeSessionId);
    }

    setMessageDraft("");
    startTransition(() => {
      dispatch({ type: "workflow:set", workflow: pendingWorkflow });
      dispatch({
        type: "messages:set",
        messages: buildPendingSendMessages(stableMessages, outgoing, currentWorkflow, liveRecords, currentContent),
      });
    });

    await withBusy(
      "Thinking...",
      async () => {
        await readEventStream(
          `/api/sessions/${encodeURIComponent(activeSessionId)}/messages/stream`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              message: outgoing,
            }),
          },
          {
            onEvent: async (eventName, payload) => {
              if (eventName === "record") {
                const nextRecord = normalizeLiveWorkflowRecord(payload, {
                  index: liveRecords.length,
                  fallbackTurnId: currentWorkflow.workflow_turn_id,
                });
                if (!nextRecord) {
                  return;
                }

                liveRecords = upsertLiveWorkflowRecord(liveRecords, nextRecord);
                startTransition(() => {
                  dispatch({
                    type: "messages:set",
                    messages: buildPendingSendMessages(
                      stableMessages,
                      outgoing,
                      currentWorkflow,
                      liveRecords,
                      currentContent
                    ),
                  });
                });
                return;
              }

              if (eventName !== "workflow") {
                return;
              }

              const workflow = normalizeWorkflow(state.meta, payload as unknown as WorkflowSnapshot);
              if (!workflow) {
                throw new Error("Workflow event payload is empty.");
              }
              currentWorkflow = workflow;
              startTransition(() => {
                dispatch({ type: "workflow:set", workflow });
                dispatch({
                  type: "messages:set",
                  messages: buildPendingSendMessages(
                    stableMessages,
                    outgoing,
                    currentWorkflow,
                    liveRecords,
                    currentContent
                  ),
                });
              });
              dispatch({
                type: "status",
                text: workflow.active_node ? `Workflow ${workflow.status} at ${workflow.active_node}.` : `Workflow ${workflow.status}.`,
                liveLabel: "Working",
              });
            },
            onChunk: async (payload) => {
              if (typeof payload.content !== "string") {
                throw new Error("Streaming chunk is missing content.");
              }
              currentContent = payload.content;
              startTransition(() => {
                dispatch({
                  type: "messages:set",
                  messages: buildPendingSendMessages(
                    stableMessages,
                    outgoing,
                    currentWorkflow,
                    liveRecords,
                    currentContent
                  ),
                });
              });
            },
            onDone: async (payload) => {
              const response = payload as unknown as ChatResponse;
              applySessionPayload(response, response.workflow ?? currentWorkflow);
              dispatch({ type: "status", text: "Reply received.", tone: "success", liveLabel: "Ready" });
            },
          }
        );
      },
      async (detail) => {
        setMessageDraft(outgoing);
        startTransition(() => {
          dispatch({ type: "messages:set", messages: stableMessages });
          dispatch({
            type: "workflow:set",
            workflow: buildFailedWorkflow(currentWorkflow, detail),
          });
        });
        dispatch({ type: "status", text: detail, tone: "error", liveLabel: "Error" });
      }
    );
  }

  async function regenerateMessage(message: MessageRecord) {
    if (!state.sessionId || state.busy) {
      return;
    }

    const messageIndex = state.messages.findIndex((item) => item.id === message.id);
    if (messageIndex < 0) {
      return;
    }

    const userIndex =
      message.role === "assistant" ? findPreviousUserIndex(state.messages, messageIndex) : messageIndex;
    const baseMessages = userIndex >= 0 ? state.messages.slice(0, userIndex + 1) : [...state.messages];
    const question = baseMessages.at(-1)?.content;
    if (!question) {
      throw new Error("Cannot regenerate without a user question.");
    }
    const pendingWorkflow = buildPendingWorkflow(state.meta, question);
    let currentWorkflow = pendingWorkflow;
    let currentContent = "Thinking...";
    let liveRecords: MessageRecord[] = [];

    startTransition(() => {
      dispatch({ type: "workflow:set", workflow: pendingWorkflow });
      dispatch({
        type: "messages:set",
        messages: buildPendingRegenerateMessages(baseMessages, currentWorkflow, liveRecords, currentContent),
      });
    });

    await withBusy(
      "Regenerating reply...",
      async () => {
        await readEventStream(
          `/api/sessions/${encodeURIComponent(state.sessionId!)}/messages/${encodeURIComponent(message.id)}/regenerate/stream`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
          },
          {
            onEvent: async (eventName, payload) => {
              if (eventName === "record") {
                const nextRecord = normalizeLiveWorkflowRecord(payload, {
                  index: liveRecords.length,
                  fallbackTurnId: currentWorkflow.workflow_turn_id,
                });
                if (!nextRecord) {
                  return;
                }

                liveRecords = upsertLiveWorkflowRecord(liveRecords, nextRecord);
                startTransition(() => {
                  dispatch({
                    type: "messages:set",
                    messages: buildPendingRegenerateMessages(
                      baseMessages,
                      currentWorkflow,
                      liveRecords,
                      currentContent
                    ),
                  });
                });
                return;
              }

              if (eventName !== "workflow") {
                return;
              }

              const workflow = normalizeWorkflow(state.meta, payload as unknown as WorkflowSnapshot);
              if (!workflow) {
                throw new Error("Workflow event payload is empty.");
              }
              currentWorkflow = workflow;
              startTransition(() => {
                dispatch({ type: "workflow:set", workflow });
                dispatch({
                  type: "messages:set",
                  messages: buildPendingRegenerateMessages(baseMessages, currentWorkflow, liveRecords, currentContent),
                });
              });
              dispatch({
                type: "status",
                text: workflow.active_node ? `Workflow ${workflow.status} at ${workflow.active_node}.` : `Workflow ${workflow.status}.`,
                liveLabel: "Working",
              });
            },
            onChunk: async (payload) => {
              if (typeof payload.content !== "string") {
                throw new Error("Streaming chunk is missing content.");
              }
              currentContent = payload.content;
              startTransition(() => {
                dispatch({
                  type: "messages:set",
                  messages: buildPendingRegenerateMessages(baseMessages, currentWorkflow, liveRecords, currentContent),
                });
              });
            },
            onDone: async (payload) => {
              const response = payload as unknown as ChatResponse;
              applySessionPayload(response, response.workflow ?? currentWorkflow);
              dispatch({ type: "status", text: "Message regenerated.", tone: "success", liveLabel: "Ready" });
            },
          }
        );
      },
      async (detail) => {
        startTransition(() => {
          dispatch({ type: "messages:set", messages: state.messages });
          dispatch({
            type: "workflow:set",
            workflow: buildFailedWorkflow(currentWorkflow, detail),
          });
        });
        dispatch({ type: "status", text: detail, tone: "error", liveLabel: "Error" });
      }
    );
  }

  return {
    workspace: {
      state,
      activeSession,
      totalStoredTokens,
      formatNumber,
    },
    settings: {
      activeModelName: settings.activeModelName,
      defaultDatabaseBackend: settings.persisted.appSettings.default_database_backend,
      embeddingModelNames: settings.embeddingModelNames,
      modelNames: settings.modelNames,
      pageOpen: settings.pageOpen,
      drafts: settings.drafts,
      actions: {
        addChatModel: settings.actions.addChatModel,
        addEmbeddingModel: settings.actions.addEmbeddingModel,
        addSkill: settings.actions.addSkill,
        close: settings.actions.closeSettingsPage,
        open: settings.actions.openSettingsPage,
        removeChatModel: settings.actions.removeChatModel,
        removeEmbeddingModel: settings.actions.removeEmbeddingModel,
        removeSkill: settings.actions.removeSkill,
        save: settings.actions.saveManagementSettings,
        selectActiveChatModel: settings.actions.selectActiveChatModel,
        setActiveChatModel: settings.actions.setActiveChatModel,
        setActiveEmbeddingModel: settings.actions.setActiveEmbeddingModel,
        testChatModel: settings.actions.testChatModel,
        testEmbeddingModel: settings.actions.testEmbeddingModel,
        updateAppSetting: settings.actions.updateAppSetting,
        updateChatModel: settings.actions.updateChatModel,
        updateEmbeddingModel: settings.actions.updateEmbeddingModel,
        updateSkill: settings.actions.updateSkill,
      },
    },
    database: {
      activeDatabaseId: state.activeDatabaseId,
      databases: state.databases,
      documents: state.databaseDocuments,
      settingsOpen: databaseSettingsOpen,
      uploadJob: state.uploadJob,
      actions: {
        closeSettings: closeDatabaseSettings,
        create: createDatabase,
        openDeleteDatabaseDialog,
        openDeleteDocumentDialog: openDeleteDatabaseDocumentDialog,
        openSettings: openDatabaseSettings,
        rename: renameDatabase,
        renameDocument: renameDatabaseDocument,
        select: selectDatabase,
        uploadDocuments: uploadDatabaseDocuments,
      },
    },
    sessions: {
      activeSession,
      activeSessionId: state.sessionId,
      sessions: state.sessions,
      actions: {
        create: createSession,
        openDeleteDialog: openDeleteSessionDialog,
        rename: renameSession,
        select: selectSession,
      },
    },
    messages: {
      draft: messageDraft,
      items: state.messages,
      setDraft: setMessageDraft,
      setSystemPromptDraft,
      systemPromptDirty,
      systemPromptDraft,
      actions: {
        applySystemPrompt,
        openDeleteDialog: openDeleteMessageDialog,
        openRollbackDialog,
        regenerate: regenerateMessage,
        resetSystemPromptDraft,
        send: sendMessage,
        updateContent: updateMessageContent,
      },
    },
    dialogs: {
      confirmDialog,
      setConfirmDialog,
    },
    refs: {
      messageInputRef,
    },
  };
}
