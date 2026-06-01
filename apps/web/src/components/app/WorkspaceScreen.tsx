import { ChatHeader, MessageComposer, MessageList } from "@/components/chat";
import {
  DatabasePanel,
  DatabaseSettingsModal,
  ModelSettingsPanel,
  SessionPanel,
  WorkflowPanel,
} from "@/components/panels";
import type { useChatWorkspace } from "@/hooks/useChatWorkspace";

import { ConfirmDialogModal } from "./ConfirmDialogModal";
import { ResponsiveWorkspaceShell } from "./ResponsiveWorkspaceShell";

type ChatWorkspace = ReturnType<typeof useChatWorkspace>;

interface WorkspaceScreenProps {
  app: ChatWorkspace;
}

export function WorkspaceScreen({ app }: WorkspaceScreenProps) {
  const { database, dialogs, messages, refs, sessions, settings, workspace } = app;
  const { activeSession, state } = workspace;
  const sessionsSlot = ({ openChat }: { openChat: () => void }) => (
    <SessionPanel
      activeSessionId={sessions.activeSessionId}
      busy={state.busy}
      onCreate={async () => {
        await sessions.actions.create();
        openChat();
      }}
      onDelete={sessions.actions.openDeleteDialog}
      onRename={(session, title) => {
        void sessions.actions.rename(session, title);
      }}
      onSelect={async (sessionId) => {
        await sessions.actions.select(sessionId);
        openChat();
      }}
      ready={state.ready}
      sessions={sessions.sessions}
    />
  );

  const databaseSlot = (
    <DatabasePanel
      activeDatabaseId={database.activeDatabaseId}
      busy={state.busy}
      databases={database.databases}
      documents={database.documents}
      uploadJob={database.uploadJob}
      onDeleteDocument={database.actions.openDeleteDocumentDialog}
      onRenameDocument={(document, name) => {
        void database.actions.renameDocument(document, name);
      }}
      onOpenSettings={database.actions.openSettings}
      onUploadFiles={(files) => {
        void database.actions.uploadDocuments(files);
      }}
    />
  );

  const chatSlot = (
    <section className="chat-workspace">
      <ChatHeader totalTokens={activeSession?.total_tokens ?? 0} title={activeSession?.title ?? "Workspace"} />

      <MessageList
        messages={messages.items}
        onDelete={messages.actions.openDeleteDialog}
        onEdit={(message, content) => {
          void messages.actions.updateContent(message, content);
        }}
        onRegenerate={(message) => {
          void messages.actions.regenerate(message);
        }}
        onRollback={messages.actions.openRollbackDialog}
        ready={state.ready}
      />

      <MessageComposer
        busy={state.busy}
        canContinue={messages.canContinue}
        canStop={messages.canStop}
        inputRef={refs.messageInputRef}
        onChange={messages.setDraft}
        onContinue={() => {
          void messages.actions.continueWorkflow();
        }}
        onStop={messages.actions.stopWorkflow}
        onSubmit={() => {
          void messages.actions.send();
        }}
        value={messages.draft}
      />
    </section>
  );

  const toolsSlot = (
    <>
      <ModelSettingsPanel
        activeModelName={settings.activeModelName}
        busy={state.busy}
        modelNames={settings.modelNames}
        onOpenSettings={() => {
          void settings.actions.open();
        }}
        onSelectActiveModel={(name) => {
          void settings.actions.selectActiveChatModel(name);
        }}
      />

      <WorkflowPanel workflow={state.workflow} />
    </>
  );

  return (
    <>
      <ResponsiveWorkspaceShell
        activeSessionId={sessions.activeSessionId}
        chatOpenRequest={state.chatOpenRequest}
        chat={chatSlot}
        database={databaseSlot}
        sessions={sessionsSlot}
        tools={toolsSlot}
      />

      <ConfirmDialogModal dialogs={dialogs} />

      <DatabaseSettingsModal
        activeDatabaseId={database.activeDatabaseId}
        busy={state.busy}
        databases={database.databases}
        defaultBackend={settings.defaultDatabaseBackend}
        embeddingModelNames={settings.embeddingModelNames}
        onClose={database.actions.closeSettings}
        onCreate={(name, embeddingModelName, backend) => {
          void database.actions.create(name, embeddingModelName, backend);
        }}
        onDelete={database.actions.openDeleteDatabaseDialog}
        onRename={(targetDatabase, name) => {
          void database.actions.rename(targetDatabase, name);
        }}
        onSelect={(databaseId) => {
          void database.actions.select(databaseId);
        }}
        open={database.settingsOpen}
      />
    </>
  );
}
