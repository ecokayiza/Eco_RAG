import {
  ArrowLeftIcon,
  Cog6ToothIcon,
  CommandLineIcon,
  CpuChipIcon,
  PuzzlePieceIcon,
  ServerStackIcon,
  XMarkIcon,
} from "@heroicons/react/24/outline";
import { useEffect, useMemo, useState } from "react";

import { Button, Field } from "@/components/common";
import { createEmptyChatModel, createEmptyEmbeddingModel } from "@/lib/model-settings";
import { createEmptySkill } from "@/lib/skill-settings";
import type {
  AppSettingsDocument,
  ChatModelConfig,
  EmbeddingModelConfig,
  JsonObject,
  ModelSettingsDocument,
  SkillRecord,
  SkillSettingsDocument,
} from "@/types/chat";

import { ModelSettingsSection } from "./ModelSettingsSection";
import { McpSettingsSection } from "./McpSettingsSection";
import { RuntimeSettingsSection } from "./RuntimeSettingsSection";
import { SkillSettingsSection } from "./SkillSettingsSection";
import {
  clampIndex,
  getDuplicateNames,
  numberValue,
  optionalNumberValue,
  optionalPositiveIntegerValue,
  resolveNextActiveName,
} from "./settings-utils";

type SettingsSection = "chat" | "embedding" | "runtime" | "mcp" | "skills";
type StatusTone = "neutral" | "success" | "error";

interface ManagementSettingsPageProps {
  appSettings: AppSettingsDocument;
  busy: boolean;
  modelSettings: ModelSettingsDocument;
  skillSettings: SkillSettingsDocument;
  statusText: string;
  statusTone: StatusTone;
  onAddChatModel: (initial?: Partial<ChatModelConfig>) => void;
  onAddEmbeddingModel: (initial?: Partial<EmbeddingModelConfig>) => void;
  onAddSkill: (initial?: Partial<SkillRecord>) => void;
  onBack: () => void;
  onChangeActiveChatModel: (name: string) => void;
  onChangeActiveEmbeddingModel: (name: string) => void;
  onRemoveChatModel: (index: number) => void;
  onRemoveEmbeddingModel: (index: number) => void;
  onRemoveSkill: (index: number) => void;
  onSave: () => void | Promise<void>;
  onTestChatModel: (model: ChatModelConfig) => void | Promise<void>;
  onTestEmbeddingModel: (model: EmbeddingModelConfig) => void | Promise<void>;
  onUpdateAppSetting: <Key extends keyof AppSettingsDocument>(key: Key, value: AppSettingsDocument[Key]) => void;
  onUpdateChatModel: <Key extends keyof ChatModelConfig>(index: number, key: Key, value: ChatModelConfig[Key]) => void;
  onUpdateEmbeddingModel: <Key extends keyof EmbeddingModelConfig>(
    index: number,
    key: Key,
    value: EmbeddingModelConfig[Key]
  ) => void;
  onUpdateSkill: <Key extends keyof SkillRecord>(index: number, key: Key, value: SkillRecord[Key]) => void;
}

const sections: Array<{ id: SettingsSection; label: string; meta: string; icon: typeof ServerStackIcon }> = [
  { id: "chat", label: "Chat Models", meta: "Generation providers", icon: ServerStackIcon },
  { id: "embedding", label: "Embedding Models", meta: "Indexing providers", icon: CpuChipIcon },
  { id: "runtime", label: "Runtime", meta: "Chunking and retrieval", icon: Cog6ToothIcon },
  { id: "mcp", label: "MCP", meta: "Tool schemas", icon: CommandLineIcon },
  { id: "skills", label: "Skills", meta: "Workflow capabilities", icon: PuzzlePieceIcon },
];

export function ManagementSettingsPage({
  appSettings,
  busy,
  modelSettings,
  skillSettings,
  statusText,
  statusTone,
  onAddChatModel,
  onAddEmbeddingModel,
  onAddSkill,
  onBack,
  onChangeActiveChatModel,
  onChangeActiveEmbeddingModel,
  onRemoveChatModel,
  onRemoveEmbeddingModel,
  onRemoveSkill,
  onSave,
  onTestChatModel,
  onTestEmbeddingModel,
  onUpdateAppSetting,
  onUpdateChatModel,
  onUpdateEmbeddingModel,
  onUpdateSkill,
}: ManagementSettingsPageProps) {
  const [section, setSection] = useState<SettingsSection>("chat");
  const [selectedChatIndex, setSelectedChatIndex] = useState(0);
  const [selectedEmbeddingIndex, setSelectedEmbeddingIndex] = useState(0);
  const [selectedSkillIndex, setSelectedSkillIndex] = useState(0);
  const [jsonValidationIssues, setJsonValidationIssues] = useState<Record<string, string>>({});
  const [dismissedStatusKey, setDismissedStatusKey] = useState<string | null>(null);

  useEffect(() => {
    setSelectedChatIndex((current) => clampIndex(current, modelSettings.chat_models.length));
  }, [modelSettings.chat_models.length]);

  useEffect(() => {
    setJsonValidationIssues({});
  }, [modelSettings.chat_models.length, selectedChatIndex]);

  useEffect(() => {
    setSelectedEmbeddingIndex((current) => clampIndex(current, modelSettings.embedding_models.length));
  }, [modelSettings.embedding_models.length]);

  useEffect(() => {
    setSelectedSkillIndex((current) => clampIndex(current, skillSettings.skills.length));
  }, [skillSettings.skills.length]);

  const stats = useMemo(
    () => ({
      chat: modelSettings.chat_models.length,
      embedding: modelSettings.embedding_models.length,
      runtime: "Settings",
      mcp: appSettings.mcp_enabled_tools.length,
      skills: skillSettings.skills.length,
    }),
    [
      appSettings.mcp_enabled_tools.length,
      modelSettings.chat_models.length,
      modelSettings.embedding_models.length,
      skillSettings.skills.length,
    ]
  );

  const validationIssues = useMemo(
    () => [...buildValidationIssues(modelSettings, skillSettings, appSettings), ...Object.values(jsonValidationIssues)],
    [appSettings, jsonValidationIssues, modelSettings, skillSettings]
  );
  const saveDisabled = busy || validationIssues.length > 0;
  const statusToastKey = `${statusTone}:${statusText}`;
  const statusIsApiTest = /\bAPI\b/i.test(statusText) && /test|testing/i.test(statusText);
  const showStatusToast = Boolean(statusIsApiTest && dismissedStatusKey !== statusToastKey);

  function setJsonValidationIssue(key: string, issue: string | null) {
    setJsonValidationIssues((current) => {
      if (!issue) {
        if (!(key in current)) {
          return current;
        }
        const { [key]: _removed, ...next } = current;
        return next;
      }
      if (current[key] === issue) {
        return current;
      }
      return { ...current, [key]: issue };
    });
  }

  return (
    <main className="settings-page" id="main-content">
      <aside className="settings-page__nav" aria-label="Settings sections">
        <div className="settings-page__brand">
          <button className="settings-page__back" onClick={onBack} type="button">
            <ArrowLeftIcon />
            Back
          </button>
          <div>
            <p className="settings-page__eyebrow">Management</p>
            <h1 className="settings-page__title">Settings</h1>
          </div>
        </div>

        <nav className="settings-nav-list">
          {sections.map((item) => {
            const Icon = item.icon;
            const isActive = section === item.id;
            const badge = stats[item.id];
            return (
              <button
                aria-current={isActive ? "page" : undefined}
                className={`settings-nav-item${isActive ? " settings-nav-item--active" : ""}`}
                key={item.id}
                onClick={() => {
                  setSection(item.id);
                }}
                type="button"
              >
                <Icon />
                <span>
                  <strong>{item.label}</strong>
                  <small>{item.meta}</small>
                </span>
                <em>{badge}</em>
              </button>
            );
          })}
        </nav>

        <div className="settings-page__nav-footer">
          {validationIssues.length > 0 ? <p className="settings-page__validation">{validationIssues[0]}</p> : null}
          <Button disabled={saveDisabled} onClick={() => void onSave()} variant="primary">
            Save Changes
          </Button>
          <Button disabled={busy} onClick={onBack} variant="ghost">
            Discard
          </Button>
        </div>
      </aside>

      <section className="settings-page__content">
        {section === "chat" ? (
          <ModelSettingsSection<ChatModelConfig>
            actionLabel="Add Chat Model"
            activeName={modelSettings.active_chat_model}
            busy={busy}
            emptyDetailTitle="No chat model selected"
            emptyListLabel="No chat models configured"
            idPrefix="chat-model"
            models={modelSettings.chat_models}
            onAdd={() => {
              const model = createEmptyChatModel(modelSettings.chat_models.length + 1);
              onAddChatModel(model);
              onChangeActiveChatModel(model.name);
              setSelectedChatIndex(modelSettings.chat_models.length);
            }}
            onChangeActive={onChangeActiveChatModel}
            onRemove={(index) => {
              onRemoveChatModel(index);
              const nextModels = modelSettings.chat_models.filter((_, itemIndex) => itemIndex !== index);
              onChangeActiveChatModel(resolveNextActiveName(nextModels, modelSettings.active_chat_model, index));
            }}
            onSelect={setSelectedChatIndex}
            onTestApi={(model) => {
              void onTestChatModel(model);
            }}
            onUpdate={onUpdateChatModel}
            renderSpecificFields={({ busy: fieldBusy, index, model, onUpdate }) => (
              <>
                <Field htmlFor="chat-model-wire-api" label="Wire API">
                  <select
                    disabled={fieldBusy}
                    id="chat-model-wire-api"
                    onChange={(event) => {
                      onUpdate(index, "wire_api", event.target.value as ChatModelConfig["wire_api"]);
                    }}
                    value={model.wire_api}
                  >
                    <option value="chat_completions">Chat Completions</option>
                    <option value="responses">Responses</option>
                  </select>
                </Field>
                <Field htmlFor="chat-model-temperature" label="Temperature">
                  <input
                    disabled={fieldBusy}
                    id="chat-model-temperature"
                    onChange={(event) => {
                      onUpdate(index, "temperature", numberValue(event.target.value, 1));
                    }}
                    step="0.1"
                    type="number"
                    value={String(model.temperature)}
                  />
                </Field>
                <Field htmlFor="chat-model-top-p" label="Top P">
                  <input
                    disabled={fieldBusy}
                    id="chat-model-top-p"
                    onChange={(event) => {
                      onUpdate(index, "top_p", optionalNumberValue(event.target.value));
                    }}
                    placeholder="Provider default"
                    step="0.1"
                    type="number"
                    value={model.top_p == null ? "" : String(model.top_p)}
                  />
                </Field>
                <JsonObjectField
                  disabled={fieldBusy}
                  id={`chat-model-${index}-custom-request-params`}
                  onChange={(value) => {
                    onUpdate(index, "custom_request_params", value);
                  }}
                  onValidationChange={(issue) => {
                    setJsonValidationIssue(`chat-${index}-custom-request-params`, issue);
                  }}
                  value={model.custom_request_params}
                />
              </>
            )}
            selectedIndex={selectedChatIndex}
            title="Chat Models"
          />
        ) : null}

        {section === "embedding" ? (
          <ModelSettingsSection<EmbeddingModelConfig>
            actionLabel="Add Embedding Model"
            activeName={modelSettings.active_embedding_model}
            busy={busy}
            emptyDetailTitle="No embedding model selected"
            emptyListLabel="No embedding models configured"
            idPrefix="embedding-model"
            models={modelSettings.embedding_models}
            onAdd={() => {
              const model = createEmptyEmbeddingModel(modelSettings.embedding_models.length + 1);
              onAddEmbeddingModel(model);
              onChangeActiveEmbeddingModel(model.name);
              setSelectedEmbeddingIndex(modelSettings.embedding_models.length);
            }}
            onChangeActive={onChangeActiveEmbeddingModel}
            onRemove={(index) => {
              onRemoveEmbeddingModel(index);
              const nextModels = modelSettings.embedding_models.filter((_, itemIndex) => itemIndex !== index);
              onChangeActiveEmbeddingModel(resolveNextActiveName(nextModels, modelSettings.active_embedding_model, index));
            }}
            onSelect={setSelectedEmbeddingIndex}
            onTestApi={(model) => {
              void onTestEmbeddingModel(model);
            }}
            onUpdate={onUpdateEmbeddingModel}
            renderSpecificFields={({ busy: fieldBusy, index, model, onUpdate }) => (
              <Field htmlFor="embedding-model-batch" label="Batch Size">
                <input
                  disabled={fieldBusy}
                  id="embedding-model-batch"
                  min={1}
                  onChange={(event) => {
                    onUpdate(index, "batch_size", optionalPositiveIntegerValue(event.target.value));
                  }}
                  placeholder="Provider default"
                  step={1}
                  type="number"
                  value={model.batch_size == null ? "" : String(model.batch_size)}
                />
              </Field>
            )}
            selectedIndex={selectedEmbeddingIndex}
            title="Embedding Models"
          />
        ) : null}

        {section === "runtime" ? (
          <RuntimeSettingsSection busy={busy} onUpdate={onUpdateAppSetting} settings={appSettings} />
        ) : null}

        {section === "mcp" ? (
          <McpSettingsSection busy={busy} onUpdate={onUpdateAppSetting} settings={appSettings} />
        ) : null}

        {section === "skills" ? (
          <SkillSettingsSection
            busy={busy}
            onAdd={() => {
              onAddSkill(createEmptySkill(skillSettings.skills.length + 1));
              setSelectedSkillIndex(skillSettings.skills.length);
            }}
            onRemove={onRemoveSkill}
            onSelect={setSelectedSkillIndex}
            onUpdate={onUpdateSkill}
            selectedIndex={selectedSkillIndex}
            skills={skillSettings.skills}
          />
        ) : null}
      </section>

      {showStatusToast ? (
        <aside
          aria-live={statusTone === "error" ? "assertive" : "polite"}
          className={`settings-toast settings-toast--${statusTone}`}
          role={statusTone === "error" ? "alert" : "status"}
        >
          <div className="settings-toast__body">
            <strong>{statusTone === "error" ? "Request Failed" : busy ? "Working" : "Result"}</strong>
            <p>{statusText}</p>
          </div>
          <button
            aria-label="Dismiss status"
            className="settings-toast__close"
            onClick={() => {
              setDismissedStatusKey(statusToastKey);
            }}
            type="button"
          >
            <XMarkIcon />
          </button>
        </aside>
      ) : null}
    </main>
  );
}

interface JsonObjectFieldProps {
  disabled: boolean;
  id: string;
  onChange: (value: JsonObject | null) => void;
  onValidationChange: (issue: string | null) => void;
  value: JsonObject | null;
}

function JsonObjectField({ disabled, id, onChange, onValidationChange, value }: JsonObjectFieldProps) {
  const [draft, setDraft] = useState(() => stringifyJsonObject(value));

  useEffect(() => {
    setDraft(stringifyJsonObject(value));
    onValidationChange(null);
  }, [id]);

  const parsed = parseJsonObjectDraft(draft);

  return (
    <div className="form-grid__full">
      <Field
        hint={
          parsed.issue ??
          "Optional JSON object merged into the provider request body through OpenAI-compatible extra_body."
        }
        htmlFor={id}
        label="Custom Request Params"
      >
        <textarea
          className={`settings-json-textarea${parsed.issue ? " settings-json-textarea--invalid" : ""}`}
          disabled={disabled}
          id={id}
          onChange={(event) => {
            const nextDraft = event.target.value;
            const nextParsed = parseJsonObjectDraft(nextDraft);
            setDraft(nextDraft);
            onValidationChange(nextParsed.issue);
            if (!nextParsed.issue) {
              onChange(nextParsed.value);
            }
          }}
          placeholder={`{\n  "max_tokens": 2048\n}`}
          spellCheck={false}
          value={draft}
        />
      </Field>
    </div>
  );
}

function stringifyJsonObject(value: JsonObject | null) {
  return value ? JSON.stringify(value, null, 2) : "";
}

function parseJsonObjectDraft(text: string): { issue: string | null; value: JsonObject | null } {
  const trimmed = text.trim();
  if (!trimmed) {
    return { issue: null, value: null };
  }

  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { issue: "Custom request params must be a JSON object.", value: null };
    }
    return { issue: null, value: parsed as JsonObject };
  } catch {
    return { issue: "Custom request params must contain valid JSON.", value: null };
  }
}

function buildValidationIssues(
  modelSettings: ModelSettingsDocument,
  skillSettings: SkillSettingsDocument,
  appSettings: AppSettingsDocument
) {
  const issues: string[] = [];
  if (getDuplicateNames(modelSettings.chat_models).length > 0) {
    issues.push("Chat model names must be unique.");
  }
  if (getDuplicateNames(modelSettings.embedding_models).length > 0) {
    issues.push("Embedding model names must be unique.");
  }
  if (getDuplicateNames(skillSettings.skills).length > 0) {
    issues.push("Skill names must be unique.");
  }
  if (appSettings.chunk_overlap >= appSettings.chunk_size) {
    issues.push("Overlap size must be smaller than chunk size.");
  }
  return issues;
}
