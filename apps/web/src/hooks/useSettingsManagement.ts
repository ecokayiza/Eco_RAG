import { useState } from "react";

import { api } from "@/lib/api";
import { defaultMcpEnabledTools, normalizeMcpEnabledTools } from "@/lib/mcp-tools";
import {
  createEmptyChatModel,
  createEmptyEmbeddingModel,
  getActiveChatModel,
  normalizeModelSettingsDocument,
} from "@/lib/model-settings";
import { createEmptySkill, normalizeSkillSettingsDocument } from "@/lib/skill-settings";
import type {
  AppSettingsDocument,
  ChatModelConfig,
  EmbeddingModelConfig,
  HealthResponse,
  MetaResponse,
  ModelSettingsDocument,
  SkillRecord,
  SkillSettingsDocument,
} from "@/types/chat";

type StatusTone = "neutral" | "success" | "error";

type BusyRunner = (
  label: string,
  action: () => Promise<void>,
  onError?: (message: string) => Promise<void> | void
) => Promise<void>;

interface UseSettingsManagementOptions {
  onRuntimeSettingsRefresh: (health: HealthResponse, meta: MetaResponse) => void;
  onStatus: (text: string, tone?: StatusTone, liveLabel?: string) => void;
  withBusy: BusyRunner;
}

const emptyModelSettings = normalizeModelSettingsDocument({
  chat_models: [],
  embedding_models: [],
});

const emptySkillSettings = normalizeSkillSettingsDocument({ skills: [] });

const defaultAppSettings: AppSettingsDocument = {
  chunk_size: 1000,
  chunk_overlap: 200,
  max_retrieve_rounds: 10,
  use_marker_pdf_loader: true,
  default_database_backend: "chroma",
  web_search_backend: "auto",
  web_fetch_screenshot_mode: false,
  mcp_enabled_tools: [...defaultMcpEnabledTools],
};

export function useSettingsManagement({
  onRuntimeSettingsRefresh,
  onStatus,
  withBusy,
}: UseSettingsManagementOptions) {
  const [modelSettings, setModelSettings] = useState<ModelSettingsDocument>(emptyModelSettings);
  const [modelSettingsDraft, setModelSettingsDraft] = useState<ModelSettingsDocument>(emptyModelSettings);
  const [skillSettings, setSkillSettings] = useState<SkillSettingsDocument>(emptySkillSettings);
  const [skillSettingsDraft, setSkillSettingsDraft] = useState<SkillSettingsDocument>(emptySkillSettings);
  const [appSettings, setAppSettings] = useState<AppSettingsDocument>(defaultAppSettings);
  const [appSettingsDraft, setAppSettingsDraft] = useState<AppSettingsDocument>(defaultAppSettings);
  const [settingsPageOpen, setSettingsPageOpen] = useState(false);

  const activeChatModel = getActiveChatModel(modelSettings);
  const activeModelName = activeChatModel?.name ?? "Not configured";
  const modelNames = modelSettings.chat_models.map((item) => item.name);
  const embeddingModelNames = modelSettings.embedding_models.map((item) => item.name);

  function applyPersistedSettings(
    models: ModelSettingsDocument,
    skills: SkillSettingsDocument,
    runtimeSettings: AppSettingsDocument = appSettings
  ) {
    const normalizedModels = normalizeModelSettingsDocument(models);
    const normalizedSkills = normalizeSkillSettingsDocument(skills);
    const normalizedAppSettings = normalizeAppSettings(runtimeSettings);
    setModelSettings(normalizedModels);
    setModelSettingsDraft(normalizedModels);
    setSkillSettings(normalizedSkills);
    setSkillSettingsDraft(normalizedSkills);
    setAppSettings(normalizedAppSettings);
    setAppSettingsDraft(normalizedAppSettings);
  }

  async function refreshRuntimeSettings() {
    const [health, meta] = await Promise.all([api.getHealth(), api.getMeta()]);
    onRuntimeSettingsRefresh(health, meta);
  }

  async function applySavedModelSettings(nextSettings: ModelSettingsDocument) {
    const normalizedSettings = normalizeModelSettingsDocument(nextSettings);
    setModelSettings(normalizedSettings);
    setModelSettingsDraft(normalizedSettings);
    await refreshRuntimeSettings();
  }

  async function applySavedManagementSettings(
    models: ModelSettingsDocument,
    skills: SkillSettingsDocument,
    runtimeSettings: AppSettingsDocument
  ) {
    applyPersistedSettings(models, skills, runtimeSettings);
    await refreshRuntimeSettings();
  }

  function resetDrafts() {
    setModelSettingsDraft(modelSettings);
    setSkillSettingsDraft(skillSettings);
    setAppSettingsDraft(appSettings);
  }

  function setActiveChatModel(name: string) {
    setModelSettingsDraft((current) =>
      normalizeModelSettingsDocument({
        ...current,
        active_chat_model: name,
      })
    );
  }

  function setActiveEmbeddingModel(name: string) {
    setModelSettingsDraft((current) =>
      normalizeModelSettingsDocument({
        ...current,
        active_embedding_model: name,
      })
    );
  }

  function updateChatModel<Key extends keyof ChatModelConfig>(
    index: number,
    key: Key,
    value: ChatModelConfig[Key]
  ) {
    setModelSettingsDraft((current) => {
      const previous = current.chat_models[index];
      if (!previous) {
        return current;
      }

      const nextName = key === "name" && typeof value === "string" ? value : previous.name;
      const nextActiveName =
        key === "name" && current.active_chat_model === previous.name && typeof value === "string"
          ? value
          : current.active_chat_model;

      return normalizeModelSettingsDocument({
        ...current,
        active_chat_model: nextActiveName,
        chat_models: replaceAtIndex(current.chat_models, index, {
          ...previous,
          [key]: value,
          ...(key === "name" ? { name: nextName } : {}),
        }),
      });
    });
  }

  function updateEmbeddingModel<Key extends keyof EmbeddingModelConfig>(
    index: number,
    key: Key,
    value: EmbeddingModelConfig[Key]
  ) {
    setModelSettingsDraft((current) => {
      const previous = current.embedding_models[index];
      if (!previous) {
        return current;
      }

      const nextName = key === "name" && typeof value === "string" ? value : previous.name;
      const nextActiveName =
        key === "name" && current.active_embedding_model === previous.name && typeof value === "string"
          ? value
          : current.active_embedding_model;

      return normalizeModelSettingsDocument({
        ...current,
        active_embedding_model: nextActiveName,
        embedding_models: replaceAtIndex(current.embedding_models, index, {
          ...previous,
          [key]: value,
          ...(key === "name" ? { name: nextName } : {}),
        }),
      });
    });
  }

  function addChatModel(initial?: Partial<ChatModelConfig>) {
    setModelSettingsDraft((current) =>
      normalizeModelSettingsDocument({
        ...current,
        chat_models: [...current.chat_models, createEmptyChatModel(current.chat_models.length + 1, initial)],
      })
    );
  }

  function addEmbeddingModel(initial?: Partial<EmbeddingModelConfig>) {
    setModelSettingsDraft((current) =>
      normalizeModelSettingsDocument({
        ...current,
        embedding_models: [
          ...current.embedding_models,
          createEmptyEmbeddingModel(current.embedding_models.length + 1, initial),
        ],
      })
    );
  }

  function removeChatModel(index: number) {
    setModelSettingsDraft((current) =>
      normalizeModelSettingsDocument({
        ...current,
        chat_models: current.chat_models.filter((_, itemIndex) => itemIndex !== index),
      })
    );
  }

  function removeEmbeddingModel(index: number) {
    setModelSettingsDraft((current) =>
      normalizeModelSettingsDocument({
        ...current,
        embedding_models: current.embedding_models.filter((_, itemIndex) => itemIndex !== index),
      })
    );
  }

  function updateSkill<Key extends keyof SkillRecord>(index: number, key: Key, value: SkillRecord[Key]) {
    setSkillSettingsDraft((current) => {
      const previous = current.skills[index];
      if (!previous) {
        return current;
      }

      return normalizeSkillSettingsDocument({
        skills: replaceAtIndex(current.skills, index, {
          ...previous,
          [key]: value,
          ...(key === "enabled" && value === false ? { default: false } : {}),
        }),
      });
    });
  }

  function addSkill(initial?: Partial<SkillRecord>) {
    setSkillSettingsDraft((current) =>
      normalizeSkillSettingsDocument({
        skills: [...current.skills, createEmptySkill(current.skills.length + 1, initial)],
      })
    );
  }

  function removeSkill(index: number) {
    setSkillSettingsDraft((current) => {
      const skill = current.skills[index];
      if (skill?.protected) {
        return current;
      }

      return normalizeSkillSettingsDocument({
        skills: current.skills.filter((_, itemIndex) => itemIndex !== index),
      });
    });
  }

  function updateAppSetting<Key extends keyof AppSettingsDocument>(key: Key, value: AppSettingsDocument[Key]) {
    setAppSettingsDraft((current) => normalizeAppSettings({ ...current, [key]: value }));
  }

  async function openSettingsPage() {
    resetDrafts();
    setSettingsPageOpen(true);
    await withBusy(
      "Loading settings...",
      async () => {
        const [freshModels, freshSkills, freshAppSettings] = await Promise.all([
          api.getModelSettings(),
          api.getSkills(),
          api.getAppSettings(),
        ]);
        applyPersistedSettings(freshModels, freshSkills, freshAppSettings);
      },
      async (detail) => {
        setSettingsPageOpen(false);
        onStatus(detail, "error", "Error");
      }
    );
  }

  function closeSettingsPage() {
    resetDrafts();
    setSettingsPageOpen(false);
  }

  async function selectActiveChatModel(name: string) {
    if (!name || modelSettings.active_chat_model === name) {
      return;
    }

    await withBusy(
      "Switching active model...",
      async () => {
        const nextSettings = normalizeModelSettingsDocument({
          ...modelSettings,
          active_chat_model: name,
        });
        const savedSettings = await api.updateModelSettings(nextSettings);
        await applySavedModelSettings(savedSettings);
        onStatus("Active model updated.", "success", "Ready");
      },
      async (detail) => {
        onStatus(detail, "error", "Error");
      }
    );
  }

  async function saveManagementSettings() {
    await withBusy(
      "Saving settings...",
      async () => {
        const [savedModels, savedSkills, savedAppSettings] = await Promise.all([
          api.updateModelSettings(normalizeModelSettingsDocument(modelSettingsDraft)),
          api.updateSkills(normalizeSkillSettingsDocument(skillSettingsDraft)),
          api.updateAppSettings(normalizeAppSettings(appSettingsDraft)),
        ]);
        await applySavedManagementSettings(savedModels, savedSkills, savedAppSettings);
        onStatus("Settings saved.", "success", "Ready");
      },
      async (detail) => {
        onStatus(detail, "error", "Error");
      }
    );
  }

  async function testChatModel(model: ChatModelConfig) {
    await withBusy(
      "Testing chat API...",
      async () => {
        const result = await api.testChatModel(model);
        onStatus(result.message, "success", "Ready");
      },
      async (detail) => {
        onStatus(`Chat API test failed: ${detail}`, "error", "Error");
      }
    );
  }

  async function testEmbeddingModel(model: EmbeddingModelConfig) {
    await withBusy(
      "Testing embedding API...",
      async () => {
        const result = await api.testEmbeddingModel(model);
        onStatus(result.message, "success", "Ready");
      },
      async (detail) => {
        onStatus(`Embedding API test failed: ${detail}`, "error", "Error");
      }
    );
  }

  return {
    pageOpen: settingsPageOpen,
    activeModelName,
    modelNames,
    embeddingModelNames,
    persisted: {
      modelSettings,
      skillSettings,
      appSettings,
    },
    drafts: {
      modelSettings: modelSettingsDraft,
      skillSettings: skillSettingsDraft,
      appSettings: appSettingsDraft,
    },
    actions: {
      addChatModel,
      addEmbeddingModel,
      addSkill,
      applyPersistedSettings,
      closeSettingsPage,
      openSettingsPage,
      removeChatModel,
      removeEmbeddingModel,
      removeSkill,
      saveManagementSettings,
      selectActiveChatModel,
      setActiveChatModel,
      setActiveEmbeddingModel,
      testChatModel,
      testEmbeddingModel,
      updateChatModel,
      updateEmbeddingModel,
      updateAppSetting,
      updateSkill,
    },
  };
}

function replaceAtIndex<T>(items: T[], index: number, nextItem: T) {
  return items.map((item, itemIndex) => (itemIndex === index ? nextItem : item));
}

function normalizeAppSettings(settings: Partial<AppSettingsDocument> | null | undefined): AppSettingsDocument {
  const chunkSize = positiveInteger(settings?.chunk_size, defaultAppSettings.chunk_size);
  const chunkOverlap = Math.min(nonNegativeInteger(settings?.chunk_overlap, defaultAppSettings.chunk_overlap), chunkSize - 1);
  return {
    chunk_size: chunkSize,
    chunk_overlap: chunkOverlap,
    max_retrieve_rounds: positiveInteger(settings?.max_retrieve_rounds, defaultAppSettings.max_retrieve_rounds),
    use_marker_pdf_loader:
      typeof settings?.use_marker_pdf_loader === "boolean"
        ? settings.use_marker_pdf_loader
        : defaultAppSettings.use_marker_pdf_loader,
    default_database_backend: normalizeDatabaseBackend(settings?.default_database_backend),
    web_search_backend: normalizeWebSearchBackend(settings?.web_search_backend),
    web_fetch_screenshot_mode:
      typeof settings?.web_fetch_screenshot_mode === "boolean"
        ? settings.web_fetch_screenshot_mode
        : defaultAppSettings.web_fetch_screenshot_mode,
    mcp_enabled_tools: normalizeMcpEnabledTools(settings?.mcp_enabled_tools),
  };
}

function normalizeWebSearchBackend(value: unknown): AppSettingsDocument["web_search_backend"] {
  return value === "duckduckgo" || value === "baidu" || value === "auto" ? value : "auto";
}

function normalizeDatabaseBackend(value: unknown): AppSettingsDocument["default_database_backend"] {
  return value === "faiss" || value === "chroma" ? value : "chroma";
}

function positiveInteger(value: unknown, fallback: number) {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : fallback;
}

function nonNegativeInteger(value: unknown, fallback: number) {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : fallback;
}
