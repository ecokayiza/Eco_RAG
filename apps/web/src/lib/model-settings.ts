import type { ChatModelConfig, EmbeddingModelConfig, JsonObject, ModelSettingsDocument } from "@/types/chat";

import { trimOrNull } from "./format";

function normalizeNumber(value: number | null | undefined, fallback: number | null) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function normalizePositiveInteger(value: number | null | undefined, fallback: number | null) {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : fallback;
}

function normalizeJsonObject(value: unknown): JsonObject | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  try {
    return JSON.parse(JSON.stringify(value)) as JsonObject;
  } catch {
    return null;
  }
}

function normalizeWireApi(value: unknown): ChatModelConfig["wire_api"] {
  return value === "responses" ? "responses" : "chat_completions";
}

export function createEmptyChatModel(index: number, initial?: Partial<ChatModelConfig>): ChatModelConfig {
  return {
    name: initial?.name ?? `Chat Model ${index}`,
    model: initial?.model ?? null,
    api_key: initial?.api_key ?? null,
    base_url: initial?.base_url ?? null,
    wire_api: normalizeWireApi(initial?.wire_api),
    temperature: initial?.temperature ?? null,
    custom_request_params: normalizeJsonObject(initial?.custom_request_params),
  };
}

export function createEmptyEmbeddingModel(index: number, initial?: Partial<EmbeddingModelConfig>): EmbeddingModelConfig {
  return {
    name: initial?.name ?? `Embedding Model ${index}`,
    model: initial?.model ?? null,
    api_key: initial?.api_key ?? null,
    base_url: initial?.base_url ?? null,
    batch_size: initial?.batch_size ?? null,
  };
}

export function normalizeChatModelConfig(
  config: Partial<ChatModelConfig> | ChatModelConfig | null | undefined,
  index = 1
): ChatModelConfig {
  return {
    name: trimOrNull(config?.name) ?? trimOrNull(config?.model) ?? `Chat Model ${index}`,
    model: trimOrNull(config?.model),
    api_key: trimOrNull(config?.api_key),
    base_url: trimOrNull(config?.base_url),
    wire_api: normalizeWireApi(config?.wire_api),
    temperature: normalizeNumber(config?.temperature, null),
    custom_request_params: normalizeJsonObject(config?.custom_request_params),
  };
}

export function normalizeEmbeddingModelConfig(
  config: Partial<EmbeddingModelConfig> | EmbeddingModelConfig | null | undefined,
  index = 1
): EmbeddingModelConfig {
  return {
    name: trimOrNull(config?.name) ?? trimOrNull(config?.model) ?? `Embedding Model ${index}`,
    model: trimOrNull(config?.model),
    api_key: trimOrNull(config?.api_key),
    base_url: trimOrNull(config?.base_url),
    batch_size: normalizePositiveInteger(config?.batch_size, null),
  };
}

export function normalizeModelSettingsDocument(
  config: Partial<ModelSettingsDocument> | ModelSettingsDocument | null | undefined
): ModelSettingsDocument {
  const chatModels = Array.isArray(config?.chat_models)
    ? config.chat_models.map((item, index) => normalizeChatModelConfig(item, index + 1))
    : [];
  const embeddingModels = Array.isArray(config?.embedding_models)
    ? config.embedding_models.map((item, index) => normalizeEmbeddingModelConfig(item, index + 1))
    : [];

  const activeChatModel = trimOrNull(config?.active_chat_model);
  const activeEmbeddingModel = trimOrNull(config?.active_embedding_model);

  return {
    active_chat_model:
      chatModels.length === 0
        ? null
        : chatModels.some((item) => item.name === activeChatModel)
          ? activeChatModel
          : chatModels[0].name,
    active_embedding_model:
      embeddingModels.length === 0
        ? null
        : embeddingModels.some((item) => item.name === activeEmbeddingModel)
          ? activeEmbeddingModel
          : embeddingModels[0].name,
    chat_models: chatModels,
    embedding_models: embeddingModels,
  };
}

export function getActiveChatModel(config: ModelSettingsDocument): ChatModelConfig | null {
  return config.chat_models.find((item) => item.name === config.active_chat_model) ?? config.chat_models[0] ?? null;
}

export function getActiveEmbeddingModel(config: ModelSettingsDocument): EmbeddingModelConfig | null {
  return (
    config.embedding_models.find((item) => item.name === config.active_embedding_model) ??
    config.embedding_models[0] ??
    null
  );
}
