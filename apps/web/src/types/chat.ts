export interface TokenUsage {
  prompt_tokens?: number;
  prompt_cache_hit_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  [key: string]: number | undefined;
}

export interface SessionSummary {
  session_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
  preview: string;
  token_usage: TokenUsage;
  total_tokens: number;
}

export type MessageRole = "system" | "user" | "assistant" | "tool";

export interface MessageAttachment {
  id?: string;
  type: "image" | string;
  kind?: string | null;
  mime_type?: string | null;
  url: string;
  path?: string | null;
  title?: string | null;
  source_url?: string | null;
  size_bytes?: number | null;
}

export interface MessageRecord {
  id: string;
  role: MessageRole;
  content: string;
  message_type?: string | null;
  workflow_turn_id?: string | null;
  tool_name?: string | null;
  token_usage?: TokenUsage | null;
  attachments?: MessageAttachment[] | null;
  pending?: boolean;
  workflow?: WorkflowSnapshot | null;
}

export interface SessionState {
  session: SessionSummary;
  messages: MessageRecord[];
  workflow_draft?: WorkflowDraftRecord | null;
}

export interface DatabaseRecord {
  id: string;
  name: string;
  collection_name: string;
  backend: "chroma" | "faiss";
  embedding_model_name: string;
  document_count: number;
  created_at: string;
  updated_at: string;
}

export interface DatabaseState {
  active_database_id: string | null;
  databases: DatabaseRecord[];
}

export interface DatabaseDocumentRecord {
  id: string;
  source_name: string;
  source_type: string;
  file_path: string | null;
  chunk_count: number;
}

export interface UploadJobFileRecord {
  id: string;
  source_name: string;
  status: string;
  chunk_count: number;
  embedded_chunks: number;
  progress: number;
  message: string;
  error: string | null;
}

export interface UploadJobRecord {
  job_id: string;
  database_id: string;
  status: string;
  message: string;
  progress: number;
  total_files: number;
  completed_files: number;
  total_chunks: number;
  embedded_chunks: number;
  current_file_name: string | null;
  files: UploadJobFileRecord[];
  error: string | null;
  error_stage: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface WorkflowNodeStatus {
  node: string;
  status: string;
  detail?: string | null;
}

export interface WorkflowLog {
  level: string;
  node?: string | null;
  message: string;
}

export type WorkflowStatus = "queued" | "running" | "completed" | "failed" | "stopped";

export interface WorkflowSnapshot {
  workflow_turn_id?: string | null;
  query: string;
  answer: string;
  status: WorkflowStatus | string;
  retrieve_round?: number | null;
  node_statuses: WorkflowNodeStatus[];
  active_node: string | null;
  tool_name?: string | null;
  logs: WorkflowLog[];
  errors: string[];
}

export interface WorkflowDraftRecord {
  session_id: string;
  user_message_id: string;
  workflow: WorkflowSnapshot;
  records: Record<string, unknown>[];
  content?: string | null;
}

export interface ChatResponse extends SessionState {
  reply: string;
  token_usage?: TokenUsage | null;
  workflow?: WorkflowSnapshot | null;
}

export interface HealthResponse {
  status: string;
  model: string | null;
}

export type JsonValue = string | number | boolean | null | JsonObject | JsonValue[];

export interface JsonObject {
  [key: string]: JsonValue;
}

export interface ChatModelConfig {
  name: string;
  model: string | null;
  api_key: string | null;
  base_url: string | null;
  wire_api: "chat_completions" | "responses";
  temperature: number | null;
  custom_request_params: JsonObject | null;
}

export interface EmbeddingModelConfig {
  name: string;
  model: string | null;
  api_key: string | null;
  base_url: string | null;
  batch_size: number | null;
}

export interface ModelSettingsDocument {
  active_chat_model: string | null;
  active_embedding_model: string | null;
  chat_models: ChatModelConfig[];
  embedding_models: EmbeddingModelConfig[];
}

export interface ModelApiTestResponse {
  ok: boolean;
  message: string;
}

export type McpToolName =
  | "load_skill"
  | "date"
  | "database_search"
  | "web_search"
  | "web_fetch"
  | "workspace_list_files"
  | "workspace_read_file"
  | "workspace_write_file"
  | "workspace_edit_file";

export interface AppSettingsDocument {
  chunk_size: number;
  chunk_overlap: number;
  max_retrieve_rounds: number;
  max_database_search_top_k: number;
  max_web_search_results: number;
  max_parallel_tool_calls: number;
  use_marker_pdf_loader: boolean;
  default_database_backend: "chroma" | "faiss";
  web_search_backend: "auto" | "duckduckgo" | "baidu";
  web_fetch_screenshot_mode: boolean;
  mcp_enabled_tools: McpToolName[];
}

export interface SkillRecord {
  name: string;
  description: string;
  content: string;
  enabled: boolean;
  default: boolean;
  protected: boolean;
}

export interface SkillSettingsDocument {
  skills: SkillRecord[];
}

export interface MetaResponse {
  workflow_statuses: string[];
  workflow_steps: string[];
  default_system_prompt: string;
}

export interface ConfirmDialogState {
  title: string;
  description: string;
  confirmLabel: string;
  tone?: "danger" | "secondary";
  onConfirm: () => Promise<void>;
}
