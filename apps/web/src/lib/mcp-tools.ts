import type { McpToolName } from "@/types/chat";

export const knownMcpTools = [
  "load_skill",
  "date",
  "database_search",
  "web_search",
  "web_fetch",
  "workspace_list_files",
  "workspace_read_file",
  "workspace_write_file",
  "workspace_edit_file",
] as const satisfies readonly McpToolName[];

export const defaultMcpEnabledTools = ["database_search", "web_search", "web_fetch"] as const satisfies readonly McpToolName[];

export const mcpToolOptions: readonly {
  name: McpToolName;
  label: string;
  detail: string;
  group: string;
}[] = [
  {
    name: "database_search",
    label: "Local Database Search",
    detail: "Indexed files and stored knowledge.",
    group: "Search",
  },
  {
    name: "web_search",
    label: "Web Search",
    detail: "Fresh web result discovery.",
    group: "Search",
  },
  {
    name: "web_fetch",
    label: "Web Fetch",
    detail: "Page retrieval by URL.",
    group: "Search",
  },
  {
    name: "load_skill",
    label: "Skill Loader",
    detail: "On-demand skill instructions.",
    group: "Skills",
  },
  {
    name: "date",
    label: "Date / Time",
    detail: "Current date and time.",
    group: "Utility",
  },
  {
    name: "workspace_list_files",
    label: "Workspace List Files",
    detail: "List files in the workspace.",
    group: "Workspace",
  },
  {
    name: "workspace_read_file",
    label: "Workspace Read File",
    detail: "Read workspace file content.",
    group: "Workspace",
  },
  {
    name: "workspace_write_file",
    label: "Workspace Write File",
    detail: "Create or overwrite files.",
    group: "Workspace",
  },
  {
    name: "workspace_edit_file",
    label: "Workspace Edit File",
    detail: "Apply text edits to files.",
    group: "Workspace",
  },
];

export function normalizeMcpEnabledTools(value: unknown): McpToolName[] {
  if (!Array.isArray(value)) {
    return [...defaultMcpEnabledTools];
  }

  const requested = new Set(value.filter(isMcpToolName));
  return knownMcpTools.filter((name) => requested.has(name));
}

function isMcpToolName(value: unknown): value is McpToolName {
  return typeof value === "string" && (knownMcpTools as readonly string[]).includes(value);
}
