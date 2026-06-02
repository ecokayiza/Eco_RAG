import { CheckCircleIcon, MagnifyingGlassIcon, NoSymbolIcon, SquaresPlusIcon } from "@heroicons/react/24/outline";

import { Button } from "@/components/common";
import { defaultMcpEnabledTools, knownMcpTools, mcpToolOptions, normalizeMcpEnabledTools } from "@/lib/mcp-tools";
import type { AppSettingsDocument, McpToolName } from "@/types/chat";

interface McpSettingsSectionProps {
  busy: boolean;
  settings: AppSettingsDocument;
  onUpdate: <Key extends keyof AppSettingsDocument>(key: Key, value: AppSettingsDocument[Key]) => void;
}

export function McpSettingsSection({ busy, settings, onUpdate }: McpSettingsSectionProps) {
  const enabled = new Set(settings.mcp_enabled_tools);

  function updateTool(name: McpToolName, checked: boolean) {
    const next = checked
      ? normalizeMcpEnabledTools([...settings.mcp_enabled_tools, name])
      : normalizeMcpEnabledTools(settings.mcp_enabled_tools.filter((item) => item !== name));
    onUpdate("mcp_enabled_tools", next);
  }

  return (
    <div className="settings-editor">
      <header className="settings-editor__header">
        <div>
          <p className="settings-page__eyebrow">MCP</p>
          <h2>Tool Schemas</h2>
        </div>
        <div className="settings-mcp-actions">
          <Button
            disabled={busy}
            onClick={() => {
              onUpdate("mcp_enabled_tools", [...defaultMcpEnabledTools]);
            }}
            size="sm"
            variant="secondary"
          >
            <MagnifyingGlassIcon />
            Search Only
          </Button>
          <Button
            disabled={busy}
            onClick={() => {
              onUpdate("mcp_enabled_tools", [...knownMcpTools]);
            }}
            size="sm"
            variant="ghost"
          >
            <SquaresPlusIcon />
            All
          </Button>
          <Button
            disabled={busy}
            onClick={() => {
              onUpdate("mcp_enabled_tools", []);
            }}
            size="sm"
            variant="ghost"
          >
            <NoSymbolIcon />
            None
          </Button>
        </div>
      </header>

      <div className="settings-runtime-grid">
        <section className="settings-detail settings-detail--mcp">
          <header className="settings-detail__header">
            <div className="settings-detail__title-block">
              <h3>MCP Tools</h3>
              <div className="settings-pills">
                <em>{settings.mcp_enabled_tools.length} sent</em>
              </div>
            </div>
          </header>

          <div className="settings-mcp-grid">
            {mcpToolOptions.map((tool) => {
              const checked = enabled.has(tool.name);
              return (
                <label className={`settings-mcp-card${checked ? " settings-mcp-card--enabled" : ""}`} key={tool.name}>
                  <input
                    checked={checked}
                    disabled={busy}
                    onChange={(event) => {
                      updateTool(tool.name, event.target.checked);
                    }}
                    type="checkbox"
                  />
                  <span className="settings-mcp-card__body">
                    <span className="settings-mcp-card__title">
                      <strong>{tool.label}</strong>
                      <em>{tool.group}</em>
                    </span>
                    <code>{tool.name}</code>
                    <small>{tool.detail}</small>
                  </span>
                  {checked ? <CheckCircleIcon aria-hidden="true" /> : null}
                </label>
              );
            })}
          </div>
        </section>
      </div>
    </div>
  );
}
