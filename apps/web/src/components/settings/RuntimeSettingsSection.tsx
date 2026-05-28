import { Field } from "@/components/common";
import type { AppSettingsDocument } from "@/types/chat";

interface RuntimeSettingsSectionProps {
  busy: boolean;
  settings: AppSettingsDocument;
  onUpdate: <Key extends keyof AppSettingsDocument>(key: Key, value: AppSettingsDocument[Key]) => void;
}

export function RuntimeSettingsSection({ busy, settings, onUpdate }: RuntimeSettingsSectionProps) {
  return (
    <div className="settings-editor">
      <header className="settings-editor__header">
        <div>
          <p className="settings-page__eyebrow">Indexing</p>
          <h2>Runtime Settings</h2>
        </div>
      </header>

      <div className="settings-runtime-grid">
        <section className="settings-detail">
          <header className="settings-detail__header">
            <div className="settings-detail__title-block">
              <h3>Embedding Pipeline</h3>
            </div>
          </header>

          <div className="form-grid form-grid--two">
            <Field htmlFor="runtime-chunk-size" label="Chunk Size">
              <input
                disabled={busy}
                id="runtime-chunk-size"
                min={1}
                onChange={(event) => {
                  onUpdate("chunk_size", positiveIntegerValue(event.target.value, settings.chunk_size));
                }}
                step={1}
                type="number"
                value={String(settings.chunk_size)}
              />
            </Field>
            <Field htmlFor="runtime-chunk-overlap" label="Overlap Size">
              <input
                disabled={busy}
                id="runtime-chunk-overlap"
                min={0}
                onChange={(event) => {
                  onUpdate("chunk_overlap", nonNegativeIntegerValue(event.target.value, settings.chunk_overlap));
                }}
                step={1}
                type="number"
                value={String(settings.chunk_overlap)}
              />
            </Field>
            <Field htmlFor="runtime-default-database-backend" label="Default Database Backend">
              <select
                disabled={busy}
                id="runtime-default-database-backend"
                onChange={(event) => {
                  onUpdate(
                    "default_database_backend",
                    event.target.value as AppSettingsDocument["default_database_backend"]
                  );
                }}
                value={settings.default_database_backend}
              >
                <option value="chroma">Chroma</option>
                <option value="faiss">FAISS</option>
              </select>
            </Field>
          </div>

          <label className="settings-toggle settings-toggle--switch">
            <input
              checked={settings.use_marker_pdf_loader}
              disabled={busy}
              onChange={(event) => {
                onUpdate("use_marker_pdf_loader", event.target.checked);
              }}
              type="checkbox"
            />
            <span>Use Marker for PDF loading</span>
          </label>

        </section>

        <section className="settings-detail">
          <header className="settings-detail__header">
            <div className="settings-detail__title-block">
              <h3>Workflow</h3>
            </div>
          </header>

          <div className="form-grid form-grid--two">
            <Field htmlFor="runtime-retrieve-rounds" label="Max Retrieve Rounds">
              <input
                disabled={busy}
                id="runtime-retrieve-rounds"
                min={1}
                onChange={(event) => {
                  onUpdate("max_retrieve_rounds", positiveIntegerValue(event.target.value, settings.max_retrieve_rounds));
                }}
                step={1}
                type="number"
                value={String(settings.max_retrieve_rounds)}
              />
            </Field>
            <Field htmlFor="runtime-web-search-backend" label="Web Search Backend">
              <select
                disabled={busy}
                id="runtime-web-search-backend"
                onChange={(event) => {
                  onUpdate("web_search_backend", event.target.value as AppSettingsDocument["web_search_backend"]);
                }}
                value={settings.web_search_backend}
              >
                <option value="auto">Auto</option>
                <option value="duckduckgo">DuckDuckGo</option>
                <option value="baidu">Baidu</option>
              </select>
            </Field>
          </div>

          <label className="settings-toggle settings-toggle--switch">
            <input
              checked={settings.web_fetch_screenshot_mode}
              disabled={busy}
              onChange={(event) => {
                onUpdate("web_fetch_screenshot_mode", event.target.checked);
              }}
              type="checkbox"
            />
            <span>Use screenshots for web fetch</span>
          </label>
        </section>
      </div>
    </div>
  );
}

function positiveIntegerValue(value: string, fallback: number) {
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function nonNegativeIntegerValue(value: string, fallback: number) {
  const parsed = Number.parseInt(value, 10);
  return Number.isInteger(parsed) && parsed >= 0 ? parsed : fallback;
}
