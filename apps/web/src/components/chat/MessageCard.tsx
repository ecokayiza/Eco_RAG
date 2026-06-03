import {
  ArrowPathIcon,
  ArrowUturnLeftIcon,
  CheckIcon,
  ClipboardDocumentIcon,
  PencilSquareIcon,
  SparklesIcon,
  TrashIcon,
  XMarkIcon,
} from "@heroicons/react/24/outline";
import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";

import { IconActionMenu } from "@/components/common/IconActionMenu";
import { formatTokenTotal, formatTokenUsage } from "@/lib/format";
import type { MessageAttachment, MessageRecord, WorkflowSnapshot } from "@/types/chat";

import { MarkdownMessage } from "./MarkdownMessage";

interface ThoughtEntry {
  label: string;
  content: string;
  attachments?: MessageAttachment[] | null;
  kind?: "tool" | "plan" | "reasoning" | "think" | "error";
  level?: string;
  summary?: string;
  summaryItems?: ThoughtSummaryItem[];
  thinkPayload?: ThinkPayload;
}

interface ThoughtSummaryItem {
  kind: "query";
  value: string;
}

interface ThinkPayload {
  reasoning: string;
  information: string[];
}

interface WorkflowSections {
  blocks: Record<string, string>;
  present: Set<string>;
}

interface MessageCardProps {
  message: MessageRecord;
  workflowMessages: MessageRecord[];
  onDelete: (message: MessageRecord) => void;
  onEdit: (message: MessageRecord, content: string) => void;
  onRegenerate: (message: MessageRecord) => void;
  onRollback: (message: MessageRecord) => void;
}

export function MessageCard({
  message,
  workflowMessages,
  onDelete,
  onEdit,
  onRegenerate,
  onRollback,
}: MessageCardProps) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);
  const [thoughtsOpen, setThoughtsOpen] = useState(!!message.pending);
  const thoughtsContentRef = useRef<HTMLDivElement>(null);
  const shouldStickThoughtsToBottomRef = useRef(true);

  const totalTokenLabel = formatTokenTotal(message.token_usage);
  const usageLabel = formatTokenUsage(message.token_usage);
  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";
  const workflowAnswer = isAssistant ? extractWorkflowBlock(message.content, "answer") : "";
  const displayContent = workflowAnswer || (message.pending && hasWorkflowDecisionBlock(message.content) ? "" : message.content);
  const isWorkflowAnswerProxy = Boolean(workflowAnswer);
  const isReadOnly =
    message.role === "tool" || (["plan", "think"].includes(message.message_type ?? "") && !isWorkflowAnswerProxy);
  const canCopy = isAssistant || message.role === "tool";
  const thoughtEntries = useMemo(() => buildThoughtEntries(message, workflowMessages), [message, workflowMessages]);
  const thoughtScrollSignature = useMemo(
    () =>
      thoughtEntries
        .map((entry) => `${entry.label}\u0000${entry.content}\u0000${entry.attachments?.length ?? 0}`)
        .join("\u0001"),
    [thoughtEntries],
  );

  useEffect(() => {
    if (message.pending) {
      setThoughtsOpen(true);
    }
  }, [message.pending]);

  useEffect(() => {
    const el = thoughtsContentRef.current;
    if (!thoughtsOpen || !el || !shouldStickThoughtsToBottomRef.current) {
      return;
    }

    el.scrollTop = el.scrollHeight;
  }, [thoughtScrollSignature, thoughtsOpen]);

  useEffect(() => {
    if (!editing) {
      setDraft(message.content);
    }
  }, [editing, message.content]);

  function stopEvent(event: MouseEvent<HTMLElement>) {
    event.preventDefault();
    event.stopPropagation();
  }

  function handleThoughtsScroll() {
    const el = thoughtsContentRef.current;
    if (!el) {
      return;
    }

    shouldStickThoughtsToBottomRef.current = isNearScrollBottom(el);
  }

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(displayContent);
    } catch {}
  }

  function submitEdit() {
    const nextContent = draft.trim();
    if (!nextContent || nextContent === message.content) {
      setEditing(false);
      setDraft(message.content);
      return;
    }

    onEdit(message, nextContent);
    setEditing(false);
  }

  const editButtons = editing ? (
    <>
      <button
        aria-label="Save message"
        className="message-action__button message-action__button--editing"
        onClick={(event) => {
          stopEvent(event);
          submitEdit();
        }}
        type="button"
      >
        <CheckIcon />
      </button>
      <button
        aria-label="Cancel edit"
        className="message-action__button message-action__button--editing"
        onClick={(event) => {
          stopEvent(event);
          setEditing(false);
          setDraft(message.content);
        }}
        type="button"
      >
        <XMarkIcon />
      </button>
    </>
  ) : null;

  const menuTrigger = !editing ? (
    <IconActionMenu
      items={[
        ...(canCopy
          ? [
              {
                key: "copy",
                label: "Copy message",
                icon: ClipboardDocumentIcon,
                onSelect: () => {
                  void handleCopy();
                },
              },
            ]
          : []),
        ...(!isReadOnly
          ? [
              {
                key: "edit",
                label: "Edit message",
                icon: PencilSquareIcon,
                onSelect: () => {
                  setEditing(true);
                  setDraft(message.content);
                },
              },
            ]
          : []),
        ...(!isReadOnly && isAssistant
          ? [
              {
                key: "regenerate",
                label: "Regenerate message",
                icon: ArrowPathIcon,
                onSelect: () => {
                  onRegenerate(message);
                },
              },
            ]
          : []),
        ...(!isReadOnly && isUser
          ? [
              {
                key: "rollback",
                label: "Rollback session",
                icon: ArrowUturnLeftIcon,
                onSelect: () => {
                  onRollback(message);
                },
              },
            ]
          : []),
        ...(!isReadOnly
          ? [
              {
                key: "delete",
                label: "Delete message",
                icon: TrashIcon,
                danger: true,
                onSelect: () => {
                  onDelete(message);
                },
              },
            ]
          : []),
      ]}
      triggerClassName="message-action__button message-action__button--subtle"
      triggerLabel="Message actions"
    />
  ) : null;

  return (
    <article className={`message-row message-row--${message.role}${message.pending ? " message-row--pending" : ""}`}>
      <div className={`message-card message-card--${message.role}${message.pending ? " message-card--pending" : ""}`}>
        {thoughtEntries.length > 0 ? (
          <details
            className="message-thoughts"
            open={thoughtsOpen}
            onToggle={(event) => setThoughtsOpen(event.currentTarget.open)}
          >
            <summary className="message-thoughts__summary">
              <SparklesIcon className="message-thoughts__icon" /> Thoughts
            </summary>
            <div className="message-thoughts__content" onScroll={handleThoughtsScroll} ref={thoughtsContentRef}>
              {thoughtEntries.map((entry, index) =>
                entry.kind === "tool" ? (
                  <details
                    key={`${entry.label}-${index}`}
                    className="message-thoughts__tool"
                    data-level={entry.level}
                    title={entry.summary}
                  >
                    <summary className="message-thoughts__tool-summary">
                      <span className="message-thoughts__log-node">&lt;{entry.label}&gt;</span>
                      {entry.summaryItems?.length ? (
                        <span className="message-thoughts__query-list">
                          {entry.summaryItems.map((item, itemIndex) => (
                            <span className="message-thoughts__query-chip" key={`${item.value}-${itemIndex}`}>
                              {item.value}
                            </span>
                          ))}
                        </span>
                      ) : null}
                    </summary>
                    <div className="message-thoughts__tool-body">
                      <MarkdownMessage className="message-thoughts__log-text" content={entry.content} />
                      <MessageAttachments attachments={entry.attachments} compact />
                    </div>
                  </details>
                ) : (
                  <div
                    key={`${entry.label}-${index}`}
                    className={`message-thoughts__log${
                      entry.kind === "plan" || entry.kind === "think" ? ` message-thoughts__log--${entry.kind}` : ""
                    }${entry.kind === "plan" || (entry.kind === "think" && entry.thinkPayload) ? " message-thoughts__log--untagged" : ""}`}
                    data-level={entry.level}
                  >
                    {entry.kind === "plan" || (entry.kind === "think" && entry.thinkPayload) ? null : (
                      <span className="message-thoughts__log-node">&lt;{entry.label}&gt;</span>
                    )}
                    {entry.kind === "plan" ? (
                      <PlanThought content={entry.content} />
                    ) : entry.kind === "think" && entry.thinkPayload ? (
                      <ThinkThought payload={entry.thinkPayload} />
                    ) : (
                      <div className="message-thoughts__log-body">
                        <MarkdownMessage className="message-thoughts__log-text" content={entry.content} />
                        <MessageAttachments attachments={entry.attachments} compact />
                      </div>
                    )}
                  </div>
                ),
              )}
            </div>
          </details>
        ) : null}

        {editing ? (
          <div
            className="message-card__body message-card__body--editing"
            contentEditable
            suppressContentEditableWarning
            onInput={(event) => {
              setDraft(event.currentTarget.innerText || "");
            }}
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                event.preventDefault();
                submitEdit();
                return;
              }
              if (event.key === "Escape") {
                event.preventDefault();
                event.currentTarget.innerText = displayContent as string;
                setEditing(false);
                setDraft(displayContent as string);
              }
            }}
            ref={(el) => {
              if (el && document.activeElement !== el) {
                el.focus();
                try {
                  const range = document.createRange();
                  const selection = window.getSelection();
                  range.selectNodeContents(el);
                  range.collapse(false);
                  selection?.removeAllRanges();
                  selection?.addRange(range);
                } catch {}
              }
            }}
          >
            {displayContent}
          </div>
        ) : (
          <MarkdownMessage className="message-card__body" content={displayContent} />
        )}
        <MessageAttachments attachments={message.attachments} />

        {totalTokenLabel || !message.pending ? (
          <div className="message-card__footer">
            {totalTokenLabel ? (
              <span className="message-card__usage" title={usageLabel || totalTokenLabel}>
                {totalTokenLabel}
              </span>
            ) : (
              <span />
            )}

            {!message.pending ? (
              !isUser || editing ? <div className="message-actions message-actions--footer">{editButtons ?? menuTrigger}</div> : null
            ) : null}
          </div>
        ) : null}
      </div>

      {!message.pending && isUser && !editing ? <div className="message-actions message-actions--outside">{menuTrigger}</div> : null}
    </article>
  );
}

function PlanThought({ content }: { content: string }) {
  return (
    <div className="message-thoughts__plan">
      <span className="message-thoughts__plan-label">Plan</span>
      <MarkdownMessage className="message-thoughts__plan-text" content={content} />
    </div>
  );
}

function ThinkThought({ payload }: { payload: ThinkPayload }) {
  return (
    <div className="message-thoughts__think">
      {payload.reasoning ? (
        <section className="message-thoughts__think-section">
          <span className="message-thoughts__think-label">Reasoning</span>
          <MarkdownMessage className="message-thoughts__think-text" content={payload.reasoning} />
        </section>
      ) : null}
      {payload.information.length > 0 ? (
        <section className="message-thoughts__think-section">
          <span className="message-thoughts__think-label">Information</span>
          <ul className="message-thoughts__think-list">
            {payload.information.map((item, index) => (
              <li className="message-thoughts__think-list-item" key={`${item}-${index}`}>
                <MarkdownMessage className="message-thoughts__think-text" content={item} />
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}

function MessageAttachments({
  attachments,
  compact = false,
}: {
  attachments?: MessageAttachment[] | null;
  compact?: boolean;
}) {
  const images = (attachments ?? []).filter((attachment) => attachment.type === "image" && attachment.url);
  if (images.length === 0) {
    return null;
  }

  return (
    <div className={`message-attachments${compact ? " message-attachments--compact" : ""}`}>
      {images.map((attachment, index) => (
        <a
          className="message-attachment"
          href={attachment.url}
          key={attachment.id ?? attachment.path ?? `${attachment.url}-${index}`}
          rel="noreferrer"
          target="_blank"
          title={attachment.source_url ?? attachment.title ?? "Open screenshot"}
        >
          <img
            alt={attachment.title ?? "web_fetch screenshot"}
            className="message-attachment__image"
            loading="lazy"
            src={attachment.url}
          />
        </a>
      ))}
    </div>
  );
}

function buildThoughtEntries(message: MessageRecord, workflowMessages: MessageRecord[]): ThoughtEntry[] {
  if (message.pending) {
    const pendingWorkflowMessages = includePendingWorkflowMessage(workflowMessages, message);
    const workflowThoughts = buildWorkflowMessageThoughtEntries(pendingWorkflowMessages);
    return workflowThoughts.length > 0 ? workflowThoughts : buildLiveThoughtEntries(message.workflow);
  }
  if (message.role !== "assistant" || !["answer", "plan", "think"].includes(message.message_type ?? "")) {
    return [];
  }
  return buildWorkflowMessageThoughtEntries(workflowMessages);
}

function includePendingWorkflowMessage(workflowMessages: MessageRecord[], message: MessageRecord) {
  if (
    message.role !== "assistant" ||
    !message.content.trim() ||
    !hasWorkflowDecisionBlock(message.content) ||
    workflowMessages.some((entry) => entry.id === message.id)
  ) {
    return workflowMessages;
  }

  return [
    ...workflowMessages,
    {
      ...message,
      message_type: message.message_type ?? firstWorkflowDecisionBlockName(message.content),
    },
  ];
}

function buildWorkflowMessageThoughtEntries(workflowMessages: MessageRecord[]): ThoughtEntry[] {
  return workflowMessages.flatMap<ThoughtEntry>((entry) => {
    if (entry.message_type === "tool") {
      const toolBlock = extractWorkflowBlock(entry.content, "tool");
      const toolThought = summarizeToolThought(entry.tool_name, toolBlock);
      return toolThought.content || entry.attachments?.length
        ? [
            {
              label: entry.tool_name || "tool",
              content: toolThought.content,
              attachments: entry.attachments,
              kind: "tool",
              summary: toolThought.summary,
              summaryItems: toolThought.summaryItems,
            },
          ]
        : [];
    }
    if (entry.role === "assistant") {
      return (["plan", "think"] as const).flatMap<ThoughtEntry>((section) => {
        const reasoningBlock = extractWorkflowBlock(entry.content, section);
        if (!reasoningBlock && !hasWorkflowBlock(entry.content, section)) {
          return [];
        }
        if (section === "think") {
          const thinkPayload = parseThinkPayload(reasoningBlock) ?? {
            reasoning: looksLikeJsonObjectStart(reasoningBlock) ? "" : reasoningBlock,
            information: [],
          };
          return [
            {
              label: section,
              content: reasoningBlock,
              kind: "think",
              thinkPayload,
            },
          ];
        }
        return [{ label: section, content: reasoningBlock, kind: "plan" }];
      });
    }
    return [];
  });
}

function parseThinkPayload(content: string): ThinkPayload | null {
  const stripped = stripJsonFence(content);
  const parsed = parseJsonObject(stripped);
  if (parsed) {
    return normalizeThinkPayload(parsed);
  }

  return parsePartialThinkPayload(stripped);
}

function normalizeThinkPayload(parsed: Record<string, unknown>): ThinkPayload | null {
  const reasoning = typeof parsed.reasoning === "string" ? parsed.reasoning.trim() : "";
  const informationValue = Array.isArray(parsed.valid_information)
    ? parsed.valid_information
    : Array.isArray(parsed.information)
      ? parsed.information
      : [];
  const information = informationValue.map(formatThinkInformationItem).filter(Boolean);
  return reasoning || information.length > 0 ? { reasoning, information } : null;
}

function parsePartialThinkPayload(content: string): ThinkPayload | null {
  const reasoning = extractPartialJsonStringField(content, "reasoning").trim();
  const informationSource =
    extractPartialJsonArrayField(content, "valid_information") ?? extractPartialJsonArrayField(content, "information");
  const information = informationSource ? extractPartialJsonStringArrayItems(informationSource) : [];
  return reasoning || information.length > 0 ? { reasoning, information } : null;
}

function parseJsonObject(content: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(content.trim()) as unknown;
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

function stripJsonFence(content: string) {
  const trimmed = content.trim();
  const fenced = trimmed.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  return fenced ? fenced[1].trim() : trimmed;
}

function looksLikeJsonObjectStart(content: string) {
  return stripJsonFence(content).trimStart().startsWith("{");
}

function extractPartialJsonStringField(content: string, key: string) {
  const valueStart = findJsonFieldValueStart(content, key);
  if (valueStart < 0 || content[valueStart] !== "\"") {
    return "";
  }
  return readPartialJsonString(content, valueStart).trim();
}

function extractPartialJsonArrayField(content: string, key: string) {
  const valueStart = findJsonFieldValueStart(content, key);
  if (valueStart < 0 || content[valueStart] !== "[") {
    return null;
  }

  let depth = 0;
  let quote: string | null = null;
  let escaping = false;
  for (let index = valueStart; index < content.length; index++) {
    const char = content[index];
    if (escaping) {
      escaping = false;
      continue;
    }
    if (char === "\\") {
      escaping = true;
      continue;
    }
    if (quote) {
      if (char === quote) {
        quote = null;
      }
      continue;
    }
    if (char === "\"") {
      quote = char;
      continue;
    }
    if (char === "[") {
      depth += 1;
      continue;
    }
    if (char === "]") {
      depth -= 1;
      if (depth === 0) {
        return content.slice(valueStart + 1, index);
      }
    }
  }

  return content.slice(valueStart + 1);
}

function extractPartialJsonStringArrayItems(content: string) {
  const items: string[] = [];
  for (let index = 0; index < content.length; index++) {
    if (content[index] !== "\"") {
      continue;
    }
    const result = readPartialJsonStringWithEnd(content, index);
    const item = result.value.trim();
    if (item) {
      items.push(item);
    }
    index = result.end;
  }
  return items;
}

function findJsonFieldValueStart(content: string, key: string) {
  const keyPattern = new RegExp(`"${escapeRegExp(key)}"\\s*:`, "i");
  const match = keyPattern.exec(content);
  if (!match || match.index === undefined) {
    return -1;
  }
  let index = match.index + match[0].length;
  while (index < content.length && /\s/.test(content[index])) {
    index += 1;
  }
  return index;
}

function readPartialJsonString(content: string, start: number) {
  return readPartialJsonStringWithEnd(content, start).value;
}

function readPartialJsonStringWithEnd(content: string, start: number) {
  let value = "";
  let escaping = false;
  for (let index = start + 1; index < content.length; index++) {
    const char = content[index];
    if (escaping) {
      value += decodeJsonEscape(char);
      escaping = false;
      continue;
    }
    if (char === "\\") {
      escaping = true;
      continue;
    }
    if (char === "\"") {
      return { value, end: index };
    }
    value += char;
  }
  return { value, end: content.length };
}

function decodeJsonEscape(char: string) {
  switch (char) {
    case "n":
      return "\n";
    case "r":
      return "\r";
    case "t":
      return "\t";
    case "b":
      return "\b";
    case "f":
      return "\f";
    default:
      return char;
  }
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function formatThinkInformationItem(value: unknown) {
  if (typeof value === "string") {
    return value.trim();
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return "";
}

function summarizeToolThought(
  toolName: string | null | undefined,
  content: string,
): { content: string; summary: string; summaryItems: ThoughtSummaryItem[] } {
  if (!content) {
    return { content: "", summary: formatToolCallSummary(toolName), summaryItems: [] };
  }
  const { body, heading } = splitToolThought(content);
  const summary = heading || formatToolCallSummary(toolName);
  const summaryItems = extractQuerySummaryItems(heading);
  if (toolName === "load_skill") {
    const loadedLine = body
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find((line) => line.toLowerCase().startsWith("loaded skill:"));
    return { content: loadedLine || "Loaded skill guidance.", summary, summaryItems };
  }
  return { content: body || content, summary, summaryItems };
}

function splitToolThought(content: string) {
  const trimmed = content.trim();
  const [firstLine = "", ...rest] = trimmed.split(/\r?\n/);
  if (!firstLine.includes("(") || !firstLine.endsWith(")")) {
    return { heading: "", body: trimmed };
  }

  return { heading: firstLine, body: rest.join("\n").trim() };
}

function formatToolCallSummary(toolName: string | null | undefined) {
  return `${toolName || "tool"}()`;
}

function extractQuerySummaryItems(heading: string): ThoughtSummaryItem[] {
  const args = parseToolCallArgs(heading);
  const values = ["query", "queries", "q", "search_query", "url", "urls"].flatMap((key) =>
    parseToolArgValues(args.get(key)),
  );
  return values.map((value) => ({ kind: "query", value }));
}

function parseToolCallArgs(heading: string) {
  const start = heading.indexOf("(");
  const end = heading.lastIndexOf(")");
  if (start < 0 || end <= start) {
    return new Map<string, string>();
  }

  const args = new Map<string, string>();
  for (const segment of splitTopLevel(heading.slice(start + 1, end), ",")) {
    const equalsIndex = findTopLevelEquals(segment);
    if (equalsIndex <= 0) {
      continue;
    }
    const key = segment.slice(0, equalsIndex).trim();
    const value = segment.slice(equalsIndex + 1).trim();
    if (key && value) {
      args.set(key, value);
    }
  }
  return args;
}

function parseToolArgValues(rawValue: string | undefined) {
  if (!rawValue) {
    return [];
  }

  const trimmed = rawValue.trim();
  if (!trimmed || ["None", "null", "undefined"].includes(trimmed)) {
    return [];
  }

  if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
    return splitTopLevel(trimmed.slice(1, -1), ",").map(cleanToolArgValue).filter(Boolean);
  }

  return [cleanToolArgValue(trimmed)].filter(Boolean);
}

function cleanToolArgValue(value: string) {
  const trimmed = value.trim();
  if (!trimmed || ["None", "null", "undefined"].includes(trimmed)) {
    return "";
  }
  if ((trimmed.startsWith("'") && trimmed.endsWith("'")) || (trimmed.startsWith("\"") && trimmed.endsWith("\""))) {
    return trimmed
      .slice(1, -1)
      .replace(/\\n/g, "\n")
      .replace(/\\r/g, "\r")
      .replace(/\\t/g, "\t")
      .replace(/\\'/g, "'")
      .replace(/\\"/g, "\"")
      .replace(/\\\\/g, "\\")
      .trim();
  }
  return trimmed;
}

function splitTopLevel(text: string, separator: string) {
  const parts: string[] = [];
  let start = 0;
  let depth = 0;
  let quote: string | null = null;
  let escaping = false;

  for (let index = 0; index < text.length; index++) {
    const char = text[index];
    if (escaping) {
      escaping = false;
      continue;
    }
    if (char === "\\") {
      escaping = true;
      continue;
    }
    if (quote) {
      if (char === quote) {
        quote = null;
      }
      continue;
    }
    if (char === "'" || char === "\"") {
      quote = char;
      continue;
    }
    if (char === "[" || char === "{" || char === "(") {
      depth += 1;
      continue;
    }
    if (char === "]" || char === "}" || char === ")") {
      depth = Math.max(0, depth - 1);
      continue;
    }
    if (char === separator && depth === 0) {
      parts.push(text.slice(start, index).trim());
      start = index + 1;
    }
  }

  parts.push(text.slice(start).trim());
  return parts.filter(Boolean);
}

function findTopLevelEquals(text: string) {
  let depth = 0;
  let quote: string | null = null;
  let escaping = false;

  for (let index = 0; index < text.length; index++) {
    const char = text[index];
    if (escaping) {
      escaping = false;
      continue;
    }
    if (char === "\\") {
      escaping = true;
      continue;
    }
    if (quote) {
      if (char === quote) {
        quote = null;
      }
      continue;
    }
    if (char === "'" || char === "\"") {
      quote = char;
      continue;
    }
    if (char === "[" || char === "{" || char === "(") {
      depth += 1;
      continue;
    }
    if (char === "]" || char === "}" || char === ")") {
      depth = Math.max(0, depth - 1);
      continue;
    }
    if (char === "=" && depth === 0) {
      return index;
    }
  }

  return -1;
}

function buildLiveThoughtEntries(workflow: WorkflowSnapshot | null | undefined): ThoughtEntry[] {
  if (!workflow) {
    return [];
  }
  return [
    ...workflow.errors.map((error) => ({ label: "error", content: error, kind: "error" as const, level: "error" })),
  ];
}

function extractWorkflowBlock(content: string, target: string): string {
  const sections = parseWorkflowSections(content);
  return sections.blocks[target]?.trim() ?? "";
}

function hasWorkflowBlock(content: string, target: string) {
  const sections = parseWorkflowSections(content);
  return sections.present.has(target);
}

function parseWorkflowSections(content: string): WorkflowSections {
  const sections: Record<string, string[]> = {};
  const present = new Set<string>();
  const codeRanges = markdownCodeRanges(content);
  let current: string | null = null;
  let currentStart = 0;

  function appendCurrent(end: number) {
    if (current) {
      sections[current] ??= [];
      sections[current].push(content.slice(currentStart, end));
    }
  }

  for (const match of content.matchAll(/<\/?\s*([a-z_]+)\s*>/gi)) {
    const rawTag = match[0];
    const index = match.index ?? 0;
    if (insideRanges(index, codeRanges)) {
      continue;
    }
    const name = canonicalWorkflowSectionName(match[1]);
    if (!name) {
      continue;
    }

    if (rawTag.startsWith("</")) {
      if (current === name) {
        appendCurrent(index);
        current = null;
        currentStart = index + rawTag.length;
      }
      continue;
    }

    if (current) {
      appendCurrent(index);
    }
    current = name;
    present.add(name);
    sections[current] = [];
    currentStart = index + rawTag.length;
  }

  if (current) {
    appendCurrent(content.length);
  }

  return {
    blocks: Object.fromEntries(Object.entries(sections).map(([key, value]) => [key, value.join("\n").trim()])),
    present,
  };
}

function hasWorkflowDecisionBlock(content: string) {
  const firstName = firstWorkflowDecisionBlockName(content);
  return firstName === "plan" || firstName === "think";
}

function firstWorkflowDecisionBlockName(content: string) {
  const codeRanges = markdownCodeRanges(content);
  for (const match of content.matchAll(/<\s*([a-z_]+)\s*>/gi)) {
    if (insideRanges(match.index ?? 0, codeRanges)) {
      continue;
    }
    const name = canonicalWorkflowSectionName(match[1]);
    if (name === "plan" || name === "think") {
      return name;
    }
  }
  return null;
}

function isWorkflowSectionName(name: string) {
  return canonicalWorkflowSectionName(name) != null;
}

function canonicalWorkflowSectionName(name: string) {
  const cleaned = name.toLowerCase();
  if (cleaned.startsWith("echo_")) {
    const canonical = cleaned.slice("echo_".length);
    return ["plan", "think", "answer", "tool"].includes(canonical) ? canonical : null;
  }
  return null;
}

function markdownCodeRanges(content: string) {
  const fenced = fencedCodeRanges(content);
  return mergeRanges([...fenced, ...inlineCodeRanges(content, fenced)]);
}

function fencedCodeRanges(content: string) {
  const ranges: Array<[number, number]> = [];
  let index = 0;
  while (index < content.length) {
    const lineEnd = content.indexOf("\n", index);
    const end = lineEnd < 0 ? content.length : lineEnd;
    const nextIndex = lineEnd < 0 ? content.length : lineEnd + 1;
    const line = content.slice(index, end);
    const opener = /^[ \t]{0,3}(```+)/.exec(line);
    if (!opener) {
      index = nextIndex;
      continue;
    }

    const markerLength = opener[1].length;
    let closeIndex = nextIndex;
    let rangeEnd = content.length;
    const closer = new RegExp(`^[ \\t]{0,3}\`{${markerLength},}`);
    while (closeIndex < content.length) {
      const closeLineEnd = content.indexOf("\n", closeIndex);
      const closeEnd = closeLineEnd < 0 ? content.length : closeLineEnd;
      const closeNextIndex = closeLineEnd < 0 ? content.length : closeLineEnd + 1;
      if (closer.test(content.slice(closeIndex, closeEnd))) {
        rangeEnd = closeNextIndex;
        break;
      }
      closeIndex = closeNextIndex;
    }

    ranges.push([index, rangeEnd]);
    index = rangeEnd;
  }
  return ranges;
}

function inlineCodeRanges(content: string, excludedRanges: Array<[number, number]>) {
  const ranges: Array<[number, number]> = [];
  let index = 0;
  while (index < content.length) {
    const excludedEnd = containingRangeEnd(index, excludedRanges);
    if (excludedEnd != null) {
      index = excludedEnd;
      continue;
    }
    if (content[index] !== "`") {
      index += 1;
      continue;
    }

    let markerEnd = index + 1;
    while (markerEnd < content.length && content[markerEnd] === "`") {
      markerEnd += 1;
    }
    const marker = content.slice(index, markerEnd);
    let searchIndex = markerEnd;
    while (true) {
      const closeIndex = content.indexOf(marker, searchIndex);
      if (closeIndex < 0) {
        index = markerEnd;
        break;
      }
      const closeExcludedEnd = containingRangeEnd(closeIndex, excludedRanges);
      if (closeExcludedEnd != null) {
        searchIndex = closeExcludedEnd;
        continue;
      }
      const closeEnd = closeIndex + marker.length;
      ranges.push([index, closeEnd]);
      index = closeEnd;
      break;
    }
  }
  return ranges;
}

function mergeRanges(ranges: Array<[number, number]>) {
  const merged: Array<[number, number]> = [];
  for (const [start, end] of [...ranges].sort(([left], [right]) => left - right)) {
    const previous = merged.at(-1);
    if (!previous || start > previous[1]) {
      merged.push([start, end]);
    } else {
      previous[1] = Math.max(previous[1], end);
    }
  }
  return merged;
}

function insideRanges(index: number, ranges: Array<[number, number]>) {
  return containingRangeEnd(index, ranges) != null;
}

function containingRangeEnd(index: number, ranges: Array<[number, number]>) {
  for (const [start, end] of ranges) {
    if (index < start) {
      return null;
    }
    if (start <= index && index < end) {
      return end;
    }
  }
  return null;
}

function isNearScrollBottom(element: HTMLElement, threshold = 48) {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= threshold;
}
