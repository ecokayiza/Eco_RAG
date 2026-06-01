import { useEffect, useRef } from "react";

import { EmptyState } from "@/components/common/EmptyState";
import type { MessageRecord } from "@/types/chat";

import { MessageCard } from "./MessageCard";

interface MessageListProps {
  ready: boolean;
  messages: MessageRecord[];
  onDelete: (message: MessageRecord) => void;
  onEdit: (message: MessageRecord, content: string) => void;
  onRegenerate: (message: MessageRecord) => void;
  onRollback: (message: MessageRecord) => void;
}

export function MessageList({
  messages,
  onDelete,
  onEdit,
  onRegenerate,
  onRollback,
  ready,
}: MessageListProps) {
  const listRef = useRef<HTMLDivElement | null>(null);
  const shouldStickToBottomRef = useRef(true);
  const workflowMessagesByTurn = new Map<string, MessageRecord[]>();
  const visibleMessages: MessageRecord[] = [];

  for (let i = 0; i < messages.length; i++) {
    const message = messages[i];
    if (message.role === "system") {
      continue;
    }
    if (message.workflow_turn_id) {
      const items = workflowMessagesByTurn.get(message.workflow_turn_id) ?? [];
      items.push(message);
      workflowMessagesByTurn.set(message.workflow_turn_id, items);

      // Only display the final assistant record (which generated the answer)
      let isLastAssistantInTurn = true;
      for (let j = i + 1; j < messages.length; j++) {
        if (messages[j].workflow_turn_id !== message.workflow_turn_id) break;
        if (messages[j].role === "assistant") {
          isLastAssistantInTurn = false;
          break;
        }
      }

      if (message.role === "assistant" && isLastAssistantInTurn) {
        visibleMessages.push(message);
      }
      continue;
    }
    visibleMessages.push(message);
  }

  useEffect(() => {
    const node = listRef.current;
    if (!node || !shouldStickToBottomRef.current) {
      return;
    }

    node.scrollTop = node.scrollHeight;
  }, [messages]);

  function handleScroll() {
    const node = listRef.current;
    if (!node) {
      return;
    }

    shouldStickToBottomRef.current = isNearScrollBottom(node);
  }

  return (
    <section className="message-list" aria-live="polite" onScroll={handleScroll} ref={listRef}>
      {visibleMessages.length === 0 ? (
        <EmptyState
          description={ready ? "Ask a question to create the first meaningful exchange in this session." : "Connecting to the backend and restoring the selected thread."}
          title={ready ? "Start A Conversation" : "Loading Workspace"}
        />
      ) : (
        visibleMessages.map((message) => (
          <MessageCard
            key={message.id}
            message={message}
            onDelete={onDelete}
            onEdit={onEdit}
            onRegenerate={onRegenerate}
            onRollback={onRollback}
            workflowMessages={message.workflow_turn_id ? workflowMessagesByTurn.get(message.workflow_turn_id) ?? [] : []}
          />
        ))
      )}
    </section>
  );
}

function isNearScrollBottom(element: HTMLElement, threshold = 48) {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= threshold;
}
