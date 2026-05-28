import type { MessageRecord, MetaResponse, SessionSummary } from "@/types/chat";

import { storage } from "./storage";

export function sortSessions(sessions: SessionSummary[]) {
  return [...sessions].sort((left, right) => right.updated_at.localeCompare(left.updated_at));
}

export function mergeSessions(sessions: SessionSummary[], session: SessionSummary) {
  const index = sessions.findIndex((item) => item.session_id === session.session_id);
  const nextSessions = [...sessions];

  if (index >= 0) {
    nextSessions[index] = session;
  } else {
    nextSessions.unshift(session);
  }

  return sortSessions(nextSessions);
}

export function getPromptFromMessages(messages: MessageRecord[], defaultPrompt: string) {
  const systemMessage = messages.find((message) => message.role === "system");
  return systemMessage?.content ?? defaultPrompt;
}

export function findPreviousUserIndex(messages: MessageRecord[], startIndex: number) {
  for (let index = startIndex - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") {
      return index;
    }
  }
  return -1;
}

export function getPreferredSessionId(sessions: SessionSummary[]) {
  const preferredId = new URLSearchParams(window.location.search).get("session") ?? storage.getSessionId();
  if (!preferredId) {
    return sessions[0]?.session_id ?? null;
  }

  const matched = sessions.find((session) => session.session_id === preferredId);
  return matched?.session_id ?? sessions[0]?.session_id ?? null;
}

export function getDefaultPrompt(meta: MetaResponse | null) {
  return meta?.default_system_prompt ?? "";
}
