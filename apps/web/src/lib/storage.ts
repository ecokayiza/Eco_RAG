const STORAGE_KEYS = {
  sessionId: "echo.session-id",
} as const;

export const storage = {
  getSessionId() {
    try {
      return window.localStorage.getItem(STORAGE_KEYS.sessionId);
    } catch {
      return null;
    }
  },
  setSessionId(value: string) {
    try {
      window.localStorage.setItem(STORAGE_KEYS.sessionId, value);
    } catch {
      // URL state is still authoritative when storage is unavailable.
    }
  },
};
