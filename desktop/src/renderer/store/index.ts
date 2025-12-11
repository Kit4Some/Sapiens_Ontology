/**
 * Zustand Store for Ontology Reasoning Desktop App
 *
 * Centralized state management with LocalStorage persistence.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import type {
  QueryResponse,
  StreamStepData,
  HealthResponse,
  AppSettings,
  LLMSettings,
  ChatSession,
  ChatMessage,
} from '../types';

// ============================================================================
// Types
// ============================================================================

export interface QueryHistoryItem {
  id: string;
  query: string;
  timestamp: string;
  response: QueryResponse | null;
  error: string | null;
}

interface QueryState {
  isLoading: boolean;
  isStreaming: boolean;
  currentQuery: string;
  response: QueryResponse | null;
  streamSteps: StreamStepData[];
  error: string | null;
}

interface HealthState {
  health: HealthResponse | null;
  isLoading: boolean;
  lastChecked: string | null;
}

interface UIState {
  isSettingsOpen: boolean;
  isSidebarOpen: boolean;
  activeTab: 'query' | 'history' | 'schema';
}

interface AppState {
  // Query State
  query: QueryState;

  // Health State
  health: HealthState;

  // Settings (persisted)
  settings: AppSettings;

  // Query History (persisted) - kept for backward compatibility
  queryHistory: QueryHistoryItem[];

  // Chat Sessions (persisted)
  sessions: ChatSession[];
  activeSessionId: string | null;

  // UI State
  ui: UIState;
}

interface AppActions {
  // Query Actions
  setLoading: (isLoading: boolean) => void;
  setStreaming: (isStreaming: boolean) => void;
  setCurrentQuery: (query: string) => void;
  setResponse: (response: QueryResponse | null) => void;
  addStreamStep: (step: StreamStepData) => void;
  clearStreamSteps: () => void;
  setError: (error: string | null) => void;
  clearQueryState: () => void;

  // Health Actions
  setHealth: (health: HealthResponse | null) => void;
  setHealthLoading: (isLoading: boolean) => void;

  // Settings Actions
  updateSettings: (settings: Partial<AppSettings>) => void;
  updateLLMSettings: (llmSettings: Partial<LLMSettings>) => void;
  resetSettings: () => void;

  // History Actions (kept for backward compatibility)
  addToHistory: (item: Omit<QueryHistoryItem, 'id' | 'timestamp'>) => void;
  removeFromHistory: (id: string) => void;
  clearHistory: () => void;

  // Session Actions
  createSession: () => string;
  setActiveSession: (id: string | null) => void;
  deleteSession: (id: string) => void;
  updateSessionTitle: (sessionId: string, title: string) => void;

  // Message Actions
  addUserMessage: (sessionId: string, content: string) => string;
  addAssistantMessage: (sessionId: string, message: Partial<ChatMessage>) => string;
  updateAssistantMessage: (sessionId: string, messageId: string, updates: Partial<ChatMessage>) => void;

  // UI Actions
  setSettingsOpen: (isOpen: boolean) => void;
  setSidebarOpen: (isOpen: boolean) => void;
  setActiveTab: (tab: UIState['activeTab']) => void;
}

// ============================================================================
// Default Values
// ============================================================================

const DEFAULT_SETTINGS: AppSettings = {
  apiUrl: 'http://localhost:8000',
  maxIterations: 5,
  theme: 'dark',
  llm: {
    provider: 'openai',
    model: 'gpt-4o-mini',
    openaiApiKey: '',
    anthropicApiKey: '',
  },
};

const DEFAULT_QUERY_STATE: QueryState = {
  isLoading: false,
  isStreaming: false,
  currentQuery: '',
  response: null,
  streamSteps: [],
  error: null,
};

const DEFAULT_HEALTH_STATE: HealthState = {
  health: null,
  isLoading: false,
  lastChecked: null,
};

const DEFAULT_UI_STATE: UIState = {
  isSettingsOpen: false,
  isSidebarOpen: false,
  activeTab: 'query',
};

// ============================================================================
// Store
// ============================================================================

export const useAppStore = create<AppState & AppActions>()(
  persist(
    (set) => ({
      // Initial State
      query: DEFAULT_QUERY_STATE,
      health: DEFAULT_HEALTH_STATE,
      settings: DEFAULT_SETTINGS,
      queryHistory: [],
      sessions: [],
      activeSessionId: null,
      ui: DEFAULT_UI_STATE,

      // Query Actions
      setLoading: (isLoading) =>
        set((state) => ({
          query: { ...state.query, isLoading },
        })),

      setStreaming: (isStreaming) =>
        set((state) => ({
          query: { ...state.query, isStreaming },
        })),

      setCurrentQuery: (currentQuery) =>
        set((state) => ({
          query: { ...state.query, currentQuery },
        })),

      setResponse: (response) =>
        set((state) => ({
          query: { ...state.query, response },
        })),

      addStreamStep: (step) =>
        set((state) => ({
          query: {
            ...state.query,
            streamSteps: [...state.query.streamSteps, step],
          },
        })),

      clearStreamSteps: () =>
        set((state) => ({
          query: { ...state.query, streamSteps: [] },
        })),

      setError: (error) =>
        set((state) => ({
          query: { ...state.query, error },
        })),

      clearQueryState: () =>
        set((state) => ({
          query: {
            ...DEFAULT_QUERY_STATE,
            currentQuery: state.query.currentQuery,
          },
        })),

      // Health Actions
      setHealth: (health) =>
        set((state) => ({
          health: {
            ...state.health,
            health,
            lastChecked: new Date().toISOString(),
          },
        })),

      setHealthLoading: (isLoading) =>
        set((state) => ({
          health: { ...state.health, isLoading },
        })),

      // Settings Actions
      updateSettings: (newSettings) =>
        set((state) => ({
          settings: { ...state.settings, ...newSettings },
        })),

      updateLLMSettings: (llmSettings) =>
        set((state) => ({
          settings: {
            ...state.settings,
            llm: { ...state.settings.llm, ...llmSettings },
          },
        })),

      resetSettings: () =>
        set(() => ({
          settings: DEFAULT_SETTINGS,
        })),

      // History Actions
      addToHistory: (item) =>
        set((state) => {
          const historyItem: QueryHistoryItem = {
            ...item,
            id: crypto.randomUUID(),
            timestamp: new Date().toISOString(),
          };
          // Keep last 50 queries
          const history = [historyItem, ...state.queryHistory].slice(0, 50);
          return { queryHistory: history };
        }),

      removeFromHistory: (id) =>
        set((state) => ({
          queryHistory: state.queryHistory.filter((item) => item.id !== id),
        })),

      clearHistory: () =>
        set(() => ({
          queryHistory: [],
        })),

      // Session Actions
      createSession: () => {
        const newSession: ChatSession = {
          id: crypto.randomUUID(),
          title: 'New Chat',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          messages: [],
        };
        set((state) => ({
          sessions: [newSession, ...state.sessions],
          activeSessionId: newSession.id,
        }));
        return newSession.id;
      },

      setActiveSession: (id) =>
        set(() => ({
          activeSessionId: id,
        })),

      deleteSession: (id) =>
        set((state) => {
          const newSessions = state.sessions.filter((s) => s.id !== id);
          const newActiveId = state.activeSessionId === id
            ? (newSessions[0]?.id || null)
            : state.activeSessionId;
          return {
            sessions: newSessions,
            activeSessionId: newActiveId,
          };
        }),

      updateSessionTitle: (sessionId, title) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId
              ? { ...s, title, updatedAt: new Date().toISOString() }
              : s
          ),
        })),

      // Message Actions
      addUserMessage: (sessionId, content) => {
        const messageId = crypto.randomUUID();
        const newMessage: ChatMessage = {
          id: messageId,
          role: 'user',
          content,
          timestamp: new Date().toISOString(),
        };
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId
              ? {
                  ...s,
                  messages: [...s.messages, newMessage],
                  updatedAt: new Date().toISOString(),
                  // Auto-set title from first message
                  title: s.messages.length === 0
                    ? content.slice(0, 50) + (content.length > 50 ? '...' : '')
                    : s.title,
                }
              : s
          ),
        }));
        return messageId;
      },

      addAssistantMessage: (sessionId, message) => {
        const messageId = message.id || crypto.randomUUID();
        const newMessage: ChatMessage = {
          id: messageId,
          role: 'assistant',
          content: message.content ?? null,
          timestamp: message.timestamp || new Date().toISOString(),
          response: message.response,
          streamSteps: message.streamSteps || [],
          error: message.error,
          isStreaming: message.isStreaming ?? false,
        };
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId
              ? {
                  ...s,
                  messages: [...s.messages, newMessage],
                  updatedAt: new Date().toISOString(),
                }
              : s
          ),
        }));
        return messageId;
      },

      updateAssistantMessage: (sessionId, messageId, updates) =>
        set((state) => ({
          sessions: state.sessions.map((s) =>
            s.id === sessionId
              ? {
                  ...s,
                  messages: s.messages.map((m) =>
                    m.id === messageId
                      ? {
                          ...m,
                          ...updates,
                          streamSteps: updates.streamSteps !== undefined
                            ? updates.streamSteps
                            : m.streamSteps,
                        }
                      : m
                  ),
                  updatedAt: new Date().toISOString(),
                }
              : s
          ),
        })),

      // UI Actions
      setSettingsOpen: (isOpen) =>
        set((state) => ({
          ui: { ...state.ui, isSettingsOpen: isOpen },
        })),

      setSidebarOpen: (isOpen) =>
        set((state) => ({
          ui: { ...state.ui, isSidebarOpen: isOpen },
        })),

      setActiveTab: (tab) =>
        set((state) => ({
          ui: { ...state.ui, activeTab: tab },
        })),
    }),
    {
      name: 'ontology-reasoning-storage',
      storage: createJSONStorage(() => localStorage),
      // Only persist settings, history, and sessions
      partialize: (state) => ({
        settings: state.settings,
        queryHistory: state.queryHistory,
        sessions: state.sessions,
        activeSessionId: state.activeSessionId,
      }),
      // Migrate old localStorage data - merge with defaults to handle missing fields
      merge: (persistedState, currentState) => {
        const persisted = persistedState as Partial<AppState> | undefined;
        if (!persisted) return currentState;

        // Deep merge settings with defaults to handle missing fields like 'llm'
        const mergedSettings: AppSettings = {
          ...DEFAULT_SETTINGS,
          ...persisted.settings,
          llm: {
            ...DEFAULT_SETTINGS.llm,
            ...(persisted.settings?.llm || {}),
          },
        };

        // Migrate old queryHistory to sessions if sessions is empty
        let sessions = persisted.sessions || [];
        let activeSessionId = persisted.activeSessionId || null;

        if (sessions.length === 0 && persisted.queryHistory && persisted.queryHistory.length > 0) {
          // Convert old history to a single session
          const messages: ChatMessage[] = [];
          persisted.queryHistory.forEach((item) => {
            // Add user message
            messages.push({
              id: crypto.randomUUID(),
              role: 'user',
              content: item.query,
              timestamp: item.timestamp,
            });
            // Add assistant message
            messages.push({
              id: crypto.randomUUID(),
              role: 'assistant',
              content: item.response?.answer || null,
              timestamp: item.timestamp,
              response: item.response || undefined,
              error: item.error,
              isStreaming: false,
            });
          });

          const migratedSession: ChatSession = {
            id: crypto.randomUUID(),
            title: 'Previous Conversations',
            createdAt: persisted.queryHistory[persisted.queryHistory.length - 1]?.timestamp || new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            messages: messages.reverse(), // Oldest first
          };

          sessions = [migratedSession];
          activeSessionId = migratedSession.id;
        }

        return {
          ...currentState,
          settings: mergedSettings,
          queryHistory: persisted.queryHistory || [],
          sessions,
          activeSessionId,
        };
      },
    }
  )
);

// ============================================================================
// Selector Hooks
// ============================================================================

export const useQueryState = () => useAppStore((state) => state.query);
export const useHealthState = () => useAppStore((state) => state.health);
export const useSettings = () => useAppStore((state) => state.settings);
export const useQueryHistory = () => useAppStore((state) => state.queryHistory);
export const useUIState = () => useAppStore((state) => state.ui);
export const useSessions = () => useAppStore((state) => state.sessions);
export const useActiveSessionId = () => useAppStore((state) => state.activeSessionId);
export const useActiveSession = () => useAppStore((state) =>
  state.sessions.find((s) => s.id === state.activeSessionId) || null
);

// Actions selectors (stable references)
export const useQueryActions = () => ({
  setLoading: useAppStore((state) => state.setLoading),
  setStreaming: useAppStore((state) => state.setStreaming),
  setCurrentQuery: useAppStore((state) => state.setCurrentQuery),
  setResponse: useAppStore((state) => state.setResponse),
  addStreamStep: useAppStore((state) => state.addStreamStep),
  clearStreamSteps: useAppStore((state) => state.clearStreamSteps),
  setError: useAppStore((state) => state.setError),
  clearQueryState: useAppStore((state) => state.clearQueryState),
});

export const useSettingsActions = () => ({
  updateSettings: useAppStore((state) => state.updateSettings),
  updateLLMSettings: useAppStore((state) => state.updateLLMSettings),
  resetSettings: useAppStore((state) => state.resetSettings),
});

export const useHistoryActions = () => ({
  addToHistory: useAppStore((state) => state.addToHistory),
  removeFromHistory: useAppStore((state) => state.removeFromHistory),
  clearHistory: useAppStore((state) => state.clearHistory),
});

export const useUIActions = () => ({
  setSettingsOpen: useAppStore((state) => state.setSettingsOpen),
  setSidebarOpen: useAppStore((state) => state.setSidebarOpen),
  setActiveTab: useAppStore((state) => state.setActiveTab),
});

export const useSessionActions = () => ({
  createSession: useAppStore((state) => state.createSession),
  setActiveSession: useAppStore((state) => state.setActiveSession),
  deleteSession: useAppStore((state) => state.deleteSession),
  updateSessionTitle: useAppStore((state) => state.updateSessionTitle),
  addUserMessage: useAppStore((state) => state.addUserMessage),
  addAssistantMessage: useAppStore((state) => state.addAssistantMessage),
  updateAssistantMessage: useAppStore((state) => state.updateAssistantMessage),
});

export default useAppStore;
