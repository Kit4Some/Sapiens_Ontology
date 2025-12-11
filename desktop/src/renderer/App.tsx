/**
 * Main App Component
 *
 * Ontology Reasoning System Desktop Application
 */

import { useCallback, useEffect, useMemo, useState, useRef } from 'react';
import { Key } from 'lucide-react';
import {
  ChatLayout,
  ChatInterface,
  QueryInput,
  SettingsModal,
  FileUpload
} from './components';
import { useQuery, useHealth } from './hooks';
import {
  useSettings,
  useUIActions,
  useQueryActions,
  useSessions,
  useActiveSessionId,
  useActiveSession,
  useSessionActions
} from './store';
import { apiClient } from './api/client';
import { ThemeProvider } from './context/ThemeContext';

type TabType = 'query' | 'ingest';

export function App() {
  const settings = useSettings();
  const sessions = useSessions();
  const activeSessionId = useActiveSessionId();
  const activeSession = useActiveSession();
  const {
    createSession,
    setActiveSession,
    deleteSession,
    addUserMessage,
    addAssistantMessage,
    updateAssistantMessage
  } = useSessionActions();
  const { setSettingsOpen } = useUIActions();
  const { clearQueryState } = useQueryActions();
  const [activeTab, setActiveTab] = useState<TabType>('query');

  // Track the current assistant message ID for streaming updates
  const currentAssistantMsgIdRef = useRef<string | null>(null);

  const { health, isLoading: healthLoading } = useHealth();
  const {
    isLoading,
    isStreaming,
    response,
    streamSteps,
    error,
    submitStreamingQuery,
    cancelQuery,
  } = useQuery();

  // Check if API key is configured
  const hasApiKey = useMemo(() => {
    if (!settings.llm) return false;
    const key = settings.llm.provider === 'openai'
      ? settings.llm.openaiApiKey
      : settings.llm.anthropicApiKey;
    return key && key.length > 0;
  }, [settings.llm]);

  // Update API client when settings change
  useEffect(() => {
    if (settings.apiUrl) {
      apiClient.setBaseUrl(settings.apiUrl);
    }
  }, [settings.apiUrl]);

  // Update LLM config when settings change
  useEffect(() => {
    if (settings.llm) {
      apiClient.setLLMConfig(settings.llm);
    }
  }, [settings.llm]);

  const handleSubmit = useCallback(
    (query: string) => {
      // Ensure we have an active session
      let sessionId = activeSessionId;
      if (!sessionId) {
        sessionId = createSession();
      }

      // Add user message to session immediately
      addUserMessage(sessionId, query);

      // Create assistant message placeholder for streaming
      const assistantMsgId = addAssistantMessage(sessionId, {
        content: null,
        isStreaming: true,
        streamSteps: [],
      });
      currentAssistantMsgIdRef.current = assistantMsgId;

      // Submit the query
      submitStreamingQuery(query, settings.maxIterations);
    },
    [activeSessionId, createSession, addUserMessage, addAssistantMessage, submitStreamingQuery, settings.maxIterations]
  );

  // Update assistant message when streaming steps arrive
  useEffect(() => {
    if (streamSteps.length > 0 && activeSessionId && currentAssistantMsgIdRef.current) {
      updateAssistantMessage(activeSessionId, currentAssistantMsgIdRef.current, {
        streamSteps: [...streamSteps],
      });
    }
  }, [streamSteps, activeSessionId, updateAssistantMessage]);

  // Update assistant message when we get a response
  useEffect(() => {
    if (response && activeSessionId && currentAssistantMsgIdRef.current) {
      updateAssistantMessage(activeSessionId, currentAssistantMsgIdRef.current, {
        content: response.answer,
        response,
        isStreaming: false,
        streamSteps: [...streamSteps],
      });
      currentAssistantMsgIdRef.current = null;
    }
  }, [response, activeSessionId, streamSteps, updateAssistantMessage]);

  // Update assistant message when we get an error
  useEffect(() => {
    if (error && activeSessionId && currentAssistantMsgIdRef.current) {
      updateAssistantMessage(activeSessionId, currentAssistantMsgIdRef.current, {
        error,
        isStreaming: false,
      });
      currentAssistantMsgIdRef.current = null;
    }
  }, [error, activeSessionId, updateAssistantMessage]);

  const handleSettingsClick = useCallback(() => {
    setSettingsOpen(true);
  }, [setSettingsOpen]);

  const handleNewChat = useCallback(() => {
    createSession();
    clearQueryState();
  }, [createSession, clearQueryState]);

  const handleSessionClick = useCallback((sessionId: string) => {
    setActiveSession(sessionId);
    clearQueryState();
  }, [setActiveSession, clearQueryState]);

  const handleDeleteSession = useCallback((sessionId: string) => {
    deleteSession(sessionId);
  }, [deleteSession]);

  const isSystemReady =
    health?.status === 'healthy' || health?.status === 'degraded';

  const canSubmit = isSystemReady && hasApiKey;

  return (
    <ThemeProvider>
      <ChatLayout
        activeTab={activeTab}
        onTabChange={setActiveTab}
        onSettingsClick={handleSettingsClick}
        onNewChat={handleNewChat}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSessionClick={handleSessionClick}
        onDeleteSession={handleDeleteSession}
      >
        {/* API Key Warning Banner */}
        {!hasApiKey && (
          <div className="bg-amber-50 border-b border-amber-100 px-6 py-2">
            <div className="max-w-3xl mx-auto flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Key className="w-4 h-4 text-amber-500" />
                <span className="text-sm text-amber-700">
                  API key not configured. Please add your {settings.llm?.provider === 'anthropic' ? 'Anthropic' : 'OpenAI'} API key in settings.
                </span>
              </div>
              <button
                onClick={handleSettingsClick}
                className="text-sm font-medium text-amber-700 hover:text-amber-800 underline"
              >
                Configure
              </button>
            </div>
          </div>
        )}

        {activeTab === 'query' ? (
          <>
            <ChatInterface
              session={activeSession}
              isStreaming={isStreaming}
            />

            <div className="p-4 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm border-t border-stone-100 dark:border-gray-800 transition-colors">
              <QueryInput
                onSubmit={handleSubmit}
                onCancel={cancelQuery}
                isLoading={isLoading}
                disabled={!canSubmit}
              />
              {!isSystemReady && !healthLoading && (
                <div className="text-center mt-2 text-xs text-red-400">
                  Backend unavailable. Check connection.
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 overflow-y-auto bg-stone-50 dark:bg-gray-950 p-8 transition-colors">
            <div className="max-w-3xl mx-auto">
              <div className="mb-8">
                <h2 className="text-2xl font-serif font-bold text-stone-800 dark:text-gray-100 mb-2">
                  Knowledge Base
                </h2>
                <p className="text-stone-500 dark:text-gray-400">
                  Upload documents to expand the ontology reasoning capabilities.
                </p>
              </div>

              <div className="bg-white dark:bg-gray-900 rounded-2xl shadow-sm border border-stone-200 dark:border-gray-800 p-6 transition-colors">
                <FileUpload />
              </div>
            </div>
          </div>
        )}

        {/* Settings Modal */}
        <SettingsModal />
      </ChatLayout>
    </ThemeProvider>
  );
}

export default App;
