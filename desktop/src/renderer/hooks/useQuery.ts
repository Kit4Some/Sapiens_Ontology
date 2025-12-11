/**
 * Custom hook for managing query state and streaming
 */

import { useState, useCallback, useRef } from 'react';
import { apiClient, isStreamStepData, isStreamAnswerData } from '../api/client';
import type {
  QueryResponse,
  StreamStepData,
  StreamAnswerData,
  StreamEvent,
} from '../types';

interface UseQueryResult {
  isLoading: boolean;
  isStreaming: boolean;
  response: QueryResponse | null;
  streamSteps: StreamStepData[];
  error: string | null;
  submitQuery: (query: string, maxIterations?: number) => Promise<void>;
  submitStreamingQuery: (query: string, maxIterations?: number) => Promise<void>;
  cancelQuery: () => void;
  clearResults: () => void;
}

export function useQuery(): UseQueryResult {
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [streamSteps, setStreamSteps] = useState<StreamStepData[]>([]);
  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  const clearResults = useCallback(() => {
    setResponse(null);
    setStreamSteps([]);
    setError(null);
  }, []);

  const cancelQuery = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
    setIsStreaming(false);
  }, []);

  // Synchronous query
  const submitQuery = useCallback(async (query: string, maxIterations = 5) => {
    clearResults();
    setIsLoading(true);

    try {
      const result = await apiClient.query(query, maxIterations);
      setResponse(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Query failed');
    } finally {
      setIsLoading(false);
    }
  }, [clearResults]);

  // Handle stream events
  const handleStreamEvent = useCallback((event: StreamEvent) => {
    switch (event.event) {
      case 'step':
        if (isStreamStepData(event.data)) {
          setStreamSteps((prev) => [...prev, event.data as StreamStepData]);
        }
        break;

      case 'answer':
        if (isStreamAnswerData(event.data)) {
          const answerData = event.data as StreamAnswerData;
          setResponse({
            id: answerData.query_id,
            query: '',
            answer: answerData.answer,
            confidence: answerData.confidence,
            answer_type: answerData.answer_type as QueryResponse['answer_type'],
            reasoning_path: answerData.reasoning_path.map((step) => ({
              iteration: step.iteration,
              node: 'reflector' as const,
              action: step.action,
              message: step.message,
              sufficiency_score: null,
              evidence_count: 0,
              timestamp: new Date().toISOString(),
            })),
            explanation: answerData.explanation,
            iteration_count: answerData.iteration_count,
            entities_found: answerData.entities_found,
            evidence_count: answerData.evidence_count,
            errors: [],
          });
        }
        break;

      case 'error':
        setError((event.data as { message: string }).message || 'Unknown error');
        break;

      case 'done':
        // Query completed
        break;
    }
  }, []);

  // Streaming query
  const submitStreamingQuery = useCallback(
    async (query: string, maxIterations = 5) => {
      clearResults();
      setIsLoading(true);
      setIsStreaming(true);

      abortControllerRef.current = new AbortController();

      try {
        const stream = apiClient.streamQuery(
          query,
          maxIterations,
          abortControllerRef.current.signal
        );

        for await (const event of stream) {
          handleStreamEvent(event);

          if (event.event === 'done' || event.event === 'error') {
            break;
          }
        }
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          setError(err instanceof Error ? err.message : 'Streaming failed');
        }
      } finally {
        setIsLoading(false);
        setIsStreaming(false);
        abortControllerRef.current = null;
      }
    },
    [clearResults, handleStreamEvent]
  );

  return {
    isLoading,
    isStreaming,
    response,
    streamSteps,
    error,
    submitQuery,
    submitStreamingQuery,
    cancelQuery,
    clearResults,
  };
}

export default useQuery;
