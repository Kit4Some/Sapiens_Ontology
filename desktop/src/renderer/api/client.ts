/**
 * API Client for Ontology Reasoning Backend
 */

import type {
  HealthResponse,
  QueryResponse,
  SchemaResponse,
  StreamEvent,
  StreamStepData,
  StreamAnswerData,
  LLMSettings,
  IngestionJobResponse,
  IngestionStatusResponse,
  SupportedFormatsResponse,
} from '../types';

const DEFAULT_API_URL = 'http://localhost:8000';

// LLM config to send with requests
export interface LLMConfig {
  provider: string;
  model: string;
  api_key: string;
}

class ApiClient {
  private baseUrl: string;
  private llmConfig: LLMConfig | null = null;

  constructor(baseUrl: string = DEFAULT_API_URL) {
    this.baseUrl = baseUrl;
  }

  setBaseUrl(url: string): void {
    this.baseUrl = url;
  }

  setLLMConfig(settings: LLMSettings): void {
    this.llmConfig = {
      provider: settings.provider,
      model: settings.model,
      api_key: settings.provider === 'openai'
        ? settings.openaiApiKey
        : settings.anthropicApiKey,
    };
  }

  getLLMConfig(): LLMConfig | null {
    return this.llmConfig;
  }

  private async fetch<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // Health check
  async getHealth(): Promise<HealthResponse> {
    return this.fetch<HealthResponse>('/api/health');
  }

  // Get graph schema
  async getSchema(): Promise<SchemaResponse> {
    return this.fetch<SchemaResponse>('/api/schema');
  }

  // Synchronous query
  async query(
    query: string,
    maxIterations: number = 5
  ): Promise<QueryResponse> {
    return this.fetch<QueryResponse>('/api/query', {
      method: 'POST',
      body: JSON.stringify({
        query,
        max_iterations: maxIterations,
        llm_config: this.llmConfig,
      }),
    });
  }

  // Streaming query with SSE
  async *streamQuery(
    query: string,
    maxIterations: number = 5,
    signal?: AbortSignal
  ): AsyncGenerator<StreamEvent, void, unknown> {
    const url = `${this.baseUrl}/api/query/stream`;

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        max_iterations: maxIterations,
        llm_config: this.llmConfig,
      }),
      signal,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            try {
              const event: StreamEvent = JSON.parse(data);
              yield event;
            } catch {
              console.warn('Failed to parse SSE event:', data);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  // Execute raw Cypher
  async executeCypher(
    cypher: string,
    parameters?: Record<string, unknown>
  ): Promise<{ results: unknown[]; count: number }> {
    return this.fetch('/api/cypher', {
      method: 'POST',
      body: JSON.stringify({
        cypher,
        parameters,
      }),
    });
  }

  // Get stats
  async getStats(): Promise<{
    graph: Record<string, unknown>;
    system: Record<string, unknown>;
  }> {
    return this.fetch('/api/stats');
  }

  // =========================================================================
  // Ingestion Methods
  // =========================================================================

  // Upload files for ingestion
  async uploadFiles(
    files: File[],
    onProgress?: (progress: number) => void
  ): Promise<IngestionJobResponse> {
    const url = `${this.baseUrl}/api/ingest`;

    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    // Add LLM config if available
    if (this.llmConfig) {
      formData.append('llm_provider', this.llmConfig.provider);
      formData.append('llm_model', this.llmConfig.model);
      formData.append('llm_api_key', this.llmConfig.api_key);
    }

    const xhr = new XMLHttpRequest();

    return new Promise((resolve, reject) => {
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress(event.loaded / event.total);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch {
            reject(new Error('Invalid response'));
          }
        } else {
          try {
            const error = JSON.parse(xhr.responseText);
            reject(new Error(error.detail || `HTTP ${xhr.status}`));
          } catch {
            reject(new Error(`HTTP ${xhr.status}`));
          }
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error'));
      });

      xhr.open('POST', url);
      xhr.send(formData);
    });
  }

  // Get ingestion job status
  async getIngestionStatus(jobId: string): Promise<IngestionStatusResponse> {
    return this.fetch<IngestionStatusResponse>(`/api/ingest/${jobId}`);
  }

  // Get supported file formats
  async getSupportedFormats(): Promise<SupportedFormatsResponse> {
    return this.fetch<SupportedFormatsResponse>('/api/ingest/formats/supported');
  }

  // Poll ingestion status until completion
  async *pollIngestionStatus(
    jobId: string,
    intervalMs: number = 1000
  ): AsyncGenerator<IngestionStatusResponse, void, unknown> {
    while (true) {
      const status = await this.getIngestionStatus(jobId);
      yield status;

      if (status.status === 'completed' || status.status === 'failed') {
        break;
      }

      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
  }

  // Stream ingestion progress via SSE
  async *streamIngestionProgress(
    jobId: string,
    signal?: AbortSignal
  ): AsyncGenerator<IngestionStreamEvent, void, unknown> {
    const url = `${this.baseUrl}/api/ingest/${jobId}/stream`;

    const response = await fetch(url, {
      method: 'GET',
      headers: {
        'Accept': 'text/event-stream',
      },
      signal,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error('No response body');
    }

    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // Parse SSE events
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            try {
              const event = JSON.parse(data) as IngestionStreamEvent;
              yield event;

              // Stop if done or error
              if (event.event === 'done' || event.event === 'error') {
                return;
              }
            } catch {
              console.warn('Failed to parse SSE event:', data);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}

// Ingestion stream event type
export interface IngestionStreamEvent {
  event: 'status' | 'progress' | 'done' | 'error';
  data: {
    job_id: string;
    step?: string;
    progress?: number;
    message?: string;
    details?: Record<string, unknown>;
    timestamp?: string;
    status?: string;
    entities_created?: number;
    relations_created?: number;
    chunks_created?: number;
  };
}

// Singleton instance
export const apiClient = new ApiClient();

// Type guards for stream events
export function isStreamStepData(data: unknown): data is StreamStepData {
  return (
    typeof data === 'object' &&
    data !== null &&
    'node' in data &&
    'iteration' in data
  );
}

export function isStreamAnswerData(data: unknown): data is StreamAnswerData {
  return (
    typeof data === 'object' &&
    data !== null &&
    'answer' in data &&
    'confidence' in data
  );
}

export default apiClient;
