/**
 * TypeScript type definitions for the Ontology Reasoning App
 */

// API Response Types
export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  neo4j_connected: boolean;
  llm_available: boolean;
  version: string;
  environment: string;
}

export interface ReasoningStep {
  iteration: number;
  node: 'constructor' | 'retriever' | 'reflector' | 'responser';
  action: string;
  message: string;
  sufficiency_score: number | null;
  evidence_count: number;
  timestamp: string;
}

export interface ThinkingStep {
  step: number;
  thought: string;
  action: string;
  observation: string;
  confidence: number;
}

export interface PatternInsight {
  pattern_type: string;
  entities: string[];
  description: string;
  significance: string;
}

export interface SupportingEvidence {
  content: string;
  source_type: string;
  relevance: number;
  entity_names: string[];
}

export interface QueryResponse {
  id: string;
  query: string;
  answer: string;
  confidence: number;
  answer_type: 'DIRECT' | 'INFERRED' | 'PARTIAL' | 'UNCERTAIN';
  reasoning_path: ReasoningStep[];
  explanation: string;
  iteration_count: number;
  entities_found: number;
  evidence_count: number;
  errors: string[];
  // Extended fields for UI display
  thinking_process?: ThinkingStep[];
  patterns?: PatternInsight[];
  supporting_evidence?: SupportingEvidence[];
  direct_answer?: string;
}

export interface StreamEvent {
  event: 'step' | 'answer' | 'error' | 'done';
  data: StreamStepData | StreamAnswerData | StreamErrorData | Record<string, never>;
}

export interface StreamStepData {
  query_id: string;
  node: string;
  iteration: number;
  sufficiency_score: number;
  current_query: string;
  evidence_count: number;
  entities_count: number;
  message: string;
  timestamp: string;
}

export interface StreamAnswerData {
  query_id: string;
  answer: string;
  confidence: number;
  answer_type: string;
  explanation: string;
  iteration_count: number;
  entities_found: number;
  evidence_count: number;
  reasoning_path: Array<{
    iteration: number;
    action: string;
    message: string;
  }>;
}

export interface StreamErrorData {
  message: string;
  query_id: string;
}

export interface SchemaResponse {
  node_labels: string[];
  relationship_types: string[];
  node_properties: Record<string, Array<{ name: string; types: string[] }>>;
  indexes: Array<Record<string, unknown>>;
}

// App State Types
export interface QueryState {
  isLoading: boolean;
  isStreaming: boolean;
  query: string;
  response: QueryResponse | null;
  streamSteps: StreamStepData[];
  error: string | null;
}

// LLM Configuration Types
export type LLMProvider = 'openai' | 'anthropic';

export interface LLMModel {
  id: string;
  name: string;
  apiName: string;
  provider: LLMProvider;
  description?: string;
}

export interface LLMSettings {
  provider: LLMProvider;
  model: string;
  openaiApiKey: string;
  anthropicApiKey: string;
}

// Available LLM Models
export const OPENAI_MODELS: LLMModel[] = [
  { id: 'gpt-4o', name: 'GPT-4o', apiName: 'gpt-4o', provider: 'openai', description: 'Latest: gpt-4o-2024-08-06' },
  { id: 'gpt-4o-mini', name: 'GPT-4o Mini', apiName: 'gpt-4o-mini', provider: 'openai', description: 'Latest: gpt-4o-mini-2024-07-18' },
  { id: 'gpt-5', name: 'GPT-5', apiName: 'gpt-5', provider: 'openai', description: 'Mini: gpt-5-mini, Nano: gpt-5-nano' },
];

export const ANTHROPIC_MODELS: LLMModel[] = [
  { id: 'claude-sonnet-4', name: 'Claude Sonnet 4', apiName: 'claude-sonnet-4-20250514', provider: 'anthropic', description: 'Default Sonnet' },
  { id: 'claude-sonnet-4-5', name: 'Claude Sonnet 4.5', apiName: 'claude-sonnet-4-5-20250929', provider: 'anthropic', description: 'Enhanced Coding' },
  { id: 'claude-sonnet-4-5-thinking', name: 'Claude Sonnet 4.5 Thinking', apiName: 'claude-sonnet-4-5-20250929', provider: 'anthropic', description: 'Extended Thinking' },
  { id: 'claude-opus-4-5', name: 'Claude Opus 4.5', apiName: 'claude-opus-4-5-20251101', provider: 'anthropic', description: 'Highest Performance' },
];

export const ALL_LLM_MODELS: LLMModel[] = [...OPENAI_MODELS, ...ANTHROPIC_MODELS];

export interface AppSettings {
  apiUrl: string;
  maxIterations: number;
  theme: 'dark' | 'light';
  llm: LLMSettings;
}

// Component Props Types
export interface QueryInputProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
  disabled?: boolean;
}

export interface ResultPanelProps {
  response: QueryResponse | null;
  streamSteps: StreamStepData[];
  isStreaming: boolean;
}

export interface ReasoningTimelineProps {
  steps: StreamStepData[];
  isActive: boolean;
}

export interface ConfidenceBadgeProps {
  confidence: number;
  size?: 'sm' | 'md' | 'lg';
}

export interface StatusIndicatorProps {
  status: 'healthy' | 'degraded' | 'unhealthy' | 'loading';
  label?: string;
}

// Ingestion Types
export type IngestionStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface IngestionJobResponse {
  job_id: string;
  status: IngestionStatus;
  message: string;
  files_count: number;
  supported_formats: string[];
}

export interface IngestionStatusResponse {
  job_id: string;
  status: IngestionStatus;
  progress: number;
  message: string;
  files_processed: number;
  files_total: number;
  entities_created: number;
  relations_created: number;
  chunks_created: number;
  errors: string[];
  started_at: string | null;
  completed_at: string | null;
}

export interface SupportedFormatsResponse {
  supported_extensions: string[];
  formats: Record<string, { name: string; description: string }>;
}

// Chat Session Types
export interface ChatSession {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string | null;
  timestamp: string;
  // Assistant-specific fields
  response?: QueryResponse;
  streamSteps?: StreamStepData[];
  error?: string | null;
  isStreaming?: boolean;
}
