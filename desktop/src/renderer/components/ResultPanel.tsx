/**
 * Result Panel Component
 *
 * Displays the query result including answer, confidence, thinking process,
 * discovered patterns, and supporting evidence in expandable sections.
 */

import { useState } from 'react';
import {
  CheckCircle,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  Brain,
  Network,
  FileText,
  Lightbulb,
} from 'lucide-react';
import { clsx } from 'clsx';
import type { QueryResponse, StreamStepData, ThinkingStep, PatternInsight, SupportingEvidence } from '../types';
import { ReasoningTimeline } from './ReasoningTimeline';

interface ResultPanelProps {
  response: QueryResponse | null;
  streamSteps: StreamStepData[];
  isStreaming: boolean;
}

function ConfidenceBadge({ confidence }: { confidence: number }) {
  const percentage = Math.round(confidence * 100);

  const getColor = () => {
    if (percentage >= 80) return 'text-green-400 bg-green-500/20';
    if (percentage >= 60) return 'text-yellow-400 bg-yellow-500/20';
    return 'text-red-400 bg-red-500/20';
  };

  return (
    <div
      className={clsx(
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full',
        'text-sm font-medium',
        getColor()
      )}
    >
      {percentage >= 80 ? (
        <CheckCircle className="w-3.5 h-3.5" />
      ) : (
        <AlertCircle className="w-3.5 h-3.5" />
      )}
      <span>{percentage}% confident</span>
    </div>
  );
}

function ExpandableSection({
  title,
  icon: Icon,
  children,
  defaultOpen = false,
  badge,
}: {
  title: string;
  icon: React.ElementType;
  children: React.ReactNode;
  defaultOpen?: boolean;
  badge?: string | number;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="bg-dark-800 rounded-xl border border-dark-700 overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-4 hover:bg-dark-700/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Icon className="w-4 h-4 text-dark-400" />
          <h3 className="text-sm font-medium text-dark-300">{title}</h3>
          {badge !== undefined && (
            <span className="px-2 py-0.5 text-xs bg-dark-700 text-dark-300 rounded-full">
              {badge}
            </span>
          )}
        </div>
        {isOpen ? (
          <ChevronUp className="w-4 h-4 text-dark-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-dark-400" />
        )}
      </button>

      {isOpen && (
        <div className="px-4 pb-4 border-t border-dark-700">
          {children}
        </div>
      )}
    </div>
  );
}

function ThinkingProcessSection({ steps }: { steps: ThinkingStep[] }) {
  if (!steps || steps.length === 0) return null;

  return (
    <div className="space-y-3 pt-3">
      {steps.map((step, idx) => (
        <div key={idx} className="pl-4 border-l-2 border-blue-500/30">
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-medium text-blue-400">Step {step.step}</span>
            <span className="text-xs text-dark-500">•</span>
            <span className="text-xs text-dark-400">{step.action}</span>
            {step.confidence > 0 && (
              <span className="text-xs text-dark-500">
                ({Math.round(step.confidence * 100)}% confidence)
              </span>
            )}
          </div>
          {step.thought && (
            <p className="text-sm text-dark-300 mb-1">
              <span className="text-dark-500">Thought: </span>
              {step.thought}
            </p>
          )}
          {step.observation && (
            <p className="text-sm text-dark-400">
              <span className="text-dark-500">Observation: </span>
              {step.observation}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}

function PatternsSection({ patterns }: { patterns: PatternInsight[] }) {
  if (!patterns || patterns.length === 0) return null;

  const getPatternIcon = (type: string) => {
    const icons: Record<string, string> = {
      hub: '🎯',
      bridge: '🌉',
      cluster: '🔮',
      chain: '⛓️',
      star: '⭐',
      hierarchy: '📊',
      cycle: '🔄',
    };
    return icons[type.toLowerCase()] || '🔷';
  };

  return (
    <div className="space-y-3 pt-3">
      {patterns.map((pattern, idx) => (
        <div key={idx} className="bg-dark-700/50 rounded-lg p-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-lg">{getPatternIcon(pattern.pattern_type)}</span>
            <span className="text-sm font-medium text-dark-200 capitalize">
              {pattern.pattern_type} Pattern
            </span>
          </div>
          <p className="text-sm text-dark-300 mb-2">{pattern.description}</p>
          {pattern.significance && (
            <p className="text-xs text-dark-400 italic">
              <Lightbulb className="w-3 h-3 inline mr-1" />
              {pattern.significance}
            </p>
          )}
          {pattern.entities.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {pattern.entities.slice(0, 5).map((entity, eidx) => (
                <span
                  key={eidx}
                  className="px-2 py-0.5 text-xs bg-dark-600 text-dark-300 rounded"
                >
                  {entity}
                </span>
              ))}
              {pattern.entities.length > 5 && (
                <span className="px-2 py-0.5 text-xs text-dark-500">
                  +{pattern.entities.length - 5} more
                </span>
              )}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function EvidenceSection({ evidence }: { evidence: SupportingEvidence[] }) {
  if (!evidence || evidence.length === 0) return null;

  const getSourceColor = (type: string) => {
    const colors: Record<string, string> = {
      DIRECT: 'text-green-400 bg-green-500/20',
      INFERRED: 'text-blue-400 bg-blue-500/20',
      CONTEXTUAL: 'text-purple-400 bg-purple-500/20',
      COMMUNITY: 'text-orange-400 bg-orange-500/20',
    };
    return colors[type] || 'text-dark-400 bg-dark-600';
  };

  return (
    <div className="space-y-3 pt-3">
      {evidence.map((ev, idx) => (
        <div key={idx} className="bg-dark-700/50 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <span className={clsx('px-2 py-0.5 text-xs rounded', getSourceColor(ev.source_type))}>
              {ev.source_type}
            </span>
            <span className="text-xs text-dark-500">
              Relevance: {Math.round(ev.relevance * 100)}%
            </span>
          </div>
          <p className="text-sm text-dark-300">{ev.content}</p>
          {ev.entity_names.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {ev.entity_names.map((name, nidx) => (
                <span
                  key={nidx}
                  className="px-1.5 py-0.5 text-xs bg-dark-600 text-dark-400 rounded"
                >
                  {name}
                </span>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

export function ResultPanel({
  response,
  streamSteps,
  isStreaming,
}: ResultPanelProps) {
  const [showExplanation, setShowExplanation] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyAnswer = () => {
    if (response?.answer) {
      navigator.clipboard.writeText(response.direct_answer || response.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  // Show timeline while streaming
  if (isStreaming || (streamSteps.length > 0 && !response)) {
    return (
      <div className="space-y-6">
        <ReasoningTimeline steps={streamSteps} isActive={isStreaming} />
      </div>
    );
  }

  if (!response) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-dark-400">
        <div className="text-6xl mb-4">🔍</div>
        <p className="text-lg">Ask a question to start reasoning</p>
        <p className="text-sm mt-2">
          The system will analyze your knowledge graph and provide evidence-based answers
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Direct Answer Section (Concise) */}
      {response.direct_answer && (
        <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl p-4 border border-blue-700/50">
          <div className="flex items-center gap-2 mb-2">
            <Lightbulb className="w-4 h-4 text-yellow-400" />
            <span className="text-sm font-medium text-dark-300">Direct Answer</span>
          </div>
          <p className="text-lg font-medium text-dark-100">{response.direct_answer}</p>
        </div>
      )}

      {/* Full Answer Section */}
      <div className="bg-dark-800 rounded-xl p-6 border border-dark-700">
        <div className="flex items-start justify-between gap-4 mb-4">
          <h2 className="text-lg font-semibold text-dark-100">Answer</h2>
          <div className="flex items-center gap-2">
            <ConfidenceBadge confidence={response.confidence} />
            <button
              onClick={copyAnswer}
              className="p-2 text-dark-400 hover:text-dark-200 hover:bg-dark-700 rounded-lg transition-colors"
              title="Copy answer"
            >
              {copied ? (
                <Check className="w-4 h-4 text-green-400" />
              ) : (
                <Copy className="w-4 h-4" />
              )}
            </button>
          </div>
        </div>

        <p className="text-dark-200 leading-relaxed whitespace-pre-wrap">
          {response.answer}
        </p>

        {/* Stats */}
        <div className="flex flex-wrap gap-4 mt-4 pt-4 border-t border-dark-700">
          <div className="text-sm">
            <span className="text-dark-400">Iterations: </span>
            <span className="text-dark-200">{response.iteration_count}</span>
          </div>
          <div className="text-sm">
            <span className="text-dark-400">Entities: </span>
            <span className="text-dark-200">{response.entities_found}</span>
          </div>
          <div className="text-sm">
            <span className="text-dark-400">Evidence: </span>
            <span className="text-dark-200">{response.evidence_count}</span>
          </div>
          <div className="text-sm">
            <span className="text-dark-400">Type: </span>
            <span className="text-dark-200">{response.answer_type}</span>
          </div>
        </div>
      </div>

      {/* Thinking Process (Expandable) */}
      {response.thinking_process && response.thinking_process.length > 0 && (
        <ExpandableSection
          title="Thinking Process"
          icon={Brain}
          badge={response.thinking_process.length}
        >
          <ThinkingProcessSection steps={response.thinking_process} />
        </ExpandableSection>
      )}

      {/* Discovered Patterns (Expandable) */}
      {response.patterns && response.patterns.length > 0 && (
        <ExpandableSection
          title="Discovered Patterns"
          icon={Network}
          badge={response.patterns.length}
        >
          <PatternsSection patterns={response.patterns} />
        </ExpandableSection>
      )}

      {/* Supporting Evidence (Expandable) */}
      {response.supporting_evidence && response.supporting_evidence.length > 0 && (
        <ExpandableSection
          title="Supporting Evidence"
          icon={FileText}
          badge={response.supporting_evidence.length}
        >
          <EvidenceSection evidence={response.supporting_evidence} />
        </ExpandableSection>
      )}

      {/* Explanation Section */}
      {response.explanation && (
        <div className="bg-dark-800 rounded-xl border border-dark-700 overflow-hidden">
          <button
            onClick={() => setShowExplanation(!showExplanation)}
            className="w-full flex items-center justify-between p-4 hover:bg-dark-700/50 transition-colors"
          >
            <h3 className="text-sm font-medium text-dark-300">
              Reasoning Explanation
            </h3>
            {showExplanation ? (
              <ChevronUp className="w-4 h-4 text-dark-400" />
            ) : (
              <ChevronDown className="w-4 h-4 text-dark-400" />
            )}
          </button>

          {showExplanation && (
            <div className="px-4 pb-4">
              <p className="text-sm text-dark-300 whitespace-pre-wrap">
                {response.explanation}
              </p>
            </div>
          )}
        </div>
      )}

      {/* Reasoning Timeline */}
      {streamSteps.length > 0 && (
        <ReasoningTimeline steps={streamSteps} isActive={false} />
      )}

      {/* Errors */}
      {response.errors.length > 0 && (
        <div className="bg-red-900/20 border border-red-800 rounded-xl p-4">
          <h3 className="text-sm font-medium text-red-400 mb-2">Errors</h3>
          <ul className="space-y-1">
            {response.errors.map((error, idx) => (
              <li key={idx} className="text-sm text-red-300">
                {error}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default ResultPanel;
