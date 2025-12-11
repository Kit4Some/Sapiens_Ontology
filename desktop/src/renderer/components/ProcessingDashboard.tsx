/**
 * Processing Dashboard Component
 *
 * Real-time dashboard showing processing progress for:
 * - Ingestion pipeline (entity extraction, embedding, loading)
 * - Query reasoning (constructor, retriever, reflector, responser)
 */

import { useState, useEffect, useCallback } from 'react';
import {
  Activity,
  Database,
  Brain,
  Search,
  FileText,
  Loader2,
  CheckCircle,
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Zap,
  Clock,
  Box,
  GitBranch,
} from 'lucide-react';
import { clsx } from 'clsx';

// Step definition for pipeline stages
interface ProcessingStep {
  id: string;
  name: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  details?: Record<string, unknown>;
  startTime?: number;
  endTime?: number;
}

// Props for the dashboard
interface ProcessingDashboardProps {
  type: 'ingestion' | 'query';
  isActive: boolean;
  steps: ProcessingStep[];
  currentStep?: string;
  overallProgress: number;
  stats?: {
    entitiesCount?: number;
    relationsCount?: number;
    chunksCount?: number;
    evidenceCount?: number;
    sufficiencyScore?: number;
  };
  error?: string;
  onClose?: () => void;
}

// Step icons mapping
const stepIcons: Record<string, React.ComponentType<{ className?: string }>> = {
  // Ingestion steps
  started: Activity,
  ingest: FileText,
  chunk: Box,
  extract_entities: Brain,
  extract_relations: GitBranch,
  embed: Zap,
  load: Database,
  completed: CheckCircle,
  failed: AlertCircle,
  // Query steps
  constructor: Search,
  retriever: Database,
  reflector: Brain,
  responser: FileText,
};

// Get icon for a step
function getStepIcon(stepId: string) {
  return stepIcons[stepId] || Activity;
}

// Format duration
function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

// Step indicator component
function StepIndicator({
  step,
  isActive,
}: {
  step: ProcessingStep;
  isActive: boolean;
}) {
  const Icon = getStepIcon(step.id);
  const duration =
    step.startTime && step.endTime
      ? formatDuration(step.endTime - step.startTime)
      : step.startTime
      ? formatDuration(Date.now() - step.startTime)
      : null;

  return (
    <div
      className={clsx(
        'flex items-center gap-3 p-3 rounded-lg transition-all',
        step.status === 'processing' && 'bg-primary-500/10 border border-primary-500/30',
        step.status === 'completed' && 'bg-green-500/10',
        step.status === 'failed' && 'bg-red-500/10',
        step.status === 'pending' && 'opacity-50'
      )}
    >
      {/* Icon */}
      <div
        className={clsx(
          'p-2 rounded-lg',
          step.status === 'processing' && 'bg-primary-500/20',
          step.status === 'completed' && 'bg-green-500/20',
          step.status === 'failed' && 'bg-red-500/20',
          step.status === 'pending' && 'bg-dark-700'
        )}
      >
        {step.status === 'processing' ? (
          <Loader2 className="w-4 h-4 text-primary-400 animate-spin" />
        ) : (
          <Icon
            className={clsx(
              'w-4 h-4',
              step.status === 'completed' && 'text-green-400',
              step.status === 'failed' && 'text-red-400',
              step.status === 'pending' && 'text-dark-500'
            )}
          />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <span
            className={clsx(
              'text-sm font-medium',
              step.status === 'processing' && 'text-primary-300',
              step.status === 'completed' && 'text-green-300',
              step.status === 'failed' && 'text-red-300',
              step.status === 'pending' && 'text-dark-400'
            )}
          >
            {step.name}
          </span>
          {duration && (
            <span className="text-xs text-dark-500 flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {duration}
            </span>
          )}
        </div>
        <p className="text-xs text-dark-400 truncate">{step.message}</p>
      </div>

      {/* Progress for active step */}
      {step.status === 'processing' && step.progress > 0 && (
        <div className="w-16 text-right">
          <span className="text-xs font-mono text-primary-400">
            {Math.round(step.progress * 100)}%
          </span>
        </div>
      )}
    </div>
  );
}

// Stats card component
function StatsCard({
  label,
  value,
  icon: Icon,
  color = 'primary',
}: {
  label: string;
  value: number | string;
  icon: React.ComponentType<{ className?: string }>;
  color?: 'primary' | 'green' | 'blue' | 'yellow';
}) {
  const colorClasses = {
    primary: 'text-primary-400 bg-primary-500/10',
    green: 'text-green-400 bg-green-500/10',
    blue: 'text-blue-400 bg-blue-500/10',
    yellow: 'text-yellow-400 bg-yellow-500/10',
  };

  return (
    <div className="flex items-center gap-3 p-3 bg-dark-800 rounded-lg">
      <div className={clsx('p-2 rounded-lg', colorClasses[color])}>
        <Icon className="w-4 h-4" />
      </div>
      <div>
        <p className="text-lg font-bold text-dark-100">{value}</p>
        <p className="text-xs text-dark-400">{label}</p>
      </div>
    </div>
  );
}

export function ProcessingDashboard({
  type,
  isActive,
  steps,
  currentStep,
  overallProgress,
  stats,
  error,
  onClose,
}: ProcessingDashboardProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [elapsedTime, setElapsedTime] = useState(0);
  const startTimeRef = useState(() => Date.now())[0];

  // Update elapsed time
  useEffect(() => {
    if (!isActive) return;

    const interval = setInterval(() => {
      setElapsedTime(Date.now() - startTimeRef);
    }, 1000);

    return () => clearInterval(interval);
  }, [isActive, startTimeRef]);

  const title = type === 'ingestion' ? 'Knowledge Graph Ingestion' : 'Query Processing';
  const Icon = type === 'ingestion' ? Database : Search;

  return (
    <div className="bg-dark-900 border border-dark-700 rounded-xl overflow-hidden">
      {/* Header */}
      <div
        className="flex items-center justify-between p-4 cursor-pointer hover:bg-dark-800/50 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-3">
          <div
            className={clsx(
              'p-2 rounded-lg',
              isActive ? 'bg-primary-500/20' : 'bg-dark-700'
            )}
          >
            {isActive ? (
              <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
            ) : (
              <Icon className="w-5 h-5 text-dark-400" />
            )}
          </div>
          <div>
            <h3 className="text-sm font-semibold text-dark-100">{title}</h3>
            <p className="text-xs text-dark-400">
              {isActive
                ? `Processing... ${formatDuration(elapsedTime)}`
                : error
                ? 'Failed'
                : overallProgress >= 1
                ? 'Completed'
                : 'Ready'}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Overall progress */}
          {isActive && (
            <div className="flex items-center gap-2">
              <div className="w-24 h-2 bg-dark-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary-500 transition-all duration-300 rounded-full"
                  style={{ width: `${overallProgress * 100}%` }}
                />
              </div>
              <span className="text-xs font-mono text-dark-300">
                {Math.round(overallProgress * 100)}%
              </span>
            </div>
          )}

          {/* Expand/collapse */}
          {isExpanded ? (
            <ChevronUp className="w-5 h-5 text-dark-400" />
          ) : (
            <ChevronDown className="w-5 h-5 text-dark-400" />
          )}
        </div>
      </div>

      {/* Expanded content */}
      {isExpanded && (
        <div className="border-t border-dark-700">
          {/* Error display */}
          {error && (
            <div className="m-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-4 h-4 text-red-400" />
                <span className="text-sm text-red-300">{error}</span>
              </div>
            </div>
          )}

          {/* Stats grid */}
          {stats && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-4">
              {type === 'ingestion' ? (
                <>
                  <StatsCard
                    label="Entities"
                    value={stats.entitiesCount ?? 0}
                    icon={Brain}
                    color="primary"
                  />
                  <StatsCard
                    label="Relations"
                    value={stats.relationsCount ?? 0}
                    icon={GitBranch}
                    color="blue"
                  />
                  <StatsCard
                    label="Chunks"
                    value={stats.chunksCount ?? 0}
                    icon={Box}
                    color="green"
                  />
                  <StatsCard
                    label="Elapsed"
                    value={formatDuration(elapsedTime)}
                    icon={Clock}
                    color="yellow"
                  />
                </>
              ) : (
                <>
                  <StatsCard
                    label="Entities"
                    value={stats.entitiesCount ?? 0}
                    icon={Brain}
                    color="primary"
                  />
                  <StatsCard
                    label="Evidence"
                    value={stats.evidenceCount ?? 0}
                    icon={Database}
                    color="blue"
                  />
                  <StatsCard
                    label="Sufficiency"
                    value={`${Math.round((stats.sufficiencyScore ?? 0) * 100)}%`}
                    icon={Activity}
                    color="green"
                  />
                  <StatsCard
                    label="Elapsed"
                    value={formatDuration(elapsedTime)}
                    icon={Clock}
                    color="yellow"
                  />
                </>
              )}
            </div>
          )}

          {/* Steps list */}
          {steps.length > 0 && (
            <div className="p-4 pt-0 space-y-2">
              <p className="text-xs font-medium text-dark-400 uppercase tracking-wider mb-2">
                Processing Steps
              </p>
              {steps.map((step) => (
                <StepIndicator
                  key={step.id}
                  step={step}
                  isActive={step.id === currentStep}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ProcessingDashboard;
