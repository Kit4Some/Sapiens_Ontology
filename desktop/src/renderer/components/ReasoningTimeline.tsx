/**
 * Reasoning Timeline Component
 *
 * Displays the step-by-step reasoning process
 */

import { useMemo } from 'react';
import {
  GitBranch,
  Search,
  Brain,
  MessageSquare,
  Loader2,
} from 'lucide-react';
import { clsx } from 'clsx';
import type { StreamStepData } from '../types';

interface ReasoningTimelineProps {
  steps: StreamStepData[];
  isActive: boolean;
}

const NODE_CONFIG = {
  constructor: {
    icon: GitBranch,
    label: 'Constructor',
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/20',
    description: 'Extracting topic entities',
  },
  retriever: {
    icon: Search,
    label: 'Retriever',
    color: 'text-green-400',
    bgColor: 'bg-green-500/20',
    description: 'Gathering evidence',
  },
  reflector: {
    icon: Brain,
    label: 'Reflector',
    color: 'text-purple-400',
    bgColor: 'bg-purple-500/20',
    description: 'Assessing sufficiency',
  },
  responser: {
    icon: MessageSquare,
    label: 'Responser',
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/20',
    description: 'Generating answer',
  },
};

export function ReasoningTimeline({ steps, isActive }: ReasoningTimelineProps) {
  const groupedSteps = useMemo(() => {
    const groups: Map<number, StreamStepData[]> = new Map();

    steps.forEach((step) => {
      const iteration = step.iteration;
      if (!groups.has(iteration)) {
        groups.set(iteration, []);
      }
      groups.get(iteration)!.push(step);
    });

    return Array.from(groups.entries()).sort(([a], [b]) => a - b);
  }, [steps]);

  if (steps.length === 0) {
    return null;
  }

  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium text-dark-300 uppercase tracking-wider">
        Reasoning Process
      </h3>

      <div className="space-y-3">
        {groupedSteps.map(([iteration, iterationSteps]) => (
          <div
            key={iteration}
            className="bg-dark-800/50 rounded-lg p-4 animate-fade-in"
          >
            <div className="flex items-center gap-2 mb-3">
              <span className="text-xs font-medium text-dark-400">
                Iteration {iteration}
              </span>
              {isActive &&
                iteration === Math.max(...steps.map((s) => s.iteration)) && (
                  <Loader2 className="w-3 h-3 animate-spin text-primary-400" />
                )}
            </div>

            <div className="space-y-2">
              {iterationSteps.map((step, idx) => {
                const config =
                  NODE_CONFIG[step.node as keyof typeof NODE_CONFIG] ||
                  NODE_CONFIG.constructor;
                const Icon = config.icon;

                return (
                  <div
                    key={`${iteration}-${step.node}-${idx}`}
                    className={clsx(
                      'flex items-start gap-3 p-2 rounded-lg',
                      'transition-colors duration-200',
                      config.bgColor
                    )}
                  >
                    <div
                      className={clsx(
                        'flex-shrink-0 p-1.5 rounded-lg',
                        'bg-dark-700/50'
                      )}
                    >
                      <Icon className={clsx('w-4 h-4', config.color)} />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <span
                          className={clsx(
                            'text-sm font-medium',
                            config.color
                          )}
                        >
                          {config.label}
                        </span>
                        {step.sufficiency_score > 0 && (
                          <span className="text-xs text-dark-400">
                            {(step.sufficiency_score * 100).toFixed(0)}%
                          </span>
                        )}
                      </div>

                      <p className="text-xs text-dark-400 mt-0.5">
                        {step.message}
                      </p>

                      {step.evidence_count > 0 && (
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-xs text-dark-500">
                            {step.evidence_count} evidence pieces
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default ReasoningTimeline;
