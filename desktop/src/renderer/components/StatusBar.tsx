/**
 * Status Bar Component
 *
 * Displays system health and connection status
 */

import { Database, Cpu, RefreshCw, CheckCircle, XCircle, AlertCircle, Brain } from 'lucide-react';
import { clsx } from 'clsx';
import type { HealthResponse } from '../types';
import { useSettings } from '../store';

interface StatusBarProps {
  health: HealthResponse | null;
  isLoading: boolean;
  onRefresh: () => void;
}

function StatusIndicator({
  connected,
  label,
  icon: Icon,
}: {
  connected: boolean | null;
  label: string;
  icon: typeof Database;
}) {
  return (
    <div className="flex items-center gap-2">
      <Icon className="w-3.5 h-3.5 text-dark-400" />
      <span className="text-xs text-dark-400">{label}</span>
      {connected === null ? (
        <AlertCircle className="w-3 h-3 text-yellow-400" />
      ) : connected ? (
        <CheckCircle className="w-3 h-3 text-green-400" />
      ) : (
        <XCircle className="w-3 h-3 text-red-400" />
      )}
    </div>
  );
}

export function StatusBar({ health, isLoading, onRefresh }: StatusBarProps) {
  const settings = useSettings();

  const getOverallStatus = () => {
    if (!health) return { color: 'bg-yellow-400', label: 'Unknown' };
    switch (health.status) {
      case 'healthy':
        return { color: 'bg-green-400', label: 'Healthy' };
      case 'degraded':
        return { color: 'bg-yellow-400', label: 'Degraded' };
      case 'unhealthy':
        return { color: 'bg-red-400', label: 'Unhealthy' };
      default:
        return { color: 'bg-gray-400', label: 'Unknown' };
    }
  };

  const status = getOverallStatus();

  // Get current LLM info
  const llmInfo = settings.llm
    ? `${settings.llm.provider === 'openai' ? 'OpenAI' : 'Anthropic'}: ${settings.llm.model}`
    : 'Not configured';

  return (
    <div className="flex items-center justify-between px-4 py-2 bg-dark-900 border-t border-dark-700">
      <div className="flex items-center gap-6">
        {/* Overall Status */}
        <div className="flex items-center gap-2">
          <div className={clsx('w-2 h-2 rounded-full', status.color)} />
          <span className="text-xs text-dark-300">{status.label}</span>
        </div>

        {/* Individual Services */}
        <StatusIndicator
          connected={health?.neo4j_connected ?? null}
          label="Neo4j"
          icon={Database}
        />
        <StatusIndicator
          connected={health?.llm_available ?? null}
          label="LLM"
          icon={Cpu}
        />

        {/* Current LLM Config */}
        <div className="flex items-center gap-2 pl-4 border-l border-dark-700">
          <Brain className="w-3.5 h-3.5 text-dark-400" />
          <span className="text-xs text-dark-400">{llmInfo}</span>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Version */}
        {health?.version && (
          <span className="text-xs text-dark-500">v{health.version}</span>
        )}

        {/* Refresh Button */}
        <button
          onClick={onRefresh}
          disabled={isLoading}
          className={clsx(
            'p-1 rounded text-dark-400 hover:text-dark-200',
            'hover:bg-dark-700 transition-colors',
            isLoading && 'animate-spin'
          )}
          title="Refresh status"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}

export default StatusBar;
