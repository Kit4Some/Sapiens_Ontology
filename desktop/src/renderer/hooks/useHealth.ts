/**
 * Custom hook for health check
 */

import { useState, useEffect, useCallback } from 'react';
import { apiClient } from '../api/client';
import type { HealthResponse } from '../types';

interface UseHealthResult {
  health: HealthResponse | null;
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
}

export function useHealth(pollInterval = 30000): UseHealthResult {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHealth = useCallback(async () => {
    try {
      const data = await apiClient.getHealth();
      setHealth(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Health check failed');
      setHealth(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();

    if (pollInterval > 0) {
      const interval = setInterval(fetchHealth, pollInterval);
      return () => clearInterval(interval);
    }
  }, [fetchHealth, pollInterval]);

  return {
    health,
    isLoading,
    error,
    refresh: fetchHealth,
  };
}

export default useHealth;
