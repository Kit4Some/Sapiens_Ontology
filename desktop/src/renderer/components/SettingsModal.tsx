/**
 * Settings Modal Component
 *
 * Allows users to configure API URL, LLM settings, max iterations, and theme.
 */

import { useState, useEffect, useMemo, useLayoutEffect } from 'react';
import { X, Save, RotateCcw, Server, Zap, Palette, Brain, Key, Eye, EyeOff } from 'lucide-react';
import { clsx } from 'clsx';
import { useSettings, useSettingsActions, useUIActions, useUIState } from '../store';
import type { AppSettings, LLMProvider, LLMSettings } from '../types';
import { OPENAI_MODELS, ANTHROPIC_MODELS } from '../types';

export function SettingsModal() {
  const settings = useSettings();
  const { isSettingsOpen } = useUIState();
  const { updateSettings, resetSettings } = useSettingsActions();
  const { setSettingsOpen } = useUIActions();

  // Local form state
  const [formData, setFormData] = useState<AppSettings>(settings);
  const [isDirty, setIsDirty] = useState(false);
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');
  const [showOpenaiKey, setShowOpenaiKey] = useState(false);
  const [showAnthropicKey, setShowAnthropicKey] = useState(false);

  // Get models for selected provider
  const availableModels = useMemo(() => {
    return formData.llm.provider === 'openai' ? OPENAI_MODELS : ANTHROPIC_MODELS;
  }, [formData.llm.provider]);

  // Sync local state when modal opens - use layout effect to avoid flicker
  useLayoutEffect(() => {
    if (isSettingsOpen) {
      setFormData(settings);
      setIsDirty(false);
      setTestStatus('idle');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isSettingsOpen]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    document.body.style.overflow = isSettingsOpen ? 'hidden' : '';
    return () => {
      document.body.style.overflow = '';
    };
  }, [isSettingsOpen]);

  // Get current API key validity status
  const currentApiKey = formData.llm.provider === 'openai'
    ? formData.llm.openaiApiKey
    : formData.llm.anthropicApiKey;
  const hasApiKey = currentApiKey.length > 0;

  // Early return AFTER all hooks
  if (!isSettingsOpen) return null;

  const handleChange = <K extends keyof AppSettings>(key: K, value: AppSettings[K]) => {
    setFormData((prev) => ({ ...prev, [key]: value }));
    setIsDirty(true);
    setTestStatus('idle');
  };

  const handleLLMChange = <K extends keyof LLMSettings>(key: K, value: LLMSettings[K]) => {
    setFormData((prev) => ({
      ...prev,
      llm: { ...prev.llm, [key]: value },
    }));
    setIsDirty(true);
  };

  const handleProviderChange = (provider: LLMProvider) => {
    const models = provider === 'openai' ? OPENAI_MODELS : ANTHROPIC_MODELS;
    setFormData((prev) => ({
      ...prev,
      llm: {
        ...prev.llm,
        provider,
        model: models[0].apiName, // Select first model of new provider
      },
    }));
    setIsDirty(true);
  };

  const handleSave = () => {
    updateSettings(formData);
    setIsDirty(false);
    setSettingsOpen(false);
  };

  const handleReset = () => {
    resetSettings();
    setFormData({
      apiUrl: 'http://localhost:8000',
      maxIterations: 5,
      theme: 'dark',
      llm: {
        provider: 'openai',
        model: 'gpt-4o-mini',
        openaiApiKey: '',
        anthropicApiKey: '',
      },
    });
    setIsDirty(false);
  };

  const handleTestConnection = async () => {
    setTestStatus('testing');
    try {
      const response = await fetch(`${formData.apiUrl}/api/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000),
      });
      if (response.ok) {
        setTestStatus('success');
      } else {
        setTestStatus('error');
      }
    } catch {
      setTestStatus('error');
    }
  };

  const handleClose = () => {
    setSettingsOpen(false);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-title"
    >
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={handleClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-lg max-h-[90vh] bg-dark-900 rounded-2xl shadow-2xl border border-dark-700 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-dark-700 flex-shrink-0">
          <h2 id="settings-title" className="text-lg font-semibold text-dark-100">
            Settings
          </h2>
          <button
            onClick={handleClose}
            className={clsx(
              'p-2 rounded-lg transition-colors',
              'text-dark-400 hover:text-dark-200 hover:bg-dark-700'
            )}
            aria-label="Close settings"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content - Scrollable */}
        <div className="px-6 py-5 space-y-6 overflow-y-auto flex-1">
          {/* LLM Provider */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium text-dark-200">
              <Brain className="w-4 h-4 text-dark-400" />
              LLM Provider
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => handleProviderChange('openai')}
                className={clsx(
                  'flex-1 px-4 py-2.5 rounded-lg font-medium transition-colors',
                  formData.llm.provider === 'openai'
                    ? 'bg-green-600 text-white'
                    : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
                )}
              >
                OpenAI
              </button>
              <button
                onClick={() => handleProviderChange('anthropic')}
                className={clsx(
                  'flex-1 px-4 py-2.5 rounded-lg font-medium transition-colors',
                  formData.llm.provider === 'anthropic'
                    ? 'bg-orange-600 text-white'
                    : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
                )}
              >
                Anthropic
              </button>
            </div>
          </div>

          {/* LLM Model Selection */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium text-dark-200">
              <Brain className="w-4 h-4 text-dark-400" />
              Model
            </label>
            <select
              value={formData.llm.model}
              onChange={(e) => handleLLMChange('model', e.target.value)}
              className={clsx(
                'w-full px-4 py-2.5 rounded-lg',
                'bg-dark-800 border border-dark-600',
                'text-dark-100',
                'focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500',
                'transition-colors'
              )}
            >
              {availableModels.map((model) => (
                <option key={model.id} value={model.apiName}>
                  {model.name} - {model.description}
                </option>
              ))}
            </select>
            <p className="text-xs text-dark-400">
              API Model: <code className="bg-dark-700 px-1 rounded">{formData.llm.model}</code>
            </p>
          </div>

          {/* OpenAI API Key */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium text-dark-200">
              <Key className="w-4 h-4 text-green-500" />
              OpenAI API Key
            </label>
            <div className="relative">
              <input
                type={showOpenaiKey ? 'text' : 'password'}
                value={formData.llm.openaiApiKey}
                onChange={(e) => handleLLMChange('openaiApiKey', e.target.value)}
                placeholder="sk-..."
                className={clsx(
                  'w-full px-4 py-2.5 pr-12 rounded-lg',
                  'bg-dark-800 border border-dark-600',
                  'text-dark-100 placeholder-dark-500',
                  'focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500',
                  'transition-colors font-mono text-sm'
                )}
              />
              <button
                type="button"
                onClick={() => setShowOpenaiKey(!showOpenaiKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-dark-400 hover:text-dark-200"
              >
                {showOpenaiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            {formData.llm.provider === 'openai' && !formData.llm.openaiApiKey && (
              <p className="text-xs text-yellow-500">Required for OpenAI provider</p>
            )}
          </div>

          {/* Anthropic API Key */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium text-dark-200">
              <Key className="w-4 h-4 text-orange-500" />
              Anthropic API Key
            </label>
            <div className="relative">
              <input
                type={showAnthropicKey ? 'text' : 'password'}
                value={formData.llm.anthropicApiKey}
                onChange={(e) => handleLLMChange('anthropicApiKey', e.target.value)}
                placeholder="sk-ant-..."
                className={clsx(
                  'w-full px-4 py-2.5 pr-12 rounded-lg',
                  'bg-dark-800 border border-dark-600',
                  'text-dark-100 placeholder-dark-500',
                  'focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500',
                  'transition-colors font-mono text-sm'
                )}
              />
              <button
                type="button"
                onClick={() => setShowAnthropicKey(!showAnthropicKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-dark-400 hover:text-dark-200"
              >
                {showAnthropicKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
            {formData.llm.provider === 'anthropic' && !formData.llm.anthropicApiKey && (
              <p className="text-xs text-yellow-500">Required for Anthropic provider</p>
            )}
          </div>

          {/* Divider */}
          <div className="border-t border-dark-700" />

          {/* API URL */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium text-dark-200">
              <Server className="w-4 h-4 text-dark-400" />
              API Server URL
            </label>
            <div className="flex gap-2">
              <input
                type="url"
                value={formData.apiUrl}
                onChange={(e) => handleChange('apiUrl', e.target.value)}
                placeholder="http://localhost:8000"
                className={clsx(
                  'flex-1 px-4 py-2.5 rounded-lg',
                  'bg-dark-800 border border-dark-600',
                  'text-dark-100 placeholder-dark-500',
                  'focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500',
                  'transition-colors'
                )}
              />
              <button
                onClick={handleTestConnection}
                disabled={testStatus === 'testing'}
                className={clsx(
                  'px-4 py-2.5 rounded-lg font-medium transition-colors',
                  testStatus === 'success' && 'bg-green-600 text-white',
                  testStatus === 'error' && 'bg-red-600 text-white',
                  testStatus === 'idle' && 'bg-dark-700 text-dark-200 hover:bg-dark-600',
                  testStatus === 'testing' && 'bg-dark-700 text-dark-400 cursor-wait'
                )}
              >
                {testStatus === 'testing' ? 'Testing...' :
                 testStatus === 'success' ? 'Connected' :
                 testStatus === 'error' ? 'Failed' : 'Test'}
              </button>
            </div>
            <p className="text-xs text-dark-400">
              The backend API server endpoint for running queries.
            </p>
          </div>

          {/* Max Iterations */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium text-dark-200">
              <Zap className="w-4 h-4 text-dark-400" />
              Max Reasoning Iterations
            </label>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="1"
                max="10"
                value={formData.maxIterations}
                onChange={(e) => handleChange('maxIterations', parseInt(e.target.value, 10))}
                className="flex-1 accent-primary-500"
              />
              <span className="w-8 text-center text-dark-200 font-mono">
                {formData.maxIterations}
              </span>
            </div>
            <p className="text-xs text-dark-400">
              Maximum number of reasoning iterations before the system concludes.
            </p>
          </div>

          {/* Theme */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-sm font-medium text-dark-200">
              <Palette className="w-4 h-4 text-dark-400" />
              Theme
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => handleChange('theme', 'dark')}
                className={clsx(
                  'flex-1 px-4 py-2.5 rounded-lg font-medium transition-colors',
                  formData.theme === 'dark'
                    ? 'bg-primary-600 text-white'
                    : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
                )}
              >
                Dark
              </button>
              <button
                onClick={() => handleChange('theme', 'light')}
                className={clsx(
                  'flex-1 px-4 py-2.5 rounded-lg font-medium transition-colors',
                  formData.theme === 'light'
                    ? 'bg-primary-600 text-white'
                    : 'bg-dark-700 text-dark-300 hover:bg-dark-600'
                )}
              >
                Light
              </button>
            </div>
            <p className="text-xs text-dark-400">
              Choose between dark and light color schemes.
            </p>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-dark-700 bg-dark-850 flex-shrink-0">
          <button
            onClick={handleReset}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg',
              'text-dark-300 hover:text-dark-100 hover:bg-dark-700',
              'transition-colors'
            )}
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>

          <div className="flex gap-3">
            <button
              onClick={handleClose}
              className={clsx(
                'px-4 py-2 rounded-lg',
                'text-dark-300 hover:text-dark-100 hover:bg-dark-700',
                'transition-colors'
              )}
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={!isDirty || !hasApiKey}
              className={clsx(
                'flex items-center gap-2 px-4 py-2 rounded-lg font-medium',
                'transition-colors',
                isDirty && hasApiKey
                  ? 'bg-primary-600 text-white hover:bg-primary-500'
                  : 'bg-dark-700 text-dark-500 cursor-not-allowed'
              )}
            >
              <Save className="w-4 h-4" />
              Save
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SettingsModal;
