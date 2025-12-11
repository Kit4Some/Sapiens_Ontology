/**
 * Chat Input Component
 *
 * Floating input field for entering natural language queries
 */

import { useState, useCallback, KeyboardEvent, useRef, useEffect } from 'react';
import { ArrowUp, StopCircle, Settings } from 'lucide-react';
import { clsx } from 'clsx';

interface ChatInputProps {
  onSubmit: (query: string) => void;
  onCancel?: () => void;
  isLoading: boolean;
  disabled?: boolean;
}

export function QueryInput({
  onSubmit,
  onCancel,
  isLoading,
  disabled = false,
}: ChatInputProps) {
  const [query, setQuery] = useState('');
  const [maxIterations, setMaxIterations] = useState(5);
  const [showSettings, setShowSettings] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = useCallback(() => {
    if (query.trim() && !isLoading && !disabled) {
      onSubmit(query.trim());
      setQuery('');
      // Reset height
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  }, [query, isLoading, disabled, onSubmit]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [handleSubmit]
  );

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`;
    }
  }, [query]);

  return (
    <div className="w-full max-w-3xl mx-auto px-4 pb-6">
      <div className="relative bg-white dark:bg-gray-900 rounded-3xl shadow-lg border border-stone-200 dark:border-gray-700 transition-all hover:shadow-xl focus-within:shadow-xl focus-within:border-primary-300 dark:focus-within:border-primary-700">

        {/* Settings Panel (Floating above) */}
        {showSettings && (
          <div className="absolute bottom-full mb-2 right-0 p-4 bg-white dark:bg-gray-900 rounded-2xl shadow-xl border border-stone-100 dark:border-gray-800 animate-slide-up z-10 w-64">
            <h3 className="text-sm font-semibold text-stone-700 dark:text-gray-200 mb-3">Reasoning Settings</h3>
            <label className="flex items-center justify-between text-sm text-stone-600 dark:text-gray-400">
              <span>Max Iterations</span>
              <input
                type="number"
                value={maxIterations}
                onChange={(e) =>
                  setMaxIterations(Math.max(1, Math.min(20, parseInt(e.target.value) || 5)))
                }
                min={1}
                max={20}
                className="w-16 px-2 py-1 bg-stone-50 dark:bg-gray-800 border border-stone-200 dark:border-gray-700 rounded text-stone-800 dark:text-gray-200 text-center focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </label>
          </div>
        )}

        <div className="flex items-end gap-2 p-3">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={clsx(
              'p-2 rounded-full transition-colors mb-1',
              'text-stone-400 hover:text-stone-600 dark:text-gray-500 dark:hover:text-gray-300 hover:bg-stone-100 dark:hover:bg-gray-800',
              showSettings && 'text-primary-600 dark:text-primary-400 bg-primary-50 dark:bg-primary-900/20'
            )}
            title="Settings"
          >
            <Settings className="w-5 h-5" />
          </button>

          <textarea
            ref={textareaRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask anything..."
            disabled={disabled || isLoading}
            rows={1}
            className={clsx(
              'flex-1 max-h-[200px] py-3 px-2',
              'bg-transparent border-none',
              'text-stone-800 dark:text-gray-100 placeholder-stone-400 dark:placeholder-gray-600 text-lg',
              'focus:outline-none focus:ring-0',
              'resize-none'
            )}
          />

          {isLoading ? (
            <button
              onClick={onCancel}
              className="p-2 rounded-full bg-stone-100 dark:bg-gray-800 text-stone-500 hover:bg-red-100 hover:text-red-600 dark:hover:bg-red-900/30 dark:hover:text-red-400 transition-colors mb-1"
              title="Stop generating"
            >
              <StopCircle className="w-6 h-6" />
            </button>
          ) : (
            <button
              onClick={handleSubmit}
              disabled={!query.trim() || disabled}
              className={clsx(
                'p-2 rounded-full transition-all duration-200 mb-1',
                query.trim()
                  ? 'bg-primary-600 text-white shadow-md hover:bg-primary-700 hover:scale-105'
                  : 'bg-stone-100 dark:bg-gray-800 text-stone-300 dark:text-gray-600 cursor-not-allowed'
              )}
            >
              <ArrowUp className="w-6 h-6" />
            </button>
          )}
        </div>
      </div>

      <div className="text-center mt-2">
        <p className="text-xs text-stone-400 dark:text-gray-600">
          AI can make mistakes. Please verify important information.
        </p>
      </div>
    </div>
  );
}

export default QueryInput;
