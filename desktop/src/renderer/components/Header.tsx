/**
 * Header Component
 */

import { BrainCircuit, Github, Settings } from 'lucide-react';
import { clsx } from 'clsx';

interface HeaderProps {
  onSettingsClick?: () => void;
}

export function Header({ onSettingsClick }: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-6 py-4 bg-dark-900 border-b border-dark-700">
      {/* Logo & Title */}
      <div className="flex items-center gap-3">
        <div className="p-2 bg-primary-500/20 rounded-xl">
          <BrainCircuit className="w-6 h-6 text-primary-400" />
        </div>
        <div>
          <h1 className="text-lg font-semibold text-dark-100">
            Ontology Reasoning
          </h1>
          <p className="text-xs text-dark-400">
            ToG 3.0 MACER + LangGraph + Neo4j
          </p>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        <a
          href="https://github.com"
          target="_blank"
          rel="noopener noreferrer"
          className={clsx(
            'p-2 rounded-lg transition-colors',
            'text-dark-400 hover:text-dark-200 hover:bg-dark-700'
          )}
          title="GitHub"
        >
          <Github className="w-5 h-5" />
        </a>
        <button
          onClick={onSettingsClick}
          className={clsx(
            'p-2 rounded-lg transition-colors',
            'text-dark-400 hover:text-dark-200 hover:bg-dark-700'
          )}
          title="Settings"
        >
          <Settings className="w-5 h-5" />
        </button>
      </div>
    </header>
  );
}

export default Header;
