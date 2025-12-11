import { MessageSquare, Upload, Settings, Plus, Trash2 } from 'lucide-react';
import { clsx } from 'clsx';
import { ThemeToggle } from './ThemeToggle';
import type { ChatSession } from '../types';

interface SidebarProps {
    activeTab: 'query' | 'ingest';
    onTabChange: (tab: 'query' | 'ingest') => void;
    onSettingsClick: () => void;
    sessions: ChatSession[];
    activeSessionId: string | null;
    onSessionClick: (sessionId: string) => void;
    onNewChat: () => void;
    onDeleteSession: (sessionId: string) => void;
}

export function Sidebar({
    activeTab,
    onTabChange,
    onSettingsClick,
    sessions,
    activeSessionId,
    onSessionClick,
    onNewChat,
    onDeleteSession
}: SidebarProps) {
    return (
        <div className="w-64 bg-stone-50 dark:bg-gray-900 border-r border-stone-200 dark:border-gray-800 h-full flex flex-col transition-colors">
            {/* Header / Logo */}
            <div className="p-4 border-b border-stone-100 dark:border-gray-800 flex items-center justify-between">
                <h1 className="text-xl font-serif font-bold text-stone-800 dark:text-gray-100 flex items-center gap-2">
                    <div className="w-8 h-8 bg-primary-500 rounded-lg flex items-center justify-center text-white">
                        OR
                    </div>
                    Ontology
                </h1>
            </div>

            {/* New Chat Button */}
            <div className="p-4">
                <button
                    onClick={onNewChat}
                    className="w-full flex items-center justify-center gap-2 bg-white dark:bg-gray-800 border border-stone-200 dark:border-gray-700 hover:bg-stone-50 dark:hover:bg-gray-700 text-stone-700 dark:text-gray-200 py-2.5 px-4 rounded-xl transition-all shadow-sm font-medium"
                >
                    <Plus className="w-4 h-4" />
                    New Chat
                </button>
            </div>

            {/* Navigation */}
            <nav className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
                <div className="px-2 text-xs font-semibold text-stone-400 dark:text-gray-500 uppercase tracking-wider mb-2 mt-2">
                    Menu
                </div>

                <button
                    onClick={() => onTabChange('query')}
                    className={clsx(
                        'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                        activeTab === 'query'
                            ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                            : 'text-stone-600 dark:text-gray-400 hover:bg-stone-100 dark:hover:bg-gray-800'
                    )}
                >
                    <MessageSquare className="w-4 h-4" />
                    Chat
                </button>

                <button
                    onClick={() => onTabChange('ingest')}
                    className={clsx(
                        'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                        activeTab === 'ingest'
                            ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300'
                            : 'text-stone-600 dark:text-gray-400 hover:bg-stone-100 dark:hover:bg-gray-800'
                    )}
                >
                    <Upload className="w-4 h-4" />
                    Knowledge Base
                </button>

                {/* Sessions Section */}
                {sessions.length > 0 && (
                    <>
                        <div className="px-2 text-xs font-semibold text-stone-400 dark:text-gray-500 uppercase tracking-wider mb-2 mt-6">
                            Recent
                        </div>
                        {sessions.map((session) => (
                            <div
                                key={session.id}
                                onClick={() => onSessionClick(session.id)}
                                className={clsx(
                                    "group w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors cursor-pointer",
                                    activeSessionId === session.id
                                        ? "bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300"
                                        : "text-stone-600 dark:text-gray-400 hover:bg-stone-100 dark:hover:bg-gray-800"
                                )}
                            >
                                <MessageSquare className="w-4 h-4 shrink-0 opacity-50" />
                                <span className="truncate flex-1">{session.title || 'New Chat'}</span>
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        onDeleteSession(session.id);
                                    }}
                                    className="opacity-0 group-hover:opacity-100 p-1 hover:bg-stone-200 dark:hover:bg-gray-700 rounded transition-opacity"
                                    title="Delete session"
                                >
                                    <Trash2 className="w-3 h-3" />
                                </button>
                            </div>
                        ))}
                    </>
                )}
            </nav>

            {/* Footer / Settings */}
            <div className="p-4 border-t border-stone-200 dark:border-gray-800 space-y-2">
                <ThemeToggle className="w-full flex items-center justify-center gap-2" />

                <button
                    onClick={onSettingsClick}
                    className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium text-stone-600 dark:text-gray-400 hover:bg-stone-100 dark:hover:bg-gray-800 transition-colors"
                >
                    <Settings className="w-4 h-4" />
                    Settings
                </button>
            </div>
        </div>
    );
}
