import React from 'react';
import { Sidebar } from './Sidebar';
import type { ChatSession } from '../types';

interface ChatLayoutProps {
    children: React.ReactNode;
    activeTab: 'query' | 'ingest';
    onTabChange: (tab: 'query' | 'ingest') => void;
    onSettingsClick: () => void;
    onNewChat: () => void;
    sessions: ChatSession[];
    activeSessionId: string | null;
    onSessionClick: (sessionId: string) => void;
    onDeleteSession: (sessionId: string) => void;
}

export function ChatLayout({
    children,
    activeTab,
    onTabChange,
    onSettingsClick,
    onNewChat,
    sessions,
    activeSessionId,
    onSessionClick,
    onDeleteSession
}: ChatLayoutProps) {
    return (
        <div className="flex h-screen bg-white dark:bg-gray-950 text-stone-900 dark:text-gray-200 font-sans transition-colors">
            <Sidebar
                activeTab={activeTab}
                onTabChange={onTabChange}
                onSettingsClick={onSettingsClick}
                onNewChat={onNewChat}
                sessions={sessions}
                activeSessionId={activeSessionId}
                onSessionClick={onSessionClick}
                onDeleteSession={onDeleteSession}
            />
            <main className="flex-1 flex flex-col h-full overflow-hidden relative bg-white dark:bg-gray-950 transition-colors">
                {children}
            </main>
        </div>
    );
}
