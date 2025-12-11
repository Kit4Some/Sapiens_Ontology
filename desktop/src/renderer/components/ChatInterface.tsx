import { useEffect, useRef, useState } from 'react';
import { clsx } from 'clsx';
import { User, Bot, AlertCircle, Clock, Brain, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';
import type { QueryResponse, StreamStepData, ChatSession } from '../types';
import { ReasoningTimeline } from './ReasoningTimeline';

interface ChatInterfaceProps {
    session: ChatSession | null;
    pendingQuery?: string; // Query being typed but not yet submitted
    isStreaming: boolean;
}

interface MessageBubbleProps {
    role: 'user' | 'assistant';
    content?: string | null;
    error?: string | null;
    steps?: StreamStepData[];
    isStreaming?: boolean;
    timestamp?: string;
    response?: QueryResponse;
}

function MessageBubble({
    role,
    content,
    error,
    steps = [],
    isStreaming = false,
    timestamp,
    response
}: MessageBubbleProps) {
    const isUser = role === 'user';
    const [showReasoning, setShowReasoning] = useState(false);

    // Determine if we have reasoning to show
    const hasReasoning = steps.length > 0 || (response?.thinking_process && response.thinking_process.length > 0);

    return (
        <div className={clsx(
            "flex w-full mb-8 animate-fade-in",
            isUser ? "justify-end" : "justify-start"
        )}>
            <div className={clsx(
                "flex max-w-[85%] md:max-w-[75%]",
                isUser ? "flex-row-reverse" : "flex-row"
            )}>
                {/* Avatar */}
                <div className={clsx(
                    "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1",
                    isUser
                        ? "ml-3 bg-stone-200 dark:bg-gray-700 text-stone-600 dark:text-gray-300"
                        : "mr-3 bg-primary-600 text-white"
                )}>
                    {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
                </div>

                {/* Content */}
                <div className={clsx(
                    "flex flex-col",
                    isUser ? "items-end" : "items-start"
                )}>
                    <div className={clsx(
                        "px-5 py-3.5 rounded-2xl text-sm leading-relaxed shadow-sm transition-colors",
                        isUser
                            ? "bg-stone-100 dark:bg-gray-800 text-stone-800 dark:text-gray-200 rounded-tr-sm"
                            : "bg-white dark:bg-gray-900 border border-stone-100 dark:border-gray-800 text-stone-800 dark:text-gray-200 rounded-tl-sm"
                    )}>
                        {content ? (
                            <div className="prose prose-stone dark:prose-invert max-w-none prose-sm">
                                <div className="whitespace-pre-wrap font-serif">{content}</div>
                            </div>
                        ) : error ? (
                            <div className="text-red-500 dark:text-red-400 flex items-center gap-2">
                                <AlertCircle className="w-4 h-4" />
                                {error}
                            </div>
                        ) : isStreaming ? (
                            <div className="flex items-center gap-2 text-stone-400 italic">
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Thinking...
                            </div>
                        ) : null}
                    </div>

                    {/* Reasoning Process (Expandable) - Only for assistant messages */}
                    {!isUser && hasReasoning && (
                        <div className="mt-2 w-full">
                            <button
                                onClick={() => setShowReasoning(!showReasoning)}
                                className="flex items-center gap-2 text-xs text-stone-400 dark:text-gray-500 hover:text-stone-600 dark:hover:text-gray-300 transition-colors py-1"
                            >
                                <Brain className="w-3.5 h-3.5" />
                                <span>Reasoning Process</span>
                                {isStreaming && (
                                    <span className="text-primary-500">
                                        ({steps.length} steps)
                                    </span>
                                )}
                                {showReasoning ? (
                                    <ChevronUp className="w-3.5 h-3.5" />
                                ) : (
                                    <ChevronDown className="w-3.5 h-3.5" />
                                )}
                            </button>

                            {showReasoning && (
                                <div className="mt-2 p-3 bg-stone-50 dark:bg-gray-800/50 rounded-lg border border-stone-100 dark:border-gray-700">
                                    <ReasoningTimeline steps={steps} isActive={isStreaming} />
                                </div>
                            )}
                        </div>
                    )}

                    {/* Streaming indicator when reasoning is collapsed */}
                    {!isUser && isStreaming && !showReasoning && steps.length > 0 && (
                        <div className="mt-1 text-xs text-primary-500 flex items-center gap-1">
                            <Loader2 className="w-3 h-3 animate-spin" />
                            Processing... ({steps.length} steps)
                        </div>
                    )}

                    {timestamp && (
                        <div className="mt-1 text-xs text-stone-300 dark:text-gray-600 flex items-center gap-1">
                            <Clock className="w-3 h-3" />
                            {new Date(timestamp).toLocaleTimeString()}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export function ChatInterface({
    session,
    pendingQuery,
    isStreaming
}: ChatInterfaceProps) {
    const bottomRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [session?.messages, isStreaming]);

    const hasMessages = session && session.messages.length > 0;

    return (
        <div className="flex-1 overflow-y-auto px-4 py-6 scroll-smooth">
            <div className="max-w-3xl mx-auto">
                {/* Welcome Message if empty */}
                {!hasMessages && !pendingQuery && (
                    <div className="flex flex-col items-center justify-center py-20 text-stone-400 text-center">
                        <div className="w-16 h-16 bg-stone-100 dark:bg-gray-800 rounded-2xl flex items-center justify-center mb-6">
                            <Bot className="w-8 h-8 text-stone-400" />
                        </div>
                        <h2 className="text-2xl font-serif font-medium text-stone-700 dark:text-gray-200 mb-2">
                            Ontology Reasoning
                        </h2>
                        <p className="max-w-md text-stone-500 dark:text-gray-400">
                            Ask complex questions about your knowledge graph. I'll reason through the data to find the answer.
                        </p>
                    </div>
                )}

                {/* Session Messages */}
                {session?.messages.map((message) => (
                    <MessageBubble
                        key={message.id}
                        role={message.role}
                        content={message.content}
                        error={message.error}
                        steps={message.streamSteps || []}
                        isStreaming={message.isStreaming}
                        timestamp={message.timestamp}
                        response={message.response}
                    />
                ))}

                <div ref={bottomRef} className="h-4" />
            </div>
        </div>
    );
}
