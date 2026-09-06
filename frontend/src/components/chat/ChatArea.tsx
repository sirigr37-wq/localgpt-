'use client';

import React, { useRef, useEffect, useState } from 'react';
import { Menu, FileText, Settings, Sparkles, Activity, Download, Check } from 'lucide-react';
import { ChatMessage as ChatMessageType, Conversation, UserSettings } from '@/types/chat';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { EmptyState } from './EmptyState';
import { LoadingState } from './LoadingState';
import { XRayPanel } from './XRayPanel';

interface ChatAreaProps {
  conversation: Conversation | null;
  messages: ChatMessageType[];
  isStreaming: boolean;
  settings: UserSettings;
  onSendMessage: (content: string) => void;
  onRegenerate: () => void;
  onRetry?: () => void;
  onEditMessage: (messageId: string, newContent: string) => void;
  onFeedback: (messageId: string, feedback: 'like' | 'dislike') => void;
  onStopStreaming: () => void;
  onToggleSidebar: () => void;
  onOpenUpload: () => void;
  onOpenSettings: () => void;
}

export const ChatArea: React.FC<ChatAreaProps> = ({
  conversation,
  messages,
  isStreaming,
  settings,
  onSendMessage,
  onRegenerate,
  onRetry,
  onEditMessage,
  onFeedback,
  onStopStreaming,
  onToggleSidebar,
  onOpenUpload,
  onOpenSettings,
}) => {
  const [isXRayOpen, setIsXRayOpen] = useState(false);
  const [isExported, setIsExported] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Export full chat conversation as a clean .txt file
  const handleExportChat = () => {
    if (messages.length === 0) return;
    const title = conversation?.title || 'conversation';
    let textContent = `====================================================\n`;
    textContent += `LOCALGPT PHASE 3 — CHAT CONVERSATION EXPORT\n`;
    textContent += `Title: ${title}\n`;
    textContent += `Date: ${new Date().toLocaleString()}\n`;
    textContent += `====================================================\n\n`;

    messages.forEach((msg, idx) => {
      const roleLabel = msg.role === 'user' ? 'USER' : 'ASSISTANT (LocalGPT)';
      textContent += `[${idx + 1}] ${roleLabel} (${new Date(msg.createdAt).toLocaleTimeString()}):\n`;
      textContent += `${msg.content}\n\n`;
      if (msg.sources && msg.sources.length > 0) {
        textContent += `  Citations:\n`;
        msg.sources.forEach((s) => {
          textContent += `  - ${s.filename} (Page ${s.page_start ?? 1}, Match: ${(s.score ? s.score * 100 : 100).toFixed(1)}%)\n`;
        });
        textContent += `\n`;
      }
      textContent += `----------------------------------------------------\n\n`;
    });

    const blob = new Blob([textContent], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${title.toLowerCase().replace(/[^a-z0-9]/gi, '_')}_chat.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    setIsExported(true);
    setTimeout(() => setIsExported(false), 2000);
  };

  // Auto-scroll to bottom on new message or during streaming tokens
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isStreaming]);

  const lastAssistantMessage = [...messages].reverse().find((m) => m.role === 'assistant') || (messages.length > 0 ? messages[messages.length - 1] : null);

  return (
    <main className="flex-1 flex flex-col h-screen overflow-hidden bg-slate-900/50 relative">
      {/* Top Navigation Bar */}
      <header className="h-14 border-b border-slate-800/80 px-4 flex items-center justify-between bg-slate-950/80 backdrop-blur-md shrink-0 z-10">
        <div className="flex items-center gap-3 min-w-0">
          {/* Mobile hamburger menu toggle */}
          <button
            type="button"
            onClick={onToggleSidebar}
            className="md:hidden p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors cursor-pointer"
            title="Toggle Sidebar"
          >
            <Menu className="w-5 h-5" />
          </button>

          {/* Active Conversation Title & Model Tag */}
          <div className="flex items-center gap-2.5 min-w-0">
            <h2 className="text-sm font-semibold text-slate-200 truncate max-w-xs md:max-w-md">
              {conversation?.title || 'New Chat'}
            </h2>
            <span className="hidden sm:inline-flex items-center gap-1 text-[11px] font-medium bg-blue-950/60 border border-blue-800/40 text-blue-400 px-2.5 py-0.5 rounded-full">
              <Sparkles className="w-3 h-3" />
              <span>Groq / Llama / Hosted</span>
            </span>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-1.5">
          {/* X-Ray Toggle Button */}
          <button
            type="button"
            onClick={() => setIsXRayOpen(!isXRayOpen)}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium border transition-all cursor-pointer ${
              isXRayOpen
                ? 'bg-purple-900/60 border-purple-500 text-purple-200 shadow-md shadow-purple-900/30'
                : 'bg-slate-800/80 hover:bg-slate-800 text-slate-300 border-slate-700/60'
            }`}
            title="Toggle LLM X-Ray Inspection Panel"
          >
            <Activity className={`w-3.5 h-3.5 ${isXRayOpen ? 'text-purple-400 animate-pulse' : 'text-purple-400'}`} />
            <span className="hidden sm:inline">X-Ray</span>
            <span className={`w-1.5 h-1.5 rounded-full ${isXRayOpen ? 'bg-purple-400' : 'bg-slate-500'}`} />
          </button>

          {/* Export / Download Full Chat History Button */}
          {messages.length > 0 && (
            <button
              type="button"
              onClick={handleExportChat}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-800 text-slate-300 text-xs font-medium border border-slate-700/60 transition-colors cursor-pointer"
              title="Download full conversation transcript (.txt)"
            >
              {isExported ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span className="text-emerald-400 hidden sm:inline">Exported</span>
                </>
              ) : (
                <>
                  <Download className="w-3.5 h-3.5 text-purple-400" />
                  <span className="hidden sm:inline">Export Chat</span>
                </>
              )}
            </button>
          )}

          <button
            type="button"
            onClick={onOpenUpload}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-800 text-slate-300 text-xs font-medium border border-slate-700/60 transition-colors cursor-pointer"
            title="Manage Knowledge Base"
          >
            <FileText className="w-3.5 h-3.5 text-blue-400" />
            <span className="hidden sm:inline">Documents</span>
          </button>

          <button
            type="button"
            onClick={onOpenSettings}
            className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors cursor-pointer"
            title="Settings"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Messages Stream Container */}
      <div className="flex-1 overflow-y-auto flex flex-col scroll-smooth">
        {/* Phase 2 LLM X-Ray Mode Active Banner */}
        {isXRayOpen && (
          <div className="mx-4 mt-3 mb-1 p-2.5 rounded-xl bg-purple-950/40 border border-purple-800/60 text-xs text-purple-200 flex items-center justify-between flex-wrap gap-2 shadow-sm animate-in fade-in duration-150">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-purple-400 animate-ping" />
              <strong className="text-purple-100">🔬 LLM X-Ray Mode: ON</strong>
              <span className="text-slate-400 text-[11px]">— Internal Generation Stages Active</span>
            </div>
            <div className="font-mono text-[11px] text-purple-300/90 hidden md:flex items-center gap-1">
              <span>User ➔ Tokens ➔ Embeddings ➔ 28 Layers ➔ Attention ➔ Logits ➔ Response</span>
            </div>
          </div>
        )}

        {messages.length === 0 ? (
          <EmptyState onSelectPrompt={onSendMessage} />
        ) : (
          <div className="flex-1 pb-4">

            {messages.map((msg) => (
              <ChatMessage
                key={msg.id}
                message={msg}
                onRegenerate={msg.role === 'assistant' ? onRegenerate : undefined}
                onRetry={msg.role === 'assistant' ? onRetry || onRegenerate : undefined}
                onEdit={
                  msg.role === 'user'
                    ? (newContent) => onEditMessage(msg.id, newContent)
                    : undefined
                }
                onFeedback={onFeedback}
              />
            ))}
            {isStreaming &&
              (!messages.length || messages[messages.length - 1].role === 'user') && (
                <LoadingState />
              )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Bottom Sticky Chat Input */}
      <ChatInput
        onSendMessage={onSendMessage}
        onOpenUpload={onOpenUpload}
        onStopStreaming={onStopStreaming}
        isStreaming={isStreaming}
      />

      {/* X-Ray Inspector Drawer */}
      <XRayPanel
        isOpen={isXRayOpen}
        onClose={() => setIsXRayOpen(false)}
        lastMessage={lastAssistantMessage}
        settings={settings}
        conversation={conversation}
        isStreaming={isStreaming}
      />
    </main>
  );
};

