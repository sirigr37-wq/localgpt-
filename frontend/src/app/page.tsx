'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import { apiClient } from '@/lib/api-client';
import { Sidebar } from '@/components/chat/Sidebar';
import { ChatArea } from '@/components/chat/ChatArea';
import { FileUploadModal } from '@/components/documents/FileUploadModal';
import { SettingsModal } from '@/components/settings/SettingsModal';
import { Conversation, ChatMessage, UserSettings } from '@/types/chat';
import { Loader2, AlertCircle, X } from 'lucide-react';

const DEFAULT_SETTINGS: UserSettings = {
  systemPrompt: 'You are a helpful, concise AI assistant specialized in document QA.',
  temperature: 0.7,
  topP: 0.9,
  maxTokens: 512,
  enableRag: true,
  theme: 'dark',
};

export default function DashboardPage() {
  const router = useRouter();
  const { user, isAuthenticated, isLoading: isAuthLoading, logout } = useAuth();

  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messagesMap, setMessagesMap] = useState<Record<string, ChatMessage[]>>({});
  const [settings, setSettings] = useState<UserSettings>(DEFAULT_SETTINGS);

  const [isLoadingConversations, setIsLoadingConversations] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Abort controller reference for stop streaming
  const abortControllerRef = useRef<AbortController | null>(null);

  // Redirect to login if user is not authenticated after initial session check
  useEffect(() => {
    if (!isAuthLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthLoading, isAuthenticated, router]);

  // Load user conversations from backend
  const refreshConversations = useCallback(
    async (searchQuery?: string, keepActiveId?: string | null) => {
      if (!isAuthenticated) return;
      try {
        const serverConvs = await apiClient.conversations.list(searchQuery);
        const mapped: Conversation[] = serverConvs.map((c) => ({
          id: c.id,
          title: c.title,
          createdAt: c.created_at,
          updatedAt: c.updated_at,
          messageCount: c.message_count,
        }));
        setConversations(mapped);

        if (mapped.length > 0) {
          const targetId =
            keepActiveId && mapped.some((c) => c.id === keepActiveId)
              ? keepActiveId
              : mapped[0].id;
          setActiveConversationId(targetId);
        } else if (!searchQuery) {
          const created = await apiClient.conversations.create({ title: 'New Chat' });
          const newConv: Conversation = {
            id: created.id,
            title: created.title,
            createdAt: created.created_at,
            updatedAt: created.updated_at,
            messageCount: 0,
          };
          setConversations([newConv]);
          setActiveConversationId(newConv.id);
          setMessagesMap({ [newConv.id]: [] });
        }
      } catch (err: unknown) {
        console.error('Failed to load conversations:', err);
        const msg = err instanceof Error ? err.message : 'Failed to load conversation history.';
        setErrorMessage(msg);
      } finally {
        setIsLoadingConversations(false);
      }
    },
    [isAuthenticated]
  );

  // Initial load of conversations
  useEffect(() => {
    let ignore = false;

    async function loadInitialData() {
      if (!isAuthenticated) return;
      try {
        const serverConvs = await apiClient.conversations.list();
        if (ignore) return;

        const mapped: Conversation[] = serverConvs.map((c) => ({
          id: c.id,
          title: c.title,
          createdAt: c.created_at,
          updatedAt: c.updated_at,
          messageCount: c.message_count,
        }));
        setConversations(mapped);

        if (mapped.length > 0) {
          setActiveConversationId(mapped[0].id);
        } else {
          const created = await apiClient.conversations.create({ title: 'New Chat' });
          if (ignore) return;
          const newConv: Conversation = {
            id: created.id,
            title: created.title,
            createdAt: created.created_at,
            updatedAt: created.updated_at,
            messageCount: 0,
          };
          setConversations([newConv]);
          setActiveConversationId(newConv.id);
          setMessagesMap({ [newConv.id]: [] });
        }
      } catch (err: unknown) {
        if (!ignore) {
          console.error('Failed to initialize conversation history:', err);
          const msg = err instanceof Error ? err.message : 'Failed to initialize conversations.';
          setErrorMessage(msg);
        }
      } finally {
        if (!ignore) {
          setIsLoadingConversations(false);
        }
      }
    }

    loadInitialData();

    return () => {
      ignore = true;
    };
  }, [isAuthenticated]);

  // Load messages whenever active conversation changes
  useEffect(() => {
    let ignore = false;

    async function loadActiveMessages() {
      if (!activeConversationId || !isAuthenticated) return;
      try {
        const detail = await apiClient.conversations.get(activeConversationId);
        if (ignore) return;

        const mappedMessages: ChatMessage[] = (detail.messages || []).map((m) => ({
          id: m.id,
          role: m.role as 'user' | 'assistant' | 'system',
          content: m.content,
          createdAt: m.created_at,
          feedback: (m.feedback as 'like' | 'dislike' | null) || undefined,
          sources:
            m.sources_json && typeof m.sources_json === 'object' && 'sources' in m.sources_json
              ? (m.sources_json as { sources: any[] }).sources // eslint-disable-line @typescript-eslint/no-explicit-any
              : undefined,
        }));

        setMessagesMap((prev) => ({
          ...prev,
          [activeConversationId]: mappedMessages,
        }));
      } catch (err: unknown) {
        if (!ignore) {
          console.error('Failed to load conversation messages:', err);
        }
      }
    }

    loadActiveMessages();

    return () => {
      ignore = true;
    };
  }, [activeConversationId, isAuthenticated]);

  // 1. Create a New Chat
  const handleNewChat = async () => {
    try {
      const created = await apiClient.conversations.create({ title: 'New Chat' });
      const newConv: Conversation = {
        id: created.id,
        title: created.title,
        createdAt: created.created_at,
        updatedAt: created.updated_at,
        messageCount: 0,
      };
      setConversations((prev) => [newConv, ...prev]);
      setActiveConversationId(newConv.id);
      setMessagesMap((prev) => ({ ...prev, [newConv.id]: [] }));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to create conversation.';
      setErrorMessage(msg);
    }
  };

  // 2. Select conversation
  const handleSelectConversation = (id: string) => {
    setActiveConversationId(id);
  };

  // 3. Rename conversation
  const handleRenameConversation = async (id: string, newTitle: string) => {
    if (!newTitle.trim()) return;
    try {
      // Optimistic update
      setConversations((prev) =>
        prev.map((c) => (c.id === id ? { ...c, title: newTitle.trim() } : c))
      );
      await apiClient.conversations.update(id, { title: newTitle.trim() });
    } catch (err: unknown) {
      console.error('Failed to rename conversation:', err);
      refreshConversations(undefined, activeConversationId);
    }
  };

  // 4. Delete conversation
  const handleDeleteConversation = async (id: string) => {
    try {
      await apiClient.conversations.delete(id);
      const remaining = conversations.filter((c) => c.id !== id);
      setConversations(remaining);
      setMessagesMap((prev) => {
        const copy = { ...prev };
        delete copy[id];
        return copy;
      });

      if (activeConversationId === id) {
        if (remaining.length > 0) {
          setActiveConversationId(remaining[0].id);
        } else {
          await handleNewChat();
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to delete conversation.';
      setErrorMessage(msg);
    }
  };

  // 5. Server-side Search handler
  const handleSearch = (query: string) => {
    refreshConversations(query.trim() || undefined, activeConversationId);
  };

  // 6. Send message & real-time streaming response from FastAPI backend
  const handleSendMessage = async (userContent: string, isRegenerate: boolean = false) => {
    if (!userContent.trim()) return;
    if (isStreaming) return;

    let convId = activeConversationId;
    // If no active conversation, create one first
    if (!convId) {
      try {
        const created = await apiClient.conversations.create({ title: 'New Chat' });
        convId = created.id;
        const newConv: Conversation = {
          id: created.id,
          title: created.title,
          createdAt: created.created_at,
          updatedAt: created.updated_at,
          messageCount: 0,
        };
        setConversations((prev) => [newConv, ...prev]);
        setActiveConversationId(convId);
      } catch {
        setErrorMessage('Failed to initiate conversation session.');
        return;
      }
    }

    const currentConvId = convId;
    const tempAssistantMsgId = `temp-asst-${Date.now() + 1}`;
    const assistantMsg: ChatMessage = {
      id: tempAssistantMsgId,
      role: 'assistant',
      content: '',
      createdAt: new Date().toISOString(),
      isStreaming: true,
    };

    if (!isRegenerate) {
      const tempUserMsgId = `temp-user-${Date.now()}`;
      const userMsg: ChatMessage = {
        id: tempUserMsgId,
        role: 'user',
        content: userContent,
        createdAt: new Date().toISOString(),
      };
      setMessagesMap((prev) => ({
        ...prev,
        [currentConvId]: [...(prev[currentConvId] || []), userMsg, assistantMsg],
      }));
    } else {
      setMessagesMap((prev) => ({
        ...prev,
        [currentConvId]: [...(prev[currentConvId] || []), assistantMsg],
      }));
    }

    setIsStreaming(true);
    setErrorMessage(null);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      let accumulatedResponse = '';

      for await (const token of apiClient.chat.streamMessage(
        currentConvId,
        userContent,
        (tok) => {
          accumulatedResponse += tok;
          setMessagesMap((prev) => ({
            ...prev,
            [currentConvId]: (prev[currentConvId] || []).map((m) =>
              m.id === tempAssistantMsgId ? { ...m, content: accumulatedResponse } : m
            ),
          }));
        },
        controller.signal,
        isRegenerate
      )) {
        // Stream reading occurs via callback above
        if (token && !accumulatedResponse) {
          accumulatedResponse += token;
        }
      }

      // Mark streaming complete on local message
      setMessagesMap((prev) => ({
        ...prev,
        [currentConvId]: (prev[currentConvId] || []).map((m) =>
          m.id === tempAssistantMsgId ? { ...m, isStreaming: false } : m
        ),
      }));

      // Refresh conversation list to capture auto-updated titles and ordering
      await refreshConversations(undefined, currentConvId);

      // Re-sync conversation messages so assistant message ID and RAG source citations appear immediately
      try {
        const detail = await apiClient.conversations.get(currentConvId);
        const mappedMessages: ChatMessage[] = (detail.messages || []).map((m) => ({
          id: m.id,
          role: m.role as 'user' | 'assistant' | 'system',
          content: m.content,
          createdAt: m.created_at,
          feedback: (m.feedback as 'like' | 'dislike' | null) || undefined,
          sources:
            m.sources_json && typeof m.sources_json === 'object' && 'sources' in m.sources_json
              ? (m.sources_json as { sources: any[] }).sources // eslint-disable-line @typescript-eslint/no-explicit-any
              : undefined,
        }));
        setMessagesMap((prev) => ({
          ...prev,
          [currentConvId]: mappedMessages,
        }));
      } catch (syncErr) {
        console.error('Failed to sync updated conversation messages:', syncErr);
      }

    } catch (err: unknown) {
      if (
        (err instanceof DOMException && err.name === 'AbortError') ||
        (err instanceof Error && err.name === 'AbortError')
      ) {
        setMessagesMap((prev) => ({
          ...prev,
          [currentConvId]: (prev[currentConvId] || []).map((m) =>
            m.id === tempAssistantMsgId ? { ...m, isStreaming: false } : m
          ),
        }));
        return;
      }
      console.error('Error during chat streaming:', err);
      const msg = err instanceof Error ? err.message : 'Error communicating with chat service.';
      setErrorMessage(msg);

      // Clean up failed assistant message
      setMessagesMap((prev) => ({
        ...prev,
        [currentConvId]: (prev[currentConvId] || []).map((m) =>
          m.id === tempAssistantMsgId
            ? {
                ...m,
                content:
                  m.content ||
                  '[Streaming error]: Could not receive response from chat endpoint.',
                isStreaming: false,
              }
            : m
        ),
      }));
    } finally {
      abortControllerRef.current = null;
      setIsStreaming(false);
    }
  };

  // 7. Stop streaming handler
  const handleStopStreaming = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsStreaming(false);
  };

  // 8. Regenerate last assistant response
  const handleRegenerate = async () => {
    if (!activeConversationId || isStreaming) return;
    const msgs = messagesMap[activeConversationId] || [];
    const lastAssistantIdx = [...msgs].reverse().findIndex((m) => m.role === 'assistant');
    if (lastAssistantIdx === -1) return;

    const targetIdx = msgs.length - 1 - lastAssistantIdx;
    const lastUserPrompt = msgs[targetIdx - 1]?.content;
    if (!lastUserPrompt) return;

    // Truncate state back to before the assistant message
    const truncated = msgs.slice(0, targetIdx);
    setMessagesMap((prev) => ({ ...prev, [activeConversationId]: truncated }));

    // Re-trigger generation with regenerate=true
    await handleSendMessage(lastUserPrompt, true);
  };

  // 9. Edit user message handler
  const handleEditMessage = async (messageId: string, newContent: string) => {
    if (!activeConversationId || !newContent.trim() || isStreaming) return;
    try {
      await apiClient.chat.editMessage(activeConversationId, messageId, newContent.trim());

      // Truncate conversation back to edited message
      const msgs = messagesMap[activeConversationId] || [];
      const targetIdx = msgs.findIndex((m) => m.id === messageId);
      if (targetIdx !== -1) {
        const truncated = msgs
          .slice(0, targetIdx + 1)
          .map((m) => (m.id === messageId ? { ...m, content: newContent.trim() } : m));
        setMessagesMap((prev) => ({ ...prev, [activeConversationId]: truncated }));
      }

      // Generate response for the edited prompt
      await handleSendMessage(newContent.trim(), true);
    } catch (err: unknown) {
      console.error('Failed to edit message:', err);
      const msg = err instanceof Error ? err.message : 'Failed to update message.';
      setErrorMessage(msg);
    }
  };

  // 10. Feedback handler (like/dislike)
  const handleFeedback = async (messageId: string, feedback: 'like' | 'dislike') => {
    if (!activeConversationId) return;
    const currentMsgs = messagesMap[activeConversationId] || [];
    const target = currentMsgs.find((m) => m.id === messageId);
    const newFeedback = target?.feedback === feedback ? null : feedback;

    // Optimistic update
    setMessagesMap((prev) => ({
      ...prev,
      [activeConversationId]: (prev[activeConversationId] || []).map((m) =>
        m.id === messageId ? { ...m, feedback: newFeedback } : m
      ),
    }));

    try {
      await apiClient.chat.updateFeedback(messageId, newFeedback);
    } catch (err: unknown) {
      console.error('Failed to save feedback:', err);
    }
  };

  // Active messages and active conversation object
  const currentMessages = activeConversationId ? messagesMap[activeConversationId] || [] : [];
  const activeConversation = conversations.find((c) => c.id === activeConversationId) || null;

  // Show auth loading state
  if (isAuthLoading) {
    return (
      <div className="h-screen w-screen flex flex-col items-center justify-center bg-slate-950 text-slate-400 gap-3">
        <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
        <p className="text-xs font-medium">Verifying authentication session...</p>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 font-sans antialiased relative">
      {/* Background Sync Indicator */}
      {isLoadingConversations && (
        <div className="absolute top-3 right-4 z-40 flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-slate-900/90 border border-slate-800 text-slate-400 text-[11px] shadow-lg backdrop-blur-md">
          <Loader2 className="w-3 h-3 animate-spin text-blue-400" />
          <span>Syncing conversations...</span>
        </div>
      )}

      {/* Global Error Banner */}
      {errorMessage && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-50 flex items-center gap-2.5 px-4 py-2.5 rounded-2xl bg-rose-950/90 border border-rose-800 text-rose-200 text-xs shadow-2xl backdrop-blur-md animate-in fade-in duration-200 max-w-lg">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
          <span className="flex-1">{errorMessage}</span>
          <button
            onClick={() => setErrorMessage(null)}
            className="p-1 rounded-lg hover:bg-rose-900/60 text-rose-400 hover:text-rose-200 transition-colors"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* 1. Sidebar */}
      <Sidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        onDeleteConversation={handleDeleteConversation}
        onRenameConversation={handleRenameConversation}
        onOpenSettings={() => setIsSettingsOpen(true)}
        onLogout={async () => {
          await logout();
          router.push('/login');
        }}
        currentUser={user}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSearch={handleSearch}
      />

      {/* 2. Main Chat Area */}
      <ChatArea
        conversation={activeConversation}
        messages={currentMessages}
        isStreaming={isStreaming}
        settings={settings}
        onSendMessage={(content) => handleSendMessage(content, false)}
        onRegenerate={handleRegenerate}
        onRetry={handleRegenerate}
        onEditMessage={handleEditMessage}
        onFeedback={handleFeedback}
        onStopStreaming={handleStopStreaming}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        onOpenUpload={() => setIsUploadOpen(true)}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {/* 3. File Upload Modal */}
      <FileUploadModal isOpen={isUploadOpen} onClose={() => setIsUploadOpen(false)} />

      {/* 4. Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        currentUser={user}
        settings={settings}
        onSaveSettings={(newSettings) => setSettings(newSettings)}
      />
    </div>
  );
}
