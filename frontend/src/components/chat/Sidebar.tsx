'use strict';
'use client';

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import {
  Plus,
  Search,
  MessageSquare,
  Trash2,
  Edit3,
  Settings,
  LogOut,
  X,
  Check,
} from 'lucide-react';
import { Conversation } from '@/types/chat';
import { User } from '@/types/auth';

interface SidebarProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewChat: () => void;
  onDeleteConversation: (id: string) => void;
  onRenameConversation: (id: string, newTitle: string) => void;
  onOpenSettings: () => void;
  onLogout?: () => void;
  currentUser: User | null;
  isOpen: boolean;
  onClose: () => void;
  onSearch?: (query: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  conversations,
  activeConversationId,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
  onRenameConversation,
  onOpenSettings,
  onLogout,
  currentUser,
  isOpen,
  onClose,
  onSearch,
}) => {
  const router = useRouter();
  const [searchQuery, setSearchQuery] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');

  // Filter conversations based on search
  const filteredConversations = conversations.filter((c) =>
    c.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const startRename = (c: Conversation, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(c.id);
    setEditTitle(c.title);
  };

  const confirmRename = (id: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    if (editTitle.trim()) {
      onRenameConversation(id, editTitle.trim());
    }
    setEditingId(null);
  };

  const cancelRename = (e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setEditingId(null);
  };

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 bg-black/60 z-40 md:hidden backdrop-blur-xs transition-opacity"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-50 w-72 md:w-64 lg:w-72 bg-slate-950 border-r border-slate-800/80 flex flex-col transition-transform duration-300 ease-in-out shrink-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        {/* Header with New Chat & Mobile Close */}
        <div className="p-3 border-b border-slate-800/60 space-y-2.5">
          <div className="flex items-center justify-between">
            <button
              onClick={() => {
                onNewChat();
                if (window.innerWidth < 768) onClose();
              }}
              className="flex-1 flex items-center justify-between gap-2 px-3 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium transition-all shadow-md shadow-blue-600/20"
            >
              <span className="flex items-center gap-2">
                <Plus className="w-4 h-4" />
                <span>New Chat</span>
              </span>
              <span className="text-[10px] bg-blue-700/60 px-1.5 py-0.5 rounded text-blue-200">
                ⌘K
              </span>
            </button>

            {/* Mobile Close Button */}
            <button
              onClick={onClose}
              className="md:hidden p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-900 ml-1"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Search Input UI */}
          <div className="relative">
            <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                if (onSearch) onSearch(e.target.value);
              }}
              placeholder="Search conversations..."
              className="w-full pl-9 pr-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500/70"
            />
            {searchQuery && (
              <button
                onClick={() => {
                  setSearchQuery('');
                  if (onSearch) onSearch('');
                }}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>

        {/* Conversation List UI */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          <div className="px-2 py-1 text-[11px] font-semibold text-slate-500 uppercase tracking-wider">
            Conversations ({filteredConversations.length})
          </div>

          {filteredConversations.length === 0 ? (
            <div className="px-3 py-6 text-center text-xs text-slate-500">
              {searchQuery ? 'No matching conversations found' : 'No conversations yet'}
            </div>
          ) : (
            filteredConversations.map((conv) => {
              const isActive = conv.id === activeConversationId;
              const isBeingEdited = editingId === conv.id;

              return (
                <div
                  key={conv.id}
                  onClick={() => {
                    onSelectConversation(conv.id);
                    if (window.innerWidth < 768) onClose();
                  }}
                  className={`group relative flex items-center justify-between gap-2 px-2.5 py-2 rounded-lg text-xs cursor-pointer transition-all ${
                    isActive
                      ? 'bg-slate-800/90 text-slate-100 font-medium border border-slate-700/60'
                      : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200'
                  }`}
                >
                  <div className="flex items-center gap-2.5 min-w-0 flex-1">
                    <MessageSquare
                      className={`w-4 h-4 shrink-0 ${
                        isActive ? 'text-blue-400' : 'text-slate-500'
                      }`}
                    />

                    {isBeingEdited ? (
                      <input
                        type="text"
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') confirmRename(conv.id);
                          if (e.key === 'Escape') cancelRename();
                        }}
                        autoFocus
                        onClick={(e) => e.stopPropagation()}
                        className="bg-slate-950 px-1.5 py-0.5 rounded border border-blue-500 text-slate-100 text-xs w-full outline-none"
                      />
                    ) : (
                      <span className="truncate">{conv.title}</span>
                    )}
                  </div>

                  {/* Actions (Rename / Delete) */}
                  <div className="flex items-center gap-1 shrink-0">
                    {isBeingEdited ? (
                      <>
                        <button
                          onClick={(e) => confirmRename(conv.id, e)}
                          title="Save title"
                          className="p-1 rounded hover:bg-slate-700 text-emerald-400"
                        >
                          <Check className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={cancelRename}
                          title="Cancel"
                          className="p-1 rounded hover:bg-slate-700 text-slate-400"
                        >
                          <X className="w-3.5 h-3.5" />
                        </button>
                      </>
                    ) : (
                      <>
                        {conv.messageCount !== undefined && conv.messageCount > 0 && (
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800/80 text-slate-500 font-mono group-hover:hidden">
                            {conv.messageCount}
                          </span>
                        )}
                        <div className={`${isActive ? 'flex' : 'hidden group-hover:flex'} items-center gap-0.5`}>
                          <button
                            onClick={(e) => startRename(conv, e)}
                            title="Rename"
                            className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-slate-200 transition-colors"
                          >
                            <Edit3 className="w-3.5 h-3.5" />
                          </button>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteConversation(conv.id);
                            }}
                            title="Delete"
                            className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-rose-400 transition-colors"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* User Profile & Settings Area */}
        <div className="p-3 border-t border-slate-800/80 bg-slate-950/60">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-8 h-8 rounded-full bg-blue-600/30 border border-blue-500/50 flex items-center justify-center font-bold text-xs text-blue-300 shrink-0">
                {currentUser?.fullName?.[0]?.toUpperCase() || 'U'}
              </div>
              <div className="min-w-0">
                <p className="text-xs font-semibold text-slate-200 truncate">
                  {currentUser?.fullName || 'User Profile'}
                </p>
                <p className="text-[11px] text-slate-500 truncate">
                  {currentUser?.email || 'user@example.com'}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-1 text-slate-400">
              <button
                onClick={onOpenSettings}
                title="Settings"
                className="p-1.5 rounded-lg hover:bg-slate-800 hover:text-slate-200 transition-colors"
              >
                <Settings className="w-4 h-4" />
              </button>
              <button
                onClick={() => {
                  if (onLogout) {
                    onLogout();
                  } else {
                    router.push('/login');
                  }
                }}
                title="Logout"
                className="p-1.5 rounded-lg hover:bg-slate-800 hover:text-rose-400 transition-colors cursor-pointer"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};
