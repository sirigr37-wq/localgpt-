'use client';

import React, { useState } from 'react';
import {
  User,
  Bot,
  Copy,
  Check,
  ThumbsUp,
  ThumbsDown,
  RotateCcw,
  Edit2,
  AlertTriangle,
  Download,
  Volume2,
  VolumeX,
  BookOpen,
} from 'lucide-react';


import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage as ChatMessageType } from '@/types/chat';
import { CodeBlock } from './CodeBlock';

interface ChatMessageProps {
  message: ChatMessageType;
  onRegenerate?: () => void;
  onRetry?: () => void;
  onEdit?: (newContent: string) => void;
  onFeedback?: (messageId: string, feedback: 'like' | 'dislike') => void;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  onRegenerate,
  onRetry,
  onEdit,
  onFeedback,
}) => {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);
  const [downloaded, setDownloaded] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(message.content);
  const [showSources, setShowSources] = useState(false);

  const wordCount = message.content ? message.content.trim().split(/\s+/).filter(Boolean).length : 0;
  const charCount = message.content ? message.content.length : 0;
  const hasDocSources = Boolean(message.sources && message.sources.length > 0);


  const handleToggleTTS = () => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      alert('Speech synthesis is not supported in this browser.');
      return;
    }
    if (window.speechSynthesis.speaking || isSpeaking) {
      window.speechSynthesis.cancel();
      setIsSpeaking(false);
    } else {
      const cleanText = message.content
        .replace(/```[\s\S]*?```/g, 'Code block omitted.')
        .replace(/`([^`]+)`/g, '$1')
        .replace(/[*_~#]/g, '');
      const utterance = new SpeechSynthesisUtterance(cleanText);
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);
      window.speechSynthesis.speak(utterance);
      setIsSpeaking(true);
    }
  };


  const isFailedResponse =
    !isUser &&
    Boolean(
      message.content.includes('[Streaming error]') ||
        message.content.includes('[LLM Provider Error]') ||
        message.content.includes('[LLM Configuration Notice]') ||
        message.content.includes('Error communicating with chat service')
    );

  const handleCopyMessage = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy message:', err);
    }
  };

  const handleDownloadText = () => {
    try {
      const blob = new Blob([message.content], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${isUser ? 'user_prompt' : 'assistant_response'}_${message.id.slice(0, 8)}.txt`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      setDownloaded(true);
      setTimeout(() => setDownloaded(false), 2000);
    } catch (err) {
      console.error('Failed to download message text:', err);
    }
  };

  const handleSaveEdit = () => {
    if (editContent.trim() && onEdit) {
      onEdit(editContent.trim());
      setIsEditing(false);
    }
  };

  const handleCancelEdit = () => {
    setEditContent(message.content);
    setIsEditing(false);
  };

  return (
    <div
      className={`w-full py-5 px-4 md:px-6 transition-colors border-b ${
        isUser
          ? 'bg-slate-900/40 border-slate-800/60'
          : isFailedResponse
          ? 'bg-rose-950/20 border-rose-900/40'
          : 'bg-slate-900/80 border-slate-800/80'
      }`}
    >
      <div className="max-w-4xl mx-auto flex gap-4 md:gap-5 items-start">
        {/* Avatar */}
        <div
          className={`w-8 h-8 rounded-lg shrink-0 flex items-center justify-center font-medium text-xs shadow-sm ${
            isUser
              ? 'bg-blue-600 text-white'
              : isFailedResponse
              ? 'bg-rose-600/30 border border-rose-500/50 text-rose-300'
              : 'bg-indigo-600/30 border border-indigo-500/40 text-indigo-400'
          }`}
        >
          {isUser ? (
            <User className="w-4 h-4" />
          ) : isFailedResponse ? (
            <AlertTriangle className="w-4 h-4 text-rose-400" />
          ) : (
            <Bot className="w-4 h-4" />
          )}
        </div>

        {/* Content & Controls */}
        <div className="flex-1 space-y-2.5 overflow-hidden">
          {/* Header Role and Time */}
          <div className="flex items-center justify-between text-xs text-slate-400">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-slate-200">
                {isUser ? 'You' : 'LocalGPT Assistant'}
              </span>
              {isFailedResponse && (
                <span className="px-1.5 py-0.5 text-[10px] font-medium bg-rose-950/80 border border-rose-800/60 text-rose-400 rounded-md">
                  Notice / Failed
                </span>
              )}
            </div>
            <span className="text-[11px] text-slate-500">
              {new Date(message.createdAt).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
              })}
            </span>
          </div>

          {/* Body Content / Inline Edit Form */}
          {isEditing ? (
            <div className="space-y-2">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full p-3 rounded-xl bg-slate-950 border border-slate-700 text-slate-100 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/50 resize-y min-h-[90px]"
                rows={3}
                autoFocus
              />
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={handleSaveEdit}
                  className="px-3.5 py-1.5 text-xs font-medium rounded-lg bg-blue-600 hover:bg-blue-500 text-white transition-colors cursor-pointer shadow-sm"
                >
                  Save & Submit
                </button>
                <button
                  type="button"
                  onClick={handleCancelEdit}
                  className="px-3.5 py-1.5 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors cursor-pointer"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : isUser ? (
            <div className="text-sm text-slate-200 whitespace-pre-wrap leading-relaxed break-words">
              {message.content}
            </div>
          ) : message.isStreaming && !message.content ? (
            <div className="flex items-center gap-1.5 h-6 py-1">
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          ) : (
            <div className="text-sm text-slate-200 break-words leading-relaxed">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    const codeString = String(children);
                    const isMultiLine = codeString.includes('\n') || Boolean(match);

                    if (isMultiLine) {
                      return (
                        <CodeBlock
                          language={match ? match[1] : undefined}
                          value={codeString}
                        />
                      );
                    }
                    return (
                      <code
                        className="px-1.5 py-0.5 rounded bg-slate-800 text-blue-300 text-xs font-mono border border-slate-700/60"
                        {...props}
                      >
                        {children}
                      </code>
                    );
                  },
                  p({ children }) {
                    return <p className="my-2 leading-relaxed text-slate-200">{children}</p>;
                  },
                  h1({ children }) {
                    return (
                      <h1 className="text-xl font-bold mt-4 mb-2 text-slate-100 border-b border-slate-800 pb-1">
                        {children}
                      </h1>
                    );
                  },
                  h2({ children }) {
                    return (
                      <h2 className="text-lg font-bold mt-3 mb-1.5 text-slate-100">
                        {children}
                      </h2>
                    );
                  },
                  h3({ children }) {
                    return (
                      <h3 className="text-base font-semibold mt-2.5 mb-1 text-slate-200">
                        {children}
                      </h3>
                    );
                  },
                  ul({ children }) {
                    return <ul className="list-disc pl-5 my-2 space-y-1 text-slate-200">{children}</ul>;
                  },
                  ol({ children }) {
                    return <ol className="list-decimal pl-5 my-2 space-y-1 text-slate-200">{children}</ol>;
                  },
                  li({ children }) {
                    return <li className="leading-relaxed">{children}</li>;
                  },
                  blockquote({ children }) {
                    return (
                      <blockquote className="border-l-4 border-blue-500/70 pl-3.5 my-2 text-slate-300 italic bg-blue-950/20 py-1.5 rounded-r">
                        {children}
                      </blockquote>
                    );
                  },
                  a({ href, children }) {
                    return (
                      <a
                        href={href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-400 underline hover:text-blue-300 transition-colors"
                      >
                        {children}
                      </a>
                    );
                  },
                  table({ children }) {
                    return (
                      <div className="my-3 overflow-x-auto rounded-lg border border-slate-700/80 shadow-md">
                        <table className="min-w-full divide-y divide-slate-700 text-xs text-left">
                          {children}
                        </table>
                      </div>
                    );
                  },
                  thead({ children }) {
                    return <thead className="bg-slate-800/90 text-slate-200">{children}</thead>;
                  },
                  th({ children }) {
                    return (
                      <th className="px-3.5 py-2 font-semibold border-b border-slate-700 text-slate-200">
                        {children}
                      </th>
                    );
                  },
                  td({ children }) {
                    return (
                      <td className="px-3.5 py-2 text-slate-300 border-b border-slate-800/60">
                        {children}
                      </td>
                    );
                  },
                  hr() {
                    return <hr className="border-slate-700/60 my-4" />;
                  },
                  strong({ children }) {
                    return <strong className="font-semibold text-slate-100">{children}</strong>;
                  },
                  em({ children }) {
                    return <em className="italic text-slate-200">{children}</em>;
                  },
                }}
              >
                {message.content}
              </ReactMarkdown>

              {/* Streaming Indicator Blinking Cursor */}
              {message.isStreaming && (
                <span
                  className="inline-block w-2 h-4 ml-1.5 bg-blue-500 animate-pulse align-middle rounded-xs"
                  aria-label="Generating..."
                />
              )}
            </div>
          )}

          {/* RAG & Generation Sources Inspection Box (Phase 2 Action Toolbar 'Sources' feature) */}
          {!isUser && showSources && (
            <div className="pt-2 animate-in fade-in duration-150">
              <div className="p-3.5 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300 space-y-2.5 shadow-lg">
                <div className="flex items-center justify-between pb-1.5 border-b border-slate-800/80">
                  <div className="flex items-center gap-2">
                    <BookOpen className="w-4 h-4 text-blue-400" />
                    <span className="font-semibold text-slate-200">
                      {hasDocSources
                        ? `Document Sources (FAISS Retrieval — ${message.sources!.length} chunk${
                            message.sources!.length > 1 ? 's' : ''
                          } used)`
                        : 'Generation Sources (Model Direct)'}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-[11px] text-slate-400">
                    <span>Words: <strong className="text-slate-200">{wordCount}</strong></span>
                    <span>•</span>
                    <span>Chars: <strong className="text-slate-200">{charCount}</strong></span>
                  </div>
                </div>

                {hasDocSources ? (
                  <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                    {message.sources!.map((src, sIdx) => {
                      const pageStr =
                        src.page_start === src.page_end
                          ? `Page ${src.page_start ?? 1}`
                          : `Pages ${src.page_start ?? 1}–${src.page_end ?? 1}`;
                      const matchPct = src.score ? Math.round(src.score * 100) : null;
                      return (
                        <div
                          key={sIdx}
                          className="p-2.5 rounded-lg bg-slate-900/90 border border-slate-800 text-xs text-slate-300 space-y-1.5"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="font-bold text-blue-400 font-mono">
                                [Source {sIdx + 1}]
                              </span>
                              <span className="font-medium text-slate-200 truncate max-w-[220px]">
                                {src.filename}
                              </span>
                              <span className="text-[11px] text-slate-400 font-mono">
                                {pageStr}
                              </span>
                            </div>
                            {matchPct !== null && (
                              <span className="px-2 py-0.5 rounded-full bg-emerald-950 border border-emerald-700 text-emerald-300 text-[10px] font-mono font-semibold">
                                Relevance: {(src.score ?? 0).toFixed(2)} ({matchPct}%)
                              </span>
                            )}
                          </div>
                          {src.excerpt && (
                            <p className="text-slate-400 text-[11px] italic bg-slate-950/60 p-2 rounded border border-slate-800/60 font-mono whitespace-pre-wrap leading-relaxed">
                              &ldquo;{src.excerpt}&rdquo;
                            </p>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="space-y-1.5 p-2.5 rounded-lg bg-slate-900/60 border border-slate-800/80 text-[11px] text-slate-400">
                    <div className="flex items-center gap-2 text-slate-200 font-medium">
                      <span>Inference Model:</span>
                      <code className="text-blue-400 bg-slate-950 px-1.5 py-0.5 rounded border border-slate-800">
                        openai/gpt-oss-120b (Hosted LLM)
                      </code>
                    </div>
                    <p className="text-slate-400 leading-relaxed">
                      Generated directly from model weights. No document sources were retrieved for this response.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}


          {/* Bottom Action Controls Bar */}
          {!isEditing && (
            <div className="flex items-center gap-1 pt-1.5 text-slate-400">
              {/* Copy Message Button */}
              <button
                type="button"
                onClick={handleCopyMessage}
                title={copied ? 'Copied message!' : 'Copy message'}
                className="p-1.5 rounded-lg hover:bg-slate-800 hover:text-slate-200 transition-colors cursor-pointer flex items-center gap-1 text-xs"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400 text-[11px]">Copied</span>
                  </>
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>

              {/* Download Text Button */}
              <button
                type="button"
                onClick={handleDownloadText}
                title={downloaded ? 'Downloaded text!' : 'Download text (.txt)'}
                className="p-1.5 rounded-lg hover:bg-slate-800 hover:text-slate-200 transition-colors cursor-pointer flex items-center gap-1 text-xs"
              >
                {downloaded ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-emerald-400" />
                    <span className="text-emerald-400 text-[11px]">Saved</span>
                  </>
                ) : (
                  <Download className="w-3.5 h-3.5" />
                )}
              </button>

              {/* Read Aloud (TTS) Button for Assistant Messages */}
              {!isUser && (
                <button
                  type="button"
                  onClick={handleToggleTTS}
                  title={isSpeaking ? 'Stop speaking' : 'Read aloud (Text-to-Speech)'}
                  className={`p-1.5 rounded-lg transition-colors cursor-pointer flex items-center gap-1 text-xs ${
                    isSpeaking
                      ? 'text-amber-400 bg-amber-950/60 border border-amber-800/60'
                      : 'hover:bg-slate-800 hover:text-slate-200'
                  }`}
                >
                  {isSpeaking ? (
                    <>
                      <VolumeX className="w-3.5 h-3.5 text-amber-400 animate-pulse" />
                      <span className="text-amber-400 text-[11px] font-medium">Stop</span>
                    </>
                  ) : (
                    <Volume2 className="w-3.5 h-3.5" />
                  )}
                </button>
              )}

              {/* User Edit Trigger Button */}

              {isUser && onEdit && (
                <button
                  type="button"
                  onClick={() => setIsEditing(true)}
                  title="Edit prompt"
                  className="p-1.5 rounded-lg hover:bg-slate-800 hover:text-slate-200 transition-colors cursor-pointer"
                >
                  <Edit2 className="w-3.5 h-3.5" />
                </button>
              )}

              {/* Assistant Action Controls */}
              {!isUser && (
                <>
                  {/* Retry Failed Response Button */}
                  {isFailedResponse && (onRetry || onRegenerate) && (
                    <button
                      type="button"
                      onClick={onRetry || onRegenerate}
                      title="Retry response"
                      className="px-2.5 py-1 rounded-lg bg-rose-900/50 hover:bg-rose-900 border border-rose-700/60 text-rose-200 text-xs font-medium flex items-center gap-1.5 transition-colors cursor-pointer ml-1"
                    >
                      <RotateCcw className="w-3.5 h-3.5 text-rose-300" />
                      <span>Retry</span>
                    </button>
                  )}

                  {/* Regenerate Button */}
                  {!isFailedResponse && onRegenerate && (
                    <button
                      type="button"
                      onClick={onRegenerate}
                      title="Regenerate response"
                      className="p-1.5 rounded-lg hover:bg-slate-800 hover:text-slate-200 transition-colors cursor-pointer"
                    >
                      <RotateCcw className="w-3.5 h-3.5" />
                    </button>
                  )}

                  {/* Like Feedback Button */}
                  {onFeedback && (
                    <button
                      type="button"
                      onClick={() => onFeedback(message.id, 'like')}
                      title="Good response"
                      className={`p-1.5 rounded-lg transition-colors cursor-pointer ${
                        message.feedback === 'like'
                          ? 'text-emerald-400 bg-emerald-950/50'
                          : 'hover:bg-slate-800 hover:text-slate-200'
                      }`}
                    >
                      <ThumbsUp className="w-3.5 h-3.5" />
                    </button>
                  )}

                  {/* Dislike Feedback Button */}
                  {onFeedback && (
                    <button
                      type="button"
                      onClick={() => onFeedback(message.id, 'dislike')}
                      title="Poor response"
                      className={`p-1.5 rounded-lg transition-colors cursor-pointer ${
                        message.feedback === 'dislike'
                          ? 'text-rose-400 bg-rose-950/50'
                          : 'hover:bg-slate-800 hover:text-slate-200'
                      }`}
                    >
                      <ThumbsDown className="w-3.5 h-3.5" />
                    </button>
                  )}

                  {/* Sources Button (Phase 2 Action Toolbar Feature Parity) */}
                  <button
                    type="button"
                    onClick={() => setShowSources(!showSources)}
                    title={showSources ? 'Hide sources' : 'View knowledge & generation sources'}
                    className={`p-1.5 rounded-lg transition-colors cursor-pointer flex items-center gap-1.5 text-xs ${
                      showSources
                        ? 'text-blue-300 bg-blue-950/80 border border-blue-700/60'
                        : hasDocSources
                        ? 'text-blue-400 hover:bg-slate-800 hover:text-blue-300'
                        : 'hover:bg-slate-800 hover:text-slate-200'
                    }`}
                  >
                    <BookOpen className="w-3.5 h-3.5" />
                    <span className="text-[11px] font-medium">
                      Sources{hasDocSources ? ` (${message.sources!.length})` : ''}
                    </span>
                  </button>
                </>
              )}

            </div>
          )}
        </div>
      </div>
    </div>
  );
};
