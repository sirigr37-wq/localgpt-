'use client';

import React, { useState, useRef, useEffect } from 'react';
import { ArrowUp, Paperclip, Square, Mic } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  onOpenUpload: () => void;
  onStopStreaming?: () => void;
  isStreaming?: boolean;
  disabled?: boolean;
}

// Extend Window interface for Web Speech API
interface SpeechRecognitionEvent extends Event {
  results: SpeechRecognitionResultList;
  resultIndex: number;
}

interface SpeechRecognitionInstance {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onstart: (() => void) | null;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: Event) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionInstance;

function getSpeechRecognitionClass(): SpeechRecognitionConstructor | null {
  if (typeof window === 'undefined') return null;
  const win = window as unknown as {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  };
  return win.SpeechRecognition || win.webkitSpeechRecognition || null;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  onOpenUpload,
  onStopStreaming,
  isStreaming = false,
  disabled = false,
}) => {
  const [input, setInput] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [micSupported, setMicSupported] = useState(true);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const recognitionRef = useRef<SpeechRecognitionInstance | null>(null);

  // Auto-resize textarea height based on content
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        200
      )}px`;
    }
  }, [input]);

  const toggleListening = () => {
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
      return;
    }

    const SpeechRecognition = getSpeechRecognitionClass();

    if (!SpeechRecognition) {
      alert('Speech Recognition is not supported in this browser. Please use Google Chrome, Edge, or Safari.');
      setMicSupported(false);
      return;
    }

    try {
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        setIsListening(true);
      };

      recognition.onresult = (event: SpeechRecognitionEvent) => {
        let currentTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          currentTranscript += event.results[i][0].transcript;
        }
        if (currentTranscript) {
          setInput((prev) => {
            const trimmed = prev.trim();
            return trimmed ? `${trimmed} ${currentTranscript}` : currentTranscript;
          });
        }
      };

      recognition.onerror = (event: SpeechRecognitionErrorEvent) => {
        console.warn('Speech recognition error:', event.error);
        setIsListening(false);
      };

      recognition.onend = () => {
        setIsListening(false);
      };

      recognitionRef.current = recognition;
      recognition.start();
    } catch (err) {
      console.error('Failed to initialize microphone:', err);
      setIsListening(false);
    }
  };

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (isListening) {
      recognitionRef.current?.stop();
      setIsListening(false);
    }
    if (input.trim() && !isStreaming && !disabled) {
      const content = input.trim();
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
      onSendMessage(content);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!isStreaming && !disabled && input.trim()) {
        handleSubmit();
      }
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-3 sm:px-4 pb-3 sm:pb-5 pt-2">
      <form
        onSubmit={handleSubmit}
        className={`relative flex flex-col rounded-2xl bg-slate-900/90 border shadow-2xl transition-all backdrop-blur-md ${
          isListening
            ? 'border-rose-500/80 ring-2 ring-rose-500/20'
            : 'border-slate-700/80 focus-within:border-blue-500/70 focus-within:ring-2 focus-within:ring-blue-500/20'
        }`}
      >
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            isListening
              ? '🎙️ Listening... speak now into your microphone...'
              : isStreaming
              ? 'Assistant is responding... click Stop to interrupt'
              : 'Message LocalGPT or ask a question about your documents... (Shift+Enter for newline)'
          }
          rows={1}
          disabled={disabled}
          className="w-full bg-transparent text-slate-100 text-sm placeholder-slate-400 py-3.5 pl-4 pr-12 md:pr-14 outline-none resize-none max-h-52 min-h-[52px]"
        />

        {/* Action icons row */}
        <div className="flex items-center justify-between px-3 pb-2.5 pt-1 border-t border-slate-800/60 text-slate-400">
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onOpenUpload}
              title="Upload document (PDF, DOCX, TXT, CSV, JSON)"
              className="p-1.5 rounded-lg hover:bg-slate-800 hover:text-slate-200 transition-colors flex items-center gap-1.5 text-xs text-slate-400 cursor-pointer"
            >
              <Paperclip className="w-4 h-4 text-slate-400" />
              <span className="hidden sm:inline">Add Documents</span>
            </button>

            {/* Microphone Voice-to-Text button */}
            {micSupported && (
              <button
                type="button"
                onClick={toggleListening}
                title={isListening ? 'Stop listening' : 'Voice input (Microphone)'}
                className={`p-1.5 rounded-lg transition-all flex items-center gap-1.5 text-xs cursor-pointer ${
                  isListening
                    ? 'bg-rose-600/30 text-rose-400 border border-rose-500/50 animate-pulse'
                    : 'hover:bg-slate-800 hover:text-slate-200 text-slate-400'
                }`}
              >
                {isListening ? (
                  <>
                    <Mic className="w-4 h-4 text-rose-400 animate-bounce" />
                    <span className="text-[11px] font-medium text-rose-400">Listening...</span>
                  </>
                ) : (
                  <>
                    <Mic className="w-4 h-4" />
                    <span className="hidden sm:inline">Voice</span>
                  </>
                )}
              </button>
            )}
          </div>

          <div className="flex items-center gap-2">
            {isStreaming ? (
              <button
                type="button"
                onClick={onStopStreaming}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 active:scale-95 text-white text-xs font-semibold transition-all shadow-md shadow-rose-900/40 cursor-pointer"
                title="Stop generation"
              >
                <Square className="w-3.5 h-3.5 fill-current" />
                <span>Stop</span>
              </button>
            ) : (
              <button
                type="submit"
                disabled={!input.trim() || disabled || isStreaming}
                className="p-2 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:hover:bg-blue-600 active:scale-95 text-white transition-all duration-150 shadow-md shadow-blue-600/30 cursor-pointer disabled:cursor-not-allowed"
                title="Send message (Enter)"
              >
                <ArrowUp className="w-4 h-4 stroke-[2.5]" />
              </button>
            )}
          </div>
        </div>
      </form>
      <div className="text-center text-[11px] text-slate-500 mt-2 select-none">
        LocalGPT Phase 3 • Grounded responses backed by hosted LLM and PostgreSQL persistence.
      </div>
    </div>
  );
};

