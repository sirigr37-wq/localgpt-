'use strict';
'use client';

import React from 'react';
import { Sparkles, FileText, Code2, Compass } from 'lucide-react';

interface EmptyStateProps {
  onSelectPrompt: (prompt: string) => void;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ onSelectPrompt }) => {
  const examplePrompts = [
    {
      icon: <FileText className="w-5 h-5 text-blue-400" />,
      title: 'Analyze Documents',
      desc: 'Ask questions about uploaded PDFs, Word docs, CSVs, or JSON files',
      prompt: 'Can you summarize the key findings in my uploaded document with page citations?',
    },
    {
      icon: <Code2 className="w-5 h-5 text-emerald-400" />,
      title: 'Write & Refactor Code',
      desc: 'Generate clean TypeScript, Python, or optimize existing functions',
      prompt: 'Write a TypeScript function to safely parse and validate JSON with error handling.',
    },
    {
      icon: <Sparkles className="w-5 h-5 text-purple-400" />,
      title: 'Explain Complex Topics',
      desc: 'Break down machine learning architectures and algorithms simply',
      prompt: 'Explain how vector embeddings and cosine similarity search work in RAG systems.',
    },
    {
      icon: <Compass className="w-5 h-5 text-amber-400" />,
      title: 'Research & Brainstorm',
      desc: 'Generate structured plans, roadmaps, and comparative analyses',
      prompt: 'Compare FastAPI vs Express.js for building scalable streaming AI backends.',
    },
  ];

  return (
    <div className="flex flex-col items-center justify-center flex-1 px-4 py-8 max-w-3xl mx-auto text-center">
      {/* Brand Icon / Logo */}
      <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20 mb-6 border border-blue-400/30">
        <Sparkles className="w-8 h-8 text-white animate-pulse" />
      </div>

      <h1 className="text-3xl font-bold text-slate-100 tracking-tight mb-3">
        How can I help you today?
      </h1>
      <p className="text-slate-400 text-sm max-w-md mb-8">
        LocalGPT Phase 3 with multi-turn memory, RAG document answering, and real-time streaming.
      </p>

      {/* Suggestion Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 w-full text-left">
        {examplePrompts.map((item, index) => (
          <button
            key={index}
            onClick={() => onSelectPrompt(item.prompt)}
            className="p-4 rounded-xl bg-slate-900/60 hover:bg-slate-800/80 border border-slate-800 hover:border-slate-700 transition-all duration-200 group flex flex-col justify-between"
          >
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 rounded-lg bg-slate-800/70 border border-slate-700/50 group-hover:scale-105 transition-transform">
                {item.icon}
              </div>
              <span className="font-medium text-slate-200 text-sm group-hover:text-blue-400 transition-colors">
                {item.title}
              </span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">{item.desc}</p>
          </button>
        ))}
      </div>
    </div>
  );
};
