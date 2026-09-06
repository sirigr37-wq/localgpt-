'use strict';
'use client';

import React from 'react';
import { Bot } from 'lucide-react';

export const LoadingState: React.FC = () => {
  return (
    <div className="flex gap-4 p-4 md:px-6 w-full max-w-4xl mx-auto items-start">
      <div className="w-8 h-8 rounded-lg bg-indigo-600/30 border border-indigo-500/40 flex items-center justify-center shrink-0 text-indigo-400">
        <Bot className="w-4 h-4" />
      </div>
      <div className="flex-1 space-y-2">
        <div className="flex items-center gap-1.5 h-6">
          <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0ms' }} />
          <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }} />
          <div className="w-2 h-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '300ms' }} />
        </div>
      </div>
    </div>
  );
};
