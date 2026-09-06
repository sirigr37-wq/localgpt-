'use client';

import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import hljs from 'highlight.js';

interface CodeBlockProps {
  language?: string;
  value: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({ language, value }) => {
  const [isCopied, setIsCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  const getHighlightedCode = () => {
    const raw = value.replace(/\n$/, '');
    if (language && hljs.getLanguage(language)) {
      try {
        return hljs.highlight(raw, { language }).value;
      } catch {
        // fallback to highlightAuto
      }
    }
    try {
      return hljs.highlightAuto(raw).value;
    } catch {
      return raw;
    }
  };

  const highlightedHtml = getHighlightedCode();

  return (
    <div className="my-3.5 rounded-xl overflow-hidden border border-slate-700/80 bg-slate-950 font-mono text-xs shadow-lg">
      <div className="flex items-center justify-between px-4 py-2 bg-slate-800/90 border-b border-slate-700/70 text-slate-400 select-none">
        <span className="font-semibold uppercase tracking-wider text-[11px] text-slate-300">
          {language || 'code'}
        </span>
        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1.5 text-xs text-slate-300 hover:text-white px-2.5 py-1 rounded bg-slate-700/50 hover:bg-slate-700 transition-all cursor-pointer"
          title="Copy code"
        >
          {isCopied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-emerald-400 font-medium">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5 text-slate-400" />
              <span>Copy code</span>
            </>
          )}
        </button>
      </div>
      <pre className="p-4 overflow-x-auto text-slate-100 leading-relaxed font-mono">
        <code
          className={`hljs ${language ? `language-${language}` : ''}`}
          dangerouslySetInnerHTML={{ __html: highlightedHtml }}
        />
      </pre>
    </div>
  );
};
