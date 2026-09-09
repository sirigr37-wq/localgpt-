'use client';

import React, { useEffect, useState, Suspense } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import { Sparkles, AlertCircle, ArrowRight } from 'lucide-react';
import Link from 'next/link';

function GoogleCallbackContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { handleGoogleCallback } = useAuth();
  const [asyncError, setAsyncError] = useState<string | null>(null);
  const processedCodeRef = React.useRef<string | null>(null);

  const queryError = searchParams.get('error');
  const code = searchParams.get('code');

  const errorMsg = queryError
    ? `Google authentication was cancelled or failed: ${queryError}`
    : (!code ? 'No authorization code provided in Google callback.' : asyncError);

  useEffect(() => {
    if (!code || queryError) {
      return;
    }

    // Google authorization codes are strictly single-use.
    // In React StrictMode (Next.js development mode), useEffect triggers twice on initial mount.
    // Ensure we exchange each authorization code exactly once.
    if (processedCodeRef.current === code) {
      return;
    }
    processedCodeRef.current = code;

    handleGoogleCallback(code)
      .then(() => {
        router.push('/');
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to complete Google authentication exchange.';
        setAsyncError(msg);
      });
  }, [code, queryError, handleGoogleCallback, router]);

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-4">
      <div className="w-full max-w-md bg-slate-900 border border-slate-800 rounded-3xl p-8 shadow-2xl text-center relative overflow-hidden">
        <div className="inline-flex items-center justify-center w-14 h-14 rounded-2xl bg-blue-600/20 text-blue-400 border border-blue-500/30 mb-4">
          <Sparkles className="w-7 h-7 animate-pulse" />
        </div>

        {errorMsg ? (
          <div className="space-y-4">
            <div className="flex items-center justify-center gap-2 text-rose-400 text-sm font-semibold">
              <AlertCircle className="w-5 h-5" />
              <span>Authentication Error</span>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed bg-rose-950/20 border border-rose-900/40 p-3 rounded-xl">
              {errorMsg}
            </p>
            <Link
              href="/login"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition-colors"
            >
              <span>Back to Login</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            <h2 className="text-lg font-bold text-slate-100">Authenticating with Google...</h2>
            <p className="text-xs text-slate-400">
              Verifying your Google session and establishing your protected account.
            </p>
            <div className="flex justify-center items-center gap-1.5 py-4">
              <div className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0ms' }} />
              <div className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '150ms' }} />
              <div className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function GoogleCallbackPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen bg-slate-950 flex items-center justify-center text-slate-400 text-xs">
          Loading authentication handler...
        </div>
      }
    >
      <GoogleCallbackContent />
    </Suspense>
  );
}
