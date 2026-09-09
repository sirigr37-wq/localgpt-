'use client';

import React from 'react';
import Link from 'next/link';
import { ArrowLeft, Sliders, User, Shield, Key } from 'lucide-react';

export default function SettingsPage() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-8">
      <div className="max-w-3xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <Link
            href="/"
            className="p-2 rounded-xl bg-slate-900 border border-slate-800 hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <h1 className="text-xl font-bold text-slate-100">User Settings & Preferences</h1>
            <p className="text-xs text-slate-400">
              Manage your LocalGPT account, security, and inference configuration.
            </p>
          </div>
        </div>

        {/* Profile Card */}
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <User className="w-4 h-4 text-blue-400" />
            <span>Profile Information</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div>
              <label className="block text-slate-400 mb-1">Full Name</label>
              <input
                type="text"
                defaultValue="Alex Developer"
                className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-slate-200"
              />
            </div>
            <div>
              <label className="block text-slate-400 mb-1">Email Address</label>
              <input
                type="email"
                disabled
                defaultValue="alex@example.com"
                className="w-full px-3 py-2 rounded-xl bg-slate-950/60 border border-slate-800 text-slate-500 cursor-not-allowed"
              />
            </div>
          </div>
        </div>

        {/* Security & OAuth */}
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <Shield className="w-4 h-4 text-emerald-400" />
            <span>Security & OAuth</span>
          </div>

          <div className="flex items-center justify-between p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs">
            <div>
              <p className="font-medium text-slate-200">Google OAuth Account</p>
              <p className="text-[11px] text-slate-400">Connected for single-sign on access</p>
            </div>
            <span className="text-[11px] px-2.5 py-1 rounded-full bg-emerald-950/70 border border-emerald-800/60 text-emerald-400 font-medium">
              Connected
            </span>
          </div>
        </div>

        {/* LLM & Model Engine */}
        <div className="p-6 rounded-2xl bg-slate-900 border border-slate-800 space-y-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <Sliders className="w-4 h-4 text-indigo-400" />
            <span>Model Defaults & API Endpoints</span>
          </div>

          <div className="space-y-3 text-xs">
            <div>
              <label className="block text-slate-400 mb-1">Active LLM Provider</label>
              <select className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 outline-none">
                <option>Hosted LLM API Endpoint (Phase 3)</option>
                <option>Local Qwen-2.5-1.5B (Phase 2 Fallback)</option>
              </select>
            </div>

            <div>
              <label className="block text-slate-400 mb-1 flex items-center gap-1.5">
                <Key className="w-3.5 h-3.5 text-slate-400" />
                <span>Hosted LLM API Key</span>
              </label>
              <input
                type="password"
                defaultValue="sk-••••••••••••••••••••••••••••••••"
                className="w-full px-3 py-2 rounded-xl bg-slate-950 border border-slate-800 text-slate-200 font-mono"
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
