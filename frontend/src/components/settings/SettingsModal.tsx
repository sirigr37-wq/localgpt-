'use strict';
'use client';

import React, { useState } from 'react';
import { X, User, Sliders, Check, Shield } from 'lucide-react';
import { UserSettings } from '@/types/chat';
import { User as UserType } from '@/types/auth';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentUser: UserType | null;
  settings: UserSettings;
  onSaveSettings: (newSettings: UserSettings) => void;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  currentUser,
  settings,
  onSaveSettings,
}) => {
  const [activeTab, setActiveTab] = useState<'profile' | 'model'>('profile');
  const [localSettings, setLocalSettings] = useState<UserSettings>(settings);
  const [savedSuccess, setSavedSuccess] = useState(false);

  if (!isOpen) return null;

  const handleSave = () => {
    onSaveSettings(localSettings);
    setSavedSuccess(true);
    setTimeout(() => {
      setSavedSuccess(false);
      onClose();
    }, 800);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-xs">
      <div className="relative w-full max-w-xl bg-slate-900 border border-slate-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[88vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
          <h3 className="font-semibold text-slate-100 text-base">Account & System Settings</h3>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex border-b border-slate-800 bg-slate-950/50 px-5">
          <button
            onClick={() => setActiveTab('profile')}
            className={`flex items-center gap-2 py-3 px-3 text-xs font-medium border-b-2 transition-colors ${
              activeTab === 'profile'
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <User className="w-4 h-4" />
            <span>Profile & Account</span>
          </button>
          <button
            onClick={() => setActiveTab('model')}
            className={`flex items-center gap-2 py-3 px-3 text-xs font-medium border-b-2 transition-colors ${
              activeTab === 'model'
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Sliders className="w-4 h-4" />
            <span>Model & RAG Controls</span>
          </button>
        </div>

        {/* Tab Content */}
        <div className="p-6 space-y-5 overflow-y-auto flex-1">
          {activeTab === 'profile' ? (
            <div className="space-y-4">
              {/* Profile Card */}
              <div className="flex items-center gap-4 p-4 rounded-xl bg-slate-950 border border-slate-800">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center font-bold text-xl text-white shadow-md">
                  {currentUser?.fullName?.[0]?.toUpperCase() || 'U'}
                </div>
                <div className="space-y-0.5">
                  <p className="font-semibold text-slate-100 text-sm">
                    {currentUser?.fullName || 'Active User'}
                  </p>
                  <p className="text-xs text-slate-400">{currentUser?.email || 'user@example.com'}</p>
                  <div className="flex items-center gap-1.5 pt-1 text-[11px] text-emerald-400">
                    <Shield className="w-3.5 h-3.5" />
                    <span>Protected Account (Phase 3)</span>
                  </div>
                </div>
              </div>

              <div className="space-y-3 pt-2">
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Display Name
                  </label>
                  <input
                    type="text"
                    defaultValue={currentUser?.fullName || 'Demo User'}
                    className="w-full px-3 py-2 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 focus:outline-none focus:border-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-slate-300 mb-1">
                    Email Address
                  </label>
                  <input
                    type="email"
                    disabled
                    value={currentUser?.email || 'user@example.com'}
                    className="w-full px-3 py-2 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-400 cursor-not-allowed"
                  />
                </div>
              </div>
            </div>
          ) : (
            <div className="space-y-5">
              {/* System Prompt */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-xs font-medium text-slate-200">
                    System Prompt
                  </label>
                  <button
                    type="button"
                    onClick={() =>
                      setLocalSettings({
                        ...localSettings,
                        systemPrompt:
                          'You are LocalGPT, a helpful and direct AI platform powered by modern inference models with deep inspection and RAG document grounding.',
                      })
                    }
                    className="text-[11px] text-blue-400 hover:text-blue-300 transition-colors cursor-pointer"
                  >
                    Reset Prompt
                  </button>
                </div>
                <textarea
                  rows={3}
                  value={localSettings.systemPrompt}
                  onChange={(e) =>
                    setLocalSettings({ ...localSettings, systemPrompt: e.target.value })
                  }
                  className="w-full p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs text-slate-200 focus:outline-none focus:border-blue-500/70"
                />
              </div>

              {/* Generation Controls */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-semibold text-slate-200">Sampling & Generation Controls</p>
                  <button
                    type="button"
                    onClick={() =>
                      setLocalSettings({
                        ...localSettings,
                        temperature: 0.7,
                        topK: 50,
                        topP: 0.9,
                        maxTokens: 512,
                      })
                    }
                    className="text-[11px] text-blue-400 hover:text-blue-300 transition-colors cursor-pointer"
                  >
                    Reset Controls
                  </button>
                </div>

                {/* Temperature */}
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">Temperature (Randomness)</span>
                    <span className="text-blue-400 font-mono">{localSettings.temperature.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.05"
                    value={localSettings.temperature}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        temperature: parseFloat(e.target.value),
                      })
                    }
                    className="w-full accent-blue-600 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                    <span>Deterministic (0.0)</span>
                    <span>Balanced (0.7)</span>
                    <span>Creative (2.0)</span>
                  </div>
                </div>

                {/* Top-K Sampling */}
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">Top-K Candidate Filtering</span>
                    <span className="text-blue-400 font-mono">{localSettings.topK ?? 50}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="100"
                    step="1"
                    value={localSettings.topK ?? 50}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        topK: parseInt(e.target.value),
                      })
                    }
                    className="w-full accent-blue-600 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                    <span>Strict (1)</span>
                    <span>Standard (50)</span>
                    <span>Broad (100)</span>
                  </div>
                </div>

                {/* Top-P Nucleus Sampling */}
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">Top-P (Nucleus Sampling)</span>
                    <span className="text-blue-400 font-mono">{localSettings.topP.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="1.0"
                    step="0.05"
                    value={localSettings.topP}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        topP: parseFloat(e.target.value),
                      })
                    }
                    className="w-full accent-blue-600 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                    <span>Focused (0.1)</span>
                    <span>Default (0.9)</span>
                    <span>Full (1.0)</span>
                  </div>
                </div>

                {/* Max New Tokens */}
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-300">Max New Tokens</span>
                    <span className="text-blue-400 font-mono">{localSettings.maxTokens}</span>
                  </div>
                  <input
                    type="range"
                    min="64"
                    max="2048"
                    step="64"
                    value={localSettings.maxTokens}
                    onChange={(e) =>
                      setLocalSettings({
                        ...localSettings,
                        maxTokens: parseInt(e.target.value),
                      })
                    }
                    className="w-full accent-blue-600 cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                    <span>Brief (64)</span>
                    <span>Standard (512)</span>
                    <span>Extended (2048)</span>
                  </div>
                </div>

                {/* RAG Toggle */}
                <div className="flex items-center justify-between p-3 rounded-xl bg-slate-950 border border-slate-800">
                  <div>
                    <p className="text-xs font-medium text-slate-200">RAG Document Grounding</p>
                    <p className="text-[11px] text-slate-400">
                      Inject relevant document chunks and page citations into context
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={localSettings.enableRag}
                    onChange={(e) =>
                      setLocalSettings({ ...localSettings, enableRag: e.target.checked })
                    }
                    className="w-4 h-4 accent-blue-600 cursor-pointer"
                  />
                </div>
              </div>
            </div>

          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-slate-800 flex items-center justify-between bg-slate-950/60">
          <span className="text-xs text-slate-500">LocalGPT v3.0</span>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-3.5 py-1.5 text-xs font-medium rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold shadow-md shadow-blue-600/30 transition-colors"
            >
              {savedSuccess ? (
                <>
                  <Check className="w-3.5 h-3.5" />
                  <span>Saved</span>
                </>
              ) : (
                <span>Save Changes</span>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
