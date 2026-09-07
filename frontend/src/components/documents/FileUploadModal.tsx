'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  X,
  UploadCloud,
  FileText,
  FileSpreadsheet,
  FileCode,
  CheckCircle2,
  AlertCircle,
  Trash2,
  Loader2,
  Layers,
  File,
  Search,
} from 'lucide-react';
import { apiClient, DocumentResponse, DocumentSearchHit } from '@/lib/api-client';


interface FileUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onFilesUpdated?: (docs: DocumentResponse[]) => void;
}

interface UploadTask {
  id: string;
  name: string;
  size: number;
  status: 'uploading' | 'success' | 'error';
  errorMessage?: string;
}

const MAX_FILE_SIZE = 25 * 1024 * 1024; // 25 MB
const VALID_EXTENSIONS = ['.pdf', '.docx', '.txt', '.csv', '.json'];

export const FileUploadModal: React.FC<FileUploadModalProps> = ({
  isOpen,
  onClose,
  onFilesUpdated,
}) => {
  const [activeTab, setActiveTab] = useState<'documents' | 'search'>('documents');
  const [documents, setDocuments] = useState<DocumentResponse[]>([]);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadTasks, setUploadTasks] = useState<UploadTask[]>([]);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [isClearingAll, setIsClearingAll] = useState(false);

  // Test Search State (Phase 2 feature parity)
  const [searchQuery, setSearchQuery] = useState('');
  const [searchTopK, setSearchTopK] = useState(3);
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<DocumentSearchHit[] | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);


  // Load documents whenever the modal opens
  useEffect(() => {
    if (!isOpen) return;
    let ignore = false;

    async function loadOnMount() {
      try {
        const res = await apiClient.documents.list();
        if (!ignore) {
          setDocuments(res.documents || []);
          setIsLoadingDocs(false);
          if (onFilesUpdated) {
            onFilesUpdated(res.documents || []);
          }
        }
      } catch (err: unknown) {
        if (!ignore) {
          console.error('Failed to load user documents:', err);
          const msg = err instanceof Error ? err.message : 'Could not load document repository.';
          setGlobalError(msg);
          setIsLoadingDocs(false);
        }
      }
    }

    loadOnMount();

    return () => {
      ignore = true;
    };
  }, [isOpen, onFilesUpdated]);

  if (!isOpen) return null;

  const getFileIcon = (fileType: string, filename: string) => {
    const ext = (fileType || filename.split('.').pop() || '').toLowerCase().replace('.', '');
    if (ext === 'pdf') {
      return <FileText className="w-5 h-5 text-rose-400" />;
    }
    if (ext === 'docx' || ext === 'doc') {
      return <FileText className="w-5 h-5 text-blue-400" />;
    }
    if (ext === 'csv') {
      return <FileSpreadsheet className="w-5 h-5 text-emerald-400" />;
    }
    if (ext === 'json') {
      return <FileCode className="w-5 h-5 text-amber-400" />;
    }
    if (ext === 'txt') {
      return <File className="w-5 h-5 text-slate-300" />;
    }
    return <FileText className="w-5 h-5 text-slate-400" />;
  };

  const formatFileSize = (bytes: number) => {
    if (!bytes || bytes === 0) return '0 B';
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // Upload handling with real-time feedback
  const handleUploadFiles = async (fileList: FileList | File[]) => {
    const filesToUpload = Array.from(fileList);
    if (filesToUpload.length === 0) return;

    for (const file of filesToUpload) {
      const ext = '.' + (file.name.split('.').pop()?.toLowerCase() || '');
      const taskId = `task-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`;

      // Client-side extension validation
      if (!VALID_EXTENSIONS.includes(ext)) {
        setUploadTasks((prev) => [
          ...prev,
          {
            id: taskId,
            name: file.name,
            size: file.size,
            status: 'error',
            errorMessage: `Unsupported file type (${ext}). Supported: PDF, DOCX, TXT, CSV, JSON.`,
          },
        ]);
        continue;
      }

      // Client-side empty file check
      if (file.size === 0) {
        setUploadTasks((prev) => [
          ...prev,
          {
            id: taskId,
            name: file.name,
            size: file.size,
            status: 'error',
            errorMessage: 'File is empty (0 bytes).',
          },
        ]);
        continue;
      }

      // Client-side size limit check (25 MB)
      if (file.size > MAX_FILE_SIZE) {
        setUploadTasks((prev) => [
          ...prev,
          {
            id: taskId,
            name: file.name,
            size: file.size,
            status: 'error',
            errorMessage: 'File size exceeds maximum limit of 25MB.',
          },
        ]);
        continue;
      }

      // Add active upload task
      setUploadTasks((prev) => [
        ...prev,
        {
          id: taskId,
          name: file.name,
          size: file.size,
          status: 'uploading',
        },
      ]);

      try {
        const uploadedDoc = await apiClient.documents.upload(file);

        // Update task to success
        setUploadTasks((prev) =>
          prev.map((t) => (t.id === taskId ? { ...t, status: 'success' } : t))
        );

        // Prepend to documents list
        setDocuments((prev) => [uploadedDoc, ...prev.filter((d) => d.id !== uploadedDoc.id)]);

        if (onFilesUpdated) {
          onFilesUpdated([uploadedDoc, ...documents]);
        }

        // Auto-dismiss completed task badge after 3 seconds
        setTimeout(() => {
          setUploadTasks((prev) => prev.filter((t) => t.id !== taskId));
        }, 3000);
      } catch (err: unknown) {
        console.warn('Failed to upload file:', err);
        const errorMsg =
          err instanceof Error ? err.message : 'Server rejected file upload.';
        setUploadTasks((prev) =>
          prev.map((t) =>
            t.id === taskId ? { ...t, status: 'error', errorMessage: errorMsg } : t
          )
        );
      }
    }
  };

  // Delete document
  const handleDelete = async (docId: string) => {
    setDeletingId(docId);
    try {
      await apiClient.documents.delete(docId);
      const remaining = documents.filter((d) => d.id !== docId);
      setDocuments(remaining);
      if (onFilesUpdated) {
        onFilesUpdated(remaining);
      }
    } catch (err: unknown) {
      console.warn('Failed to delete document:', err);
      const msg = err instanceof Error ? err.message : 'Failed to delete document.';
      setGlobalError(msg);
    } finally {
      setDeletingId(null);
    }
  };

  const handleClearAll = async () => {
    if (!window.confirm('Are you sure you want to delete all documents and clear your vector store?')) {
      return;
    }
    setIsClearingAll(true);
    try {
      await apiClient.documents.clearAll();
      setDocuments([]);
      setSearchResults(null);
      if (onFilesUpdated) {
        onFilesUpdated([]);
      }
    } catch (err: unknown) {
      console.warn('Failed to clear documents:', err);
      const msg = err instanceof Error ? err.message : 'Failed to clear documents.';
      setGlobalError(msg);
    } finally {
      setIsClearingAll(false);
    }
  };

  const handleRunSearch = async () => {
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    try {
      const res = await apiClient.documents.search(searchQuery.trim(), searchTopK);
      setSearchResults(res.matches || []);
    } catch (err: unknown) {
      console.warn('Failed to run test search:', err);
      const msg = err instanceof Error ? err.message : 'Vector search failed.';
      setGlobalError(msg);
    } finally {
      setIsSearching(false);
    }
  };


  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleUploadFiles(e.dataTransfer.files);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="relative w-full max-w-xl bg-slate-900 border border-slate-700/80 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800 bg-slate-950/60">
          <div>
            <h3 className="font-semibold text-slate-100 text-base">Knowledge Base & Documents</h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Upload documents to ground responses with RAG citations.
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-colors cursor-pointer"
            title="Close modal"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Navigation (Phase 2 feature parity) */}
        <div className="flex border-b border-slate-800 bg-slate-950/50 px-5">
          <button
            type="button"
            onClick={() => setActiveTab('documents')}
            className={`flex items-center gap-2 py-3 px-3 text-xs font-medium border-b-2 transition-colors ${
              activeTab === 'documents'
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <UploadCloud className="w-4 h-4" />
            <span>Documents ({documents.length})</span>
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('search')}
            className={`flex items-center gap-2 py-3 px-3 text-xs font-medium border-b-2 transition-colors ${
              activeTab === 'search'
                ? 'border-blue-500 text-blue-400'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            <Search className="w-4 h-4" />
            <span>Test Document Search</span>
          </button>
        </div>

        {/* Global Error Banner */}
        {globalError && (
          <div className="mx-5 mt-4 p-3 rounded-xl bg-rose-950/80 border border-rose-800 text-rose-200 text-xs flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
              <span>{globalError}</span>
            </div>
            <button
              type="button"
              onClick={() => setGlobalError(null)}
              className="text-rose-400 hover:text-rose-200 p-0.5"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* Modal Body - Tab 1: Documents & Upload */}
        {activeTab === 'documents' && (
          <div className="p-5 space-y-4 overflow-y-auto flex-1">

          {/* Drag and Drop Upload Zone */}
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
              isDragging
                ? 'border-blue-500 bg-blue-950/30 ring-4 ring-blue-500/20'
                : 'border-slate-700/80 hover:border-slate-600 bg-slate-950/50 hover:bg-slate-950/80'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.docx,.txt,.csv,.json"
              className="hidden"
              onChange={(e) => {
                if (e.target.files) handleUploadFiles(e.target.files);
                e.target.value = '';
              }}
            />
            <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-slate-800/90 border border-slate-700 flex items-center justify-center text-blue-400 shadow-md">
              <UploadCloud className="w-6 h-6" />
            </div>
            <p className="text-sm font-medium text-slate-200">
              Click or drag documents to upload
            </p>
            <p className="text-xs text-slate-400 mt-1">
              Supports <span className="text-slate-300 font-medium">PDF, DOCX, TXT, CSV, JSON</span> (up to 25MB)
            </p>
          </div>

          {/* Active Upload Tasks (In-progress & Instant errors) */}
          {uploadTasks.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs font-medium text-slate-400 px-1">Upload Queue</p>
              <div className="space-y-1.5">
                {uploadTasks.map((task) => (
                  <div
                    key={task.id}
                    className={`flex items-center justify-between p-2.5 rounded-xl border text-xs transition-all ${
                      task.status === 'uploading'
                        ? 'bg-blue-950/40 border-blue-800/60 text-blue-200'
                        : task.status === 'success'
                        ? 'bg-emerald-950/40 border-emerald-800/60 text-emerald-200'
                        : 'bg-rose-950/40 border-rose-800/60 text-rose-200'
                    }`}
                  >
                    <div className="flex items-center gap-2.5 min-w-0 flex-1">
                      {task.status === 'uploading' ? (
                        <Loader2 className="w-4 h-4 animate-spin text-blue-400 shrink-0" />
                      ) : task.status === 'success' ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                      ) : (
                        <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
                      )}
                      <div className="min-w-0 flex-1">
                        <p className="font-medium truncate">{task.name}</p>
                        {task.errorMessage ? (
                          <p className="text-[11px] text-rose-300">{task.errorMessage}</p>
                        ) : (
                          <p className="text-[11px] text-slate-400">
                            {formatFileSize(task.size)} • {task.status === 'uploading' ? 'Parsing & uploading...' : 'Uploaded!'}
                          </p>
                        )}
                      </div>
                    </div>
                    {task.status === 'error' && (
                      <button
                        type="button"
                        onClick={() =>
                          setUploadTasks((prev) => prev.filter((t) => t.id !== task.id))
                        }
                        className="p-1 text-slate-400 hover:text-slate-200"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Persisted Uploaded Documents List */}
          <div className="space-y-2 pt-1">
            <div className="flex items-center justify-between text-xs text-slate-400 px-1">
              <span>Your Documents ({documents.length})</span>
              <div className="flex items-center gap-3">
                {documents.length > 0 && (
                  <button
                    type="button"
                    onClick={handleClearAll}
                    disabled={isClearingAll}
                    className="text-rose-400 hover:text-rose-300 text-[11px] font-medium transition-colors flex items-center gap-1 cursor-pointer disabled:opacity-50"
                    title="Delete all documents and clear FAISS vector store"
                  >
                    {isClearingAll ? (
                      <Loader2 className="w-3 h-3 animate-spin" />
                    ) : (
                      <Trash2 className="w-3 h-3" />
                    )}
                    <span>Clear All</span>
                  </button>
                )}
                <span className="text-emerald-400 text-[11px] font-medium flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  User-Isolated Storage
                </span>
              </div>
            </div>

            {isLoadingDocs ? (
              <div className="py-8 flex flex-col items-center justify-center text-xs text-slate-400 gap-2 border border-slate-800/80 rounded-xl bg-slate-950/40">
                <Loader2 className="w-5 h-5 animate-spin text-blue-500" />
                <span>Loading your documents...</span>
              </div>
            ) : documents.length === 0 ? (
              <div className="py-8 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-xl bg-slate-950/30">
                No documents uploaded yet. Add files above to begin grounding your queries.
              </div>
            ) : (
              <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                {documents.map((doc) => (
                  <div
                    key={doc.id}
                    className="flex items-center justify-between p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-200 group hover:border-slate-700 transition-colors shadow-sm"
                  >
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                      <div className="p-2 rounded-lg bg-slate-900 border border-slate-800 shrink-0">
                        {getFileIcon(doc.file_type, doc.filename)}
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="font-medium text-slate-200 truncate pr-2">{doc.filename}</p>
                        <div className="flex items-center gap-2 text-[11px] text-slate-500 mt-0.5">
                          <span>{formatFileSize(doc.file_size_bytes)}</span>
                          <span>•</span>
                          <span className="flex items-center gap-1">
                            <Layers className="w-3 h-3 text-slate-400" />
                            {doc.num_pages} {doc.num_pages === 1 ? 'page' : 'pages'}
                          </span>
                          <span>•</span>
                          <span className="uppercase text-[10px] font-semibold text-slate-400 bg-slate-900 px-1.5 py-0.2 rounded border border-slate-800">
                            {doc.file_type}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Delete action */}
                    <button
                      type="button"
                      onClick={() => handleDelete(doc.id)}
                      disabled={deletingId === doc.id}
                      className="p-2 rounded-lg text-slate-500 hover:text-rose-400 hover:bg-rose-950/30 transition-colors cursor-pointer disabled:opacity-50"
                      title="Delete document"
                    >
                      {deletingId === doc.id ? (
                        <Loader2 className="w-4 h-4 animate-spin text-rose-400" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
        )}

        {/* Modal Body - Tab 2: Test Document Search (Phase 2 feature parity) */}
        {activeTab === 'search' && (
          <div className="p-5 space-y-4 overflow-y-auto flex-1">
            <div className="p-3.5 rounded-xl bg-blue-950/30 border border-blue-800/40 text-xs text-slate-300 space-y-1">
              <div className="flex items-center gap-2 text-blue-400 font-semibold">
                <Search className="w-4 h-4" />
                <span>FAISS Semantic Similarity Search</span>
              </div>
              <p className="text-[11px] text-slate-400">
                Direct Phase 2 tester: Test query embeddings against your stored document chunks using cosine similarity.
              </p>
            </div>

            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Search Query
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleRunSearch();
                    }}
                    placeholder="e.g. artificial intelligence algorithms, quantum superposition..."
                    className="flex-1 px-3 py-2 rounded-xl bg-slate-950 border border-slate-700 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  />
                  <button
                    type="button"
                    onClick={handleRunSearch}
                    disabled={isSearching || !searchQuery.trim()}
                    className="px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer disabled:opacity-50 shadow-md shadow-blue-600/30"
                  >
                    {isSearching ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Search className="w-4 h-4" />
                    )}
                    <span>Search</span>
                  </button>
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-300">Top-K Matching Chunks</span>
                  <span className="text-blue-400 font-mono">{searchTopK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  step="1"
                  value={searchTopK}
                  onChange={(e) => setSearchTopK(parseInt(e.target.value))}
                  className="w-full accent-blue-600 cursor-pointer"
                />
              </div>
            </div>

            {/* Results */}
            <div className="space-y-2 pt-2">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>Search Results {searchResults !== null ? `(${searchResults.length})` : ''}</span>
              </div>

              {isSearching ? (
                <div className="py-8 flex flex-col items-center justify-center text-xs text-slate-400 gap-2 border border-slate-800 rounded-xl bg-slate-950/40">
                  <Loader2 className="w-5 h-5 animate-spin text-blue-500" />
                  <span>Computing query embedding and searching FAISS index...</span>
                </div>
              ) : searchResults === null ? (
                <div className="py-8 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-xl bg-slate-950/30">
                  Enter a query and click Search to test retrieval against your knowledge base.
                </div>
              ) : searchResults.length === 0 ? (
                <div className="py-8 text-center text-xs text-amber-400/80 border border-amber-900/40 rounded-xl bg-amber-950/20">
                  No matching chunks found. Upload documents first or try another search term.
                </div>
              ) : (
                <div className="space-y-2.5 max-h-72 overflow-y-auto pr-1">
                  {searchResults.map((hit) => (
                    <div
                      key={hit.rank}
                      className="p-3 rounded-xl bg-slate-950 border border-slate-800 text-xs text-slate-300 space-y-2"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-blue-400 font-mono">
                            Result #{hit.rank}
                          </span>
                          <span className="font-medium text-slate-200 truncate max-w-[200px]">
                            {hit.filename}
                          </span>
                          <span className="text-[11px] text-slate-500">
                            {hit.page_start === hit.page_end
                              ? `Page ${hit.page_start}`
                              : `Pages ${hit.page_start}–${hit.page_end}`}
                          </span>
                        </div>
                        <span className="px-2 py-0.5 rounded-full bg-emerald-950 border border-emerald-700 text-emerald-300 text-[10px] font-mono font-semibold">
                          Score: {hit.score.toFixed(4)} ({Math.round(hit.score * 100)}%)
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 leading-relaxed italic bg-slate-900/80 p-2 rounded-lg border border-slate-800/80 font-mono whitespace-pre-wrap">
                        {hit.text}
                      </p>
                      <div className="flex gap-3 text-[10px] text-slate-500">
                        <span>Words: {hit.word_count}</span>
                        <span>Characters: {hit.char_count}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}


        {/* Modal Footer */}
        <div className="px-5 py-3 border-t border-slate-800 bg-slate-950/60 flex items-center justify-between text-xs text-slate-500">
          <span>Metadata stored in PostgreSQL with isolated filesystem partitions.</span>
          <button
            type="button"
            onClick={onClose}
            className="px-3.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition-colors cursor-pointer"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
