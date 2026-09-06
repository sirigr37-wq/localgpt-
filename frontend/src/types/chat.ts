export interface Citation {
  id?: string;
  filename: string;
  page_start?: number;
  page_end?: number;
  score?: number;
  excerpt?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  createdAt: string;
  feedback?: 'like' | 'dislike' | null;
  sources?: Citation[];
  isStreaming?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messageCount?: number;
  lastMessageSnippet?: string;
}

export interface UploadedFileItem {
  id: string;
  name: string;
  size: number;
  type: string;
  progress: number;
  status: 'uploading' | 'completed' | 'error';
  errorMessage?: string;
}

export interface UserSettings {
  systemPrompt: string;
  temperature: number;
  topK?: number;
  topP: number;
  maxTokens: number;
  enableRag: boolean;
  theme: 'dark' | 'light' | 'system';
}

