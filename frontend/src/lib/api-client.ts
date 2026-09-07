export function getApiBaseUrl(): string {
  // If an explicit API base URL is provided (e.g. Render backend on Vercel), prioritize it
  if (process.env.NEXT_PUBLIC_API_BASE_URL) {
    return process.env.NEXT_PUBLIC_API_BASE_URL.replace(/\/$/, '');
  }

  if (typeof window !== 'undefined') {
    // When running locally on localhost or 127.0.0.1, connect directly to FastAPI backend
    // to prevent Next.js dev server proxy socket hangups (ECONNRESET) on streaming.
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
      return 'http://localhost:8000/api/v1';
    }
    // When accessed via remote tunnels or HTTPS domains, use relative '/api/v1' path.
    return '/api/v1';
  }
  return 'http://127.0.0.1:8000/api/v1';
}

export interface UserResponse {
  id: string;
  email: string;
  full_name?: string | null;
  avatar_url?: string | null;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
  updated_at: string;
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: string;
  expires_in_minutes: number;
}

export interface GoogleAuthUrlResponse {
  auth_url: string | null;
  configured: boolean;
  message?: string;
}

export interface ConversationResponse {
  id: string;
  title: string;
  user_id: string;
  system_prompt?: string | null;
  created_at: string;
  updated_at: string;
  message_count?: number;
}

export interface ConversationDetailResponse extends ConversationResponse {
  messages: ChatMessageResponse[];
}

export interface DocumentResponse {
  id: string;
  user_id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  num_pages: number;
  chunk_count: number;
  status: string;
  created_at: string;
}

export interface DocumentSearchHit {
  rank: number;
  filename: string;
  page_start: number;
  page_end: number;
  score: number;
  text: string;
  word_count: number;
  char_count: number;
}

export interface DocumentSearchResponse {
  query: string;
  total_matches: number;
  matches: DocumentSearchHit[];
}


export interface ChatMessageResponse {
  id: string;
  role: string;
  content: string;
  sources?: Array<Record<string, unknown>>;
  sources_json?: Record<string, unknown> | null;
  feedback?: 'like' | 'dislike' | null;
  created_at: string;
}

export class ApiError extends Error {
  status: number;
  data?: unknown;

  constructor(message: string, status: number, data?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

export function getStoredToken(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('localgpt_token');
}

export function setStoredToken(token: string | null): void {
  if (typeof window === 'undefined') return;
  if (token) {
    localStorage.setItem('localgpt_token', token);
  } else {
    localStorage.removeItem('localgpt_token');
  }
}

export interface RequestOptions extends RequestInit {
  timeoutMs?: number;
}

async function request<T>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const token = getStoredToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  // Attach auth token if available
  if (token && !headers['Authorization']) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  // Default content type to JSON unless FormData is being sent
  if (!(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  const url = `${getApiBaseUrl()}${endpoint}`;
  const isUpload = options.body instanceof FormData || endpoint.includes('/upload');
  const timeoutMs = options.timeoutMs ?? (isUpload ? 300000 : 30000); // 5 min for file upload, 30s standard

  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort(new Error(`Request timed out after ${Math.round(timeoutMs / 1000)} seconds.`));
  }, timeoutMs);

  if (options.signal) {
    if (options.signal.aborted) {
      controller.abort(options.signal.reason);
    } else {
      options.signal.addEventListener('abort', () => {
        controller.abort(options.signal?.reason);
      });
    }
  }

  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      headers,
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeoutId);
  }

  if (response.status === 204) {
    return null as T;
  }

  let data: unknown;
  const contentType = response.headers.get('content-type');
  if (contentType && contentType.includes('application/json')) {
    data = await response.json();
  } else {
    data = await response.text();
  }

  if (!response.ok) {
    const errorDetail =
      data && typeof data === 'object' && 'detail' in data
        ? String((data as { detail: unknown }).detail)
        : typeof data === 'string'
        ? data
        : 'API request failed';
    throw new ApiError(errorDetail, response.status, data);
  }

  return data as T;
}

export const apiClient = {
  // Authentication & OAuth
  auth: {
    register: (data: { email: string; password: string; full_name?: string }) =>
      request<UserResponse>('/auth/register', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    login: (data: { email: string; password: string }) =>
      request<AuthTokenResponse>('/auth/login', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    getMe: () => request<UserResponse>('/auth/me'),
    logout: () =>
      request<{ message: string }>('/auth/logout', {
        method: 'POST',
      }),
    getGoogleAuthUrl: (redirectUri?: string) => {
      const q = redirectUri ? `?redirect_uri=${encodeURIComponent(redirectUri)}` : '';
      return request<GoogleAuthUrlResponse>(`/auth/google/url${q}`);
    },
    googleCallback: (code: string, redirectUri?: string) =>
      request<AuthTokenResponse>('/auth/google/callback', {
        method: 'POST',
        body: JSON.stringify({ code, redirect_uri: redirectUri }),
      }),
    requestPasswordReset: (email: string) =>
      request<{ message: string; reset_token?: string | null }>(
        '/auth/password-reset/request',
        {
          method: 'POST',
          body: JSON.stringify({ email }),
        }
      ),
    confirmPasswordReset: (token: string, new_password: string) =>
      request<{ status: string; message: string }>('/auth/password-reset/confirm', {
        method: 'POST',
        body: JSON.stringify({ token, new_password }),
      }),
  },

  // Conversations
  conversations: {
    list: (query?: string) => {
      const q = query ? `?q=${encodeURIComponent(query)}` : '';
      return request<ConversationResponse[]>(`/conversations${q}`);
    },
    create: (data: { title?: string; system_prompt?: string }) =>
      request<ConversationResponse>('/conversations', {
        method: 'POST',
        body: JSON.stringify(data),
      }),
    get: (id: string) => request<ConversationDetailResponse>(`/conversations/${id}`),
    update: (id: string, data: { title?: string; system_prompt?: string }) =>
      request<ConversationResponse>(`/conversations/${id}`, {
        method: 'PUT',
        body: JSON.stringify(data),
      }),
    delete: (id: string) =>
      request<void>(`/conversations/${id}`, {
        method: 'DELETE',
      }),
  },
  // Documents

  documents: {
    list: () => request<{ total: number; documents: DocumentResponse[] }>('/documents'),
    upload: (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      return request<DocumentResponse>('/documents/upload', {
        method: 'POST',
        body: formData,
      });
    },
    search: (query: string, top_k: number = 3) =>
      request<DocumentSearchResponse>('/documents/search', {
        method: 'POST',
        body: JSON.stringify({ query, top_k }),
      }),
    get: (id: string) => request<DocumentResponse>(`/documents/${id}`),
    delete: (id: string) =>
      request<void>(`/documents/${id}`, {
        method: 'DELETE',
      }),
    clearAll: () =>
      request<{ count: number; message: string }>('/documents', {
        method: 'DELETE',
      }),
  },


  // Chat & Messages
  chat: {
    postMessage: (
      conversationId: string,
      message: { role: string; content: string; sources?: Array<Record<string, unknown>> }
    ) =>
      request<ChatMessageResponse>(`/chat/${conversationId}/messages`, {
        method: 'POST',
        body: JSON.stringify(message),
      }),
    streamMessage: async function* (
      conversationId: string,
      content: string,
      onToken?: (token: string) => void,
      signal?: AbortSignal,
      regenerate: boolean = false
    ): AsyncGenerator<string, void, unknown> {
      const token = getStoredToken();
      let response: Response;
      const url = `${getApiBaseUrl()}/chat/${conversationId}/stream${
        regenerate ? '?regenerate=true' : ''
      }`;
      try {
        response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: JSON.stringify({ role: 'user', content }),
          signal,
        });
      } catch (err: unknown) {
        if (
          (err instanceof DOMException && err.name === 'AbortError') ||
          (err instanceof Error && err.name === 'AbortError')
        ) {
          return;
        }
        throw err;
      }

      if (!response.ok) {
        let errDetail = 'Failed to stream response';
        try {
          const errData = await response.json();
          errDetail = errData.detail || errDetail;
        } catch {
          // fallback
        }
        throw new ApiError(errDetail, response.status);
      }

      if (!response.body) {
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || !trimmed.startsWith('data: ')) continue;
            const dataStr = trimmed.slice(6);
            if (dataStr === '[DONE]') return;
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.token) {
                if (onToken) onToken(parsed.token);
                yield parsed.token;
              }
            } catch {
              // ignore malformed SSE token json
            }
          }
        }
      } catch (err: unknown) {
        if (
          (err instanceof DOMException && err.name === 'AbortError') ||
          (err instanceof Error && err.name === 'AbortError')
        ) {
          return;
        }
        throw err;
      }
    },
    editMessage: (conversationId: string, messageId: string, content: string) =>
      request<ChatMessageResponse>(`/chat/${conversationId}/messages/${messageId}`, {
        method: 'PUT',
        body: JSON.stringify({ role: 'user', content }),
      }),
    updateFeedback: (messageId: string, feedback: 'like' | 'dislike' | null) =>
      request<{ status: string }>(`/chat/messages/${messageId}/feedback`, {
        method: 'PATCH',
        body: JSON.stringify({ feedback }),
      }),
  },
};
