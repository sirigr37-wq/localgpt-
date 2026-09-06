export interface User {
  id: string;
  email: string;
  fullName: string;
  avatarUrl?: string | null;
  provider: 'local' | 'google';
  createdAt: string;
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}
