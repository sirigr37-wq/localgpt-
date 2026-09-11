'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';
import { User } from '@/types/auth';
import { apiClient, setStoredToken, getStoredToken, ApiError } from '@/lib/api-client';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string) => Promise<void>;
  logout: () => Promise<void>;
  initiateGoogleLogin: () => Promise<{ configured: boolean; message?: string }>;
  handleGoogleCallback: (code: string, redirectUri?: string) => Promise<void>;
  requestPasswordReset: (email: string) => Promise<{ message: string; resetToken?: string | null }>;
  confirmPasswordReset: (token: string, newPassword: string) => Promise<void>;
  clearError: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const clearError = () => setError(null);

  // Initialize auth state by checking stored token
  // NOTE: This is a silent session restore — errors must NEVER surface to the UI here.
  useEffect(() => {
    const initAuth = async () => {
      const storedToken = getStoredToken();
      if (storedToken) {
        setToken(storedToken);
        try {
          const userData = await apiClient.auth.getMe();
          setUser({
            id: userData.id,
            email: userData.email,
            fullName: userData.full_name || userData.email.split('@')[0],
            avatarUrl: userData.avatar_url,
            provider: 'local',
            createdAt: userData.created_at,
          });
        } catch (err) {
          // Token invalid/expired — clear silently, do NOT show error to user
          console.warn('Session verification failed; clearing invalid token.', err);
          setStoredToken(null);
          setToken(null);
          setUser(null);
          setError(null); // ensure no stale errors show
        }
      }
      setIsLoading(false);
    };

    initAuth();
  }, []);

  const login = async (email: string, password: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const tokenData = await apiClient.auth.login({ email, password });
      setStoredToken(tokenData.access_token);
      setToken(tokenData.access_token);

      const userData = await apiClient.auth.getMe();
      setUser({
        id: userData.id,
        email: userData.email,
        fullName: userData.full_name || userData.email.split('@')[0],
        avatarUrl: userData.avatar_url,
        provider: 'local',
        createdAt: userData.created_at,
      });
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : (err instanceof Error ? err.message : 'Invalid email or password.');
      setError(msg);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (email: string, password: string, fullName?: string) => {
    setIsLoading(true);
    setError(null);
    try {
      await apiClient.auth.register({ email, password, full_name: fullName });
      // Automatically log in upon successful registration
      await login(email, password);
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : (err instanceof Error ? err.message : 'Registration failed.');
      setError(msg);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    try {
      await apiClient.auth.logout();
    } catch {
      // Ignore network errors on logout
    } finally {
      setStoredToken(null);
      setToken(null);
      setUser(null);
    }
  };

  const initiateGoogleLogin = async () => {
    setError(null);
    try {
      const callbackUrl = `${window.location.origin}/auth/callback/google`;
      const res = await apiClient.auth.getGoogleAuthUrl(callbackUrl);

      if (res.configured && res.auth_url) {
        window.location.href = res.auth_url;
        return { configured: true };
      } else {
        // 'not configured' is informational only — show as notice, NOT as red error
        return {
          configured: false,
          message:
            res.message ||
            'Google OAuth credentials not configured. Please set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.',
        };
      }
    } catch (err: unknown) {
      // Only show red error banner for genuine network/server failures (not 404 cold-start)
      const apiErr = err instanceof ApiError ? err : null;
      if (apiErr && (apiErr.status === 404 || apiErr.status === 503)) {
        // Backend is sleeping or route missing — treat as 'not configured' notice
        return {
          configured: false,
          message: 'Backend is starting up, please wait a moment and try again.',
        };
      }
      const msg = apiErr ? apiErr.message : (err instanceof Error ? err.message : 'Failed to connect to Google OAuth service.');
      setError(msg);
      return { configured: false, message: msg };
    }
  };

  const handleGoogleCallback = React.useCallback(async (code: string, redirectUri?: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const callbackUrl = redirectUri || `${window.location.origin}/auth/callback/google`;
      const tokenData = await apiClient.auth.googleCallback(code, callbackUrl);
      setStoredToken(tokenData.access_token);
      setToken(tokenData.access_token);

      const userData = await apiClient.auth.getMe();
      setUser({
        id: userData.id,
        email: userData.email,
        fullName: userData.full_name || userData.email.split('@')[0],
        avatarUrl: userData.avatar_url,
        provider: 'google',
        createdAt: userData.created_at,
      });
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : (err instanceof Error ? err.message : 'Google authentication exchange failed.');
      setError(msg);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const requestPasswordReset = async (email: string) => {
    setError(null);
    try {
      const res = await apiClient.auth.requestPasswordReset(email);
      return { message: res.message, resetToken: res.reset_token };
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : (err instanceof Error ? err.message : 'Failed to request password reset.');
      setError(msg);
      throw err;
    }
  };

  const confirmPasswordReset = async (token: string, newPassword: string) => {
    setError(null);
    try {
      await apiClient.auth.confirmPasswordReset(token, newPassword);
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.message : (err instanceof Error ? err.message : 'Failed to reset password.');
      setError(msg);
      throw err;
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user,
        isLoading,
        error,
        login,
        register,
        logout,
        initiateGoogleLogin,
        handleGoogleCallback,
        requestPasswordReset,
        confirmPasswordReset,
        clearError,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
