import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import { apiClient, ApiError } from '../lib/api';
import { User } from '../lib/types';

interface AuthContextType {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  isAdmin: boolean;
  login: (email: string, password: string) => Promise<User>;
  register: (email: string, password: string) => Promise<User>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<User | null>;
  hasAcknowledgedDisclaimer: boolean;
  acknowledgeDisclaimer: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [hasAcknowledgedDisclaimer, setHasAcknowledgedDisclaimer] = useState<boolean>(true);

  const checkDisclaimerAck = useCallback((userId?: string) => {
    if (!userId) {
      setHasAcknowledgedDisclaimer(true);
      return;
    }
    const ack = localStorage.getItem(`stockpulse_disclaimer_ack_${userId}`);
    setHasAcknowledgedDisclaimer(ack === 'true');
  }, []);

  const refreshUser = useCallback(async (): Promise<User | null> => {
    try {
      const data = await apiClient<any>('/auth/me');
      const u: User = data?.user ? data.user : data;
      if (u && (u.id !== undefined || u.email)) {
        setUser(u);
        checkDisclaimerAck(String(u.id));
        return u;
      }
      setUser(null);
      return null;
    } catch {
      setUser(null);
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [checkDisclaimerAck]);

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  const login = async (email: string, password: string): Promise<User> => {
    await apiClient('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    const refreshed = await refreshUser();
    if (!refreshed) {
      throw new ApiError(401, 'LOGIN_FAILED', 'Failed to retrieve session after login');
    }
    return refreshed;
  };

  const register = async (email: string, password: string): Promise<User> => {
    await apiClient('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    // Auto-login after registration
    return await login(email, password);
  };

  const logout = async (): Promise<void> => {
    try {
      await apiClient('/auth/logout', { method: 'POST' });
    } catch {
      // Ignore logout errors
    } finally {
      setUser(null);
    }
  };

  const acknowledgeDisclaimer = () => {
    if (user?.id) {
      localStorage.setItem(`stockpulse_disclaimer_ack_${user.id}`, 'true');
    }
    setHasAcknowledgedDisclaimer(true);
  };

  const isAuthenticated = !!user;
  const isAdmin = user?.role === 'admin';

  return (
    <AuthContext.Provider
      value={{
        user,
        isLoading,
        isAuthenticated,
        isAdmin,
        login,
        register,
        logout,
        refreshUser,
        hasAcknowledgedDisclaimer,
        acknowledgeDisclaimer,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
