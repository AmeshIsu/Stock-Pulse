import React, { useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { ApiError } from '../lib/api';
import { Eye, EyeOff, Lock, Mail, AlertCircle, Info } from 'lucide-react';

export const Login: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const returnUrl = searchParams.get('returnUrl') || '/';

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showForgotModal, setShowForgotModal] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMsg(null);
    setIsLoading(true);

    try {
      await login(email, password);
      navigate(returnUrl, { replace: true });
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        if (err.status === 429) {
          setErrorMsg('Too many failed login attempts. Please wait a few minutes before trying again.');
        } else if (err.code === 'USER_LOCKED') {
          setErrorMsg('This account has been temporarily locked due to multiple failed attempts. Please contact an administrator.');
        } else {
          setErrorMsg(err.message || 'Invalid email or password.');
        }
      } else {
        setErrorMsg('An unexpected error occurred during sign in.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-[75vh] flex items-center justify-center py-10 px-4">
      <div className="w-full max-w-md bg-card border border-border rounded-xl shadow-lg p-6 sm:p-8 space-y-6">
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-primary text-primary-foreground font-black text-xl shadow">
            SP
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">Sign In to Stock Pulse</h1>
          <p className="text-xs text-muted-foreground">
            Access Colombo Stock Exchange upside rankings & analytics
          </p>
        </div>

        {errorMsg && (
          <div
            className="p-3 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-200 text-xs flex items-start gap-2"
            role="alert"
            data-testid="login-error-message"
          >
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>{errorMsg}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-foreground" htmlFor="login-email">
              Email Address
            </label>
            <div className="relative">
              <Mail className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                id="login-email"
                type="email"
                required
                data-testid="login-email-input"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@example.com"
                className="w-full pl-9 pr-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between items-center">
              <label className="text-xs font-semibold text-foreground" htmlFor="login-password">
                Password
              </label>
              <button
                type="button"
                data-testid="forgot-password-link"
                onClick={() => setShowForgotModal(true)}
                className="text-xs text-primary hover:underline"
              >
                Forgot password?
              </button>
            </div>
            <div className="relative">
              <Lock className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                id="login-password"
                type={showPassword ? 'text' : 'password'}
                required
                data-testid="login-password-input"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-9 pr-10 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <button
                type="button"
                data-testid="toggle-password-visibility-btn"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-2.5 text-muted-foreground hover:text-foreground"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            data-testid="login-submit-btn"
            className="w-full py-2.5 px-4 rounded-lg bg-primary text-primary-foreground font-medium text-sm hover:bg-primary-700 transition shadow-sm disabled:opacity-50"
          >
            {isLoading ? 'Signing in...' : 'Sign In'}
          </button>
        </form>

        <div className="text-center pt-2 border-t">
          <p className="text-xs text-muted-foreground">
            Don't have an account?{' '}
            <Link to="/register" className="font-semibold text-primary hover:underline" data-testid="register-link">
              Create an account
            </Link>
          </p>
        </div>
      </div>

      {/* Forgot Password Modal */}
      {showForgotModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          data-testid="forgot-password-modal"
        >
          <div className="bg-card border border-border rounded-xl p-6 max-w-sm w-full space-y-4">
            <div className="flex items-center gap-2 text-primary font-semibold text-sm">
              <Info className="w-5 h-5" />
              <span>Password Reset Assistance</span>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              To maintain system integrity, password resets are processed by system administrators. Please contact your system administrator or send a request to support to receive a secure, one-time temporary password.
            </p>
            <button
              type="button"
              data-testid="close-forgot-modal-btn"
              onClick={() => setShowForgotModal(false)}
              className="w-full py-2 bg-primary text-primary-foreground text-xs font-semibold rounded-lg hover:bg-primary-700"
            >
              Understood
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
