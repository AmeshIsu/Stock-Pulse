import React, { useState, useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { ApiError } from '../lib/api';
import { Eye, EyeOff, Lock, Mail, Check, X, AlertCircle } from 'lucide-react';

export const Register: React.FC = () => {
  const { register } = useAuth();
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  // Live password policy checks
  const policy = useMemo(() => {
    return {
      minLength: password.length >= 8,
      hasUpper: /[A-Z]/.test(password),
      hasLower: /[a-z]/.test(password),
      hasDigit: /[0-9]/.test(password),
      hasSpecial: /[^A-Za-z0-9]/.test(password),
      matchesConfirm: password.length > 0 && password === confirmPassword,
    };
  }, [password, confirmPassword]);

  const isFormValid =
    policy.minLength &&
    policy.hasUpper &&
    policy.hasLower &&
    policy.hasDigit &&
    policy.hasSpecial &&
    policy.matchesConfirm &&
    email.includes('@');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isFormValid) return;

    setErrorMsg(null);
    setIsLoading(true);

    try {
      await register(email, password);
      navigate('/', { replace: true });
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        if (err.status === 429) {
          setErrorMsg('Registration rate limit exceeded. Please try again shortly.');
        } else {
          setErrorMsg(err.message || 'Registration failed. Email may already be in use.');
        }
      } else {
        setErrorMsg('An unexpected error occurred during account creation.');
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
          <h1 className="text-2xl font-bold tracking-tight text-foreground">Create Your Account</h1>
          <p className="text-xs text-muted-foreground">
            Join Stock Pulse for objective CSE equity upside rankings
          </p>
        </div>

        {errorMsg && (
          <div
            className="p-3 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-200 text-xs flex items-start gap-2"
            role="alert"
            data-testid="register-error-message"
          >
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>{errorMsg}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-foreground" htmlFor="reg-email">
              Email Address
            </label>
            <div className="relative">
              <Mail className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                id="reg-email"
                type="email"
                required
                data-testid="register-email-input"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@example.com"
                className="w-full pl-9 pr-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-foreground" htmlFor="reg-password">
              Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                id="reg-password"
                type={showPassword ? 'text' : 'password'}
                required
                data-testid="register-password-input"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Create strong password"
                className="w-full pl-9 pr-10 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
              />
              <button
                type="button"
                data-testid="toggle-register-password-btn"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-2.5 text-muted-foreground hover:text-foreground"
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-foreground" htmlFor="reg-confirm-password">
              Confirm Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                id="reg-confirm-password"
                type={showPassword ? 'text' : 'password'}
                required
                data-testid="register-confirm-password-input"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Repeat password"
                className="w-full pl-9 pr-3 py-2 text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
              />
            </div>
          </div>

          {/* Live Password Policy List */}
          <div className="p-3 bg-muted/40 rounded-lg space-y-1.5 border" data-testid="password-policy-list">
            <p className="text-[11px] font-semibold text-foreground">Password Requirements:</p>
            <ul className="text-[11px] space-y-1">
              <li className={`flex items-center gap-1.5 ${policy.minLength ? 'text-emerald-800 dark:text-emerald-300 font-medium' : 'text-muted-foreground'}`}>
                {policy.minLength ? <Check className="w-3.5 h-3.5" /> : <X className="w-3.5 h-3.5" />}
                <span>At least 8 characters</span>
              </li>
              <li className={`flex items-center gap-1.5 ${policy.hasUpper ? 'text-emerald-800 dark:text-emerald-300 font-medium' : 'text-muted-foreground'}`}>
                {policy.hasUpper ? <Check className="w-3.5 h-3.5" /> : <X className="w-3.5 h-3.5" />}
                <span>Uppercase letter (A–Z)</span>
              </li>
              <li className={`flex items-center gap-1.5 ${policy.hasLower ? 'text-emerald-800 dark:text-emerald-300 font-medium' : 'text-muted-foreground'}`}>
                {policy.hasLower ? <Check className="w-3.5 h-3.5" /> : <X className="w-3.5 h-3.5" />}
                <span>Lowercase letter (a–z)</span>
              </li>
              <li className={`flex items-center gap-1.5 ${policy.hasDigit ? 'text-emerald-800 dark:text-emerald-300 font-medium' : 'text-muted-foreground'}`}>
                {policy.hasDigit ? <Check className="w-3.5 h-3.5" /> : <X className="w-3.5 h-3.5" />}
                <span>Number (0–9)</span>
              </li>
              <li className={`flex items-center gap-1.5 ${policy.hasSpecial ? 'text-emerald-800 dark:text-emerald-300 font-medium' : 'text-muted-foreground'}`}>
                {policy.hasSpecial ? <Check className="w-3.5 h-3.5" /> : <X className="w-3.5 h-3.5" />}
                <span>Special character (!@#$%^&*)</span>
              </li>
              <li className={`flex items-center gap-1.5 ${policy.matchesConfirm ? 'text-emerald-800 dark:text-emerald-300 font-medium' : 'text-muted-foreground'}`}>
                {policy.matchesConfirm ? <Check className="w-3.5 h-3.5" /> : <X className="w-3.5 h-3.5" />}
                <span>Passwords match</span>
              </li>
            </ul>
          </div>

          <button
            type="submit"
            disabled={!isFormValid || isLoading}
            data-testid="register-submit-btn"
            className="w-full py-2.5 px-4 rounded-lg bg-primary text-primary-foreground font-medium text-sm hover:bg-primary-700 transition shadow-sm disabled:opacity-50"
          >
            {isLoading ? 'Creating account...' : 'Create Account'}
          </button>
        </form>

        <div className="text-center pt-2 border-t">
          <p className="text-xs text-muted-foreground">
            Already have an account?{' '}
            <Link to="/login" className="font-semibold text-primary hover:underline" data-testid="login-link">
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};
