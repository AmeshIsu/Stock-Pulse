import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { useTheme } from '../contexts/ThemeContext';
import { apiClient } from '../lib/api';
import {
  User,
  Lock,
  Sun,
  Moon,
  LogOut,
  Shield,
  Check,
  AlertCircle,
  KeyRound,
} from 'lucide-react';

export const Account: React.FC = () => {
  const { user, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();

  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [passwordStatus, setPasswordStatus] = useState<{
    type: 'success' | 'error';
    message: string;
  } | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();
    if (newPassword !== confirmPassword) {
      setPasswordStatus({ type: 'error', message: 'New passwords do not match.' });
      return;
    }
    if (newPassword.length < 8) {
      setPasswordStatus({
        type: 'error',
        message: 'Password must be at least 8 characters with upper, lower, number, and symbol.',
      });
      return;
    }

    setIsSubmitting(true);
    setPasswordStatus(null);

    try {
      await apiClient('/auth/change-password', {
        method: 'POST',
        body: JSON.stringify({
          current_password: currentPassword,
          new_password: newPassword,
        }),
      });
      setPasswordStatus({
        type: 'success',
        message: 'Password changed successfully.',
      });
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
    } catch {
      setPasswordStatus({
        type: 'error',
        message: 'Current password incorrect or new password does not meet security requirements.',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="space-y-6 max-w-2xl mx-auto" data-testid="account-page">
      {/* Header */}
      <div className="pb-2 border-b">
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground flex items-center gap-2">
          <User className="w-6 h-6 text-primary" />
          <span>Account Settings</span>
        </h1>
        <p className="text-xs text-muted-foreground mt-1">
          Manage authentication credentials and visual preferences
        </p>
      </div>

      {/* Account Info Profile */}
      <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4">
        <h2 className="text-sm font-bold text-foreground">User Profile</h2>
        <div className="space-y-2 text-xs">
          <div className="flex justify-between py-1.5 border-b">
            <span className="text-muted-foreground">Email Address</span>
            <span className="font-semibold text-foreground">{user?.email}</span>
          </div>
          <div className="flex justify-between py-1.5 border-b">
            <span className="text-muted-foreground">Access Role</span>
            <span className="font-bold uppercase text-primary tracking-wide">
              {user?.role}
            </span>
          </div>
          <div className="flex justify-between py-1.5 border-b">
            <span className="text-muted-foreground">Account Status</span>
            <span className="text-emerald-800 dark:text-emerald-400 font-semibold">Active</span>
          </div>
        </div>
      </div>

      {/* Theme Preference */}
      <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-3">
        <h2 className="text-sm font-bold text-foreground">Appearance</h2>
        <div className="flex items-center justify-between text-xs">
          <div>
            <span className="font-semibold text-foreground block">Visual Theme</span>
            <span className="text-muted-foreground">Switch between light and dark modes</span>
          </div>
          <button
            type="button"
            data-testid="account-theme-toggle"
            onClick={toggleTheme}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border bg-muted text-foreground font-semibold hover:bg-muted/80"
          >
            {theme === 'light' ? (
              <>
                <Moon className="w-4 h-4" />
                <span>Dark Mode</span>
              </>
            ) : (
              <>
                <Sun className="w-4 h-4" />
                <span>Light Mode</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Change Password */}
      <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex items-center gap-2">
          <KeyRound className="w-4 h-4 text-primary" />
          <h2 className="text-sm font-bold text-foreground">Change Password</h2>
        </div>

        {passwordStatus && (
          <div
            className={`p-3 rounded-lg border text-xs flex items-center gap-2 ${
              passwordStatus.type === 'success'
                ? 'bg-emerald-50 dark:bg-emerald-950/40 border-emerald-200 text-emerald-800 dark:text-emerald-200'
                : 'bg-rose-50 dark:bg-rose-950/40 border-rose-200 text-rose-800 dark:text-rose-200'
            }`}
          >
            {passwordStatus.type === 'success' ? (
              <Check className="w-4 h-4" />
            ) : (
              <AlertCircle className="w-4 h-4" />
            )}
            <span>{passwordStatus.message}</span>
          </div>
        )}

        <form onSubmit={handlePasswordChange} className="space-y-3 text-xs">
          <div className="space-y-1">
            <label className="font-semibold text-foreground" htmlFor="current-password-input">
              Current Password
            </label>
            <input
              id="current-password-input"
              aria-label="Current Password"
              type="password"
              required
              data-testid="current-password-input"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
              className="w-full p-2 border rounded-lg bg-background"
            />
          </div>

          <div className="space-y-1">
            <label className="font-semibold text-foreground" htmlFor="new-password-input">
              New Password
            </label>
            <input
              id="new-password-input"
              aria-label="New Password"
              type="password"
              required
              data-testid="new-password-input"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              className="w-full p-2 border rounded-lg bg-background"
            />
          </div>

          <div className="space-y-1">
            <label className="font-semibold text-foreground" htmlFor="confirm-new-password-input">
              Confirm New Password
            </label>
            <input
              id="confirm-new-password-input"
              aria-label="Confirm New Password"
              type="password"
              required
              data-testid="confirm-new-password-input"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="w-full p-2 border rounded-lg bg-background"
            />
          </div>

          <button
            type="submit"
            disabled={isSubmitting}
            data-testid="change-password-submit-btn"
            className="w-full py-2 bg-primary text-primary-foreground font-semibold rounded-lg hover:bg-primary-700 disabled:opacity-50"
          >
            {isSubmitting ? 'Updating Password...' : 'Update Password'}
          </button>
        </form>
      </div>

      {/* Sign Out */}
      <div className="bg-card border border-border rounded-xl p-6 shadow-sm flex items-center justify-between">
        <div>
          <span className="font-bold text-sm text-foreground block">Session</span>
          <span className="text-xs text-muted-foreground">Sign out of current active session</span>
        </div>
        <button
          type="button"
          data-testid="account-signout-btn"
          onClick={() => {
            logout();
            navigate('/login');
          }}
          className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-rose-50 text-rose-700 border border-rose-200 dark:bg-rose-950/40 dark:text-rose-200 text-xs font-semibold hover:bg-rose-100 transition"
        >
          <LogOut className="w-4 h-4" />
          <span>Sign Out</span>
        </button>
      </div>
    </div>
  );
};
