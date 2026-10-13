import React, { useState, useEffect } from 'react';
import { apiClient, ApiError } from '../../lib/api';
import { User } from '../../lib/types';
import {
  Users,
  UserPlus,
  Search,
  KeyRound,
  Shield,
  Check,
  Copy,
  AlertTriangle,
  UserCheck,
  UserX,
} from 'lucide-react';

export const AdminUsers: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [search, setSearch] = useState('');
  const [isLoading, setIsLoading] = useState(true);

  // Create User Modal
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newEmail, setNewEmail] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [newRole, setNewRole] = useState<'user' | 'admin'>('user');

  // Reset Password Modal
  const [tempPassword, setTempPassword] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Errors / Messages
  const [actionError, setActionError] = useState<string | null>(null);

  const loadUsers = async () => {
    setIsLoading(true);
    try {
      const res = await apiClient<{ users: User[] }>('/admin/users');
      setUsers(res.users || []);
    } catch (err) {
      console.error('Failed to load users:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadUsers();
  }, []);

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault();
    setActionError(null);
    try {
      await apiClient('/admin/users', {
        method: 'POST',
        body: JSON.stringify({
          email: newEmail,
          password: newPassword,
          role: newRole,
        }),
      });
      setShowCreateModal(false);
      setNewEmail('');
      setNewPassword('');
      loadUsers();
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setActionError(err.message);
      } else {
        setActionError('Failed to create user account.');
      }
    }
  };

  const handleToggleStatus = async (targetUser: User) => {
    setActionError(null);
    try {
      await apiClient(`/admin/users/${targetUser.id}/status`, {
        method: 'PUT',
        body: JSON.stringify({ is_active: !targetUser.is_active }),
      });
      loadUsers();
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setActionError(err.message);
      } else {
        setActionError('Failed to update status (last admin guard active).');
      }
    }
  };

  const handleChangeRole = async (targetUser: User, nextRole: 'user' | 'admin') => {
    setActionError(null);
    try {
      await apiClient(`/admin/users/${targetUser.id}/role`, {
        method: 'PUT',
        body: JSON.stringify({ role: nextRole }),
      });
      loadUsers();
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setActionError(err.message);
      } else {
        setActionError('Failed to change role (cannot remove last administrator).');
      }
    }
  };

  const handleResetPassword = async (targetUser: User) => {
    setActionError(null);
    try {
      const res = await apiClient<{ temporary_password: string }>(
        `/admin/users/${targetUser.id}/reset-password`,
        { method: 'POST' }
      );
      setTempPassword(res.temporary_password);
      setCopied(false);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setActionError(err.message);
      } else {
        setActionError('Failed to generate reset password.');
      }
    }
  };

  const filteredUsers = users.filter((u) =>
    u.email.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6" data-testid="admin-users-page">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <Users className="w-6 h-6 text-primary" />
            <span>User Management &amp; Roles</span>
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            RBAC administration, account lockouts, credential resets, and last-admin safety guardrails
          </p>
        </div>

        <button
          type="button"
          data-testid="open-create-user-modal-btn"
          onClick={() => {
            setActionError(null);
            setShowCreateModal(true);
          }}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-primary text-primary-foreground text-xs font-semibold rounded-lg hover:bg-primary-700 shadow-sm transition"
        >
          <UserPlus className="w-4 h-4" />
          <span>Create New User</span>
        </button>
      </div>

      {actionError && (
        <div
          className="p-3 bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-800 text-rose-800 dark:text-rose-200 text-xs rounded-xl flex items-center gap-2"
          role="alert"
          data-testid="admin-user-action-error"
        >
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{actionError}</span>
        </div>
      )}

      {/* Search Toolbar */}
      <div className="relative max-w-sm">
        <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
        <input
          type="text"
          data-testid="admin-user-search-input"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search by email..."
          className="w-full pl-9 pr-3 py-2 text-xs rounded-lg border bg-card"
        />
      </div>

      {/* Users Table */}
      <div className="bg-card border border-border rounded-xl shadow-sm overflow-hidden" data-testid="admin-users-table-container">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="bg-muted/40 border-b text-muted-foreground font-semibold uppercase tracking-wider text-[11px]">
              <th className="py-3 px-4">User Email</th>
              <th className="py-3 px-4">Role</th>
              <th className="py-3 px-4">Status</th>
              <th className="py-3 px-4">Created Date</th>
              <th className="py-3 px-4">Last Login</th>
              <th className="py-3 px-4 text-center">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {filteredUsers.map((u) => (
              <tr key={u.id} data-testid={`admin-user-row-${u.id}`} className="hover:bg-muted/30">
                <td className="py-3 px-4 font-semibold text-foreground">{u.email}</td>
                <td className="py-3 px-4">
                  <select
                    value={u.role}
                    data-testid={`select-role-${u.id}`}
                    onChange={(e) => handleChangeRole(u, e.target.value as 'user' | 'admin')}
                    className="p-1 rounded border text-xs bg-background font-medium"
                  >
                    <option value="user">User</option>
                    <option value="admin">Admin</option>
                  </select>
                </td>
                <td className="py-3 px-4">
                  <span
                    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-bold ${
                      u.is_active
                        ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300'
                        : 'bg-rose-100 text-rose-800 dark:bg-rose-950/40 dark:text-rose-300'
                    }`}
                  >
                    {u.is_active ? 'Active' : 'Disabled'}
                  </span>
                </td>
                <td className="py-3 px-4 text-muted-foreground tabular-nums">
                  {new Date(u.created_at).toLocaleDateString()}
                </td>
                <td className="py-3 px-4 text-muted-foreground tabular-nums">
                  {u.last_login_at ? new Date(u.last_login_at).toLocaleString() : 'Never'}
                </td>
                <td className="py-3 px-4 text-center">
                  <div className="flex items-center justify-center gap-2">
                    <button
                      type="button"
                      data-testid={`toggle-status-btn-${u.id}`}
                      onClick={() => handleToggleStatus(u)}
                      className="p-1.5 rounded hover:bg-muted text-muted-foreground hover:text-foreground"
                      title={u.is_active ? 'Disable user' : 'Enable user'}
                    >
                      {u.is_active ? <UserX className="w-4 h-4 text-rose-600" /> : <UserCheck className="w-4 h-4 text-emerald-600" />}
                    </button>
                    <button
                      type="button"
                      data-testid={`reset-pwd-btn-${u.id}`}
                      onClick={() => handleResetPassword(u)}
                      className="p-1.5 rounded hover:bg-muted text-muted-foreground hover:text-primary"
                      title="Generate one-time reset password"
                    >
                      <KeyRound className="w-4 h-4" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Temporary Password Modal */}
      {tempPassword && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          data-testid="temp-password-modal"
        >
          <div className="bg-card border border-border rounded-xl p-6 max-w-md w-full space-y-4 animate-in zoom-in-95">
            <div className="flex items-center gap-2 text-primary font-bold text-sm">
              <KeyRound className="w-5 h-5" />
              <span>One-Time Temporary Password</span>
            </div>

            <p className="text-xs text-muted-foreground">
              Provide this one-time password to the user. This credential will only be shown once and expires upon initial sign-in.
            </p>

            <div className="flex items-center justify-between p-3 bg-muted rounded-lg font-mono text-sm font-bold text-foreground">
              <span data-testid="temp-password-display">{tempPassword}</span>
              <button
                type="button"
                data-testid="copy-temp-password-btn"
                onClick={() => {
                  navigator.clipboard.writeText(tempPassword);
                  setCopied(true);
                  setTimeout(() => setCopied(false), 2000);
                }}
                className="p-1.5 rounded hover:bg-card text-muted-foreground hover:text-foreground flex items-center gap-1 text-xs"
              >
                {copied ? <Check className="w-4 h-4 text-emerald-600" /> : <Copy className="w-4 h-4" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>

            <button
              type="button"
              data-testid="close-temp-pwd-modal-btn"
              onClick={() => setTempPassword(null)}
              className="w-full py-2 bg-primary text-primary-foreground text-xs font-semibold rounded-lg hover:bg-primary-700"
            >
              Done &amp; Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Create User Modal */}
      {showCreateModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          data-testid="create-user-modal"
        >
          <div className="bg-card border border-border rounded-xl p-6 max-w-md w-full space-y-4 animate-in zoom-in-95">
            <div className="flex items-center justify-between pb-2 border-b">
              <h3 className="font-bold text-foreground text-sm">Create User Account</h3>
              <button
                onClick={() => setShowCreateModal(false)}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                Cancel
              </button>
            </div>

            <form onSubmit={handleCreateUser} className="space-y-4 text-xs">
              <div className="space-y-1">
                <label className="font-semibold text-foreground">Email Address</label>
                <input
                  type="email"
                  required
                  data-testid="new-user-email-input"
                  value={newEmail}
                  onChange={(e) => setNewEmail(e.target.value)}
                  placeholder="analyst@example.com"
                  className="w-full p-2 border rounded-lg bg-background"
                />
              </div>

              <div className="space-y-1">
                <label className="font-semibold text-foreground">Initial Password</label>
                <input
                  type="password"
                  required
                  data-testid="new-user-password-input"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  placeholder="At least 8 chars..."
                  className="w-full p-2 border rounded-lg bg-background"
                />
              </div>

              <div className="space-y-1">
                <label className="font-semibold text-foreground">Role</label>
                <select
                  value={newRole}
                  data-testid="new-user-role-select"
                  onChange={(e) => setNewRole(e.target.value as 'user' | 'admin')}
                  className="w-full p-2 border rounded-lg bg-background"
                >
                  <option value="user">Standard User</option>
                  <option value="admin">System Administrator</option>
                </select>
              </div>

              <button
                type="submit"
                data-testid="submit-create-user-btn"
                className="w-full py-2.5 bg-primary text-primary-foreground font-semibold rounded-lg hover:bg-primary-700"
              >
                Create Account
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
