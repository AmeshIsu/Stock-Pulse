import React, { useState, useEffect, useRef } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { useTheme } from '../../contexts/ThemeContext';
import { apiClient } from '../../lib/api';
import { NotificationItem } from '../../lib/types';
import {
  TrendingUp,
  BarChart2,
  Bookmark,
  Bell,
  HelpCircle,
  Shield,
  User as UserIcon,
  Sun,
  Moon,
  Menu,
  X,
  LogOut,
  Check,
} from 'lucide-react';

export const Navbar: React.FC = () => {
  const { user, isAuthenticated, isAdmin, logout } = useAuth();
  const { theme, toggleTheme } = useTheme();
  const location = useLocation();
  const navigate = useNavigate();

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [notifications, setNotifications] = useState<NotificationItem[]>([]);
  const notifRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isAuthenticated) {
      apiClient<NotificationItem[]>('/notifications')
        .then((data) => setNotifications(data || []))
        .catch(() => setNotifications([]));
    }
  }, [isAuthenticated, location.pathname]);

  // Close notification popover on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (notifRef.current && !notifRef.current.contains(event.target as Node)) {
        setNotificationsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const unreadCount = notifications.filter((n) => !n.is_read).length;

  const handleMarkRead = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await apiClient(`/notifications/${id}/read`, { method: 'POST' });
      setNotifications((prev) =>
        prev.map((n) => (n.id === id ? { ...n, is_read: true } : n))
      );
    } catch {
      // ignore
    }
  };

  const navLinks = [
    { label: 'Dashboard', path: '/', icon: TrendingUp },
    { label: 'Rankings', path: '/rankings', icon: BarChart2 },
    { label: 'Watchlist', path: '/watchlist', icon: Bookmark },
    { label: 'Alerts', path: '/alerts', icon: Bell },
    { label: 'How It Works', path: '/how-it-works', icon: HelpCircle },
  ];

  if (isAdmin) {
    navLinks.push({ label: 'Admin Console', path: '/admin', icon: Shield });
  }

  const isActive = (path: string) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  return (
    <header className="sticky top-0 z-40 w-full border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/75">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand Logo */}
        <div className="flex items-center gap-8">
          <Link
            to="/"
            className="flex items-center gap-2.5 font-bold text-lg text-primary tracking-tight hover:opacity-90"
            data-testid="navbar-brand-logo"
          >
            <div className="w-8 h-8 rounded-lg bg-primary text-primary-foreground flex items-center justify-center font-extrabold shadow-sm">
              SP
            </div>
            <span className="text-foreground">Stock<span className="text-primary font-light">Pulse</span></span>
          </Link>

          {/* Desktop Navigation */}
          {isAuthenticated && (
            <nav className="hidden md:flex items-center gap-1" aria-label="Main Navigation">
              {navLinks.map((link) => {
                const Icon = link.icon;
                const active = isActive(link.path);
                return (
                  <Link
                    key={link.path}
                    to={link.path}
                    data-testid={`nav-link-${link.label.toLowerCase().replace(/\s+/g, '-')}`}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                      active
                        ? 'bg-primary/10 text-primary font-semibold'
                        : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
                    }`}
                  >
                    <Icon className="w-4 h-4" aria-hidden="true" />
                    <span>{link.label}</span>
                  </Link>
                );
              })}
            </nav>
          )}
        </div>

        {/* Right Actions */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Theme Toggle */}
          <button
            type="button"
            onClick={toggleTheme}
            data-testid="theme-toggle-btn"
            aria-label={`Switch to ${theme === 'light' ? 'dark' : 'light'} mode`}
            className="p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
          >
            {theme === 'light' ? (
              <Moon className="w-4 h-4" aria-hidden="true" />
            ) : (
              <Sun className="w-4 h-4" aria-hidden="true" />
            )}
          </button>

          {isAuthenticated ? (
            <>
              {/* Notification Bell */}
              <div className="relative" ref={notifRef}>
                <button
                  type="button"
                  data-testid="navbar-notification-bell"
                  onClick={() => setNotificationsOpen(!notificationsOpen)}
                  aria-label="View notifications"
                  className="relative p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
                >
                  <Bell className="w-4 h-4" aria-hidden="true" />
                  {unreadCount > 0 && (
                    <span
                      data-testid="notification-unread-count"
                      className="absolute top-1 right-1 flex h-4 w-4 items-center justify-center rounded-full bg-rose-500 text-[10px] font-bold text-white"
                    >
                      {unreadCount}
                    </span>
                  )}
                </button>

                {/* Notifications Dropdown */}
                {notificationsOpen && (
                  <div
                    className="absolute right-0 mt-2 w-80 sm:w-96 rounded-xl border bg-card p-3 shadow-xl z-50 animate-in fade-in zoom-in-95"
                    data-testid="notifications-dropdown"
                  >
                    <div className="flex items-center justify-between pb-2 mb-2 border-b">
                      <span className="font-semibold text-sm text-foreground">Notifications</span>
                      <Link
                        to="/notifications"
                        onClick={() => setNotificationsOpen(false)}
                        className="text-xs text-primary hover:underline"
                        data-testid="see-all-notifications-link"
                      >
                        View all
                      </Link>
                    </div>
                    <div className="max-h-72 overflow-y-auto space-y-2">
                      {notifications.length === 0 ? (
                        <p className="text-xs text-muted-foreground text-center py-4">No notifications yet</p>
                      ) : (
                        notifications.slice(0, 5).map((notif) => (
                          <div
                            key={notif.id}
                            onClick={() => {
                              setNotificationsOpen(false);
                              navigate(`/stocks/${notif.stock_code}`);
                            }}
                            className={`p-2.5 rounded-lg text-xs cursor-pointer transition-colors ${
                              notif.is_read ? 'bg-muted/30 text-muted-foreground' : 'bg-primary/5 text-foreground font-medium'
                            } hover:bg-muted`}
                          >
                            <div className="flex justify-between items-start gap-2">
                              <span className="font-semibold text-foreground">{notif.title}</span>
                              {!notif.is_read && (
                                <button
                                  type="button"
                                  onClick={(e) => handleMarkRead(notif.id, e)}
                                  aria-label="Mark notification as read"
                                  className="text-muted-foreground hover:text-primary p-0.5"
                                >
                                  <Check className="w-3 h-3" />
                                </button>
                              )}
                            </div>
                            <p className="text-[11px] text-muted-foreground mt-0.5 line-clamp-2">{notif.message}</p>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Account Link */}
              <Link
                to="/account"
                data-testid="nav-account-link"
                className="flex items-center gap-2 p-1.5 rounded-lg text-sm text-muted-foreground hover:text-foreground hover:bg-muted"
                aria-label="User Account"
              >
                <div className="w-7 h-7 rounded-full bg-primary/20 text-primary flex items-center justify-center font-bold text-xs">
                  {user?.email ? user.email.charAt(0).toUpperCase() : <UserIcon className="w-3.5 h-3.5" />}
                </div>
                <span className="hidden sm:inline-block text-xs font-medium max-w-[120px] truncate text-foreground">
                  {user?.email}
                </span>
              </Link>

              {/* Logout Button */}
              <button
                type="button"
                data-testid="nav-logout-btn"
                onClick={() => {
                  logout();
                  navigate('/login');
                }}
                aria-label="Sign out"
                className="p-2 rounded-lg text-muted-foreground hover:text-rose-600 hover:bg-rose-50 dark:hover:bg-rose-950/40 transition-colors"
              >
                <LogOut className="w-4 h-4" aria-hidden="true" />
              </button>
            </>
          ) : (
            <div className="flex items-center gap-2">
              <Link
                to="/login"
                data-testid="nav-login-btn"
                className="px-3 py-1.5 text-xs sm:text-sm font-medium text-foreground hover:text-primary"
              >
                Log in
              </Link>
              <Link
                to="/register"
                data-testid="nav-register-btn"
                className="px-3 py-1.5 text-xs sm:text-sm font-medium bg-primary text-primary-foreground rounded-lg hover:bg-primary-700 shadow-sm"
              >
                Register
              </Link>
            </div>
          )}

          {/* Mobile Menu Toggle */}
          {isAuthenticated && (
            <button
              type="button"
              data-testid="mobile-menu-toggle"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 rounded-lg text-muted-foreground hover:text-foreground"
              aria-label="Toggle navigation menu"
            >
              {mobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
          )}
        </div>
      </div>

      {/* Mobile Navigation Drawer */}
      {mobileMenuOpen && isAuthenticated && (
        <div className="md:hidden border-t px-4 pt-2 pb-4 space-y-1 bg-card">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const active = isActive(link.path);
            return (
              <Link
                key={link.path}
                to={link.path}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium ${
                  active
                    ? 'bg-primary/10 text-primary font-semibold'
                    : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                }`}
              >
                <Icon className="w-4 h-4" aria-hidden="true" />
                <span>{link.label}</span>
              </Link>
            );
          })}
        </div>
      )}
    </header>
  );
};
