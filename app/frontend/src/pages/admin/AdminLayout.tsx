import React from 'react';
import { Link, Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import {
  LayoutDashboard,
  Cpu,
  Users,
  Database,
  History,
  BookOpen,
  ArrowLeft,
  Shield,
  LogOut,
} from 'lucide-react';

export const AdminLayout: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const navItems = [
    { label: 'Overview', path: '/admin', exact: true, icon: LayoutDashboard },
    { label: 'Model Governance', path: '/admin/models', icon: Cpu },
    { label: 'User Management', path: '/admin/users', icon: Users },
    { label: 'Data Management', path: '/admin/data', icon: Database },
    { label: 'Audit Log', path: '/admin/audit', icon: History },
    { label: 'Technical Methodology', path: '/admin/methodology', icon: BookOpen },
  ];

  const isActive = (item: (typeof navItems)[0]) => {
    if (item.exact) return location.pathname === item.path;
    return location.pathname.startsWith(item.path);
  };

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-background text-foreground" data-testid="admin-layout">
      {/* Admin Sidebar */}
      <aside className="w-full md:w-64 border-r bg-card/60 backdrop-blur p-4 flex flex-col justify-between shrink-0">
        <div className="space-y-6">
          {/* Admin Header */}
          <div className="flex items-center justify-between pb-4 border-b">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-primary text-primary-foreground flex items-center justify-center font-bold">
                <Shield className="w-4 h-4" />
              </div>
              <div>
                <span className="font-extrabold text-sm block tracking-tight">Admin Console</span>
                <span className="text-[10px] text-muted-foreground uppercase tracking-widest font-semibold">Governance</span>
              </div>
            </div>
            <Link
              to="/"
              data-testid="admin-back-to-app-btn"
              className="text-xs text-primary hover:underline flex items-center gap-1 font-medium"
              title="Return to user application"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>App</span>
            </Link>
          </div>

          {/* Navigation Items */}
          <nav className="space-y-1" aria-label="Admin Navigation">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = isActive(item);
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  data-testid={`admin-nav-${item.label.toLowerCase().replace(/\s+/g, '-')}`}
                  className={`flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-medium transition ${
                    active
                      ? 'bg-primary text-primary-foreground shadow-xs font-semibold'
                      : 'text-muted-foreground hover:text-foreground hover:bg-muted'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Footer User Info */}
        <div className="pt-4 border-t space-y-3">
          <div className="flex items-center justify-between text-xs">
            <div className="truncate max-w-[140px]">
              <span className="font-bold block text-foreground truncate">{user?.email}</span>
              <span className="text-[10px] text-emerald-800 dark:text-emerald-400 font-semibold uppercase">Administrator</span>
            </div>
            <button
              type="button"
              data-testid="admin-signout-btn"
              onClick={() => {
                logout();
                navigate('/login');
              }}
              className="p-1.5 text-muted-foreground hover:text-rose-600"
              title="Sign out"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* Main Admin Content */}
      <main className="flex-1 p-6 md:p-8 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
};
