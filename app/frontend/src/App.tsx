import React, { Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from './contexts/ThemeContext';
import { AuthProvider } from './contexts/AuthContext';

// Layout & Guards
import { AppLayout } from './components/layout/AppLayout';
import { RequireAuth } from './components/guards/RequireAuth';
import { RequireAdmin } from './components/guards/RequireAdmin';

// User Pages (Eager or Lazy)
import { Dashboard } from './pages/Dashboard';
import { Rankings } from './pages/Rankings';
import { StockDetail } from './pages/StockDetail';
import { Watchlist } from './pages/Watchlist';
import { Alerts } from './pages/Alerts';
import { Notifications } from './pages/Notifications';
import { HowItWorks } from './pages/HowItWorks';
import { Account } from './pages/Account';
import { Login } from './pages/Login';
import { Register } from './pages/Register';
import { Forbidden } from './pages/Forbidden';
import { NotFound } from './pages/NotFound';

// Code-Split Admin Pages (Prefixed admin-*)
const AdminLayout = React.lazy(() =>
  import('./pages/admin/AdminLayout').then((m) => ({ default: m.AdminLayout }))
);
const AdminOverview = React.lazy(() =>
  import('./pages/admin/AdminOverview').then((m) => ({ default: m.AdminOverview }))
);
const AdminModels = React.lazy(() =>
  import('./pages/admin/AdminModels').then((m) => ({ default: m.AdminModels }))
);
const AdminUsers = React.lazy(() =>
  import('./pages/admin/AdminUsers').then((m) => ({ default: m.AdminUsers }))
);
const AdminData = React.lazy(() =>
  import('./pages/admin/AdminData').then((m) => ({ default: m.AdminData }))
);
const AdminAudit = React.lazy(() =>
  import('./pages/admin/AdminAudit').then((m) => ({ default: m.AdminAudit }))
);
const AdminMethodology = React.lazy(() =>
  import('./pages/admin/AdminMethodology').then((m) => ({ default: m.AdminMethodology }))
);

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000,
      retry: 1,
    },
  },
});

export const App: React.FC = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <AuthProvider>
          <BrowserRouter>
            <Suspense
              fallback={
                <div className="flex h-screen items-center justify-center">
                  <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
                </div>
              }
            >
              <Routes>
                {/* Public Auth Routes */}
                <Route path="/login" element={<Login />} />
                <Route path="/register" element={<Register />} />

                {/* Authenticated User Routes */}
                <Route
                  path="/"
                  element={
                    <RequireAuth>
                      <AppLayout />
                    </RequireAuth>
                  }
                >
                  <Route index element={<Dashboard />} />
                  <Route path="rankings" element={<Rankings />} />
                  <Route path="stocks/:code" element={<StockDetail />} />
                  <Route path="watchlist" element={<Watchlist />} />
                  <Route path="alerts" element={<Alerts />} />
                  <Route path="notifications" element={<Notifications />} />
                  <Route path="how-it-works" element={<HowItWorks />} />
                  <Route path="account" element={<Account />} />
                </Route>

                {/* Admin Console (Strictly Admin Only & Code-Split) */}
                <Route
                  path="/admin"
                  element={
                    <RequireAdmin>
                      <AdminLayout />
                    </RequireAdmin>
                  }
                >
                  <Route index element={<AdminOverview />} />
                  <Route path="models" element={<AdminModels />} />
                  <Route path="users" element={<AdminUsers />} />
                  <Route path="data" element={<AdminData />} />
                  <Route path="audit" element={<AdminAudit />} />
                  <Route path="methodology" element={<AdminMethodology />} />
                </Route>

                {/* Error Pages */}
                <Route path="/403" element={<Forbidden />} />
                <Route path="/404" element={<NotFound />} />
                <Route path="*" element={<Navigate to="/404" replace />} />
              </Routes>
            </Suspense>
          </BrowserRouter>
        </AuthProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

export default App;
