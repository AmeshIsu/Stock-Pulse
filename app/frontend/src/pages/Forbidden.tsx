import React from 'react';
import { Link } from 'react-router-dom';
import { ShieldAlert, ArrowLeft } from 'lucide-react';

export const Forbidden: React.FC = () => {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center text-center px-4" data-testid="forbidden-page">
      <div className="w-16 h-16 rounded-2xl bg-rose-50 dark:bg-rose-950/40 text-rose-600 dark:text-rose-400 flex items-center justify-center mb-6">
        <ShieldAlert className="w-8 h-8" aria-hidden="true" />
      </div>
      <h1 className="text-3xl font-bold tracking-tight text-foreground mb-2">403 — Access Denied</h1>
      <p className="text-muted-foreground max-w-md mb-8">
        You do not have administrative privileges to access this console. This area is reserved for designated system administrators.
      </p>
      <Link
        to="/"
        data-testid="forbidden-return-home-btn"
        className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium text-sm hover:bg-primary-700 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Return to Dashboard</span>
      </Link>
    </div>
  );
};
