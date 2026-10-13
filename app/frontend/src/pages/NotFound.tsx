import React from 'react';
import { Link } from 'react-router-dom';
import { FileQuestion, ArrowLeft } from 'lucide-react';

export const NotFound: React.FC = () => {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center text-center px-4" data-testid="not-found-page">
      <div className="w-16 h-16 rounded-2xl bg-slate-100 dark:bg-slate-800 text-muted-foreground flex items-center justify-center mb-6">
        <FileQuestion className="w-8 h-8" aria-hidden="true" />
      </div>
      <h1 className="text-3xl font-bold tracking-tight text-foreground mb-2">404 — Page Not Found</h1>
      <p className="text-muted-foreground max-w-md mb-8">
        The requested page does not exist or may have been moved.
      </p>
      <Link
        to="/"
        data-testid="not-found-return-home-btn"
        className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium text-sm hover:bg-primary-700 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Return to Dashboard</span>
      </Link>
    </div>
  );
};
