import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiClient } from '../../lib/api';
import { BookOpen, Printer } from 'lucide-react';

export const AdminMethodology: React.FC = () => {
  const [content, setContent] = useState<string>('');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    apiClient<{ markdown_content: string }>('/admin/methodology')
      .then((res) => setContent(res.markdown_content))
      .catch((err) => console.error('Failed to load admin methodology:', err))
      .finally(() => setIsLoading(false));
  }, []);

  return (
    <div className="space-y-6 max-w-5xl mx-auto" data-testid="admin-methodology-page">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <BookOpen className="w-6 h-6 text-primary" />
            <span>Technical Methodology Specification</span>
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Statistical documentation, fold correlations, Holm-Bonferroni correction tables, and backtest results
          </p>
        </div>

        <button
          type="button"
          onClick={() => window.print()}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-semibold hover:bg-muted"
        >
          <Printer className="w-4 h-4 text-primary" />
          <span>Print Spec</span>
        </button>
      </div>

      <div className="bg-card border border-border rounded-xl p-6 sm:p-8 shadow-sm prose dark:prose-invert max-w-none text-xs sm:text-sm leading-relaxed">
        {isLoading ? (
          <div className="space-y-4 animate-pulse">
            <div className="h-6 bg-muted rounded w-1/3" />
            <div className="h-20 bg-muted rounded" />
            <div className="h-40 bg-muted rounded" />
          </div>
        ) : (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {content || '# Technical Methodology\n\nNo content loaded.'}
          </ReactMarkdown>
        )}
      </div>
    </div>
  );
};
