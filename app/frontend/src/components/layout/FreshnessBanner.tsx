import React from 'react';
import { AlertCircle, AlertTriangle } from 'lucide-react';

interface FreshnessBannerProps {
  asOfDate?: string;
}

export const FreshnessBanner: React.FC<FreshnessBannerProps> = ({ asOfDate }) => {
  if (!asOfDate) return null;

  const asOf = new Date(asOfDate);
  // Compare to simulated market reference or current date
  const now = new Date('2026-01-15T00:00:00Z'); // Reference time relative to 2025-12-31 data
  const diffDays = Math.floor((now.getTime() - asOf.getTime()) / (1000 * 60 * 60 * 24));

  if (diffDays <= 7) {
    return null; // Fresh within 7 days
  }

  const isRed = diffDays > 30;

  return (
    <div
      role="alert"
      className={`w-full py-2 px-4 text-xs font-medium flex items-center justify-center gap-2 border-b ${
        isRed
          ? 'bg-rose-50 text-rose-900 border-rose-200 dark:bg-rose-950/60 dark:text-rose-200 dark:border-rose-800'
          : 'bg-amber-50 text-amber-900 border-amber-200 dark:bg-amber-950/60 dark:text-amber-200 dark:border-amber-800'
      }`}
      data-testid="freshness-banner"
    >
      {isRed ? (
        <AlertCircle className="w-4 h-4 shrink-0" aria-hidden="true" />
      ) : (
        <AlertTriangle className="w-4 h-4 shrink-0" aria-hidden="true" />
      )}
      <span>Prices were last updated on {asOfDate}</span>
    </div>
  );
};
