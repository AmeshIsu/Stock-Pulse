import React from 'react';
import { HelpCircle, AlertTriangle } from 'lucide-react';
import { ConfidenceLevel } from '../../lib/types';

interface ConfidencePillProps {
  level: ConfidenceLevel;
  className?: string;
}

export const ConfidencePill: React.FC<ConfidencePillProps> = ({ level = 'Low confidence', className = '' }) => {
  const safeLevel = level || 'Low confidence';
  // Confidence labels MUST use calm neutral or amber pill, NEVER green.
  const isAmber = safeLevel.includes('Low');
  
  const colorClasses = isAmber
    ? 'bg-amber-50 text-amber-900 border-amber-300 dark:bg-amber-950/40 dark:text-amber-200 dark:border-amber-800'
    : 'bg-slate-100 text-slate-800 border-slate-300 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-700';

  const Icon = isAmber ? AlertTriangle : HelpCircle;

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full border ${colorClasses} ${className}`}
      data-testid="confidence-pill"
    >
      <Icon className="w-3.5 h-3.5" aria-hidden="true" />
      <span>{safeLevel}</span>
    </span>
  );
};
