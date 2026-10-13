import React from 'react';
import { ShieldCheck, ShieldAlert } from 'lucide-react';
import { RiskLevel } from '../../lib/types';

interface RiskBadgeProps {
  level: RiskLevel;
  className?: string;
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({ level, className = '' }) => {
  const isElevated = level === 'Elevated';
  
  const colorClasses = isElevated
    ? 'bg-amber-50 text-amber-900 border-amber-300 dark:bg-amber-950/40 dark:text-amber-200 dark:border-amber-800'
    : 'bg-slate-100 text-slate-700 border-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700';

  const Icon = isElevated ? ShieldAlert : ShieldCheck;

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 text-xs font-medium rounded border ${colorClasses} ${className}`}
      data-testid={`risk-badge-${level.toLowerCase()}`}
    >
      <Icon className="w-3.5 h-3.5" aria-hidden="true" />
      <span>{level} risk</span>
    </span>
  );
};
