import React from 'react';
import { TrendingUp, Minus, TrendingDown } from 'lucide-react';
import { Tier } from '../../lib/types';

interface TierBadgeProps {
  tier: Tier;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const TierBadge: React.FC<TierBadgeProps> = ({ tier, className = '', size = 'md' }) => {
  let label = 'Neutral';
  let colorClasses = 'bg-slate-100 text-slate-700 border-slate-300 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700';
  let Icon = Minus;

  if (tier === 'High') {
    label = 'High upside';
    colorClasses = 'bg-emerald-50 text-emerald-800 border-emerald-300 dark:bg-emerald-950/40 dark:text-emerald-300 dark:border-emerald-800';
    Icon = TrendingUp;
  } else if (tier === 'Low') {
    label = 'Low upside';
    colorClasses = 'bg-rose-50 text-rose-800 border-rose-300 dark:bg-rose-950/40 dark:text-rose-300 dark:border-rose-800';
    Icon = TrendingDown;
  }

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5 gap-1',
    md: 'text-xs sm:text-sm px-2.5 py-1 gap-1.5',
    lg: 'text-sm px-3 py-1.5 gap-2',
  }[size];

  return (
    <span
      className={`inline-flex items-center font-medium rounded-full border ${colorClasses} ${sizeClasses} ${className}`}
      data-testid={`tier-badge-${tier.toLowerCase()}`}
    >
      <Icon className={size === 'sm' ? 'w-3 h-3' : 'w-4 h-4'} aria-hidden="true" />
      <span>{label}</span>
    </span>
  );
};
