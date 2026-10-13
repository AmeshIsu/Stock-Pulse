import React from 'react';
import { Tier } from '../../lib/types';

interface ScoreGaugeProps {
  score: number; // 0 - 100
  tier?: Tier;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}

export const ScoreGauge: React.FC<ScoreGaugeProps> = ({
  score,
  tier = 'Neutral',
  size = 'md',
  showLabel = true,
}) => {
  const clampedScore = Math.max(0, Math.min(100, Math.round(score)));

  let color = '#64748b'; // Neutral
  if (tier === 'High') color = '#10b981'; // Green
  else if (tier === 'Low') color = '#ef4444'; // Red

  if (size === 'sm') {
    return (
      <div className="flex items-center gap-2" data-testid="score-gauge-sm">
        <div className="w-16 h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-300"
            style={{ width: `${clampedScore}%`, backgroundColor: color }}
          />
        </div>
        {showLabel && (
          <span className="text-xs font-semibold tabular-nums text-foreground">
            {clampedScore}
          </span>
        )}
      </div>
    );
  }

  // Circular / semi-arc presentation for detailed cards
  const strokeWidth = size === 'lg' ? 8 : 6;
  const radius = size === 'lg' ? 42 : 30;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (clampedScore / 100) * circumference;

  return (
    <div
      className={`relative inline-flex items-center justify-center ${
        size === 'lg' ? 'w-24 h-24' : 'w-18 h-18'
      }`}
      data-testid="score-gauge"
    >
      <svg
        className="transform -rotate-90"
        width={radius * 2 + strokeWidth * 2}
        height={radius * 2 + strokeWidth * 2}
      >
        <circle
          cx={radius + strokeWidth}
          cy={radius + strokeWidth}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          className="text-slate-200 dark:text-slate-800"
          fill="transparent"
        />
        <circle
          cx={radius + strokeWidth}
          cy={radius + strokeWidth}
          r={radius}
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-500 ease-out"
          fill="transparent"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span
          className={`font-bold tabular-nums text-foreground ${
            size === 'lg' ? 'text-2xl' : 'text-lg'
          }`}
        >
          {clampedScore}
        </span>
        {size === 'lg' && (
          <span className="text-[10px] text-muted-foreground uppercase tracking-wider">
            / 100
          </span>
        )}
      </div>
    </div>
  );
};
