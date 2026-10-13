import React from 'react';

interface SparklineProps {
  data?: number[];
  width?: number;
  height?: number;
  className?: string;
}

export const Sparkline: React.FC<SparklineProps> = ({
  data = [100, 102, 101, 104, 103, 107, 106, 110, 108, 112],
  width = 80,
  height = 24,
  className = '',
}) => {
  if (!data || data.length < 2) {
    return <div className="w-20 h-6 bg-slate-100 dark:bg-slate-800 rounded" />;
  }

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data
    .map((val, idx) => {
      const x = (idx / (data.length - 1)) * (width - 4) + 2;
      const y = height - 2 - ((val - min) / range) * (height - 6);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');

  const isUp = data[data.length - 1] >= data[0];
  const strokeColor = isUp ? '#10b981' : '#ef4444';

  return (
    <svg
      width={width}
      height={height}
      className={`overflow-visible inline-block ${className}`}
      data-testid="sparkline-svg"
      aria-hidden="true"
    >
      <polyline
        fill="none"
        stroke={strokeColor}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        points={points}
      />
    </svg>
  );
};

interface Range52wProps {
  current: number;
  low?: number;
  high?: number;
}

export const Range52w: React.FC<Range52wProps> = ({ current, low, high }) => {
  if (low === undefined || high === undefined || high <= low) {
    return <span className="text-xs text-muted-foreground">—</span>;
  }

  const pct = Math.max(0, Math.min(100, ((current - low) / (high - low)) * 100));

  return (
    <div className="flex flex-col gap-1 w-28" data-testid="range-52w">
      <div className="flex justify-between text-[10px] text-muted-foreground tabular-nums">
        <span>{low.toFixed(1)}</span>
        <span>{high.toFixed(1)}</span>
      </div>
      <div className="relative w-full h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
        <div
          className="absolute top-0 bottom-0 w-2 bg-primary rounded-full -ml-1 transition-all"
          style={{ left: `${pct}%` }}
        />
      </div>
    </div>
  );
};
