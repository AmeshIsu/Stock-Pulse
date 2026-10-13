import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { apiClient } from '../lib/api';
import {
  HelpCircle,
  FileText,
  Printer,
  Table,
  ShieldAlert,
  Sliders,
  CheckCircle,
} from 'lucide-react';

interface MethodologyData {
  markdown_content: string;
  thresholds: Record<string, unknown>;
}

export const HowItWorks: React.FC = () => {
  const [data, setData] = useState<MethodologyData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    apiClient<MethodologyData>('/methodology')
      .then((res) => setData(res))
      .catch((err) => console.error('Failed to load methodology:', err))
      .finally(() => setIsLoading(false));
  }, []);

  const handlePrint = () => {
    window.print();
  };

  const thresholdsInventory = [
    {
      name: 'High Upside Tier Cutoff',
      value: 'Top 33.3% (Rank 1–7)',
      derivedFrom: 'Pre-registered cross-sectional tercile split',
      dataUsed: 'Historical 20-stock out-of-sample forward returns',
      purpose: 'Categorizes equities into High, Neutral, and Low relative upside tiers',
    },
    {
      name: 'Elevated Volatility Threshold',
      value: '> 30.0% annualized',
      derivedFrom: 'Cross-sectional historical median + dispersion buffer',
      dataUsed: '63-day standard deviation of daily log returns',
      purpose: 'Screens out or flags stocks undergoing volatile price fluctuations',
    },
    {
      name: 'Market Liquidity Warning Threshold',
      value: '< LKR 500,000 median turnover or > 20% zero-volume days',
      derivedFrom: 'CSE institutional execution viability test',
      dataUsed: '63-day daily trading turnover and recorded tick volume',
      purpose: 'Identifies illiquid securities where quoted prices may not represent executable depth',
    },
    {
      name: 'Horizon Rebalance Frequency',
      value: 'Non-overlapping multi-week rebalancing',
      derivedFrom: 'Autocorrelation decay analysis',
      dataUsed: 'Historical price series 2014–2025',
      purpose: 'Eliminates statistical bias from overlapping return intervals',
    },
  ];

  return (
    <div className="space-y-8 max-w-5xl mx-auto" data-testid="how-it-works-page">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground flex items-center gap-2">
            <HelpCircle className="w-6 h-6 text-primary" />
            <span>Methodology &amp; System Rules</span>
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Institutional transparency, threshold derivation, historical testing framework, and risk boundaries
          </p>
        </div>

        <button
          type="button"
          data-testid="print-methodology-btn"
          onClick={handlePrint}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg border border-border bg-card text-xs font-semibold text-foreground hover:bg-muted shadow-xs transition no-print"
        >
          <Printer className="w-4 h-4 text-primary" />
          <span>Print Document</span>
        </button>
      </div>

      {/* Quick Navigation Anchor Links */}
      <div className="bg-card border border-border rounded-xl p-4 shadow-sm flex flex-wrap items-center gap-4 text-xs font-medium no-print">
        <span className="text-muted-foreground font-semibold">Jump to:</span>
        <a href="#thresholds" className="text-primary hover:underline flex items-center gap-1">
          <Table className="w-3.5 h-3.5" />
          <span>Operational Thresholds</span>
        </a>
        <a href="#methodology-text" className="text-primary hover:underline flex items-center gap-1">
          <FileText className="w-3.5 h-3.5" />
          <span>Full Methodology Text</span>
        </a>
        <a href="#limitations" className="text-primary hover:underline flex items-center gap-1">
          <ShieldAlert className="w-3.5 h-3.5" />
          <span>Governance &amp; Limitations</span>
        </a>
      </div>

      {/* Thresholds Table Section */}
      <section id="thresholds" className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4">
        <div>
          <h2 className="text-lg font-bold text-foreground flex items-center gap-2">
            <Sliders className="w-5 h-5 text-primary" />
            <span>Operational Thresholds &amp; Derivation Matrix</span>
          </h2>
          <p className="text-xs text-muted-foreground mt-1">
            Every cutoff employed across Stock Pulse is strictly pre-registered and historically audited.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-muted/40 border-b text-muted-foreground font-semibold uppercase tracking-wider text-[11px]">
                <th className="py-3 px-3">Threshold Metric</th>
                <th className="py-3 px-3">Cutoff Value</th>
                <th className="py-3 px-3">Derivation Basis</th>
                <th className="py-3 px-3">Data Used</th>
                <th className="py-3 px-3">System Purpose</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {thresholdsInventory.map((t, idx) => (
                <tr key={idx} className="hover:bg-muted/30">
                  <td className="py-3 px-3 font-bold text-foreground">{t.name}</td>
                  <td className="py-3 px-3 font-semibold text-primary tabular-nums">{t.value}</td>
                  <td className="py-3 px-3 text-muted-foreground">{t.derivedFrom}</td>
                  <td className="py-3 px-3 text-muted-foreground">{t.dataUsed}</td>
                  <td className="py-3 px-3 text-muted-foreground">{t.purpose}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Rendered Methodology Markdown */}
      <section
        id="methodology-text"
        className="bg-card border border-border rounded-xl p-6 sm:p-8 shadow-sm space-y-4 prose dark:prose-invert max-w-none text-xs sm:text-sm leading-relaxed"
      >
        {isLoading ? (
          <div className="space-y-4 animate-pulse">
            <div className="h-6 bg-muted rounded w-1/3" />
            <div className="h-20 bg-muted rounded" />
            <div className="h-40 bg-muted rounded" />
          </div>
        ) : (
          <ReactMarkdown remarkPlugins={[remarkGfm]}>
            {data?.markdown_content || '# User Methodology\n\nNo methodology content loaded.'}
          </ReactMarkdown>
        )}
      </section>

      {/* Regulatory Limitations & Disclaimer */}
      <section
        id="limitations"
        className="p-6 bg-slate-50 dark:bg-slate-900 border border-border rounded-xl space-y-3"
      >
        <div className="flex items-center gap-2 text-foreground font-bold text-sm">
          <ShieldAlert className="w-5 h-5 text-amber-600 dark:text-amber-400" />
          <span>Statutory Governance &amp; Limitations Notice</span>
        </div>
        <p className="text-xs text-muted-foreground leading-relaxed">
          Stock Pulse rankings represent cross-sectional characteristics measured over non-overlapping historical periods across the 20 equities comprising the Colombo Stock Exchange S&amp;P SL20 index. Historical out-of-sample validation exhibits modest predictive strength and non-trivial variance across market cycles.
        </p>
        <div className="p-3 bg-card border rounded-lg text-xs font-semibold text-foreground">
          “Stock Pulse provides data-driven rankings for information only. It is not financial advice. Signals shown have limited historical reliability.”
        </div>
      </section>
    </div>
  );
};
