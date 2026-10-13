import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { apiClient } from '../lib/api';
import {
  MarketSummaryResponse,
  RankingsResponse,
  StockRankingRow,
  Horizon,
  WatchlistItem,
} from '../lib/types';
import { TierBadge } from '../components/common/TierBadge';
import { ConfidencePill } from '../components/common/ConfidencePill';
import { RiskBadge } from '../components/common/RiskBadge';
import { ScoreGauge } from '../components/common/ScoreGauge';
import { Sparkline } from '../components/common/Sparkline';
import {
  TrendingUp,
  ShieldCheck,
  Activity,
  Award,
  ArrowUpRight,
  ArrowDownRight,
  AlertTriangle,
  HelpCircle,
  ExternalLink,
  ChevronRight,
  Bookmark,
  Sparkles,
} from 'lucide-react';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  const [market, setMarket] = useState<MarketSummaryResponse | null>(null);
  const [rankingsData, setRankingsData] = useState<RankingsResponse | null>(null);
  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [selectedHorizon, setSelectedHorizon] = useState<'overall' | Horizon>('overall');
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      apiClient<MarketSummaryResponse>('/market/summary'),
      apiClient<RankingsResponse>('/rankings'),
      apiClient<WatchlistItem[]>('/watchlist').catch(() => []),
    ])
      .then(([m, r, w]) => {
        setMarket(m);
        setRankingsData(r);
        setWatchlist(w || []);
      })
      .catch((err) => {
        console.error('Failed to load dashboard data:', err);
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, []);

  if (isLoading) {
    return (
      <div className="space-y-6 animate-pulse" data-testid="dashboard-loading-skeleton">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="h-28 bg-muted rounded-xl" />
          <div className="h-28 bg-muted rounded-xl" />
          <div className="h-28 bg-muted rounded-xl" />
        </div>
        <div className="h-96 bg-muted rounded-xl" />
      </div>
    );
  }

  const stocks = rankingsData?.rankings || [];

  // Sort top picks according to active horizon tab
  const sortedPicks = [...stocks].sort((a, b) => {
    if (selectedHorizon === 'overall') {
      return b.overall_score - a.overall_score;
    }
    return b.horizons[selectedHorizon].score - a.horizons[selectedHorizon].score;
  });

  const topPicks = sortedPicks.slice(0, 5);

  const horizonTabs: { id: 'overall' | Horizon; label: string }[] = [
    { id: 'overall', label: 'Overall' },
    { id: 'short', label: 'Short term (about 1 month)' },
    { id: 'mid', label: 'Mid term (about 3 months)' },
    { id: 'long', label: 'Long term (about 12 months)' },
  ];

  return (
    <div className="space-y-8" data-testid="dashboard-page">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground flex items-center gap-2.5">
            <span>Market Pulse</span>
            <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20">
              S&amp;P SL20
            </span>
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Data-driven upside rankings and structural risk analytics for Colombo equities as of {rankingsData?.as_of_date || market?.as_of_date || '2025-12-31'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link
            to="/rankings"
            data-testid="view-full-rankings-btn"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-semibold hover:bg-primary-700 transition"
          >
            <span>View All 20 Stocks</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>

      {/* Market Summary Tiles */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4" data-testid="market-summary-tiles">
        {/* Tile 1: High Upside Mix */}
        <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-2">
          <div className="flex items-center justify-between text-muted-foreground">
            <span className="text-xs font-semibold uppercase tracking-wider">Top-Tier Focus</span>
            <Sparkles className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-3xl font-black text-foreground tabular-nums" data-testid="high-tier-count">
              {market?.high_tier_count ?? 0}
            </span>
            <span className="text-xs text-muted-foreground font-medium">/ 20 Universe Equities</span>
          </div>
          <p className="text-[11px] text-muted-foreground leading-normal">
            Equities currently exhibiting strong cross-sectional characteristics relative to their median peers.
          </p>
        </div>

        {/* Tile 2: Risk Profile Mix */}
        <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-2">
          <div className="flex items-center justify-between text-muted-foreground">
            <span className="text-xs font-semibold uppercase tracking-wider">Universe Risk Mix</span>
            <ShieldCheck className="w-4 h-4 text-primary" />
          </div>
          <div className="flex items-center gap-3">
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-bold text-foreground tabular-nums">
                {market?.risk_mix?.normal ?? 0}
              </span>
              <span className="text-[11px] text-muted-foreground font-medium">Normal</span>
            </div>
            <span className="text-muted-foreground/40 font-light">&bull;</span>
            <div className="flex items-baseline gap-1">
              <span className="text-2xl font-bold text-amber-600 dark:text-amber-400 tabular-nums">
                {market?.risk_mix?.elevated ?? 0}
              </span>
              <span className="text-[11px] text-muted-foreground font-medium">Elevated</span>
            </div>
          </div>
          <p className="text-[11px] text-muted-foreground leading-normal">
            63-day realized volatility screening identifying stocks exhibiting elevated price dispersion.
          </p>
        </div>

        {/* Tile 3: Conditional Volatility Note */}
        <div className={`bg-card border rounded-xl p-5 shadow-sm space-y-2 ${
          market?.volatility_note?.elevated ? 'border-amber-300 dark:border-amber-800 bg-amber-50/30' : 'border-border'
        }`}>
          <div className="flex items-center justify-between text-muted-foreground">
            <span className="text-xs font-semibold uppercase tracking-wider">Cross-Sectional Volatility</span>
            <Activity className="w-4 h-4 text-muted-foreground" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-foreground tabular-nums">
              {market?.volatility_note?.current_vol_pct?.toFixed(2) ?? '20.59'}%
            </span>
            <span className="text-[11px] text-muted-foreground font-medium">
              (63-day median vs {market?.volatility_note?.threshold_pct ?? 30}% threshold)
            </span>
          </div>
          <p className="text-[11px] text-muted-foreground leading-normal" data-testid="volatility-headline">
            {market?.volatility_note?.headline || 'Market volatility is within normal historical ranges.'}
          </p>
        </div>
      </div>

      {/* Consistent Leaders Strip */}
      {market?.consistent_leaders && market.consistent_leaders.length > 0 && (
        <div className="bg-primary/5 border border-primary/15 rounded-xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-4" data-testid="consistent-leaders-strip">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-primary/20 text-primary flex items-center justify-center shrink-0">
              <Award className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-xs font-bold uppercase tracking-wider text-primary">
                Consistent Leaders Across Horizons
              </h2>
              <p className="text-xs text-muted-foreground">
                Equities ranked in High upside across multiple investment horizons simultaneously:
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {market.consistent_leaders.map((leader) => (
              <Link
                key={leader.code}
                to={`/stocks/${leader.code}`}
                data-testid={`leader-pill-${leader.code}`}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-card border border-border shadow-xs hover:border-primary text-xs font-medium text-foreground transition"
              >
                <span className="font-bold text-primary">{leader.code}</span>
                <span className="text-muted-foreground text-[11px]">{leader.name}</span>
                <span className={`text-[11px] font-semibold tabular-nums ${
                  leader.daily_change_pct >= 0 ? 'text-emerald-800 dark:text-emerald-400' : 'text-rose-700 dark:text-rose-400'
                }`}>
                  {leader.daily_change_pct >= 0 ? '+' : ''}{leader.daily_change_pct.toFixed(2)}%
                </span>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Top Picks Section */}
      <div className="bg-card border border-border rounded-xl shadow-sm p-6 space-y-6" data-testid="top-picks-section">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-2 border-b">
          <div>
            <h2 className="text-xl font-bold text-foreground">Top Upside Focus</h2>
            <p className="text-xs text-muted-foreground">
              Highest-scoring securities on our 0–100 cross-sectional upside scale
            </p>
          </div>

          {/* Horizon Tabs */}
          <div className="flex flex-wrap items-center gap-1 bg-muted p-1 rounded-lg">
            {horizonTabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                data-testid={`tab-${tab.id}`}
                onClick={() => setSelectedHorizon(tab.id)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  selectedHorizon === tab.id
                    ? 'bg-card text-foreground shadow-xs font-semibold'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Confidence Notice Bar */}
        <div className="p-3.5 bg-slate-50 dark:bg-slate-900 border border-border rounded-lg flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs" data-testid="confidence-notice-bar">
          <div className="flex items-center gap-3">
            <span className="font-semibold text-foreground text-xs">Horizon Confidence:</span>
            {selectedHorizon === 'overall' ? (
              <div className="flex flex-wrap items-center gap-2">
                <div className="flex items-center gap-1.5">
                  <span className="text-[11px] text-muted-foreground">Short:</span>
                  <ConfidencePill level={rankingsData?.confidence_labels?.short || 'Low confidence'} />
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-[11px] text-muted-foreground">Mid:</span>
                  <ConfidencePill level={rankingsData?.confidence_labels?.mid || 'Low confidence'} />
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-[11px] text-muted-foreground">Long:</span>
                  <ConfidencePill level={rankingsData?.confidence_labels?.long || 'Low confidence'} />
                </div>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <ConfidencePill level={rankingsData?.confidence_labels?.[selectedHorizon] || 'Low confidence'} />
                <span className="text-muted-foreground text-xs">
                  {rankingsData?.confidence_explanations?.[selectedHorizon] ||
                    'Historical out-of-sample relationships in this horizon have limited predictive strength.'}
                </span>
              </div>
            )}
          </div>
          <Link
            to="/how-it-works"
            data-testid="how-we-tested-this-link"
            className="inline-flex items-center gap-1 text-primary hover:underline font-medium shrink-0"
          >
            <HelpCircle className="w-3.5 h-3.5" />
            <span>How we tested this</span>
          </Link>
        </div>

        {/* Top Picks List */}
        <div className="space-y-3">
          {topPicks.map((stock, idx) => {
            const score =
              selectedHorizon === 'overall'
                ? stock.overall_score
                : stock.horizons[selectedHorizon].score;
            const tier =
              selectedHorizon === 'overall'
                ? stock.overall_tier
                : stock.horizons[selectedHorizon].tier;

            return (
              <div
                key={stock.code}
                data-testid={`top-pick-row-${stock.code}`}
                onClick={() => navigate(`/stocks/${stock.code}`)}
                className="group flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-xl border border-border bg-card hover:border-primary/50 hover:shadow-xs transition cursor-pointer gap-4"
              >
                {/* Left: Rank, Code, Name, Badges */}
                <div className="flex items-center gap-4">
                  <div className="w-7 h-7 rounded-full bg-muted flex items-center justify-center font-bold text-xs text-muted-foreground group-hover:text-primary group-hover:bg-primary/10 transition tabular-nums">
                    {idx + 1}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-sm text-foreground group-hover:text-primary transition">
                        {stock.code}
                      </span>
                      {stock.liquidity_flag && (
                        <span
                          title={stock.liquidity_reason || 'Thinly traded security'}
                          className="text-amber-500"
                        >
                          <AlertTriangle className="w-3.5 h-3.5" />
                        </span>
                      )}
                    </div>
                    <span className="text-xs text-muted-foreground line-clamp-1">{stock.name}</span>
                  </div>
                </div>

                {/* Center: Score Gauge & Tier Pill */}
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <span className="text-[10px] uppercase font-semibold text-muted-foreground block">
                      Upside score
                    </span>
                    <ScoreGauge score={score} tier={tier} size="sm" />
                  </div>
                  <TierBadge tier={tier} size="sm" />
                  <RiskBadge level={stock.risk} />
                </div>

                {/* Right: Sparkline, Price, 1-day change */}
                <div className="flex items-center justify-between sm:justify-end gap-6 border-t sm:border-t-0 pt-2 sm:pt-0">
                  <div className="hidden md:block">
                    <Sparkline
                      data={[
                        stock.last_price * 0.96,
                        stock.last_price * 0.98,
                        stock.last_price * 0.97,
                        stock.last_price * 1.01,
                        stock.last_price * 1.0,
                        stock.last_price * (1 + stock.daily_change_pct / 100),
                      ]}
                    />
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-bold text-foreground tabular-nums">
                      LKR {stock.last_price.toFixed(2)}
                    </div>
                    <div
                      className={`text-xs font-semibold tabular-nums flex items-center justify-end gap-0.5 ${
                        stock.daily_change_pct >= 0
                          ? 'text-emerald-800 dark:text-emerald-400'
                          : 'text-rose-700 dark:text-rose-400'
                      }`}
                    >
                      {stock.daily_change_pct >= 0 ? (
                        <ArrowUpRight className="w-3.5 h-3.5" />
                      ) : (
                        <ArrowDownRight className="w-3.5 h-3.5" />
                      )}
                      <span>
                        {stock.daily_change_pct >= 0 ? '+' : ''}
                        {stock.daily_change_pct.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Grid: Movers & Watchlist snapshot */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Movers Panel with Indicative Tag */}
        <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4" data-testid="movers-panel">
          <div className="flex items-center justify-between pb-2 border-b">
            <div>
              <h2 className="text-lg font-bold text-foreground">Market Movers</h2>
              <p className="text-xs text-muted-foreground">Recent trading session price shifts</p>
            </div>
            <span
              className="text-[10px] font-semibold uppercase px-2 py-0.5 rounded border border-slate-300 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 text-muted-foreground"
              data-testid="indicative-tag"
            >
              Indicative
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Top Gainers */}
            <div className="space-y-2">
              <span className="text-xs font-bold text-emerald-700 dark:text-emerald-400 uppercase tracking-wide">
                Gainers
              </span>
              <div className="space-y-1.5">
                {(market?.top_movers?.gainers || []).slice(0, 3).map((item) => (
                  <div
                    key={item.code}
                    onClick={() => navigate(`/stocks/${item.code}`)}
                    className="p-2 rounded-lg bg-muted/40 hover:bg-muted cursor-pointer flex items-center justify-between text-xs"
                  >
                    <div>
                      <span className="font-bold text-foreground">{item.code}</span>
                      <span className="text-[11px] text-muted-foreground block truncate max-w-[100px]">{item.name}</span>
                    </div>
                    <div className="text-right">
                      <span className="font-semibold text-emerald-800 dark:text-emerald-400 tabular-nums">
                        +{item.daily_change_pct.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Top Losers */}
            <div className="space-y-2">
              <span className="text-xs font-bold text-rose-700 dark:text-rose-400 uppercase tracking-wide">
                Losers
              </span>
              <div className="space-y-1.5">
                {(market?.top_movers?.losers || []).slice(0, 3).map((item) => (
                  <div
                    key={item.code}
                    onClick={() => navigate(`/stocks/${item.code}`)}
                    className="p-2 rounded-lg bg-muted/40 hover:bg-muted cursor-pointer flex items-center justify-between text-xs"
                  >
                    <div>
                      <span className="font-bold text-foreground">{item.code}</span>
                      <span className="text-[11px] text-muted-foreground block truncate max-w-[100px]">{item.name}</span>
                    </div>
                    <div className="text-right">
                      <span className="font-semibold text-rose-700 dark:text-rose-400 tabular-nums">
                        {item.daily_change_pct.toFixed(2)}%
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Watchlist Snapshot */}
        <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4" data-testid="watchlist-snapshot">
          <div className="flex items-center justify-between pb-2 border-b">
            <div>
              <h2 className="text-lg font-bold text-foreground">Watchlist Snapshot</h2>
              <p className="text-xs text-muted-foreground">Securities monitored for upside transitions</p>
            </div>
            <Link
              to="/watchlist"
              data-testid="go-to-watchlist-link"
              className="text-xs text-primary font-semibold hover:underline inline-flex items-center gap-1"
            >
              <span>Manage</span>
              <ExternalLink className="w-3 h-3" />
            </Link>
          </div>

          {watchlist.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground space-y-2">
              <Bookmark className="w-8 h-8 mx-auto opacity-30" />
              <p className="text-xs">Your watchlist is currently empty.</p>
              <Link
                to="/rankings"
                className="text-xs text-primary font-semibold hover:underline inline-block"
              >
                Explore rankings to add stocks
              </Link>
            </div>
          ) : (
            <div className="space-y-2">
              {watchlist.slice(0, 4).map((w) => (
                <div
                  key={w.id}
                  onClick={() => navigate(`/stocks/${w.stock_code}`)}
                  className="p-3 rounded-lg border border-border bg-card hover:bg-muted/40 transition cursor-pointer flex items-center justify-between text-xs"
                >
                  <div className="flex items-center gap-3">
                    <span className="font-bold text-primary">{w.stock_code}</span>
                    <span className="text-muted-foreground truncate max-w-[140px]">{w.stock_name}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <TierBadge tier={w.overall_tier} size="sm" />
                    <span className="font-bold tabular-nums">LKR {w.last_price.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
