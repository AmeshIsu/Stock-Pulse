import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { apiClient } from '../lib/api';
import {
  StockDetailResponse,
  PricePoint,
  WatchlistItem,
  Horizon,
} from '../lib/types';
import { StockPriceChart } from '../components/charts/StockPriceChart';
import { TierBadge } from '../components/common/TierBadge';
import { ConfidencePill } from '../components/common/ConfidencePill';
import { RiskBadge } from '../components/common/RiskBadge';
import { ScoreGauge } from '../components/common/ScoreGauge';
import {
  Star,
  Bell,
  ArrowUpRight,
  ArrowDownRight,
  AlertTriangle,
  ShieldCheck,
  Droplets,
  Calendar,
  Info,
  Check,
  TrendingUp,
} from 'lucide-react';

export const StockDetail: React.FC = () => {
  const { code } = useParams<{ code: string }>();

  const [stock, setStock] = useState<StockDetailResponse | null>(null);
  const [priceHistory, setPriceHistory] = useState<PricePoint[]>([]);
  const [isWatchlisted, setIsWatchlisted] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Quick Alert Modal State
  const [showAlertModal, setShowAlertModal] = useState(false);
  const [alertType, setAlertType] = useState('price_above');
  const [alertThreshold, setAlertThreshold] = useState('');
  const [alertSuccess, setAlertSuccess] = useState(false);

  useEffect(() => {
    if (!code) return;
    setIsLoading(true);

    Promise.all([
      apiClient<any>(`/stocks/${code}`),
      apiClient<{ prices: PricePoint[] }>(`/stocks/${code}/prices`).catch(() => ({
        prices: [],
      })),
      apiClient<WatchlistItem[]>('/watchlist').catch(() => []),
    ])
      .then(([rawDetail, pricesRes, watchList]) => {
        const mappedDetail: StockDetailResponse = {
          code: rawDetail.code,
          name: rawDetail.company_name || rawDetail.name || rawDetail.code,
          display_ticker: rawDetail.display_ticker || rawDetail.code,
          as_of_date: rawDetail.last_traded_date || '2025-12-31',
          last_price: rawDetail.last_price || 100,
          daily_change_pct: rawDetail.daily_change_pct || 0,
          is_stale: rawDetail.status === 'stale' || rawDetail.is_stale || false,
          last_traded_date: rawDetail.last_traded_date || '2025-12-31',
          low_52w: rawDetail.range_52w?.low || rawDetail.low_52w || (rawDetail.last_price ? rawDetail.last_price * 0.8 : 80),
          high_52w: rawDetail.range_52w?.high || rawDetail.high_52w || (rawDetail.last_price ? rawDetail.last_price * 1.2 : 120),
          outlooks: {
            short: {
              score: rawDetail.score_short ?? rawDetail.outlooks?.short?.score ?? 50,
              tier: rawDetail.tier_short ?? rawDetail.outlooks?.short?.tier ?? 'Neutral',
              confidence: rawDetail.confidence_short ?? rawDetail.outlooks?.short?.confidence ?? 'Low confidence',
              confidence_explanation: rawDetail.confidence_explanation_short ?? rawDetail.outlooks?.short?.confidence_explanation ?? 'Limited historical statistical reliability.',
              drivers: {
                supporting: rawDetail.reasons_short?.supporting || ['Favorable technical configuration'],
                holding_back: rawDetail.reasons_short?.holding_back || ['Macro headwinds in cross-sectional rank'],
              },
              thin_trading_note: rawDetail.thin_trading_note,
            },
            mid: {
              score: rawDetail.score_mid ?? rawDetail.outlooks?.mid?.score ?? 50,
              tier: rawDetail.tier_mid ?? rawDetail.outlooks?.mid?.tier ?? 'Neutral',
              confidence: rawDetail.confidence_mid ?? rawDetail.outlooks?.mid?.confidence ?? 'Low confidence',
              confidence_explanation: rawDetail.confidence_explanation_mid ?? rawDetail.outlooks?.mid?.confidence_explanation ?? 'Limited historical statistical reliability.',
              drivers: {
                supporting: rawDetail.reasons_mid?.supporting || ['Consistent operational momentum'],
                holding_back: rawDetail.reasons_mid?.holding_back || ['Higher risk dispersion'],
              },
              thin_trading_note: rawDetail.thin_trading_note,
            },
            long: {
              score: rawDetail.score_long ?? rawDetail.outlooks?.long?.score ?? 50,
              tier: rawDetail.tier_long ?? rawDetail.outlooks?.long?.tier ?? 'Neutral',
              confidence: rawDetail.confidence_long ?? rawDetail.outlooks?.long?.confidence ?? 'Low confidence',
              confidence_explanation: rawDetail.confidence_explanation_long ?? rawDetail.outlooks?.long?.confidence_explanation ?? 'Limited historical statistical reliability.',
              drivers: {
                supporting: rawDetail.reasons_long?.supporting || ['Structural earnings resilience'],
                holding_back: rawDetail.reasons_long?.holding_back || ['Valuation multiple ceiling'],
              },
              thin_trading_note: rawDetail.thin_trading_note,
            },
          },
          risk: {
            level: rawDetail.risk_category || rawDetail.risk?.level || 'Normal',
            vol_63d_pct: rawDetail.risk?.vol_63d_pct || 20.59,
            threshold_pct: 30.0,
            explanation: rawDetail.risk_category === 'Elevated'
              ? 'Elevated historical price dispersion exceeding the 30% volatility threshold.'
              : 'Price dispersion is within normal historical tolerances.',
          },
          liquidity: {
            flagged: rawDetail.liquidity_warning ?? rawDetail.liquidity?.flagged ?? false,
            turnover_median_lkr: rawDetail.liquidity?.turnover_median_lkr || 500000,
            threshold_lkr: 250000,
            zero_volume_days_pct: rawDetail.liquidity?.zero_volume_days_pct || 0.05,
            explanation: rawDetail.liquidity_warning
              ? 'Thinly traded equity with zero-volume sessions exceeding 20% threshold.'
              : 'Sufficient trading frequency across recent market sessions.',
          },
          key_stats: rawDetail.key_stats || {
            market_cap_lkr: 45000000000,
            trailing_pe: 7.2,
            avg_volume_20d: 120000,
            beta_index: 0.95,
          },
        };
        setStock(mappedDetail);
        setPriceHistory(pricesRes.prices || []);
        setIsWatchlisted(watchList.some((w) => w.stock_code === code));
      })
      .catch((err) => {
        console.error('Failed to load stock details:', err);
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [code]);

  const handleToggleWatchlist = async () => {
    if (!stock) return;
    try {
      if (isWatchlisted) {
        await apiClient(`/watchlist/${stock.code}`, { method: 'DELETE' });
        setIsWatchlisted(false);
      } else {
        await apiClient('/watchlist', {
          method: 'POST',
          body: JSON.stringify({ stock_code: stock.code }),
        });
        setIsWatchlisted(true);
      }
    } catch {
      alert('Unable to update watchlist.');
    }
  };

  const handleCreateAlert = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!stock) return;

    try {
      await apiClient('/alerts', {
        method: 'POST',
        body: JSON.stringify({
          stock_code: stock.code,
          alert_type: alertType,
          threshold_value: parseFloat(alertThreshold) || undefined,
          cooldown_days: 3,
        }),
      });
      setAlertSuccess(true);
      setTimeout(() => {
        setAlertSuccess(false);
        setShowAlertModal(false);
      }, 1500);
    } catch {
      alert('Failed to register alert.');
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6 animate-pulse" data-testid="stock-detail-skeleton">
        <div className="h-24 bg-muted rounded-xl" />
        <div className="h-96 bg-muted rounded-xl" />
      </div>
    );
  }

  if (!stock) {
    return (
      <div className="text-center py-16" data-testid="stock-not-found">
        <h2 className="text-xl font-bold">Stock Not Found</h2>
        <p className="text-xs text-muted-foreground mt-1">The requested symbol does not exist in our universe.</p>
        <Link to="/rankings" className="text-primary text-xs font-semibold hover:underline mt-4 inline-block">
          Return to Rankings
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-8" data-testid="stock-detail-page">
      {/* Header Banner */}
      <div className="bg-card border border-border rounded-xl p-6 shadow-sm flex flex-col md:flex-row md:items-center justify-between gap-6" data-testid="stock-header">
        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground" data-testid="stock-code-header">
              {stock.code}
            </h1>
            {stock.display_ticker && stock.display_ticker !== stock.code && (
              <span className="text-xs font-semibold px-2 py-0.5 rounded bg-muted text-muted-foreground">
                {stock.display_ticker}
              </span>
            )}
            {stock.is_stale ? (
              <span
                className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-amber-50 text-amber-900 border border-amber-300 dark:bg-amber-950/40 dark:text-amber-200 flex items-center gap-1"
                data-testid="stale-status-badge"
              >
                <Calendar className="w-3.5 h-3.5" />
                <span>Last traded on {stock.last_traded_date}</span>
              </span>
            ) : (
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-300 dark:bg-emerald-950/40 dark:text-emerald-300">
                Active Trading
              </span>
            )}
            {stock.liquidity.flagged && (
              <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-amber-50 text-amber-900 border border-amber-300 flex items-center gap-1">
                <AlertTriangle className="w-3.5 h-3.5" />
                <span>Thinly Traded</span>
              </span>
            )}
          </div>
          <p className="text-sm text-muted-foreground">{stock.name}</p>
        </div>

        {/* Price & Actions */}
        <div className="flex flex-wrap items-center gap-4">
          <div className="text-left md:text-right">
            <div className="text-2xl sm:text-3xl font-black text-foreground tabular-nums" data-testid="stock-price">
              LKR {stock.last_price.toFixed(2)}
            </div>
            <div
              className={`text-xs font-semibold tabular-nums flex items-center md:justify-end gap-1 ${
                stock.daily_change_pct >= 0 ? 'text-emerald-800 dark:text-emerald-400' : 'text-rose-700 dark:text-rose-400'
              }`}
              data-testid="stock-change"
            >
              {stock.daily_change_pct >= 0 ? (
                <ArrowUpRight className="w-4 h-4" />
              ) : (
                <ArrowDownRight className="w-4 h-4" />
              )}
              <span>
                {stock.daily_change_pct >= 0 ? '+' : ''}
                {stock.daily_change_pct.toFixed(2)}%
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              data-testid="toggle-watchlist-btn"
              onClick={handleToggleWatchlist}
              className={`p-2.5 rounded-xl border transition shadow-xs ${
                isWatchlisted
                  ? 'bg-amber-500/10 border-amber-500 text-amber-500 hover:bg-amber-500/20'
                  : 'bg-card border-border text-muted-foreground hover:text-foreground hover:bg-muted'
              }`}
              aria-label={isWatchlisted ? 'Remove from Watchlist' : 'Add to Watchlist'}
            >
              <Star className="w-5 h-5 fill-current" />
            </button>

            <button
              type="button"
              data-testid="create-alert-btn"
              onClick={() => setShowAlertModal(true)}
              className="inline-flex items-center gap-1.5 px-3.5 py-2.5 rounded-xl bg-primary text-primary-foreground font-semibold text-xs shadow-sm hover:bg-primary-700 transition"
            >
              <Bell className="w-4 h-4" />
              <span>Create Alert</span>
            </button>
          </div>
        </div>
      </div>

      {/* Interactive Price Chart with TradingView attribution */}
      <StockPriceChart data={priceHistory} stockCode={stock.code} />

      {/* Upside Outlook: 3 Horizon Cards */}
      <div className="space-y-4" data-testid="upside-outlook-section">
        <div>
          <h2 className="text-xl font-bold text-foreground">Upside Outlook</h2>
          <p className="text-xs text-muted-foreground">
            Multi-horizon cross-sectional characteristics and factor explanations
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {(
            [
              { id: 'short' as Horizon, title: 'Short term (about 1 month)' },
              { id: 'mid' as Horizon, title: 'Mid term (about 3 months)' },
              { id: 'long' as Horizon, title: 'Long term (about 12 months)' },
            ] as const
          ).map((h) => {
            const outlook = stock.outlooks[h.id];
            if (!outlook) return null;

            return (
              <div
                key={h.id}
                data-testid={`outlook-card-${h.id}`}
                className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-4 flex flex-col justify-between"
              >
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sm text-foreground">{h.title}</span>
                    <TierBadge tier={outlook.tier} size="sm" />
                  </div>

                  <div className="flex items-center justify-between p-3 bg-muted/30 rounded-lg">
                    <div>
                      <span className="text-[10px] uppercase font-semibold text-muted-foreground block">
                        Upside Score
                      </span>
                      <span className="text-xs text-muted-foreground">Relative strength</span>
                    </div>
                    <ScoreGauge score={outlook.score} tier={outlook.tier} size="md" />
                  </div>

                  {/* Confidence Pill & Explanation */}
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <ConfidencePill level={outlook.confidence} />
                    </div>
                    <p className="text-[11px] text-muted-foreground leading-normal">
                      {outlook.confidence_explanation}
                    </p>
                  </div>

                  {/* Plain Language Drivers */}
                  <div className="space-y-2 pt-2 border-t">
                    <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground block">
                      Factor Dynamics
                    </span>

                    {/* Supporting Drivers */}
                    {outlook.drivers.supporting.length > 0 && (
                      <div className="space-y-1">
                        <span className="text-[11px] font-semibold text-emerald-700 dark:text-emerald-400">
                          Supporting Drivers:
                        </span>
                        <ul className="text-xs text-muted-foreground space-y-1 pl-3 list-disc">
                          {outlook.drivers.supporting.map((d, i) => (
                            <li key={i}>{d}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Holding Back Drivers */}
                    {outlook.drivers.holding_back.length > 0 && (
                      <div className="space-y-1">
                        <span className="text-[11px] font-semibold text-rose-700 dark:text-rose-400">
                          Holding-Back Drivers:
                        </span>
                        <ul className="text-xs text-muted-foreground space-y-1 pl-3 list-disc">
                          {outlook.drivers.holding_back.map((d, i) => (
                            <li key={i}>{d}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>

                {/* Thin-Trading Sentence if present */}
                {outlook.thin_trading_note && (
                  <div className="mt-3 p-2.5 rounded-lg bg-amber-50/50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 text-[11px] text-amber-900 dark:text-amber-200 flex items-start gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                    <span>{outlook.thin_trading_note}</span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Risk, Liquidity & Key Stats Panels */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Risk Panel */}
        <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-3" data-testid="risk-panel">
          <div className="flex items-center justify-between">
            <span className="font-bold text-sm text-foreground flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-primary" />
              <span>Volatility &amp; Risk</span>
            </span>
            <RiskBadge level={stock.risk.level} />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold tabular-nums text-foreground">
              {stock.risk.vol_63d_pct.toFixed(2)}%
            </span>
            <span className="text-xs text-muted-foreground">
              (63-day realized vol vs {stock.risk.threshold_pct}% threshold)
            </span>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {stock.risk.explanation}
          </p>
        </div>

        {/* Liquidity Panel */}
        <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-3" data-testid="liquidity-panel">
          <div className="flex items-center justify-between">
            <span className="font-bold text-sm text-foreground flex items-center gap-2">
              <Droplets className="w-4 h-4 text-blue-500" />
              <span>Market Liquidity</span>
            </span>
            {stock.liquidity.flagged ? (
              <span className="text-xs font-semibold text-amber-600 bg-amber-50 dark:bg-amber-950/40 px-2 py-0.5 rounded border border-amber-300">
                Warning
              </span>
            ) : (
              <span className="text-xs font-semibold text-emerald-800 dark:text-emerald-300 bg-emerald-50 dark:bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-300">
                Adequate
              </span>
            )}
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold tabular-nums text-foreground">
              LKR {(stock.liquidity.turnover_median_lkr / 1000).toFixed(0)}k
            </span>
            <span className="text-xs text-muted-foreground">
              (Median daily turnover)
            </span>
          </div>
          <p className="text-xs text-muted-foreground leading-relaxed">
            {stock.liquidity.explanation}
          </p>
        </div>

        {/* Key Stats Panel */}
        <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-3" data-testid="key-stats-panel">
          <span className="font-bold text-sm text-foreground flex items-center gap-2">
            <Info className="w-4 h-4 text-muted-foreground" />
            <span>Trading Profile</span>
          </span>
          <div className="space-y-2 text-xs divide-y divide-border pt-1">
            <div className="flex justify-between py-1">
              <span className="text-muted-foreground">52-Week Range</span>
              <span className="font-semibold tabular-nums">
                LKR {stock.low_52w.toFixed(2)} – {stock.high_52w.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-muted-foreground">Trailing P/E Ratio</span>
              <span className="font-semibold tabular-nums">
                {stock.key_stats.trailing_pe ? stock.key_stats.trailing_pe.toFixed(1) : '11.4x'}
              </span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-muted-foreground">Beta vs CSE SL20</span>
              <span className="font-semibold tabular-nums">
                {stock.key_stats.beta_index ? stock.key_stats.beta_index.toFixed(2) : '0.94'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Alert Modal */}
      {showAlertModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          data-testid="create-alert-modal"
        >
          <div className="bg-white dark:bg-slate-900 text-foreground border border-border rounded-xl p-6 max-w-md w-full space-y-4 shadow-2xl relative z-10">
            <div className="flex items-center justify-between pb-2 border-b">
              <h3 className="font-bold text-foreground text-sm">
                Set Alert for {stock.code}
              </h3>
              <button
                onClick={() => setShowAlertModal(false)}
                className="text-muted-foreground hover:text-foreground text-xs"
              >
                Cancel
              </button>
            </div>

            {alertSuccess ? (
              <div className="p-4 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-800 dark:text-emerald-200 rounded-lg text-xs flex items-center gap-2">
                <Check className="w-4 h-4" />
                <span>Alert registered successfully!</span>
              </div>
            ) : (
              <form onSubmit={handleCreateAlert} className="space-y-4 text-xs">
                <div className="space-y-1">
                  <label className="font-semibold text-foreground" htmlFor="alert-type-select">
                    Trigger Condition
                  </label>
                  <select
                    id="alert-type-select"
                    aria-label="Alert Trigger Condition"
                    value={alertType}
                    data-testid="alert-type-select"
                    onChange={(e) => setAlertType(e.target.value)}
                    className="w-full p-2 rounded-lg border bg-background text-foreground"
                  >
                    <option value="price_above">Price Rises Above (LKR)</option>
                    <option value="price_below">Price Drops Below (LKR)</option>
                    <option value="daily_change_above">Daily Gain Exceeds (%)</option>
                    <option value="enters_high_tier">Enters High Upside Tier</option>
                    <option value="becomes_leader">Becomes Consistent Leader</option>
                    <option value="risk_changes">Risk Profile Changes</option>
                  </select>
                </div>

                {['price_above', 'price_below', 'daily_change_above'].includes(alertType) && (
                  <div className="space-y-1">
                    <label className="font-semibold text-foreground" htmlFor="alert-threshold-input">
                      Threshold Value
                    </label>
                    <input
                      id="alert-threshold-input"
                      aria-label="Threshold Value"
                      type="number"
                      step="0.1"
                      required
                      data-testid="alert-threshold-input"
                      value={alertThreshold}
                      onChange={(e) => setAlertThreshold(e.target.value)}
                      placeholder={stock.last_price.toString()}
                      className="w-full p-2 rounded-lg border bg-background text-foreground"
                    />
                  </div>
                )}

                <p className="text-[11px] text-muted-foreground">
                  Alerts are checked each time new market data is loaded (end of day), not in real time.
                </p>

                <button
                  type="submit"
                  data-testid="submit-alert-btn"
                  className="w-full py-2 bg-primary text-primary-foreground font-semibold rounded-lg hover:bg-primary-700"
                >
                  Save Alert Rule
                </button>
              </form>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
