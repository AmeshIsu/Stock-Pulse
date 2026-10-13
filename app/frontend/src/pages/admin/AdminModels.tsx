import React, { useState, useEffect } from 'react';
import { apiClient, ApiError } from '../../lib/api';
import { Horizon } from '../../lib/types';
import {
  Cpu,
  ShieldAlert,
  AlertTriangle,
  RotateCcw,
  CheckCircle,
  X,
  Layers,
  ChevronRight,
  Info,
  Check,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';

interface HorizonMetricItem {
  dev_mean_ic?: number;
  hit_rate?: number;
  hit_rate_ci?: [number, number];
  folds_positive_count?: number;
  folds_total_count?: number;
  fold_t_stat?: number;
  raw_p_value?: number;
  holm_adjusted_p_value?: number;
  evidence_level?: 'GREEN' | 'AMBER' | 'RED';
  net_excess?: number;
  holdout_ic?: number;
  holdout_n_eff?: number;
  holdout_status?: string;
  production_config_metrics?: {
    dev_mean_ic?: number;
    fold_t_stat?: number;
    folds_positive_count?: number;
    folds_total_count?: number;
    top_tier_hit_rate?: number;
  };
  per_fold_ic?: Record<string, number>;
  is_auto_default?: boolean;
  is_currently_active?: boolean;
}

interface ModelItem {
  model_id: string;
  admin_display_name: string;
  family: string;
  short_description: string;
  servable: boolean;
  servable_reason?: string;
  horizons: Record<Horizon, HorizonMetricItem>;
}

interface AdminModelRegistryResponse {
  active_models: Record<Horizon, string>;
  models: ModelItem[];
}

interface ModelPreviewData {
  model_id: string;
  horizon: string;
  active_model_id: string;
  top_5_model: string[];
  top_5_active: string[];
  changed_count: number;
}

export const AdminModels: React.FC = () => {
  const [data, setData] = useState<AdminModelRegistryResponse | null>(null);
  const [activeHorizon, setActiveHorizon] = useState<Horizon>('short');
  const [isLoading, setIsLoading] = useState(true);

  // Drawer state for model details
  const [selectedModel, setSelectedModel] = useState<ModelItem | null>(null);

  // Model switch confirmation modal state
  const [switchTarget, setSwitchTarget] = useState<{ model: ModelItem; horizon: Horizon } | null>(null);
  const [switchReason, setSwitchReason] = useState('');
  const [previewData, setPreviewData] = useState<ModelPreviewData | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [ackWarning, setAckWarning] = useState(false);
  const [switchError, setSwitchError] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const loadModels = async () => {
    setIsLoading(true);
    try {
      const res = await apiClient<AdminModelRegistryResponse>('/admin/models');
      setData(res);
    } catch (err) {
      console.error('Failed to load admin models:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const handleOpenSwitchModal = async (model: ModelItem, horizon: Horizon) => {
    setSwitchTarget({ model, horizon });
    setSwitchReason('');
    setAckWarning(false);
    setSwitchError(null);
    setPreviewLoading(true);

    try {
      const p = await apiClient<ModelPreviewData>(
        `/admin/models/${model.model_id}/preview?horizon=${horizon}`
      );
      setPreviewData(p);
    } catch (err) {
      console.error('Failed to load preview:', err);
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleConfirmSwitch = async () => {
    if (!switchTarget) return;
    if (switchReason.trim().length < 5) {
      setSwitchError('Please enter a valid justification reason (at least 5 characters).');
      return;
    }

    setSwitchError(null);
    try {
      await apiClient('/admin/model-settings', {
        method: 'PUT',
        body: JSON.stringify({
          horizon: switchTarget.horizon,
          mode: 'manual',
          model_id: switchTarget.model.model_id,
          confirm: true,
          reason: switchReason,
          acknowledge_warning: ackWarning,
        }),
      });

      setToastMessage(
        `Successfully activated ${switchTarget.model.admin_display_name} for ${switchTarget.horizon} horizon.`
      );
      setSwitchTarget(null);
      loadModels();
      setTimeout(() => setToastMessage(null), 4000);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        setSwitchError(err.message);
      } else {
        setSwitchError('Failed to execute model switch.');
      }
    }
  };

  const handleResetToAuto = async (horizon?: Horizon) => {
    try {
      const horizonsToReset = horizon ? [horizon] : (['short', 'mid', 'long'] as Horizon[]);
      for (const h of horizonsToReset) {
        await apiClient('/admin/model-settings', {
          method: 'PUT',
          body: JSON.stringify({
            horizon: h,
            mode: 'auto',
            confirm: true,
            reason: 'Administrative reset to auto-selected default',
          }),
        });
      }
      setToastMessage(`Reset ${horizon ? `${horizon} horizon` : 'all horizons'} to Auto model.`);
      loadModels();
      setTimeout(() => setToastMessage(null), 4000);
    } catch {
      alert('Failed to reset to auto model.');
    }
  };

  // Determine if chosen model needs warning relative to Auto model
  const autoModelId = data?.models.find(
    (m) => m.horizons[activeHorizon]?.is_auto_default
  )?.model_id;

  const autoMetrics = data?.models.find((m) => m.model_id === autoModelId)?.horizons[activeHorizon];
  const targetMetrics = switchTarget?.model.horizons[switchTarget.horizon];

  const needsWarning =
    targetMetrics &&
    autoMetrics &&
    ((targetMetrics.evidence_level === 'RED' && autoMetrics.evidence_level !== 'RED') ||
      (targetMetrics.evidence_level === 'AMBER' && autoMetrics.evidence_level === 'GREEN') ||
      (targetMetrics.fold_t_stat || 0) < (autoMetrics.fold_t_stat || 0));

  // Transform per_fold_ic to bar chart data
  const foldChartData = selectedModel?.horizons[activeHorizon]?.per_fold_ic
    ? Object.entries(selectedModel.horizons[activeHorizon].per_fold_ic!).map(
        ([fold, ic]) => ({
          fold,
          ic: parseFloat(ic.toFixed(4)),
        })
      )
    : [];

  return (
    <div className="space-y-6" data-testid="admin-models-page">
      {/* Toast Alert */}
      {toastMessage && (
        <div
          className="fixed top-4 right-4 z-50 p-4 rounded-xl bg-emerald-800 text-white shadow-xl flex items-center gap-2 text-xs font-semibold animate-in slide-in-from-top-2"
          role="status"
          data-testid="admin-toast-message"
        >
          <CheckCircle className="w-4 h-4" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <Cpu className="w-6 h-6 text-primary" />
            <span>Model Governance &amp; Registry</span>
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Statistical validation metrics, active algorithm overrides, and pre-registered serving defaults
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            data-testid="reset-all-auto-btn"
            onClick={() => handleResetToAuto()}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-card text-xs font-semibold text-foreground hover:bg-muted transition"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset All to Auto</span>
          </button>
        </div>
      </div>

      {/* Mandatory Statutory Governance Banner */}
      <div
        className="p-4 bg-amber-50 dark:bg-amber-950/40 border border-amber-300 dark:border-amber-800 rounded-xl flex items-start gap-3 text-xs text-amber-900 dark:text-amber-200"
        role="alert"
        data-testid="governance-warning-banner"
      >
        <ShieldAlert className="w-5 h-5 shrink-0 text-amber-600 dark:text-amber-400 mt-0.5" />
        <div className="space-y-1">
          <span className="font-bold block">Methodological Notice &amp; Multiple-Testing Warning</span>
          <p className="leading-relaxed">
            No model shows statistically reliable predictive power in our testing; metrics are historical, and choosing among many models carries a multiple-testing bias.
          </p>
        </div>
      </div>

      {/* Plain-English Legend */}
      <div className="p-3 bg-muted/30 border border-border rounded-lg text-xs text-muted-foreground flex items-center gap-2" data-testid="hit-rate-legend">
        <Info className="w-4 h-4 text-primary shrink-0" />
        <span>
          <strong>Hit Rate:</strong> The share of High-tier stock-dates, using non-overlapping rebalance periods, whose forward return beat that date's cross-sectional median, on development out-of-sample predictions only.
        </span>
      </div>

      {/* Horizon Tabs & Active Reset */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-1 bg-muted p-1 rounded-lg">
          {(['short', 'mid', 'long'] as Horizon[]).map((h) => (
            <button
              key={h}
              type="button"
              data-testid={`admin-tab-${h}`}
              onClick={() => setActiveHorizon(h)}
              className={`px-3.5 py-1.5 rounded-md text-xs font-semibold transition ${
                activeHorizon === h
                  ? 'bg-card text-foreground shadow-xs'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <span className="capitalize">{h} Horizon</span>
            </button>
          ))}
        </div>

        <button
          type="button"
          data-testid={`reset-current-auto-btn-${activeHorizon}`}
          onClick={() => handleResetToAuto(activeHorizon)}
          className="text-xs text-primary font-semibold hover:underline inline-flex items-center gap-1"
        >
          <RotateCcw className="w-3 h-3" />
          <span>Reset {activeHorizon} to Auto Default</span>
        </button>
      </div>

      {/* Models Table */}
      <div className="bg-card border border-border rounded-xl shadow-sm overflow-hidden" data-testid="admin-models-table-container">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-muted/40 border-b text-muted-foreground font-semibold uppercase tracking-wider text-[11px]">
                <th className="py-3 px-3">Model</th>
                <th className="py-3 px-3">Family</th>
                <th className="py-3 px-2 text-center">Status</th>
                <th className="py-3 px-3">Evidence</th>
                <th className="py-3 px-3 text-right">Dev IC</th>
                <th className="py-3 px-3 text-right">Hit Rate (90% CI)</th>
                <th className="py-3 px-3 text-center">Positive Folds</th>
                <th className="py-3 px-3 text-right">t-stat</th>
                <th className="py-3 px-3 text-right">Holm p</th>
                <th className="py-3 px-3 text-right">Prod IC</th>
                <th className="py-3 px-3 text-right">Holdout</th>
                <th className="py-3 px-3 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {data?.models.map((m) => {
                const metric = m.horizons[activeHorizon];
                if (!metric) return null;

                const isActive = metric.is_currently_active;
                const isAuto = metric.is_auto_default;

                return (
                  <tr
                    key={m.model_id}
                    data-testid={`admin-model-row-${m.model_id}`}
                    onClick={() => setSelectedModel(m)}
                    className="hover:bg-muted/30 cursor-pointer transition-colors"
                  >
                    <td className="py-3 px-3">
                      <div className="font-bold text-foreground font-mono">{m.admin_display_name}</div>
                      <span className="text-[10px] text-muted-foreground font-mono">{m.model_id}</span>
                    </td>
                    <td className="py-3 px-3 text-muted-foreground">{m.family}</td>
                    <td className="py-3 px-2 text-center space-y-1">
                      {isActive && (
                        <span
                          data-testid={`badge-active-${m.model_id}`}
                          className="inline-block px-1.5 py-0.5 rounded text-[10px] font-bold bg-primary text-primary-foreground"
                        >
                          ACTIVE
                        </span>
                      )}
                      {isAuto && (
                        <span
                          data-testid={`badge-auto-${m.model_id}`}
                          className="inline-block px-1.5 py-0.5 rounded text-[10px] font-bold bg-emerald-800 text-white ml-1"
                        >
                          AUTO
                        </span>
                      )}
                      {!m.servable && (
                        <span className="inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold bg-muted text-muted-foreground">
                          Non-Servable
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-3">
                      <span
                        className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold ${
                          metric.evidence_level === 'GREEN'
                            ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300'
                            : metric.evidence_level === 'AMBER'
                            ? 'bg-amber-100 text-amber-800 dark:bg-amber-950/40 dark:text-amber-300'
                            : 'bg-rose-100 text-rose-800 dark:bg-rose-950/40 dark:text-rose-300'
                        }`}
                      >
                        {metric.evidence_level || 'RED'}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-right font-semibold tabular-nums">
                      {metric.dev_mean_ic != null ? metric.dev_mean_ic.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums">
                      {metric.hit_rate != null ? (
                        <div>
                          <span className="font-semibold">{(metric.hit_rate * 100).toFixed(2)}%</span>
                          {metric.hit_rate_ci && (
                            <span className="text-[10px] text-muted-foreground block">
                              [{(metric.hit_rate_ci[0] * 100).toFixed(1)}%, {(metric.hit_rate_ci[1] * 100).toFixed(1)}%]
                            </span>
                          )}
                        </div>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="py-3 px-3 text-center tabular-nums font-semibold">
                      {metric.folds_positive_count != null && metric.folds_total_count != null
                        ? `${metric.folds_positive_count}/${metric.folds_total_count}`
                        : '—'}
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums font-semibold">
                      {metric.fold_t_stat != null ? metric.fold_t_stat.toFixed(2) : '—'}
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums text-muted-foreground">
                      {metric.holm_adjusted_p_value != null
                        ? metric.holm_adjusted_p_value.toFixed(4)
                        : '—'}
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums">
                      {metric.production_config_metrics?.dev_mean_ic != null
                        ? metric.production_config_metrics.dev_mean_ic.toFixed(4)
                        : '—'}
                    </td>
                    <td className="py-3 px-3 text-right tabular-nums">
                      {metric.holdout_ic != null ? (
                        <div>
                          <span>{metric.holdout_ic.toFixed(4)}</span>
                          <span className="text-[10px] text-muted-foreground block">
                            (n_eff: {metric.holdout_n_eff || 2})
                          </span>
                        </div>
                      ) : (
                        '—'
                      )}
                    </td>
                    <td className="py-3 px-3 text-center" onClick={(e) => e.stopPropagation()}>
                      {m.servable && !isActive ? (
                        <button
                          type="button"
                          data-testid={`activate-model-btn-${m.model_id}`}
                          onClick={() => handleOpenSwitchModal(m, activeHorizon)}
                          className="px-2.5 py-1 bg-primary text-primary-foreground font-semibold rounded text-[11px] hover:bg-primary-700 shadow-xs"
                        >
                          Use Model
                        </button>
                      ) : (
                        <span className="text-[11px] text-muted-foreground">
                          {isActive ? 'Active' : 'Locked'}
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Model Inspection Drawer */}
      {selectedModel && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-end bg-black/50"
          role="dialog"
          aria-modal="true"
          data-testid="admin-model-drawer"
        >
          <div className="bg-card border-l border-border h-full w-full max-w-xl p-6 overflow-y-auto space-y-6 shadow-2xl animate-in slide-in-from-right duration-200">
            <div className="flex items-center justify-between pb-3 border-b">
              <div>
                <h3 className="text-base font-bold text-foreground font-mono">
                  {selectedModel.admin_display_name}
                </h3>
                <span className="text-xs text-muted-foreground">{selectedModel.family} Family</span>
              </div>
              <button
                type="button"
                data-testid="close-model-drawer-btn"
                onClick={() => setSelectedModel(null)}
                className="p-1 text-muted-foreground hover:text-foreground"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <p className="text-xs text-muted-foreground leading-relaxed">
              {selectedModel.short_description || 'No detailed architectural notes recorded.'}
            </p>

            {/* Annual Fold IC Chart */}
            {foldChartData.length > 0 && (
              <div className="space-y-2">
                <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground block">
                  Per-Fold Annual Rank Correlation ({activeHorizon})
                </span>
                <div className="h-48 w-full pt-2 bg-muted/20 rounded-xl p-2 border">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={foldChartData}>
                      <XAxis dataKey="fold" fontSize={10} stroke="#888888" />
                      <YAxis fontSize={10} stroke="#888888" domain={[-0.1, 0.15]} />
                      <Tooltip />
                      <ReferenceLine y={0} stroke="#666" />
                      <Bar dataKey="ic" fill="#2563eb" radius={[2, 2, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* All-Horizon Metrics Grid */}
            <div className="space-y-2">
              <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground block">
                Cross-Horizon Summary Matrix
              </span>
              <div className="grid grid-cols-3 gap-3 text-xs">
                {(['short', 'mid', 'long'] as Horizon[]).map((h) => {
                  const m = selectedModel.horizons[h];
                  if (!m) return null;

                  return (
                    <div key={h} className="p-3 bg-muted/40 rounded-xl border border-border space-y-1">
                      <span className="font-bold text-foreground uppercase tracking-wide block">
                        {h}
                      </span>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">IC:</span>
                        <span className="font-semibold tabular-nums">{m.dev_mean_ic?.toFixed(4) || '—'}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Hit:</span>
                        <span className="font-semibold tabular-nums">
                          {m.hit_rate ? `${(m.hit_rate * 100).toFixed(1)}%` : '—'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">t-stat:</span>
                        <span className="font-semibold tabular-nums">{m.fold_t_stat?.toFixed(2) || '—'}</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Validated vs Production-Configuration Metrics */}
            <div className="space-y-2">
              <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground block">
                Validated vs. Production Configuration ({activeHorizon})
              </span>
              <div className="p-3 bg-muted/40 rounded-xl border border-border text-xs space-y-2">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Validated Dev IC:</span>
                  <span className="font-bold tabular-nums">
                    {selectedModel.horizons[activeHorizon]?.dev_mean_ic?.toFixed(4) || '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Production-Config Dev IC:</span>
                  <span className="font-bold tabular-nums">
                    {selectedModel.horizons[activeHorizon]?.production_config_metrics?.dev_mean_ic?.toFixed(4) || '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Validated t-stat:</span>
                  <span className="font-bold tabular-nums">
                    {selectedModel.horizons[activeHorizon]?.fold_t_stat?.toFixed(2) || '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Production-Config t-stat:</span>
                  <span className="font-bold tabular-nums">
                    {selectedModel.horizons[activeHorizon]?.production_config_metrics?.fold_t_stat?.toFixed(2) || '—'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Switch Confirmation Modal */}
      {switchTarget && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          data-testid="model-switch-modal"
        >
          <div className="bg-card border border-border rounded-xl shadow-2xl max-w-lg w-full p-6 space-y-5 animate-in zoom-in-95">
            <div className="flex items-center justify-between pb-2 border-b">
              <h3 className="font-bold text-foreground text-sm">
                Confirm Model Switch ({switchTarget.horizon.toUpperCase()})
              </h3>
              <button
                type="button"
                onClick={() => setSwitchTarget(null)}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                Cancel
              </button>
            </div>

            {switchError && (
              <div
                className="p-3 rounded-lg bg-rose-50 dark:bg-rose-950/40 border border-rose-200 text-rose-800 dark:text-rose-200 text-xs flex items-center gap-2"
                role="alert"
                data-testid="switch-error-message"
              >
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{switchError}</span>
              </div>
            )}

            {/* Ranking Preview Section */}
            <div className="p-4 bg-muted/40 rounded-xl border border-border space-y-3 text-xs" data-testid="preview-section">
              <span className="font-bold text-foreground block">Top 5 Picks Impact Preview</span>
              {previewLoading ? (
                <div className="text-xs text-muted-foreground animate-pulse py-2">Computing preview...</div>
              ) : previewData ? (
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Candidate ({switchTarget.model.admin_display_name}):</span>
                    <span className="font-mono font-bold text-primary">
                      {previewData.top_5_model.join(', ')}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Active ({previewData.active_model_id}):</span>
                    <span className="font-mono text-muted-foreground">
                      {previewData.top_5_active.join(', ')}
                    </span>
                  </div>
                  <div className="pt-1 border-t flex justify-between font-semibold">
                    <span>Top 5 Position Shifts:</span>
                    <span className="tabular-nums text-foreground">{previewData.changed_count} changed</span>
                  </div>
                </div>
              ) : null}
            </div>

            {/* Warning Checkbox if Evidence/t-stat is lower than Auto */}
            {needsWarning && (
              <div className="p-3 rounded-xl bg-amber-50 dark:bg-amber-950/40 border border-amber-300 dark:border-amber-800 space-y-2 text-xs">
                <div className="flex items-center gap-2 font-bold text-amber-900 dark:text-amber-200">
                  <AlertTriangle className="w-4 h-4 text-amber-600" />
                  <span>Lower Evidence Override Warning</span>
                </div>
                <p className="text-amber-800 dark:text-amber-300 text-[11px] leading-relaxed">
                  The selected candidate model exhibits a lower evidence tier or lower fold t-statistic than the pre-registered Auto benchmark.
                </p>
                <label className="flex items-start gap-2 pt-1 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    data-testid="acknowledge-warning-checkbox"
                    checked={ackWarning}
                    onChange={(e) => setAckWarning(e.target.checked)}
                    className="mt-0.5 rounded text-primary focus:ring-primary"
                  />
                  <span className="font-semibold text-amber-950 dark:text-amber-100 text-[11px]">
                    I acknowledge that I am overriding the pre-registered Auto selection with a lower-evidence algorithm.
                  </span>
                </label>
              </div>
            )}

            {/* Justification Reason Form */}
            <div className="space-y-1.5 text-xs">
              <label className="font-semibold text-foreground">
                Justification Reason <span className="text-rose-500">*</span>
              </label>
              <textarea
                required
                rows={3}
                data-testid="switch-reason-textarea"
                value={switchReason}
                onChange={(e) => setSwitchReason(e.target.value)}
                placeholder="Explain the operational rationale for this model change..."
                className="w-full p-2.5 rounded-lg border bg-background text-xs"
              />
              <span className="text-[10px] text-muted-foreground block">
                This reason will be committed permanently to the administrative audit trail.
              </span>
            </div>

            <div className="flex justify-end gap-2 pt-2 border-t">
              <button
                type="button"
                onClick={() => setSwitchTarget(null)}
                className="px-3 py-1.5 rounded-lg border text-xs font-semibold text-muted-foreground hover:text-foreground"
              >
                Cancel
              </button>
              <button
                type="button"
                data-testid="confirm-model-switch-btn"
                disabled={needsWarning && !ackWarning}
                onClick={handleConfirmSwitch}
                className="px-4 py-1.5 bg-primary text-primary-foreground text-xs font-semibold rounded-lg hover:bg-primary-700 disabled:opacity-50"
              >
                Confirm Switch
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
