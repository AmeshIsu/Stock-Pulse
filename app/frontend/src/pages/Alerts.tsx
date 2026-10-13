import React, { useState, useEffect } from 'react';
import { apiClient } from '../lib/api';
import { AlertRule, Horizon, RankingsResponse } from '../lib/types';
import {
  Bell,
  Plus,
  Trash2,
  Edit,
  Check,
  AlertCircle,
  ToggleLeft,
  ToggleRight,
  Clock,
} from 'lucide-react';

export const Alerts: React.FC = () => {
  const [alerts, setAlerts] = useState<AlertRule[]>([]);
  const [allStocks, setAllStocks] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Modal State
  const [showModal, setShowModal] = useState(false);
  const [editingAlertId, setEditingAlertId] = useState<string | null>(null);

  // Form Fields
  const [stockCode, setStockCode] = useState('');
  const [alertType, setAlertType] = useState<AlertRule['alert_type']>('price_above');
  const [thresholdVal, setThresholdVal] = useState('');
  const [selectedHorizon, setSelectedHorizon] = useState<Horizon>('short');
  const [cooldownDays, setCooldownDays] = useState(3);

  const loadAlerts = async () => {
    setIsLoading(true);
    try {
      const [rules, rankings] = await Promise.all([
        apiClient<AlertRule[]>('/alerts'),
        apiClient<RankingsResponse>('/rankings'),
      ]);
      setAlerts(rules || []);
      setAllStocks(rankings?.rankings?.map((r) => r.code) || []);
      if (!stockCode && rankings?.rankings?.[0]?.code) {
        setStockCode(rankings.rankings[0].code);
      }
    } catch (err) {
      console.error('Failed to load alerts:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadAlerts();
  }, []);

  const handleOpenCreate = () => {
    setEditingAlertId(null);
    setAlertType('price_above');
    setThresholdVal('');
    setSelectedHorizon('short');
    setCooldownDays(3);
    setShowModal(true);
  };

  const handleOpenEdit = (rule: AlertRule) => {
    setEditingAlertId(rule.id);
    setStockCode(rule.stock_code);
    setAlertType(rule.alert_type);
    setThresholdVal(rule.threshold_value?.toString() || '');
    setSelectedHorizon(rule.horizon || 'short');
    setCooldownDays(rule.cooldown_days || 3);
    setShowModal(true);
  };

  const handleToggleActive = async (rule: AlertRule) => {
    try {
      await apiClient(`/alerts/${rule.id}`, {
        method: 'PUT',
        body: JSON.stringify({ is_active: !rule.is_active }),
      });
      setAlerts((prev) =>
        prev.map((a) => (a.id === rule.id ? { ...a, is_active: !a.is_active } : a))
      );
    } catch {
      alert('Failed to update alert state.');
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await apiClient(`/alerts/${id}`, { method: 'DELETE' });
      setAlerts((prev) => prev.filter((a) => a.id !== id));
    } catch {
      alert('Failed to delete alert.');
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const payload: Partial<AlertRule> = {
      stock_code: stockCode,
      alert_type: alertType,
      cooldown_days: Number(cooldownDays) || 3,
      threshold_value: ['price_above', 'price_below', 'daily_change_above', 'daily_change_below'].includes(alertType)
        ? parseFloat(thresholdVal)
        : undefined,
      horizon: alertType === 'enters_high_tier' ? selectedHorizon : undefined,
    };

    try {
      if (editingAlertId) {
        await apiClient(`/alerts/${editingAlertId}`, {
          method: 'PUT',
          body: JSON.stringify(payload),
        });
      } else {
        await apiClient('/alerts', {
          method: 'POST',
          body: JSON.stringify(payload),
        });
      }
      setShowModal(false);
      loadAlerts();
    } catch {
      alert('Error saving alert rule.');
    }
  };

  return (
    <div className="space-y-6" data-testid="alerts-page">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground flex items-center gap-2">
            <Bell className="w-6 h-6 text-primary" />
            <span>Alerts &amp; Notifications</span>
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Configure automated signals for price movements, tier migrations, and universe volatility shifts
          </p>
        </div>

        <button
          type="button"
          data-testid="create-new-alert-btn"
          onClick={handleOpenCreate}
          className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-primary text-primary-foreground text-xs font-semibold rounded-lg hover:bg-primary-700 shadow-sm transition"
        >
          <Plus className="w-4 h-4" />
          <span>New Alert</span>
        </button>
      </div>

      {/* Helper Disclaimer Notice */}
      <div className="p-3 bg-muted/40 border border-border rounded-lg flex items-center gap-2.5 text-xs text-muted-foreground">
        <Clock className="w-4 h-4 shrink-0 text-primary" />
        <span>
          Alerts are checked each time new market data is loaded (end of day), not in real time.
        </span>
      </div>

      {/* Alerts List */}
      {alerts.length === 0 && !isLoading ? (
        <div className="bg-card border border-border rounded-xl p-12 text-center space-y-3" data-testid="alerts-empty-state">
          <Bell className="w-10 h-10 mx-auto text-muted-foreground opacity-30" />
          <h3 className="font-bold text-foreground">No active alerts configured</h3>
          <p className="text-xs text-muted-foreground max-w-sm mx-auto">
            Create automated threshold checks to receive notifications when a stock enters High upside or exceeds target prices.
          </p>
        </div>
      ) : (
        <div className="bg-card border border-border rounded-xl shadow-sm overflow-hidden" data-testid="alerts-table-container">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-muted/40 border-b text-muted-foreground font-semibold uppercase tracking-wider text-[11px]">
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4">Stock</th>
                <th className="py-3 px-4">Condition</th>
                <th className="py-3 px-4">Threshold</th>
                <th className="py-3 px-4">Cooldown</th>
                <th className="py-3 px-4 text-center">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {alerts.map((rule) => (
                <tr key={rule.id} data-testid={`alert-row-${rule.id}`} className="hover:bg-muted/30">
                  <td className="py-3 px-4">
                    <button
                      type="button"
                      data-testid={`toggle-alert-status-${rule.id}`}
                      onClick={() => handleToggleActive(rule)}
                      className={`inline-flex items-center gap-1.5 font-semibold text-xs ${
                        rule.is_active ? 'text-emerald-600' : 'text-muted-foreground'
                      }`}
                    >
                      {rule.is_active ? (
                        <>
                          <ToggleRight className="w-5 h-5 text-primary" />
                          <span>Active</span>
                        </>
                      ) : (
                        <>
                          <ToggleLeft className="w-5 h-5 text-muted-foreground" />
                          <span>Paused</span>
                        </>
                      )}
                    </button>
                  </td>
                  <td className="py-3 px-4 font-bold text-primary">{rule.stock_code}</td>
                  <td className="py-3 px-4 capitalize">
                    {rule.alert_type.replace(/_/g, ' ')}
                    {rule.horizon && ` (${rule.horizon})`}
                  </td>
                  <td className="py-3 px-4 tabular-nums font-semibold">
                    {rule.threshold_value !== undefined ? rule.threshold_value : '—'}
                  </td>
                  <td className="py-3 px-4 tabular-nums text-muted-foreground">
                    {rule.cooldown_days} days
                  </td>
                  <td className="py-3 px-4 text-center">
                    <div className="flex items-center justify-center gap-2">
                      <button
                        type="button"
                        data-testid={`edit-alert-btn-${rule.id}`}
                        onClick={() => handleOpenEdit(rule)}
                        className="p-1 text-muted-foreground hover:text-primary"
                        aria-label="Edit alert"
                      >
                        <Edit className="w-3.5 h-3.5" />
                      </button>
                      <button
                        type="button"
                        data-testid={`delete-alert-btn-${rule.id}`}
                        onClick={() => handleDelete(rule.id)}
                        className="p-1 text-muted-foreground hover:text-rose-600"
                        aria-label="Delete alert"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Create / Edit Alert Modal */}
      {showModal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          data-testid="alert-config-modal"
        >
          <div className="bg-white dark:bg-slate-900 text-foreground border border-border rounded-xl p-6 max-w-md w-full space-y-4 shadow-2xl relative z-10 animate-in zoom-in-95">
            <div className="flex items-center justify-between pb-2 border-b">
              <h3 className="font-bold text-foreground text-sm">
                {editingAlertId ? 'Edit Alert Rule' : 'Create New Alert Rule'}
              </h3>
              <button
                onClick={() => setShowModal(false)}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                Cancel
              </button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4 text-xs">
              <div className="space-y-1">
                <label className="font-semibold text-foreground" htmlFor="modal-stock-select">
                  Select Stock
                </label>
                <select
                  id="modal-stock-select"
                  aria-label="Select Stock"
                  value={stockCode}
                  data-testid="modal-stock-select"
                  onChange={(e) => setStockCode(e.target.value)}
                  className="w-full p-2 rounded-lg border bg-background text-foreground"
                >
                  {allStocks.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-1">
                <label className="font-semibold text-foreground" htmlFor="modal-type-select">
                  Alert Condition
                </label>
                <select
                  id="modal-type-select"
                  aria-label="Alert Condition"
                  value={alertType}
                  data-testid="modal-type-select"
                  onChange={(e) => setAlertType(e.target.value as AlertRule['alert_type'])}
                  className="w-full p-2 rounded-lg border bg-background text-foreground"
                >
                  <option value="price_above">Price Rises Above (LKR)</option>
                  <option value="price_below">Price Drops Below (LKR)</option>
                  <option value="daily_change_above">Daily Gain Exceeds (%)</option>
                  <option value="daily_change_below">Daily Loss Exceeds (%)</option>
                  <option value="enters_high_tier">Enters High Upside Tier</option>
                  <option value="becomes_leader">Becomes Consistent Leader</option>
                  <option value="risk_changes">Risk Profile Changes (Normal &harr; Elevated)</option>
                </select>
              </div>

              {alertType === 'enters_high_tier' && (
                <div className="space-y-1">
                  <label className="font-semibold text-foreground" htmlFor="modal-horizon-select">
                    Target Horizon
                  </label>
                  <select
                    id="modal-horizon-select"
                    aria-label="Target Horizon"
                    value={selectedHorizon}
                    data-testid="modal-horizon-select"
                    onChange={(e) => setSelectedHorizon(e.target.value as Horizon)}
                    className="w-full p-2 rounded-lg border bg-background text-foreground"
                  >
                    <option value="short">Short term (about 1 month)</option>
                    <option value="mid">Mid term (about 3 months)</option>
                    <option value="long">Long term (about 12 months)</option>
                  </select>
                </div>
              )}

              {['price_above', 'price_below', 'daily_change_above', 'daily_change_below'].includes(alertType) && (
                <div className="space-y-1">
                  <label className="font-semibold text-foreground" htmlFor="modal-threshold-input">
                    Threshold Value
                  </label>
                  <input
                    id="modal-threshold-input"
                    aria-label="Threshold Value"
                    type="number"
                    step="0.1"
                    required
                    data-testid="modal-threshold-input"
                    value={thresholdVal}
                    onChange={(e) => setThresholdVal(e.target.value)}
                    placeholder="e.g. 110.0"
                    className="w-full p-2 rounded-lg border bg-background"
                  />
                </div>
              )}

              <div className="space-y-1">
                <label className="font-semibold text-foreground" htmlFor="modal-cooldown-input">
                  Cooldown Period (Days)
                </label>
                <input
                  id="modal-cooldown-input"
                  aria-label="Cooldown Period (Days)"
                  type="number"
                  min="1"
                  max="30"
                  data-testid="modal-cooldown-input"
                  value={cooldownDays}
                  onChange={(e) => setCooldownDays(parseInt(e.target.value) || 1)}
                  className="w-full p-2 rounded-lg border bg-background"
                />
                <span className="text-[10px] text-muted-foreground block">
                  Avoids repeated alerts on consecutive market days.
                </span>
              </div>

              <div className="pt-2">
                <button
                  type="submit"
                  data-testid="save-alert-rule-btn"
                  className="w-full py-2.5 bg-primary text-primary-foreground font-semibold rounded-lg hover:bg-primary-700 shadow-sm"
                >
                  Save Alert Rule
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
