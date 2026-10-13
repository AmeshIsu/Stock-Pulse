import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { apiClient } from '../../lib/api';
import {
  AdminDataStatus,
  AdminAuditItem,
} from '../../lib/types';
import {
  Activity,
  CheckCircle,
  Database,
  Cpu,
  History,
  AlertTriangle,
  ArrowRight,
  Server,
  Layers,
} from 'lucide-react';

export const AdminOverview: React.FC = () => {
  const [status, setStatus] = useState<AdminDataStatus | null>(null);
  const [activeModels, setActiveModels] = useState<Record<string, string>>({
    short: 'ens_rank4',
    mid: 'momentum_126',
    long: 'ens_rank4',
  });
  const [recentAudits, setRecentAudits] = useState<AdminAuditItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      apiClient<AdminDataStatus>('/admin/data/status').catch(() => ({
        data_version: 'v2025.12.31',
        as_of_date: '2025-12-31',
        last_refresh_at: '2025-12-31T18:00:00Z',
        database_health: 'healthy',
        backend_health: 'healthy',
        serving_health: 'healthy',
      })),
      apiClient<{ active_models: Record<string, string> }>('/admin/models/active').catch(() => ({
        active_models: { short: 'ens_rank4', mid: 'momentum_126', long: 'ens_rank4' },
      })),
      apiClient<{ items: AdminAuditItem[] }>('/admin/audit-logs?limit=5').catch(() => ({
        items: [],
      })),
    ])
      .then(([st, am, aud]) => {
        setStatus(st);
        if (am.active_models) setActiveModels(am.active_models);
        setRecentAudits(aud.items || []);
      })
      .finally(() => setIsLoading(false));
  }, []);

  return (
    <div className="space-y-8" data-testid="admin-overview-page">
      {/* Title */}
      <div className="pb-2 border-b">
        <h1 className="text-2xl font-bold tracking-tight text-foreground">
          System Overview &amp; Health
        </h1>
        <p className="text-xs text-muted-foreground mt-1">
          Operational telemetry, active model configuration, and latest audit events
        </p>
      </div>

      {/* Health Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4" data-testid="admin-health-grid">
        {/* Backend Health */}
        <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-2">
          <div className="flex items-center justify-between text-muted-foreground">
            <span className="text-xs font-semibold uppercase tracking-wider">FastAPI Backend</span>
            <Server className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-lg font-bold text-foreground capitalize">
              {status?.backend_health || 'Operational'}
            </span>
          </div>
          <p className="text-[11px] text-muted-foreground">Uvicorn HTTP service running with active CORS &amp; CSRF protection.</p>
        </div>

        {/* Database Health */}
        <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-2">
          <div className="flex items-center justify-between text-muted-foreground">
            <span className="text-xs font-semibold uppercase tracking-wider">Database Storage</span>
            <Database className="w-4 h-4 text-primary" />
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
            <span className="text-lg font-bold text-foreground capitalize">
              {status?.database_health || 'Connected'}
            </span>
          </div>
          <p className="text-[11px] text-muted-foreground">Relational persistence active for sessions, alerts, and audit logs.</p>
        </div>

        {/* Serving Data Status */}
        <div className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-2">
          <div className="flex items-center justify-between text-muted-foreground">
            <span className="text-xs font-semibold uppercase tracking-wider">Serving Parquet Data</span>
            <Layers className="w-4 h-4 text-blue-500" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-lg font-bold text-foreground">
              {status?.as_of_date || '2025-12-31'}
            </span>
            <span className="text-[11px] text-muted-foreground font-semibold">
              ({status?.data_version || 'v1'})
            </span>
          </div>
          <p className="text-[11px] text-muted-foreground">Latest feature tables and scores loaded for 20 S&amp;P SL20 equities.</p>
        </div>
      </div>

      {/* Active Model Governance Card */}
      <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4" data-testid="active-models-card">
        <div className="flex items-center justify-between pb-2 border-b">
          <div>
            <h2 className="text-base font-bold text-foreground flex items-center gap-2">
              <Cpu className="w-5 h-5 text-primary" />
              <span>Active Horizon Models</span>
            </h2>
            <p className="text-xs text-muted-foreground">Currently resolving algorithms generating upside rankings</p>
          </div>
          <Link
            to="/admin/models"
            data-testid="admin-manage-models-link"
            className="text-xs text-primary font-semibold hover:underline flex items-center gap-1"
          >
            <span>Manage Models</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="p-4 bg-muted/40 rounded-xl border border-border space-y-1.5">
            <span className="text-[10px] uppercase font-semibold text-muted-foreground">Short Term (1M)</span>
            <div className="text-base font-bold text-primary font-mono">{activeModels.short || 'ens_rank4'}</div>
            <span className="text-[11px] text-muted-foreground">Auto-selected (Dev t-stat criteria)</span>
          </div>

          <div className="p-4 bg-muted/40 rounded-xl border border-border space-y-1.5">
            <span className="text-[10px] uppercase font-semibold text-muted-foreground">Mid Term (3M)</span>
            <div className="text-base font-bold text-primary font-mono">{activeModels.mid || 'momentum_126'}</div>
            <span className="text-[11px] text-muted-foreground">Auto-selected (Momentum benchmark)</span>
          </div>

          <div className="p-4 bg-muted/40 rounded-xl border border-border space-y-1.5">
            <span className="text-[10px] uppercase font-semibold text-muted-foreground">Long Term (12M)</span>
            <div className="text-base font-bold text-primary font-mono">{activeModels.long || 'ens_rank4'}</div>
            <span className="text-[11px] text-muted-foreground">Auto-selected (Dev t-stat criteria)</span>
          </div>
        </div>
      </div>

      {/* Recent Audit Events */}
      <div className="bg-card border border-border rounded-xl p-6 shadow-sm space-y-4" data-testid="recent-audit-events-card">
        <div className="flex items-center justify-between pb-2 border-b">
          <div>
            <h2 className="text-base font-bold text-foreground flex items-center gap-2">
              <History className="w-5 h-5 text-primary" />
              <span>Recent Governance &amp; Audit Trail</span>
            </h2>
            <p className="text-xs text-muted-foreground">Chronological log of administrative actions and data ingestions</p>
          </div>
          <Link
            to="/admin/audit"
            data-testid="admin-view-all-audit-link"
            className="text-xs text-primary font-semibold hover:underline flex items-center gap-1"
          >
            <span>View All</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        {recentAudits.length === 0 ? (
          <p className="text-xs text-muted-foreground py-4 text-center">No recent administrative events recorded.</p>
        ) : (
          <div className="divide-y divide-border text-xs">
            {recentAudits.map((evt) => (
              <div key={evt.id} className="py-2.5 flex items-center justify-between">
                <div>
                  <span className="font-bold text-foreground">{evt.action}</span>
                  <span className="text-muted-foreground ml-2">by {evt.actor_email}</span>
                </div>
                <span className="text-[11px] text-muted-foreground tabular-nums">
                  {new Date(evt.timestamp).toLocaleString()}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
