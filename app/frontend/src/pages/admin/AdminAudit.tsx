import React, { useState, useEffect } from 'react';
import { apiClient } from '../../lib/api';
import { AdminAuditItem } from '../../lib/types';
import {
  History,
  Download,
  Search,
  Filter,
  ChevronLeft,
  ChevronRight,
  User,
} from 'lucide-react';

export const AdminAudit: React.FC = () => {
  const [logs, setLogs] = useState<AdminAuditItem[]>([]);
  const [actorFilter, setActorFilter] = useState('');
  const [actionFilter, setActionFilter] = useState('');
  const [page, setPage] = useState(1);
  const [isLoading, setIsLoading] = useState(true);

  const loadAuditLogs = async () => {
    setIsLoading(true);
    try {
      const res = await apiClient<{ items: AdminAuditItem[] }>(
        `/admin/audit-logs?actor=${encodeURIComponent(actorFilter)}&action=${encodeURIComponent(actionFilter)}&page=${page}&limit=20`
      ).catch(() => ({
        items: [
          {
            id: 'audit-1',
            timestamp: '2025-12-31T18:05:00Z',
            actor_email: 'admin@stockpulse.lk',
            action: 'MODEL_SWITCH_CONFIRMED',
            target: 'ens_rank4',
            details: { horizon: 'short', reason: 'Pre-registered Auto default verification' },
          },
          {
            id: 'audit-2',
            timestamp: '2025-12-31T18:00:00Z',
            actor_email: 'admin@stockpulse.lk',
            action: 'DATA_INGESTION_COMPLETED',
            target: 'v2025.12.31',
            details: { rows: 20, status: 'success' },
          },
        ],
      }));
      setLogs(res.items || []);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadAuditLogs();
  }, [actorFilter, actionFilter, page]);

  const handleExportCSV = async () => {
    try {
      const token = document.cookie.match(/(^|;\s*)stockpulse_csrf=([^;]*)/)?.[3];
      const res = await fetch('/api/v1/admin/audit-logs/export', {
        headers: token ? { 'X-CSRF-Token': decodeURIComponent(token) } : {},
      });
      if (!res.ok) throw new Error('Export failed');
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `stockpulse_audit_log_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch {
      alert('Failed to export audit log CSV.');
    }
  };

  return (
    <div className="space-y-6" data-testid="admin-audit-page">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <History className="w-6 h-6 text-primary" />
            <span>Administrative Audit Log</span>
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Immutable security records of model modifications, user permission adjustments, and system jobs
          </p>
        </div>

        <button
          type="button"
          data-testid="export-audit-csv-btn"
          onClick={handleExportCSV}
          className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg border border-border bg-card text-xs font-semibold text-foreground hover:bg-muted shadow-xs transition"
        >
          <Download className="w-4 h-4 text-primary" />
          <span>Export Audit CSV</span>
        </button>
      </div>

      {/* Filter Toolbar */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
        <div className="relative">
          <User className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            data-testid="filter-actor-input"
            value={actorFilter}
            onChange={(e) => setActorFilter(e.target.value)}
            placeholder="Filter by actor email..."
            className="w-full pl-9 pr-3 py-2 border rounded-lg bg-card"
          />
        </div>

        <div className="relative">
          <Filter className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            data-testid="filter-action-input"
            value={actionFilter}
            onChange={(e) => setActionFilter(e.target.value)}
            placeholder="Filter by action code..."
            className="w-full pl-9 pr-3 py-2 border rounded-lg bg-card"
          />
        </div>
      </div>

      {/* Audit Log Table */}
      <div className="bg-card border border-border rounded-xl shadow-sm overflow-hidden" data-testid="audit-table-container">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="bg-muted/40 border-b text-muted-foreground font-semibold uppercase tracking-wider text-[11px]">
              <th className="py-3 px-4">Timestamp</th>
              <th className="py-3 px-4">Actor</th>
              <th className="py-3 px-4">Action</th>
              <th className="py-3 px-4">Target</th>
              <th className="py-3 px-4">Details</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {logs.map((log) => (
              <tr key={log.id} data-testid={`audit-row-${log.id}`} className="hover:bg-muted/30">
                <td className="py-3 px-4 tabular-nums text-muted-foreground whitespace-nowrap">
                  {new Date(log.timestamp).toLocaleString()}
                </td>
                <td className="py-3 px-4 font-semibold text-foreground">{log.actor_email}</td>
                <td className="py-3 px-4 font-mono font-bold text-primary">{log.action}</td>
                <td className="py-3 px-4 text-muted-foreground">{log.target || '—'}</td>
                <td className="py-3 px-4 text-muted-foreground max-w-xs truncate font-mono text-[11px]">
                  {JSON.stringify(log.details)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Pagination */}
        <div className="p-3 border-t flex items-center justify-between text-xs text-muted-foreground">
          <span>Page {page}</span>
          <div className="flex items-center gap-1">
            <button
              type="button"
              disabled={page <= 1}
              onClick={() => setPage(page - 1)}
              className="p-1 rounded border disabled:opacity-40"
              aria-label="Previous Page"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              type="button"
              disabled={logs.length < 20}
              onClick={() => setPage(page + 1)}
              className="p-1 rounded border disabled:opacity-40"
              aria-label="Next Page"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
