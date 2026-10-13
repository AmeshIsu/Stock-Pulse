import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { apiClient } from '../lib/api';
import { WatchlistItem, RankingsResponse } from '../lib/types';
import { TierBadge } from '../components/common/TierBadge';
import { ScoreGauge } from '../components/common/ScoreGauge';
import {
  Bookmark,
  Trash2,
  Edit2,
  Check,
  Plus,
  Bell,
  ArrowUpRight,
  ArrowDownRight,
  Search,
} from 'lucide-react';

export const Watchlist: React.FC = () => {
  const navigate = useNavigate();

  const [watchlist, setWatchlist] = useState<WatchlistItem[]>([]);
  const [allStocks, setAllStocks] = useState<{ code: string; name: string }[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Add stock combobox
  const [selectedStockToAdd, setSelectedStockToAdd] = useState('');
  const [searchFilter, setSearchFilter] = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [noteText, setNoteText] = useState('');

  const loadData = async () => {
    setIsLoading(true);
    try {
      const [wList, rankings] = await Promise.all([
        apiClient<WatchlistItem[]>('/watchlist'),
        apiClient<RankingsResponse>('/rankings'),
      ]);
      setWatchlist(wList || []);
      setAllStocks(
        rankings?.rankings?.map((r) => ({ code: r.code, name: r.name })) || []
      );
    } catch (err) {
      console.error('Failed to load watchlist:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleAddStock = async () => {
    if (!selectedStockToAdd) return;
    try {
      await apiClient('/watchlist', {
        method: 'POST',
        body: JSON.stringify({ stock_code: selectedStockToAdd }),
      });
      setSelectedStockToAdd('');
      loadData();
    } catch {
      alert('Could not add stock to watchlist.');
    }
  };

  const handleRemove = async (stockCode: string) => {
    try {
      await apiClient(`/watchlist/${stockCode}`, { method: 'DELETE' });
      setWatchlist((prev) => prev.filter((w) => w.stock_code !== stockCode));
    } catch {
      alert('Could not remove stock.');
    }
  };

  const handleSaveNote = async (item: WatchlistItem) => {
    try {
      await apiClient(`/watchlist/${item.stock_code}`, {
        method: 'PUT',
        body: JSON.stringify({ notes: noteText }),
      });
      setWatchlist((prev) =>
        prev.map((w) => (w.id === item.id ? { ...w, notes: noteText } : w))
      );
      setEditingId(null);
    } catch {
      alert('Failed to update note.');
    }
  };

  const availableStocksToAdd = allStocks.filter(
    (s) => !watchlist.some((w) => w.stock_code === s.code)
  );

  return (
    <div className="space-y-6" data-testid="watchlist-page">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground flex items-center gap-2">
            <Bookmark className="w-6 h-6 text-primary" />
            <span>Watchlist</span>
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Monitor selected Colombo equities for upside ranking adjustments and tier migrations
          </p>
        </div>

        {/* Add stock combobox */}
        <div className="flex items-center gap-2">
          <select
            data-testid="add-stock-select"
            aria-label="Select a stock to watch"
            value={selectedStockToAdd}
            onChange={(e) => setSelectedStockToAdd(e.target.value)}
            className="text-xs p-2 rounded-lg border bg-card text-foreground"
          >
            <option value="">Select a stock to watch...</option>
            {availableStocksToAdd.map((s) => (
              <option key={s.code} value={s.code}>
                {s.code} — {s.name}
              </option>
            ))}
          </select>
          <button
            type="button"
            data-testid="add-stock-btn"
            disabled={!selectedStockToAdd}
            onClick={handleAddStock}
            className="inline-flex items-center gap-1 px-3 py-2 bg-primary text-primary-foreground text-xs font-semibold rounded-lg hover:bg-primary-700 disabled:opacity-50"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Add</span>
          </button>
        </div>
      </div>

      {/* Main Watchlist Table */}
      {watchlist.length === 0 && !isLoading ? (
        <div className="bg-card border border-border rounded-xl p-12 text-center space-y-4" data-testid="watchlist-empty-state">
          <div className="w-12 h-12 rounded-2xl bg-muted mx-auto flex items-center justify-center text-muted-foreground">
            <Bookmark className="w-6 h-6" />
          </div>
          <div className="space-y-1">
            <h3 className="font-bold text-foreground">Your watchlist is currently empty</h3>
            <p className="text-xs text-muted-foreground max-w-sm mx-auto">
              Add equities from our 20-stock universe to track their cross-sectional scores and receive timely tier alerts.
            </p>
          </div>
          <Link
            to="/rankings"
            data-testid="explore-rankings-btn"
            className="inline-flex items-center gap-1.5 px-4 py-2 bg-primary text-primary-foreground text-xs font-semibold rounded-lg hover:bg-primary-700"
          >
            <span>Browse Full Rankings</span>
          </Link>
        </div>
      ) : (
        <div className="bg-card border border-border rounded-xl shadow-sm overflow-hidden" data-testid="watchlist-table-container">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-muted/40 border-b text-muted-foreground font-semibold uppercase tracking-wider text-[11px]">
                <th className="py-3 px-4">Stock</th>
                <th className="py-3 px-4">Overall Upside</th>
                <th className="py-3 px-4 text-right">Last Price</th>
                <th className="py-3 px-4 text-right">1D %</th>
                <th className="py-3 px-4">Notes</th>
                <th className="py-3 px-4 text-center">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {watchlist.map((item) => (
                <tr key={item.id} data-testid={`watchlist-row-${item.stock_code}`} className="hover:bg-muted/30">
                  <td className="py-3 px-4 cursor-pointer" onClick={() => navigate(`/stocks/${item.stock_code}`)}>
                    <div className="font-bold text-primary hover:underline">{item.stock_code}</div>
                    <div className="text-[11px] text-muted-foreground line-clamp-1">{item.stock_name}</div>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <ScoreGauge score={item.overall_score} tier={item.overall_tier} size="sm" />
                      <TierBadge tier={item.overall_tier} size="sm" />
                    </div>
                  </td>
                  <td className="py-3 px-4 text-right font-bold text-foreground tabular-nums">
                    LKR {item.last_price.toFixed(2)}
                  </td>
                  <td
                    className={`py-3 px-4 text-right font-semibold tabular-nums ${
                      item.daily_change_pct >= 0 ? 'text-emerald-600' : 'text-rose-600'
                    }`}
                  >
                    {item.daily_change_pct >= 0 ? '+' : ''}
                    {item.daily_change_pct.toFixed(2)}%
                  </td>
                  <td className="py-3 px-4 max-w-xs">
                    {editingId === item.id ? (
                      <div className="flex items-center gap-1">
                        <input
                          type="text"
                          data-testid={`edit-note-input-${item.stock_code}`}
                          value={noteText}
                          onChange={(e) => setNoteText(e.target.value)}
                          className="w-full text-xs p-1 border rounded bg-background"
                          placeholder="Add investment note..."
                        />
                        <button
                          type="button"
                          data-testid={`save-note-btn-${item.stock_code}`}
                          onClick={() => handleSaveNote(item)}
                          className="p-1 text-emerald-600 hover:text-emerald-700"
                          aria-label="Save note"
                        >
                          <Check className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    ) : (
                      <div
                        className="flex items-center justify-between group cursor-pointer"
                        onClick={() => {
                          setEditingId(item.id);
                          setNoteText(item.notes || '');
                        }}
                      >
                        <span className="text-muted-foreground text-xs line-clamp-1 italic">
                          {item.notes || 'Click to add note...'}
                        </span>
                        <Edit2 className="w-3 h-3 text-muted-foreground opacity-0 group-hover:opacity-100 ml-2" />
                      </div>
                    )}
                  </td>
                  <td className="py-3 px-4 text-center">
                    <div className="flex items-center justify-center gap-2">
                      <button
                        type="button"
                        data-testid={`quick-alert-btn-${item.stock_code}`}
                        onClick={() => navigate(`/stocks/${item.stock_code}`)}
                        className="p-1 text-muted-foreground hover:text-primary"
                        title="Set alert"
                        aria-label={`Set alert for ${item.stock_code}`}
                      >
                        <Bell className="w-3.5 h-3.5" />
                      </button>
                      <button
                        type="button"
                        data-testid={`remove-watchlist-btn-${item.stock_code}`}
                        onClick={() => handleRemove(item.stock_code)}
                        className="p-1 text-muted-foreground hover:text-rose-600"
                        title="Remove from watchlist"
                        aria-label={`Remove ${item.stock_code} from watchlist`}
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
    </div>
  );
};
