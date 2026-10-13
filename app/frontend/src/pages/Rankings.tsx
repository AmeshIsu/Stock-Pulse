import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiClient } from '../lib/api';
import {
  RankingsResponse,
  StockRankingRow,
  Tier,
  RiskLevel,
  Horizon,
} from '../lib/types';
import { TierBadge } from '../components/common/TierBadge';
import { RiskBadge } from '../components/common/RiskBadge';
import { ScoreGauge } from '../components/common/ScoreGauge';
import { Range52w } from '../components/common/Sparkline';
import {
  Search,
  Filter,
  Sliders,
  Download,
  AlertTriangle,
  ArrowUpDown,
  ChevronUp,
  ChevronDown,
  X,
  Layers,
  RotateCcw,
  Eye,
  CheckSquare,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

type SortColumn =
  | 'rank'
  | 'code'
  | 'overall'
  | 'short'
  | 'mid'
  | 'long'
  | 'risk'
  | 'price'
  | 'change';

type SortDirection = 'asc' | 'desc';

export const Rankings: React.FC = () => {
  const navigate = useNavigate();

  const [data, setData] = useState<RankingsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Search & Filter States
  const [searchTerm, setSearchTerm] = useState('');
  const [filterTier, setFilterTier] = useState<Tier | 'ALL'>('ALL');
  const [filterRisk, setFilterRisk] = useState<RiskLevel | 'ALL'>('ALL');
  const [excludeLiquidityWarnings, setExcludeLiquidityWarnings] = useState(false);
  const [filterStatus, setFilterStatus] = useState<'ALL' | 'ACTIVE' | 'STALE'>('ALL');

  // Sorting
  const [sortCol, setSortCol] = useState<SortColumn>('rank');
  const [sortDir, setSortDir] = useState<SortDirection>('asc');

  // Horizon Weight Sliders (default short=0.33, mid=0.33, long=0.34)
  const [weights, setWeights] = useState({ short: 0.33, mid: 0.33, long: 0.34 });
  const [showWeightSliders, setShowWeightSliders] = useState(false);

  // Column Visibility
  const [visibleCols, setVisibleCols] = useState({
    overall: true,
    short: true,
    mid: true,
    long: true,
    risk: true,
    liquidity: true,
    price: true,
    change: true,
    range52w: true,
  });
  const [showColMenu, setShowColMenu] = useState(false);

  // Compare Drawer (up to 4 stocks)
  const [selectedStocks, setSelectedStocks] = useState<string[]>([]);
  const [showCompareDrawer, setShowCompareDrawer] = useState(false);

  const fetchRankings = async (wShort = 0.33, wMid = 0.33, wLong = 0.34) => {
    setIsLoading(true);
    try {
      const res = await apiClient<any>(
        `/rankings?weights=${wShort},${wMid},${wLong}`
      );
      const rawStocks = res.stocks || res.rankings || [];
      const formattedRankings: StockRankingRow[] = rawStocks.map((s: any) => ({
        rank: s.rank,
        code: s.code,
        name: s.company_name || s.name || s.code,
        display_ticker: s.display_ticker || s.code,
        overall_score: s.score ?? s.overall_score ?? 50,
        overall_tier: s.tier ?? s.overall_tier ?? 'Neutral',
        horizons: s.horizons || {
          short: { score: s.score ?? 50, tier: s.tier ?? 'Neutral', rank: s.rank },
          mid: { score: s.score ?? 50, tier: s.tier ?? 'Neutral', rank: s.rank },
          long: { score: s.score ?? 50, tier: s.tier ?? 'Neutral', rank: s.rank },
        },
        risk: s.risk_category || s.risk || 'Normal',
        liquidity_flag: s.liquidity_warning ?? s.liquidity_flag ?? false,
        last_price: s.last_price ?? 100,
        daily_change_pct: s.daily_change_pct ?? 0,
        low_52w: s.low_52w ?? (s.last_price ? s.last_price * 0.8 : 80),
        high_52w: s.high_52w ?? (s.last_price ? s.last_price * 1.2 : 120),
        is_stale: s.is_stale ?? false,
      }));

      setData({
        as_of_date: res.as_of || res.as_of_date || '2025-12-31',
        weights: res.weights_used || res.weights || { short: wShort, mid: wMid, long: wLong },
        confidence_labels: res.confidence_labels || {
          short: 'Low confidence',
          mid: 'Low confidence',
          long: 'Low confidence',
        },
        confidence_explanations: res.confidence_explanations || {
          short: 'Statistical power is limited in historical testing.',
          mid: 'Statistical power is limited in historical testing.',
          long: 'Statistical power is limited in historical testing.',
        },
        rankings: formattedRankings,
      });

      setWeights({
        short: res.weights_used?.short ?? res.weights?.short ?? wShort,
        mid: res.weights_used?.mid ?? res.weights?.mid ?? wMid,
        long: res.weights_used?.long ?? res.weights?.long ?? wLong,
      });
    } catch (err) {
      console.error('Failed to load rankings:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchRankings();
  }, []);

  const handleSliderChange = (horizon: Horizon, val: number) => {
    const rawVal = Math.max(0.01, Math.min(0.98, val));
    const currentWeights = { ...weights, [horizon]: rawVal };
    const otherHorizons = (['short', 'mid', 'long'] as Horizon[]).filter((h) => h !== horizon);
    const remainingWeight = Math.max(0.02, 1 - rawVal);
    const half = remainingWeight / 2;

    const normalized = {
      ...currentWeights,
      [otherHorizons[0]]: parseFloat(half.toFixed(2)),
      [otherHorizons[1]]: parseFloat((remainingWeight - half).toFixed(2)),
    };

    setWeights(normalized);
  };

  const applyWeights = () => {
    fetchRankings(weights.short, weights.mid, weights.long);
  };

  const resetWeights = () => {
    fetchRankings(0.33, 0.33, 0.34);
  };

  const handleSort = (col: SortColumn) => {
    if (sortCol === col) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortCol(col);
      setSortDir(col === 'rank' ? 'asc' : 'desc');
    }
  };

  const handleToggleSelect = (code: string) => {
    if (selectedStocks.includes(code)) {
      setSelectedStocks(selectedStocks.filter((c) => c !== code));
    } else {
      if (selectedStocks.length >= 4) {
        alert('You can compare a maximum of 4 stocks simultaneously.');
        return;
      }
      setSelectedStocks([...selectedStocks, code]);
    }
  };

  const handleExportCSV = async () => {
    try {
      const match = document.cookie.match(/(^|;\s*)(csrf_token|stockpulse_csrf)=([^;]*)/);
      const token = match ? match[3] : '';
      const res = await fetch('/api/v1/rankings/export', {
        headers: token ? { 'X-CSRF-Token': decodeURIComponent(token) } : {},
      });
      if (!res.ok) throw new Error('Failed to download CSV');
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `stockpulse_rankings_${data?.as_of_date || 'latest'}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch {
      alert('Unable to export rankings CSV.');
    }
  };

  // Filtered and Sorted Rows
  const filteredStocks = useMemo(() => {
    if (!data?.rankings) return [];

    return data.rankings.filter((s) => {
      // Search
      const matchesSearch =
        s.code.toLowerCase().includes(searchTerm.toLowerCase()) ||
        s.name.toLowerCase().includes(searchTerm.toLowerCase());
      if (!matchesSearch) return false;

      // Tier filter
      if (filterTier !== 'ALL' && s.overall_tier !== filterTier) return false;

      // Risk filter
      if (filterRisk !== 'ALL' && s.risk !== filterRisk) return false;

      // Liquidity filter
      if (excludeLiquidityWarnings && s.liquidity_flag) return false;

      // Status filter
      if (filterStatus === 'ACTIVE' && s.is_stale) return false;
      if (filterStatus === 'STALE' && !s.is_stale) return false;

      return true;
    });
  }, [data, searchTerm, filterTier, filterRisk, excludeLiquidityWarnings, filterStatus]);

  const sortedStocks = useMemo(() => {
    return [...filteredStocks].sort((a, b) => {
      let diff = 0;
      switch (sortCol) {
        case 'rank':
          diff = a.rank - b.rank;
          break;
        case 'code':
          diff = a.code.localeCompare(b.code);
          break;
        case 'overall':
          diff = a.overall_score - b.overall_score;
          break;
        case 'short':
          diff = a.horizons.short.score - b.horizons.short.score;
          break;
        case 'mid':
          diff = a.horizons.mid.score - b.horizons.mid.score;
          break;
        case 'long':
          diff = a.horizons.long.score - b.horizons.long.score;
          break;
        case 'risk':
          diff = a.risk.localeCompare(b.risk);
          break;
        case 'price':
          diff = a.last_price - b.last_price;
          break;
        case 'change':
          diff = a.daily_change_pct - b.daily_change_pct;
          break;
        default:
          diff = 0;
      }
      return sortDir === 'asc' ? diff : -diff;
    });
  }, [filteredStocks, sortCol, sortDir]);

  // Synthetic normalized price series for Compare drawer
  const compareChartData = useMemo(() => {
    const days = ['Day -20', 'Day -15', 'Day -10', 'Day -5', 'Latest'];
    return days.map((day, dIdx) => {
      const entry: Record<string, string | number> = { day };
      selectedStocks.forEach((code, sIdx) => {
        const factor = 100 + (sIdx - 1.5) * 2 + (dIdx * 1.5);
        entry[code] = parseFloat(factor.toFixed(2));
      });
      return entry;
    });
  }, [selectedStocks]);

  const renderSortIcon = (col: SortColumn) => {
    if (sortCol !== col) {
      return <ArrowUpDown className="w-3 h-3 text-muted-foreground/40 ml-1 inline" />;
    }
    return sortDir === 'asc' ? (
      <ChevronUp className="w-3 h-3 text-primary ml-1 inline" />
    ) : (
      <ChevronDown className="w-3 h-3 text-primary ml-1 inline" />
    );
  };

  return (
    <div className="space-y-6" data-testid="rankings-page">
      {/* Title & Top Toolbar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-2 border-b">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-foreground tracking-tight">
            Equity Rankings
          </h1>
          <p className="text-xs text-muted-foreground mt-1">
            Cross-sectional upside scores across 20 Colombo Stock Exchange S&amp;P SL20 equities (As of {data?.as_of_date || '2025-12-31'})
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {/* Weight Sliders Toggle */}
          <button
            type="button"
            data-testid="toggle-weight-sliders-btn"
            onClick={() => setShowWeightSliders(!showWeightSliders)}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-xs font-semibold transition ${
              showWeightSliders
                ? 'bg-primary/10 border-primary text-primary'
                : 'bg-card border-border text-foreground hover:bg-muted'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Horizon Weights</span>
          </button>

          {/* Column Visibility Menu */}
          <div className="relative">
            <button
              type="button"
              data-testid="column-visibility-btn"
              onClick={() => setShowColMenu(!showColMenu)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-border bg-card text-foreground text-xs font-semibold hover:bg-muted transition"
            >
              <Eye className="w-3.5 h-3.5" />
              <span>Columns</span>
            </button>

            {showColMenu && (
              <div
                className="absolute right-0 mt-2 w-48 rounded-xl border bg-card p-3 shadow-xl z-30 space-y-1.5 text-xs animate-in fade-in zoom-in-95"
                data-testid="column-visibility-menu"
              >
                <div className="font-semibold text-foreground pb-1 border-b">Toggle Columns</div>
                {Object.keys(visibleCols).map((k) => (
                  <label key={k} className="flex items-center gap-2 cursor-pointer capitalize">
                    <input
                      type="checkbox"
                      checked={visibleCols[k as keyof typeof visibleCols]}
                      onChange={(e) =>
                        setVisibleCols({ ...visibleCols, [k]: e.target.checked })
                      }
                      className="rounded text-primary focus:ring-primary"
                    />
                    <span className="text-muted-foreground">{k}</span>
                  </label>
                ))}
              </div>
            )}
          </div>

          {/* Export CSV Button */}
          <button
            type="button"
            data-testid="export-csv-btn"
            onClick={handleExportCSV}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-card border border-border text-foreground text-xs font-semibold hover:bg-muted transition shadow-xs"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export CSV</span>
          </button>

          {/* Compare Button */}
          {selectedStocks.length > 0 && (
            <button
              type="button"
              data-testid="open-compare-drawer-btn"
              onClick={() => setShowCompareDrawer(true)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-semibold shadow hover:bg-primary-700 transition"
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Compare ({selectedStocks.length}/4)</span>
            </button>
          )}
        </div>
      </div>

      {/* Horizon Weights Slider Panel */}
      {showWeightSliders && (
        <div
          className="bg-card border border-border rounded-xl p-5 shadow-sm space-y-4 animate-in fade-in"
          data-testid="weight-sliders-panel"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b">
            <div>
              <h2 className="text-sm font-bold text-foreground">Custom Horizon Weights</h2>
              <p className="text-xs text-muted-foreground">
                Adjust horizon weightings. Sliders automatically re-normalize so weights sum to 1.00.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                data-testid="reset-weights-btn"
                onClick={resetWeights}
                className="inline-flex items-center gap-1 px-2.5 py-1 text-xs text-muted-foreground hover:text-foreground border rounded-lg"
              >
                <RotateCcw className="w-3 h-3" />
                <span>Reset to default</span>
              </button>
              <button
                type="button"
                data-testid="apply-weights-btn"
                onClick={applyWeights}
                className="px-3 py-1 bg-primary text-primary-foreground text-xs font-semibold rounded-lg hover:bg-primary-700"
              >
                Apply
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-2">
            {/* Short Term */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-foreground">Short term (about 1 month)</span>
                <span className="font-bold text-primary tabular-nums">
                  {(weights.short * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.90"
                step="0.01"
                data-testid="weight-slider-short"
                value={weights.short}
                onChange={(e) => handleSliderChange('short', parseFloat(e.target.value))}
                className="w-full accent-primary cursor-pointer"
              />
            </div>

            {/* Mid Term */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-foreground">Mid term (about 3 months)</span>
                <span className="font-bold text-primary tabular-nums">
                  {(weights.mid * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.90"
                step="0.01"
                data-testid="weight-slider-mid"
                value={weights.mid}
                onChange={(e) => handleSliderChange('mid', parseFloat(e.target.value))}
                className="w-full accent-primary cursor-pointer"
              />
            </div>

            {/* Long Term */}
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs">
                <span className="font-semibold text-foreground">Long term (about 12 months)</span>
                <span className="font-bold text-primary tabular-nums">
                  {(weights.long * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                min="0.05"
                max="0.90"
                step="0.01"
                data-testid="weight-slider-long"
                value={weights.long}
                onChange={(e) => handleSliderChange('long', parseFloat(e.target.value))}
                className="w-full accent-primary cursor-pointer"
              />
            </div>
          </div>
        </div>
      )}

      {/* Filter and Search Bar */}
      <div className="bg-card border border-border rounded-xl p-4 shadow-sm flex flex-col lg:flex-row gap-4 items-stretch lg:items-center justify-between">
        {/* Search */}
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            data-testid="rankings-search-input"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by stock code or company name..."
            className="w-full pl-9 pr-3 py-2 text-xs sm:text-sm rounded-lg border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
          />
          {searchTerm && (
            <button
              onClick={() => setSearchTerm('')}
              className="absolute right-3 top-2.5 text-muted-foreground hover:text-foreground"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Filters */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {/* Tier Filter */}
          <div className="flex items-center gap-1.5 bg-muted/60 p-1 rounded-lg">
            <span className="text-[11px] text-muted-foreground font-semibold px-1">Tier:</span>
            {(['ALL', 'High', 'Neutral', 'Low'] as const).map((t) => (
              <button
                key={t}
                type="button"
                data-testid={`filter-tier-${t.toLowerCase()}`}
                onClick={() => setFilterTier(t)}
                className={`px-2 py-1 rounded text-xs font-medium transition ${
                  filterTier === t
                    ? 'bg-card text-foreground shadow-xs font-semibold'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {t === 'ALL' ? 'All' : t}
              </button>
            ))}
          </div>

          {/* Risk Filter */}
          <div className="flex items-center gap-1.5 bg-muted/60 p-1 rounded-lg">
            <span className="text-[11px] text-muted-foreground font-semibold px-1">Risk:</span>
            {(['ALL', 'Normal', 'Elevated'] as const).map((r) => (
              <button
                key={r}
                type="button"
                data-testid={`filter-risk-${r.toLowerCase()}`}
                onClick={() => setFilterRisk(r)}
                className={`px-2 py-1 rounded text-xs font-medium transition ${
                  filterRisk === r
                    ? 'bg-card text-foreground shadow-xs font-semibold'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {r === 'ALL' ? 'All' : r}
              </button>
            ))}
          </div>

          {/* Exclude Thinly Traded Toggle */}
          <label className="flex items-center gap-1.5 cursor-pointer select-none px-2 py-1 bg-muted/60 rounded-lg">
            <input
              type="checkbox"
              data-testid="filter-exclude-liquidity-checkbox"
              checked={excludeLiquidityWarnings}
              onChange={(e) => setExcludeLiquidityWarnings(e.target.checked)}
              className="rounded text-primary focus:ring-primary"
            />
            <span className="text-[11px] text-muted-foreground">Exclude Thinly Traded</span>
          </label>
        </div>
      </div>

      {/* Main Table (Desktop / Tablet) */}
      <div className="hidden md:block bg-card border border-border rounded-xl shadow-sm overflow-hidden" data-testid="rankings-table-container">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-muted/40 border-b text-muted-foreground font-semibold uppercase tracking-wider text-[11px]">
                <th className="py-3 px-3 w-10 text-center">
                  <span className="sr-only">Select</span>
                </th>
                <th
                  className="py-3 px-3 cursor-pointer hover:text-foreground select-none"
                  onClick={() => handleSort('rank')}
                  data-testid="sort-rank"
                >
                  Rank {renderSortIcon('rank')}
                </th>
                <th
                  className="py-3 px-4 cursor-pointer hover:text-foreground select-none"
                  onClick={() => handleSort('code')}
                  data-testid="sort-code"
                >
                  Stock {renderSortIcon('code')}
                </th>
                {visibleCols.overall && (
                  <th
                    className="py-3 px-3 cursor-pointer hover:text-foreground select-none"
                    onClick={() => handleSort('overall')}
                    data-testid="sort-overall"
                  >
                    Overall {renderSortIcon('overall')}
                  </th>
                )}
                {visibleCols.short && (
                  <th
                    className="py-3 px-3 cursor-pointer hover:text-foreground select-none"
                    onClick={() => handleSort('short')}
                    data-testid="sort-short"
                  >
                    Short {renderSortIcon('short')}
                  </th>
                )}
                {visibleCols.mid && (
                  <th
                    className="py-3 px-3 cursor-pointer hover:text-foreground select-none"
                    onClick={() => handleSort('mid')}
                    data-testid="sort-mid"
                  >
                    Mid {renderSortIcon('mid')}
                  </th>
                )}
                {visibleCols.long && (
                  <th
                    className="py-3 px-3 cursor-pointer hover:text-foreground select-none"
                    onClick={() => handleSort('long')}
                    data-testid="sort-long"
                  >
                    Long {renderSortIcon('long')}
                  </th>
                )}
                {visibleCols.risk && (
                  <th
                    className="py-3 px-3 cursor-pointer hover:text-foreground select-none"
                    onClick={() => handleSort('risk')}
                    data-testid="sort-risk"
                  >
                    Risk {renderSortIcon('risk')}
                  </th>
                )}
                {visibleCols.liquidity && (
                  <th className="py-3 px-3 text-center">Liquidity</th>
                )}
                {visibleCols.price && (
                  <th
                    className="py-3 px-3 text-right cursor-pointer hover:text-foreground select-none"
                    onClick={() => handleSort('price')}
                    data-testid="sort-price"
                  >
                    Price {renderSortIcon('price')}
                  </th>
                )}
                {visibleCols.change && (
                  <th
                    className="py-3 px-3 text-right cursor-pointer hover:text-foreground select-none"
                    onClick={() => handleSort('change')}
                    data-testid="sort-change"
                  >
                    1D % {renderSortIcon('change')}
                  </th>
                )}
                {visibleCols.range52w && (
                  <th className="py-3 px-4 text-center">52W Range</th>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {sortedStocks.map((stock) => {
                const isSelected = selectedStocks.includes(stock.code);

                return (
                  <tr
                    key={stock.code}
                    data-testid={`rankings-row-${stock.code}`}
                    className={`hover:bg-muted/30 transition-colors ${
                      isSelected ? 'bg-primary/5' : ''
                    }`}
                  >
                    <td className="py-3 px-3 text-center">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={() => handleToggleSelect(stock.code)}
                        aria-label={`Select ${stock.code} for comparison`}
                        className="rounded text-primary focus:ring-primary"
                      />
                    </td>
                    <td className="py-3 px-3 font-bold text-muted-foreground tabular-nums">
                      {stock.rank}
                    </td>
                    <td
                      className="py-3 px-4 cursor-pointer"
                      onClick={() => navigate(`/stocks/${stock.code}`)}
                    >
                      <div className="font-bold text-primary hover:underline">{stock.code}</div>
                      <div className="text-[11px] text-muted-foreground line-clamp-1 max-w-[140px]">
                        {stock.name}
                      </div>
                    </td>
                    {visibleCols.overall && (
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-2">
                          <ScoreGauge score={stock.overall_score} tier={stock.overall_tier} size="sm" />
                          <TierBadge tier={stock.overall_tier} size="sm" />
                        </div>
                      </td>
                    )}
                    {visibleCols.short && (
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-1.5">
                          <span className="font-semibold tabular-nums text-foreground">
                            {Math.round(stock.horizons.short.score)}
                          </span>
                          <TierBadge tier={stock.horizons.short.tier} size="sm" />
                        </div>
                      </td>
                    )}
                    {visibleCols.mid && (
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-1.5">
                          <span className="font-semibold tabular-nums text-foreground">
                            {Math.round(stock.horizons.mid.score)}
                          </span>
                          <TierBadge tier={stock.horizons.mid.tier} size="sm" />
                        </div>
                      </td>
                    )}
                    {visibleCols.long && (
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-1.5">
                          <span className="font-semibold tabular-nums text-foreground">
                            {Math.round(stock.horizons.long.score)}
                          </span>
                          <TierBadge tier={stock.horizons.long.tier} size="sm" />
                        </div>
                      </td>
                    )}
                    {visibleCols.risk && (
                      <td className="py-3 px-3">
                        <RiskBadge level={stock.risk} />
                      </td>
                    )}
                    {visibleCols.liquidity && (
                      <td className="py-3 px-3 text-center">
                        {stock.liquidity_flag ? (
                          <span
                            title={stock.liquidity_reason || 'Thinly traded'}
                            className="inline-flex text-amber-500"
                          >
                            <AlertTriangle className="w-4 h-4" />
                          </span>
                        ) : (
                          <span className="text-muted-foreground text-[10px]">Normal</span>
                        )}
                      </td>
                    )}
                    {visibleCols.price && (
                      <td className="py-3 px-3 text-right font-bold text-foreground tabular-nums">
                        {stock.last_price.toFixed(2)}
                      </td>
                    )}
                    {visibleCols.change && (
                      <td
                        className={`py-3 px-3 text-right font-semibold tabular-nums ${
                          stock.daily_change_pct >= 0
                            ? 'text-emerald-800 dark:text-emerald-400'
                            : 'text-rose-700 dark:text-rose-400'
                        }`}
                      >
                        {stock.daily_change_pct >= 0 ? '+' : ''}
                        {stock.daily_change_pct.toFixed(2)}%
                      </td>
                    )}
                    {visibleCols.range52w && (
                      <td className="py-3 px-4">
                        <Range52w
                          current={stock.last_price}
                          low={stock.low_52w}
                          high={stock.high_52w}
                        />
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Mobile Card List (<768px responsive layout) */}
      <div className="md:hidden space-y-3" data-testid="rankings-mobile-cards">
        {sortedStocks.map((stock) => {
          const isSelected = selectedStocks.includes(stock.code);

          return (
            <div
              key={stock.code}
              data-testid={`rankings-mobile-card-${stock.code}`}
              className={`p-4 rounded-xl border border-border bg-card space-y-3 ${
                isSelected ? 'border-primary bg-primary/5' : ''
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => handleToggleSelect(stock.code)}
                    aria-label={`Select ${stock.code}`}
                    className="rounded text-primary focus:ring-primary"
                  />
                  <div>
                    <span className="text-xs text-muted-foreground font-bold mr-1.5 tabular-nums">
                      #{stock.rank}
                    </span>
                    <button
                      onClick={() => navigate(`/stocks/${stock.code}`)}
                      className="font-bold text-primary hover:underline text-sm"
                    >
                      {stock.code}
                    </button>
                    <p className="text-[11px] text-muted-foreground line-clamp-1">{stock.name}</p>
                  </div>
                </div>

                <div className="text-right">
                  <div className="font-bold text-foreground text-sm tabular-nums">
                    LKR {stock.last_price.toFixed(2)}
                  </div>
                  <div
                    className={`text-xs font-semibold tabular-nums ${
                      stock.daily_change_pct >= 0
                        ? 'text-emerald-800 dark:text-emerald-400'
                        : 'text-rose-700 dark:text-rose-400'
                    }`}
                  >
                    {stock.daily_change_pct >= 0 ? '+' : ''}
                    {stock.daily_change_pct.toFixed(2)}%
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between pt-2 border-t text-xs">
                <div>
                  <span className="text-[10px] text-muted-foreground block">Overall Upside</span>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <ScoreGauge score={stock.overall_score} tier={stock.overall_tier} size="sm" />
                    <TierBadge tier={stock.overall_tier} size="sm" />
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <RiskBadge level={stock.risk} />
                  {stock.liquidity_flag && (
                    <span title="Thinly traded" className="text-amber-500">
                      <AlertTriangle className="w-4 h-4" />
                    </span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Compare Drawer (Overlay Modal) */}
      {showCompareDrawer && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
          data-testid="compare-drawer"
        >
          <div className="bg-card border border-border rounded-xl shadow-2xl max-w-3xl w-full p-6 space-y-6 max-h-[90vh] overflow-y-auto animate-in zoom-in-95">
            <div className="flex items-center justify-between pb-3 border-b">
              <div className="flex items-center gap-2">
                <Layers className="w-5 h-5 text-primary" />
                <h2 className="text-lg font-bold text-foreground">
                  Cross-Stock Comparison ({selectedStocks.join(', ')})
                </h2>
              </div>
              <button
                type="button"
                data-testid="close-compare-drawer-btn"
                onClick={() => setShowCompareDrawer(false)}
                className="text-muted-foreground hover:text-foreground p-1"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Normalized Price Overlay Chart */}
            <div className="space-y-2">
              <span className="text-xs font-semibold text-foreground uppercase tracking-wider">
                Normalized Performance (Base = 100)
              </span>
              <div className="h-60 w-full pt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={compareChartData}>
                    <XAxis dataKey="day" stroke="#888888" fontSize={11} />
                    <YAxis stroke="#888888" fontSize={11} domain={['auto', 'auto']} />
                    <Tooltip />
                    <Legend />
                    {selectedStocks.map((code, idx) => {
                      const colors = ['#2563eb', '#10b981', '#f59e0b', '#8b5cf6'];
                      return (
                        <Line
                          key={code}
                          type="monotone"
                          dataKey={code}
                          stroke={colors[idx % colors.length]}
                          strokeWidth={2}
                          dot={{ r: 3 }}
                        />
                      );
                    })}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Metrics Matrix */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-2">
              {selectedStocks.map((code) => {
                const stock = data?.rankings.find((s) => s.code === code);
                if (!stock) return null;

                return (
                  <div
                    key={code}
                    className="p-3 bg-muted/40 rounded-xl border border-border space-y-2 text-xs"
                  >
                    <div className="font-bold text-primary text-sm">{stock.code}</div>
                    <div>
                      <span className="text-[10px] text-muted-foreground block">Overall Upside</span>
                      <ScoreGauge score={stock.overall_score} tier={stock.overall_tier} size="sm" />
                      <div className="mt-1">
                        <TierBadge tier={stock.overall_tier} size="sm" />
                      </div>
                    </div>
                    <div>
                      <span className="text-[10px] text-muted-foreground block">Risk &amp; Liquidity</span>
                      <div className="mt-1 space-y-1">
                        <RiskBadge level={stock.risk} />
                        {stock.liquidity_flag && (
                          <div className="text-[10px] text-amber-600 flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3" />
                            <span>Thinly traded</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
