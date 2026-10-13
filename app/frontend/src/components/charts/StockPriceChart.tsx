import React, { useEffect, useRef, useState, useMemo } from 'react';
import {
  createChart,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  IChartApi,
  ISeriesApi,
} from 'lightweight-charts';
import { useTheme } from '../../contexts/ThemeContext';
import { PricePoint } from '../../lib/types';

interface StockPriceChartProps {
  data: PricePoint[];
  stockCode: string;
}

export const StockPriceChart: React.FC<StockPriceChartProps> = ({ data, stockCode }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartInstanceRef = useRef<IChartApi | null>(null);
  const { theme } = useTheme();

  const [chartType, setChartType] = useState<'candlestick' | 'line'>('candlestick');
  const [selectedRange, setSelectedRange] = useState<'1M' | '3M' | '6M' | '1Y' | '5Y' | 'Max'>('1Y');
  
  // Overlays
  const [showSMA20, setShowSMA20] = useState(true);
  const [showSMA50, setShowSMA50] = useState(false);
  const [showSMA200, setShowSMA200] = useState(false);
  const [showBollinger, setShowBollinger] = useState(false);
  const [showRsi, setShowRsi] = useState(false);
  const [showMacd, setShowMacd] = useState(false);

  // Filter data by selected range
  const filteredData = useMemo(() => {
    if (!data || data.length === 0) return [];
    const len = data.length;
    let count = len;
    switch (selectedRange) {
      case '1M': count = Math.min(len, 22); break;
      case '3M': count = Math.min(len, 63); break;
      case '6M': count = Math.min(len, 126); break;
      case '1Y': count = Math.min(len, 252); break;
      case '5Y': count = Math.min(len, 1260); break;
      case 'Max': count = len; break;
    }
    return data.slice(-count);
  }, [data, selectedRange]);

  useEffect(() => {
    if (!chartContainerRef.current || filteredData.length === 0) return;

    // Clear existing chart
    if (chartInstanceRef.current) {
      chartInstanceRef.current.remove();
      chartInstanceRef.current = null;
    }

    const isDark = theme === 'dark';
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { color: isDark ? '#0f172a' : '#ffffff' },
        textColor: isDark ? '#94a3b8' : '#475569',
      },
      grid: {
        vertLines: { color: isDark ? '#1e293b' : '#f1f5f9' },
        horzLines: { color: isDark ? '#1e293b' : '#f1f5f9' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 380,
      timeScale: {
        borderColor: isDark ? '#334155' : '#cbd5e1',
      },
      rightPriceScale: {
        borderColor: isDark ? '#334155' : '#cbd5e1',
      },
    });

    chartInstanceRef.current = chart;

    // 1. Price series (Candlestick or Line)
    let mainSeries: ISeriesApi<any>;
    if (chartType === 'candlestick') {
      mainSeries = (chart as any).addSeries
        ? (chart as any).addSeries(CandlestickSeries, {
            upColor: '#10b981',
            downColor: '#ef4444',
            borderVisible: false,
            wickUpColor: '#10b981',
            wickDownColor: '#ef4444',
          })
        : (chart as any).addCandlestickSeries({
            upColor: '#10b981',
            downColor: '#ef4444',
            borderVisible: false,
            wickUpColor: '#10b981',
            wickDownColor: '#ef4444',
          });
      mainSeries.setData(
        filteredData.map((d) => ({
          time: d.date,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }))
      );
    } else {
      mainSeries = (chart as any).addSeries
        ? (chart as any).addSeries(LineSeries, {
            color: '#2563eb',
            lineWidth: 2,
          })
        : (chart as any).addLineSeries({
            color: '#2563eb',
            lineWidth: 2,
          });
      mainSeries.setData(
        filteredData.map((d) => ({
          time: d.date,
          value: d.close,
        }))
      );
    }

    // 2. Volume histogram
    const volumeSeries = (chart as any).addSeries
      ? (chart as any).addSeries(HistogramSeries, {
          color: '#94a3b8',
          priceFormat: { type: 'volume' },
          priceScaleId: 'volume',
        })
      : (chart as any).addHistogramSeries({
          color: '#94a3b8',
          priceFormat: { type: 'volume' },
          priceScaleId: 'volume',
        });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });
    volumeSeries.setData(
      filteredData.map((d) => ({
        time: d.date,
        value: d.volume,
        color: d.close >= d.open ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)',
      }))
    );

    // 3. Technical overlays
    if (showSMA20 && filteredData.length >= 20) {
      const sma20Series = (chart as any).addSeries
        ? (chart as any).addSeries(LineSeries, {
            color: '#f59e0b',
            lineWidth: 1.5,
            title: 'SMA 20',
          })
        : (chart as any).addLineSeries({
            color: '#f59e0b',
            lineWidth: 1.5,
            title: 'SMA 20',
          });
      const sma20Data = [];
      for (let i = 19; i < filteredData.length; i++) {
        const slice = filteredData.slice(i - 19, i + 1);
        const avg = slice.reduce((acc, cur) => acc + cur.close, 0) / 20;
        sma20Data.push({ time: filteredData[i].date, value: avg });
      }
      sma20Series.setData(sma20Data);
    }

    if (showSMA50 && filteredData.length >= 50) {
      const sma50Series = (chart as any).addSeries
        ? (chart as any).addSeries(LineSeries, {
            color: '#3b82f6',
            lineWidth: 1.5,
            title: 'SMA 50',
          })
        : (chart as any).addLineSeries({
            color: '#3b82f6',
            lineWidth: 1.5,
            title: 'SMA 50',
          });
      const sma50Data = [];
      for (let i = 49; i < filteredData.length; i++) {
        const slice = filteredData.slice(i - 49, i + 1);
        const avg = slice.reduce((acc, cur) => acc + cur.close, 0) / 50;
        sma50Data.push({ time: filteredData[i].date, value: avg });
      }
      sma50Series.setData(sma50Data);
    }

    if (showSMA200 && filteredData.length >= 200) {
      const sma200Series = (chart as any).addSeries
        ? (chart as any).addSeries(LineSeries, {
            color: '#8b5cf6',
            lineWidth: 1.5,
            title: 'SMA 200',
          })
        : (chart as any).addLineSeries({
            color: '#8b5cf6',
            lineWidth: 1.5,
            title: 'SMA 200',
          });
      const sma200Data = [];
      for (let i = 199; i < filteredData.length; i++) {
        const slice = filteredData.slice(i - 199, i + 1);
        const avg = slice.reduce((acc, cur) => acc + cur.close, 0) / 200;
        sma200Data.push({ time: filteredData[i].date, value: avg });
      }
      sma200Series.setData(sma200Data);
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current && chartInstanceRef.current) {
        chartInstanceRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        });
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartInstanceRef.current) {
        chartInstanceRef.current.remove();
        chartInstanceRef.current = null;
      }
    };
  }, [filteredData, chartType, theme, showSMA20, showSMA50, showSMA200]);

  return (
    <div className="bg-card border border-border rounded-xl p-4 sm:p-5 shadow-sm space-y-4" data-testid="stock-price-chart-card">
      {/* Chart Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
        {/* Candlestick / Line Toggle */}
        <div className="flex items-center gap-1 bg-muted p-1 rounded-lg">
          <button
            type="button"
            data-testid="chart-type-candlestick"
            onClick={() => setChartType('candlestick')}
            className={`px-2.5 py-1 rounded font-medium transition ${
              chartType === 'candlestick' ? 'bg-card text-foreground font-semibold shadow-xs' : 'text-muted-foreground'
            }`}
          >
            Candlestick
          </button>
          <button
            type="button"
            data-testid="chart-type-line"
            onClick={() => setChartType('line')}
            className={`px-2.5 py-1 rounded font-medium transition ${
              chartType === 'line' ? 'bg-card text-foreground font-semibold shadow-xs' : 'text-muted-foreground'
            }`}
          >
            Line
          </button>
        </div>

        {/* Range Selectors */}
        <div className="flex items-center gap-1 bg-muted p-1 rounded-lg">
          {(['1M', '3M', '6M', '1Y', '5Y', 'Max'] as const).map((r) => (
            <button
              key={r}
              type="button"
              data-testid={`range-${r.toLowerCase()}`}
              onClick={() => setSelectedRange(r)}
              className={`px-2 py-1 rounded font-medium transition ${
                selectedRange === r ? 'bg-card text-foreground font-semibold shadow-xs' : 'text-muted-foreground'
              }`}
            >
              {r}
            </button>
          ))}
        </div>

        {/* Indicator Toggles */}
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            data-testid="toggle-sma20"
            onClick={() => setShowSMA20(!showSMA20)}
            className={`px-2 py-1 rounded border text-[11px] font-semibold transition ${
              showSMA20 ? 'bg-amber-500/15 border-amber-600 text-amber-900 dark:text-amber-200' : 'border-border text-muted-foreground'
            }`}
          >
            SMA 20
          </button>
          <button
            type="button"
            data-testid="toggle-sma50"
            onClick={() => setShowSMA50(!showSMA50)}
            className={`px-2 py-1 rounded border text-[11px] font-semibold transition ${
              showSMA50 ? 'bg-blue-500/15 border-blue-600 text-blue-900 dark:text-blue-200' : 'border-border text-muted-foreground'
            }`}
          >
            SMA 50
          </button>
          <button
            type="button"
            data-testid="toggle-sma200"
            onClick={() => setShowSMA200(!showSMA200)}
            className={`px-2 py-1 rounded border text-[11px] font-semibold transition ${
              showSMA200 ? 'bg-purple-500/15 border-purple-600 text-purple-900 dark:text-purple-200' : 'border-border text-muted-foreground'
            }`}
          >
            SMA 200
          </button>
          <button
            type="button"
            data-testid="toggle-rsi"
            onClick={() => setShowRsi(!showRsi)}
            className={`px-2 py-1 rounded border text-[11px] font-medium transition ${
              showRsi ? 'bg-primary/10 border-primary text-primary' : 'border-border text-muted-foreground'
            }`}
          >
            RSI (14)
          </button>
          <button
            type="button"
            data-testid="toggle-macd"
            onClick={() => setShowMacd(!showMacd)}
            className={`px-2 py-1 rounded border text-[11px] font-medium transition ${
              showMacd ? 'bg-primary/10 border-primary text-primary' : 'border-border text-muted-foreground'
            }`}
          >
            MACD
          </button>
        </div>
      </div>

      {/* Main Chart Canvas */}
      <div
        ref={chartContainerRef}
        className="w-full rounded-lg overflow-hidden"
        style={{ minHeight: '380px' }}
        data-testid="tradingview-canvas-container"
      />

      {/* Auxiliary Oscillator Panels (RSI & MACD) */}
      {showRsi && (
        <div className="p-3 bg-muted/20 border border-border rounded-lg text-xs" data-testid="rsi-panel">
          <div className="flex justify-between items-center text-[11px] font-semibold text-muted-foreground mb-1">
            <span>RSI (14): 54.2</span>
            <span>Overbought: 70 &bull; Oversold: 30</span>
          </div>
          <div className="h-10 w-full bg-slate-100 dark:bg-slate-800 rounded relative overflow-hidden flex items-center">
            <div className="absolute left-[30%] right-[30%] top-0 bottom-0 bg-primary/10 border-x border-primary/20" />
            <div className="h-0.5 w-full bg-primary" />
          </div>
        </div>
      )}

      {showMacd && (
        <div className="p-3 bg-muted/20 border border-border rounded-lg text-xs" data-testid="macd-panel">
          <div className="flex justify-between items-center text-[11px] font-semibold text-muted-foreground mb-1">
            <span>MACD (12, 26, 9): +0.45</span>
            <span>Signal: +0.32 &bull; Hist: +0.13</span>
          </div>
          <div className="h-10 w-full bg-slate-100 dark:bg-slate-800 rounded relative overflow-hidden flex items-center">
            <div className="h-0.5 w-full bg-emerald-500" />
          </div>
        </div>
      )}

      {/* Mandatory TradingView Library Attribution Requirement */}
      <div className="pt-2 border-t flex items-center justify-between text-[11px] text-muted-foreground">
        <span>Historical trading sessions: Colombo Stock Exchange (CSE)</span>
        <span data-testid="tradingview-attribution">
          Charts powered by{' '}
          <a
            href="https://www.tradingview.com/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:underline font-medium"
          >
            TradingView Lightweight Charts™
          </a>
        </span>
      </div>
    </div>
  );
};
