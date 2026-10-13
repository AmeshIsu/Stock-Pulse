# Stock Pulse — Frontend Design & Interaction Note

This document outlines the architectural, visual, and interaction choices adopted for the Stock Pulse web application, informed by an analysis of leading public financial and analytics platforms.

---

## 1. Comparative Site Analysis & Design Inspirations

### Yahoo Finance (Stock Page Structure & Market Hierarchy)
* **Adopted Concepts**:
  * **Header Architecture**: Compact, information-dense stock banner featuring canonical series ID (`code`), company name, real-time price badge, percentage change, and explicit trading state badges (e.g., active vs. stale with last traded date).
  * **Quick Actions**: Prominent, accessible buttons for Watchlist management and Alert triggering positioned alongside the stock header.
  * **Tabbed Information Breakdown**: Logical division into Overview, Upside Outlook, Risk & Liquidity diagnostics, and Technical charts.

### TradingView (Interactive Charting & Technical Visuals)
* **Adopted Concepts**:
  * **Canvas Architecture**: Embedded TradingView Lightweight Charts with support for both Candlestick and Line price visualizations.
  * **Attribution Compliance**: Explicit, dignified TradingView library attribution rendered on chart footers.
  * **Overlay & Sub-panel Controls**: Smooth toggles for key technical overlays (SMA 20, SMA 50, SMA 200, Bollinger Bands) and oscillator panels (RSI, MACD) with volume histogram bars.
  * **Time Range Navigation**: Instant zoom periods (`1M`, `3M`, `6M`, `1Y`, `5Y`, `Max`).

### Koyfin & Finviz (Screener Density & Tabular Analytics)
* **Adopted Concepts**:
  * **Dense, Scan-Friendly Grid**: High-density 20-stock rankings table optimized for analytical scanning without visual clutter.
  * **Interactive Horizon Weighting**: Live horizon weight sliders (Short, Mid, Long) with instant re-normalization and "Reset to default" functionality.
  * **Miniaturized Metrics**: Embedded 52-week price range visual bars, sparklines, and liquidity warning flags directly within tabular rows.
  * **Cross-Stock Comparison**: Checkbox multi-selection activating a dedicated Compare Drawer (up to 4 stocks) normalizing price histories to base 100.

### Robinhood (Clean Simplicity & Mobile Accessibility)
* **Adopted Concepts**:
  * **Visual Restraint**: Clean white and deep-slate surfaces, generous padding, and typography-first hierarchy using tabular numerals for numbers.
  * **Responsive Card Transformation**: On mobile screens (<768px), wide analytical tables gracefully collapse into touch-friendly cards without losing data fidelity.
  * **Clear Disclaimers**: Frictionless first-login acknowledgment modal reinforcing informational non-advice boundaries.

### Simply Wall St (Visual Summaries & Plain-English Explanations)
* **Adopted Concepts**:
  * **Upside Score Gauges**: Intuitive radial and bar gauges (0–100 scale) replacing raw statistical scores.
  * **Tier Badges with Iconography**: Strict multi-modal tier indicators—High upside (emerald green + arrow up icon), Neutral (slate gray + horizontal dash icon), and Low upside (rose red/amber + arrow down icon)—ensuring contrast and accessibility without color alone.
  * **Plain-Language Factor Cards**: Horizon outlook cards featuring clear "Supporting factors" and "Holding-back factors" translating quantitative inputs into plain English.

### Stripe & Linear Dashboards (Admin Polish & Command Experience)
* **Adopted Concepts**:
  * **Strict 8px Grid & Subdued Palettes**: Monochromatic slate-gray neutral scales, subtle 1px border dividers (`border-border`), and crisp focus rings (`ring-2 ring-primary`).
  * **Model Management Safeguards**: Administrative model switches guarded by confirmation modals requiring stated reasons, previewing top-5 ranking shifts, and demanding explicit acknowledgment checkboxes for lower-evidence overrides.
  * **Auditing & Job Feedback**: Clear progress bars, dry-run toggles, and comprehensive audit log tables with CSV export.

---

## 2. Design System Tokens & Accessibility Standards

* **Color Palette**:
  * **Primary Accent**: Deep Royal Blue (`#1e40af` / `hsl(221, 83%, 53%)`).
  * **Backgrounds**: Pure Crisp Light (`#f8fafc` / `#ffffff`) and Sleek Dark Slate (`#090d16` / `#0f172a`).
  * **Tiers**:
    * High upside: Emerald (`#059669` / `#10b981`) with Lucide `TrendingUp` icon.
    * Neutral: Slate (`#64748b`) with Lucide `Minus` icon.
    * Low upside: Amber/Rose (`#e11d48` / `#f43f5e`) with Lucide `TrendingDown` icon.
  * **Confidence**: Neutral/Amber pills only (Never green), signaling caution and historical limitations.
* **Typography**:
  * Family: `Inter`, system-ui, sans-serif.
  * Numbers: Font feature settings `tnum` (tabular numbers) across all metric cards and tables.
* **Compliance**:
  * Fully compliant with WCAG AA contrast standards.
  * Keyboard navigable with explicit ARIA tags and visible focus outlines.
  * Persistent disclaimer in the application footer:
    *"Stock Pulse provides data-driven rankings for information only. It is not financial advice. Signals shown have limited historical reliability."*
