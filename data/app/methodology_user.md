# Stock Pulse: Investor Methodology & Guide

> **Data As-Of Date**: 2024-12-10 | **Universe**: Colombo Stock Exchange (S&P SL20)

> **Important Disclaimer**: Stock Pulse is an automated quantitative research and ranking platform built strictly for informational purposes. It does not constitute personalized investment advice, a portfolio recommendation, or an endorsement to buy or sell securities. Past out-of-sample performance does not guarantee future results. Frontier equity investing carries substantial risk of principal loss.

---

## 1. What Stock Pulse Does (and Does Not Do)

### What Stock Pulse Does:
- **Ranks Stocks Relative to Peers**: Every trading session, Stock Pulse analyzes constituents of the Colombo Stock Exchange (CSE) S&P SL20 index and ranks them by upward upside potential.
- **Multi-Horizon Perspective**: Provides distinct rankings across three horizons:
  - **Short Horizon (21 Trading Days)**: Tactical ~1-month outlook.
  - **Mid Horizon (63 Trading Days)**: Quarterly ~3-month trend outlook.
  - **Long Horizon (252 Trading Days)**: Strategic ~1-year capital appreciation outlook.
- **Actionable Tiers**: Assigns companies into **High Upside**, **Neutral**, and **Low Upside** tiers, accompanied by risk volatility categories, a liquidity warning, and a thin-trading alert.

### What Stock Pulse Does NOT Do:
- It does not forecast exact price targets (e.g. 'Target LKR 150').
- It does not evaluate company earnings reports, financial statements, or qualitative management news.
- It does not guarantee trading returns or protect against systemic frontier market drawdowns.

---

## 2. The Data & Ranking Universe

- **Index Coverage**: The 20 member companies of the CSE S&P SL20 index.
- **Share Class Selection**: Analyzes voting ordinary shares (`.N`) for all companies, except Seylan Bank where the more liquid non-voting ordinary share line (`SEYB.X`) is selected.
- **Historical Coverage**: Spans over two decades of official CSE daily trading sessions (2001 through 2025).

---

## 3. How the Scores Work

The system evaluates each company each day using several price-based signals (including trend momentum, trailing volatility, and trading activity). The resulting upside score is expressed as a percentile rank from 0 to 100:
- **100**: Highest upside potential in the index relative to peers.
- **50**: Median stock.
- **0**: Lowest upside potential relative to peers.

### Overall Upside Score
The overall score is a weighted combination of the tactical, quarterly, and annual horizons:
- **Short Horizon**: `25.0%` weight
- **Mid Horizon**: `50.0%` weight
- **Long Horizon**: `25.0%` weight

---

## 4. Confidence Labels

Each horizon carries a formal statistical confidence rating:
- **Short Horizon**: **Very low confidence** (RED) - Signals at the 1-month horizon do not demonstrate statistically confirmed edge over random noise after transaction frictions.
- **Mid Horizon**: **Low confidence** (AMBER) - Medium-term 3-month momentum demonstrates suggestive historical outperformance, though subject to frontier regime shifts.
- **Long Horizon**: **Very low confidence** (RED) - 1-year annual horizons have limited non-overlapping historical evaluation cycles.

---

## 5. Tiers and How They Were Derived

Companies are sorted into three tiers based on pre-specified cutoff percentiles:
- **Short Horizon Cutoff**: Top `25%` are High tier, bottom `25%` are Low tier (convention - no evidence-based cutoff).
- **Mid Horizon Cutoff**: Top `30%` are High tier, bottom `30%` are Low tier (empirical).
- **Long Horizon Cutoff**: Top `30%` are High tier, bottom `30%` are Low tier (empirical).

---

## 6. Historical Performance of Tiers

> [!WARNING]
> **Optimism Warning**: The cutoffs and the score were chosen using the same historical years shown here, so these hit rates are optimistic. Realized future performance is likely to be lower.

Performance is reported as **hit rate** (the percentage of selections that beat the median stock in the index):

| Horizon | Partition | High Tier Hit Rate (90% CI) | Neutral Tier Hit Rate | Low Tier Hit Rate | High Tier Median Excess | Evaluation Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Short (21d)** | **Historical Dev** | **51.5%** ([48.7%, 53.8%]) | 50.9% | 47.0% | +0.00% | Selected on Dev (n=138) |
| **Short (21d)** | **Holdout** | **57.7%** (small sample) | 52.3% | 37.5% | +1.59% | Holdout (n=16) |
| **Mid (63d)** | **Historical Dev** | **50.7%** ([45.2%, 55.6%]) | 52.0% | 46.7% | +0.00% | Selected on Dev (n=46) |
| **Mid (63d)** | **Holdout** | **37.0%** (small sample) | 59.5% | 50.0% | -5.13% | Holdout (n=5) |
| **Long (252d)** | **Historical Dev** | **54.9%** ([46.1%, 64.1%]) | 54.6% | 38.8% | +0.58% | Selected on Dev (n=10) |
| **Long (252d)** | **Holdout** | **60.0%** (small sample) | 50.0% | 40.0% | +0.51% | Holdout (n=1) |

---

## 7. Volatility Categories & Warnings

### Realized Volatility Categories
Calculated from trailing 63-day annualized return volatility:
- **Low Volatility**: Annualized volatility $\le$ `21.76%`.
- **Medium Volatility**: Annualized volatility between `21.76%` and `32.98%`.
- **High Volatility**: Annualized volatility $>$ `32.98%`.

### Liquidity Warning
- A stock receives a **Liquidity Warning** when its trailing 63 market-calendar trading day participation is at or below `88.9%` (the 20th percentile of market participation). This warns investors that the stock has experienced frequent non-trading sessions.

### Thin Trading Warning (Stretched Horizon)
- Stocks flagged for thin trading: **HNB.N**.
- For thinly traded stocks, 21 consecutive trading rows span significantly more calendar days than the market median (> 1.5x ratio). All operational windows in Stock Pulse count the stock's own trading observations, meaning calendar holding periods are longer for these companies.

---

## 8. Limitations & Key Assumptions

1. **Survivorship Conditioning**: The historical universe comprises companies surviving into the modern S&P SL20 index.
2. **No Fundamental or Sentiment Data**: The engine evaluates only historical price and volume. Quarterly financial statements, dividend announcements, macroeconomic indicators, and news events are omitted.
3. **Frontier Market Transaction Frictions**: We assume an institutional round-trip trading cost of **200 basis points** (100 bps buy, 100 bps sell) covering brokerage commissions, CSE statutory charges, and bid-ask spread impact.
4. **Own-Trading-Day Horizons**: Observation windows count each stock's own active trading days rather than fixed calendar days.
5. **Frontier Liquidity Concentration**: Trading volume on the Colombo Stock Exchange is heavily concentrated in a small subset of blue-chip counters.

---

## 9. Regulatory & Legal Disclaimer

This platform is provided solely for academic research, education, and technology demonstration. Under no circumstances should any score, tier, or metric produced by Stock Pulse be interpreted as investment recommendations or portfolio advice.