# Stock Pulse: Investor Methodology & Guide

> **Data As-Of Date**: 2024-12-10 | **Universe**: Colombo Stock Exchange (S&P SL20)

> **Important Disclaimer**: Stock Pulse is an automated quantitative research and ranking platform built strictly for informational purposes. It does not constitute personalized investment advice, a portfolio recommendation, or an endorsement to buy or sell securities. Past out-of-sample performance does not guarantee future results. Frontier equity investing carries significant risk of principal loss.

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
- **50**: Average upside potential.
- **0**: Lowest upside potential relative to peers.

### Overall Weighted Composite Score
The system combines scores across all three timeframes using evidence-derived weighting:
- **Short Horizon**: `25.0%` weight.
- **Mid Horizon**: `50.0%` weight.
- **Long Horizon**: `25.0%` weight.

---

## 4. Ranking Tiers (High, Neutral, Low)

Percentile score thresholds define three action tiers:

| Horizon | Candidate Fraction | High Upside Cutoff | Neutral Range | Low Upside Cutoff |
| :--- | :---: | :---: | :---: | :---: |
| **Short (21d)** | Top/Bottom 25% | Score $\ge 75$ | 25 to 75 | Score $\le 25$ |
| **Mid (63d)** | Top/Bottom 30% | Score $\ge 70$ | 30 to 70 | Score $\le 30$ |
| **Long (252d)** | Top/Bottom 30% | Score $\ge 70$ | 30 to 70 | Score $\le 30$ |

---

## 5. Consistent Leaders

A company earns the **Consistent Leader** badge only when it satisfies three strict simultaneous rules:
1. **High Upside** in at least two of the three time horizons.
2. **Zero Low Upside** rankings across all three horizons.
3. **No Liquidity Warning** on that trading day.

---

## 6. Model Confidence Levels & Historical Performance

### Confidence Levels by Horizon
- **Short Horizon (21d)**: `Very low confidence` (Evidence Level: `RED`).
- **Mid Horizon (63d)**: `Low confidence` (Evidence Level: `AMBER`).
- **Long Horizon (252d)**: `Very low confidence` (Evidence Level: `RED`).

### Historical Top-Tier Out-of-Sample Hit Rates
> [!WARNING]
> **Optimism Warning**: Historical hit rates reflect simulation over development periods and were selected retrospectively. Real-world performance is typically lower.

| Horizon | Evaluation Period | High Tier Hit Rate (90% CI) | Neutral Hit Rate | Low Hit Rate | High Tier Median Excess | Evaluation Flag |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Short (21d)** | **Historical Dev** | **49.4%** ([47.0%, 51.8%]) | 51.7% | 47.4% | +0.00% | Selected on Dev (n=138) |
| **Short (21d)** | **Holdout** | **61.0%** (small sample) | 53.0% | 33.1% | +1.65% | Holdout (n=16) |
| **Mid (63d)** | **Historical Dev** | **50.7%** ([44.6%, 56.8%]) | 52.7% | 45.3% | +0.00% | Selected on Dev (n=46) |
| **Mid (63d)** | **Holdout** | **63.0%** (small sample) | 43.2% | 46.3% | +1.19% | Holdout (n=5) |
| **Long (252d)** | **Historical Dev** | **60.2%** ([50.9%, 71.1%]) | 49.2% | 40.8% | +3.23% | Selected on Dev (n=10) |
| **Long (252d)** | **Holdout** | **80.0%** (small sample) | 33.3% | 40.0% | +9.38% | Holdout (n=1) |

---

## 7. Volatility Categories & Warnings

### Realized Volatility Categories
Calculated from trailing 63-day annualized return volatility:
- **Low Volatility**: Annualized volatility $\le$ `21.76%`.
- **Medium Volatility**: Annualized volatility between `21.76%` and `32.98%`.
- **High Volatility**: Annualized volatility $>$ `32.98%`.

### Liquidity Warning
- A stock receives a **Liquidity Warning** when its trailing 63 market-calendar trading day participation is at or below `88.9%` (the 20th percentile of market participation). This warns investors that the stock has experienced frequent non-trading sessions.
- The liquidity warning is an informational flag indicating lower historical trading frequency, not a recommendation to avoid a stock.

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