# Stock Pulse: Feature Engineering & Target Data Dictionary

**Document**: `reports/02_feature_dictionary.md`  
**Scope**: In-Universe Voting Equity Series (`MainType == 'N'`)  
**Strict Temporal Invariant**: All features use data strictly up to and including trading date $t$. Windows never bridge hard segment boundaries.

---

## 1. Feature Specifications

| Feature Name | Category | Plain-English Formula | Financial / Empirical Rationale |
| :--- | :--- | :--- | :--- |
| `Open` | Engineered | `Open` | Technical predictor. |
| `log_ret_5` | Momentum | `ln(Close_t / Close_{t-5})` | Short-term 1-week momentum reflecting immediate price velocity. |
| `log_ret_10` | Momentum | `ln(Close_t / Close_{t-10})` | 2-week price momentum capturing bi-weekly trend momentum. |
| `log_ret_21` | Momentum | `ln(Close_t / Close_{t-21})` | Monthly 1-month momentum; institutional tactical horizon benchmark. |
| `log_ret_63` | Momentum | `ln(Close_t / Close_{t-63})` | Quarterly 3-month momentum; aligns with corporate earnings cycle. |
| `log_ret_126` | Momentum | `ln(Close_t / Close_{t-126})` | Semi-annual 6-month intermediate momentum. |
| `log_ret_252` | Momentum | `ln(Close_t / Close_{t-252})` | Annual 1-year total price momentum; classic long-term trend factor. |
| `mom_12_1` | Momentum | `ln(Close_{t-21} / Close_{t-252})` | Classic Jegadeesh-Titman 12-1 momentum excluding short-term reversal month. |
| `vol_21d` | Risk | `std(log_ret, 21) * sqrt(252)` | Annualized 1-month realized volatility measuring tactical risk. |
| `vol_63d` | Risk | `std(log_ret, 63) * sqrt(252)` | Annualized 3-month quarterly realized volatility measuring medium-term risk. |
| `downside_vol_63d` | Risk | `sqrt(252 * sum(min(log_ret, 0)^2) / 62)` | Downside semi-volatility penalizing negative tail variance only. |
| `parkinson_vol_21d` | Risk | `sqrt(252 / (21 * 4 * ln 2) * sum(ln(High/Low)^2))` | Extreme-value volatility estimator utilizing intraday High and Low ranges. |
| `atr14_norm` | Risk | `ATR_14 / Close_t` | 14-day Average True Range normalized by price for cross-sectional volatility comparability. |
| `max_drawdown_63d` | Risk | `(Close_t - max(Close_{t-63..t})) / max(Close_{t-63..t})` | Peak-to-trough decline over preceding quarter; assesses drawdown severity. |
| `max_drawdown_252d` | Risk | `(Close_t - max(Close_{t-252..t})) / max(Close_{t-252..t})` | Maximum peak-to-trough drawdown over 1-year window. |
| `dist_high_126d` | Risk/Range | `(Close_t - max(High_{t-126..t})) / max(High_{t-126..t})` | Proximity to 6-month price ceiling. |
| `dist_low_126d` | Risk/Range | `(Close_t - min(Low_{t-126..t})) / min(Low_{t-126..t})` | Proximity to 6-month price floor. |
| `dist_high_252d` | Risk/Range | `(Close_t - max(High_{t-252..t})) / max(High_{t-252..t})` | Proximity to 52-week annual high; George-Hwang anchor indicator. |
| `dist_low_252d` | Risk/Range | `(Close_t - min(Low_{t-252..t})) / min(Low_{t-252..t})` | Proximity to 52-week annual low; bounce and distress indicator. |
| `close_to_sma20` | Trend | `Close_t / SMA_20(Close) - 1` | Percentage divergence from 20-day tactical moving average. |
| `close_to_sma50` | Trend | `Close_t / SMA_50(Close) - 1` | Percentage divergence from 50-day intermediate moving average. |
| `close_to_sma200` | Trend | `Close_t / SMA_200(Close) - 1` | Percentage divergence from 200-day institutional regime trend line. |
| `sma50_to_sma200` | Trend | `SMA_50 / SMA_200 - 1` | Golden Cross / Death Cross moving average divergence spread. |
| `rsi_14` | Trend/Oscillator | `100 - (100 / (1 + avg_gain / avg_loss))` | 14-day Relative Strength Index measuring overbought/oversold momentum. |
| `macd_hist_norm` | Trend | `(MACD - Signal) / Close_t` | Moving Average Convergence Divergence normalized oscillator histogram. |
| `bollinger_pct_b_20` | Trend/Volatility | `(Close - LowerBand) / (UpperBand - LowerBand)` | Bollinger Band %B position within 20-day 2-sigma envelope. |
| `log_trade_vol_21d` | Liquidity | `ln(1 + SMA_21(TradeVolume))` | Log-transformed 1-month average daily trade transaction count. |
| `vol_ratio_21_126` | Liquidity | `SMA_21(ShareVolume) / SMA_126(ShareVolume)` | Volume surge ratio identifying institutional accumulation or distribution. |
| `log_turnover_21d` | Liquidity | `ln(1 + SMA_21(Turnover))` | Log-transformed 1-month average daily rupee trading turnover. |
| `amihud_illiq_63d` | Liquidity | `mean( |daily_ret| / (Turnover / 1M) )` | Amihud price impact ratio measuring illiquidity cost per rupee traded. |
| `zero_trade_share_63d` | Liquidity | `mean( I(TradeVolume == 0 or ShareVolume == 0) )` | **Uninformative (constant 0.0)**: Cleaned dataset only contains rows on days a stock traded, so this feature is constant zero across all historical observations. Replaced by market-calendar `trade_day_share_63m` for liquidity warnings; models are not retrained. |
| `log_turnover_21d_ownz` | Engineered | `log_turnover_21d_ownz` | Technical predictor. |
| `log_trade_vol_21d_ownz` | Engineered | `log_trade_vol_21d_ownz` | Technical predictor. |
| `amihud_illiq_63d_ownz` | Engineered | `amihud_illiq_63d_ownz` | Technical predictor. |
| `vol_21d_ownz` | Engineered | `vol_21d_ownz` | Technical predictor. |
| `vol_63d_ownz` | Engineered | `vol_63d_ownz` | Technical predictor. |
| `index_ret_63d` | Macro Regime | `sum(mkt_ret, 63)` | Trailing 63-day Colombo market index return representing market regime. |
| `index_vol_63d` | Macro Regime | `std(mkt_ret, 63) * sqrt(252)` | Trailing 63-day Colombo market volatility representing market risk regime. |
| `rel_ret_21d` | Market-Relative | `log_ret_21 - mkt_cum_ret_21` | 1-month alpha / excess return relative to Colombo equal-weighted index. |
| `rel_ret_63d` | Market-Relative | `log_ret_63 - mkt_cum_ret_63` | 3-month alpha / excess return relative to Colombo equal-weighted index. |
| `beta_252d` | Market-Relative | `Cov(ret_i, ret_mkt) / Var(ret_mkt)` | 252-day trailing systematic market beta to Colombo S&P SL20 index. |
| `cs_rank_mom_63d` | Cross-Sectional | `rank(log_ret_63, pct=True)` | Cross-sectional percentile ranking of 3-month momentum across universe on day t. |
| `cs_rank_vol_63d` | Cross-Sectional | `rank(vol_63d, pct=True)` | Cross-sectional percentile ranking of volatility across universe on day t. |
| `cs_rank_vol_21d` | Cross-Sectional | `rank(log_trade_vol_21d, pct=True)` | Cross-sectional percentile ranking of trading activity across universe on day t. |

---

## 2. Target Specifications

All targets look forward strictly within the same segment. Offsets: $h \in \{21, 63, 252\}$ trading days.

| Target Variable | Horizon | Formula | Purpose |
| :--- | :---: | :--- | :--- |
| `fwd_ret_short` | 21 days | $Close[t+21] / Close[t] - 1$ | 1-month forward raw return |
| `fwd_excess_short` | 21 days | $fwd\_ret - \text{median}_{date}(fwd\_ret)$ | 1-month forward return relative to cross-sectional median |
| `fwd_rankpct_short` | 21 days | $\text{rank}_{date}(fwd\_ret, \text{pct}=True)$ | 1-month cross-sectional ranking percentile $[0, 1]$ |
| `outperform_short` | 21 days | $\mathbb{I}(fwd\_excess > 0)$ | Binary upside outperformance label |
| `fwd_ret_mid` | 63 days | $Close[t+63] / Close[t] - 1$ | 3-month forward raw return |
| `fwd_excess_mid` | 63 days | $fwd\_ret - \text{median}_{date}(fwd\_ret)$ | 3-month forward return relative to cross-sectional median |
| `fwd_rankpct_mid` | 63 days | $\text{rank}_{date}(fwd\_ret, \text{pct}=True)$ | 3-month cross-sectional ranking percentile $[0, 1]$ |
| `outperform_mid` | 63 days | $\mathbb{I}(fwd\_excess > 0)$ | Binary upside outperformance label |
| `fwd_ret_long` | 252 days | $Close[t+252] / Close[t] - 1$ | 1-year forward raw return |
| `fwd_excess_long` | 252 days | $fwd\_ret - \text{median}_{date}(fwd\_ret)$ | 1-year forward return relative to cross-sectional median |
| `fwd_rankpct_long` | 252 days | $\text{rank}_{date}(fwd\_ret, \text{pct}=True)$ | 1-year cross-sectional ranking percentile $[0, 1]$ |
| `outperform_long` | 252 days | $\mathbb{I}(fwd\_excess > 0)$ | Binary upside outperformance label |

---
*Generated by Stock Pulse Feature Engine.*
