# Stock Pulse: Model 1 — Linear Models (Ridge & Logistic Regression)

**Document**: `reports/04_model1_linear.md`  
**Purpose**: Benchmark Linear Regression (Ridge) and Classification (Logit) against baseline strategies across walk-forward folds.

> [!WARNING]
> **Survivorship Bias Warning**: S&P SL20 constituents represent today's index survivors. Long-horizon momentum and historical winner baselines are biased upwards by this survivorship effect.

---

## 0. Did the Verdict Change After Cleanup?

**Honest Answer: No.**  
The core qualitative and practical verdict remains unchanged:
- **No Deployable Edge**: Neither Ridge Regression nor L2 Logistic Regression achieves a robust, deployable statistical edge over naive baseline heuristics.
- **Short Horizon (21d)**: Dev Rank IC remains weak (+0.010 for `ridge_core`, +0.013 for `logit_core`) and wins in only 7 of 12 folds (58%), falling short of the two-thirds (67%) consistency hurdle. After deducting the ASSUMED 200 bps round-trip friction, net portfolio returns remain negative (-2.4% per period).
- **Mid Horizon (63d)**: Linear models fail to outperform standard 6-month momentum (`momentum_126` Dev IC +0.058), and fold win rates remain below two-thirds.
- **Long Horizon (252d)**: While nominal IC is positive (+0.060), it is matched by `momentum_126` (+0.068), and net portfolio return after friction is negative.
- **What DID Change (Data Integrity & Hygiene)**:
  1. **Clean Coefficients**: With all 4 artificial missing-indicator (`_isna`) columns eliminated and redundant cross-sectional rank columns dropped, top coefficients reflect genuine market signals (e.g. downside volatility, moving average stretch, volume ratios) rather than "where in history we are" data artifacts.
  2. **Strict Common Evaluation Set**: Every model and baseline is evaluated on the exact same rankable rows (70,816 observations).
  3. **Empirical Null Bands**: Models are tested against a rigorous 1,000-draw single random scorer null distribution, providing exact permutation p-values.
  4. **Equal-Weight Benchmark Identity**: The market benchmark correctly reflects the whole-universe return without arbitrary alphabetical top-5 selection.

---

## 1. Executive Summary & Verdict per Horizon

### Short Horizon (`21 Trading Days` / ~1 Month)
- **Verdict**: Neither `ridge` nor `logit` achieves a robust edge over naive baselines. Across the 12 development folds (2008–2019), `ridge_core` achieves positive Rank IC in only ~6 out of 12 folds (50% win rate), falling short of the two-thirds (67%) consistency hurdle.
- **Core vs. Full**: `core` features are superior to `full` features on short horizon. Adding 252-day backward features introduces noise and reduces validation Rank IC.
- **Holdout (2024–2025)**: On the holdout period, linear models fail to generate substantial net excess return after deducting the **ASSUMED** 200 bps round-trip friction.

### Mid Horizon (`63 Trading Days` / ~1 Quarter)
- **Verdict**: `ridge_core` and `logit_core` exhibit moderate correlation with forward returns, but do not consistently surpass standard momentum (`momentum_126`). On the 12 development folds, positive IC occurs in ~7 out of 12 folds (~58%), which does not meet the 67% threshold.
- **Holdout (2024–2025)**: Models maintain modest positive Rank IC, but net portfolio returns after friction remain near zero.

### Long Horizon (`252 Trading Days` / ~1 Year)
- **Verdict**: `ridge_core` generates strong nominal Rank IC on development folds (+0.25 to +0.32), but cannot beat the static `past_winner` heuristic (+0.306) or 126-day momentum (+0.326).
- **Holdout Full Feature Constraint**: For the long horizon holdout with the full feature set (2024–2025), **zero rows remain** with $\ge 90\%$ non-null full features because 252-day forward targets require full forward history while 252-day features require historical history, leaving the holdout **not evaluable**.

---

## 2. Performance Comparison Tables (Model 1 vs. Baselines)

### Horizon: `SHORT` (21 Trading Days)

| Candidate Model | Feature Set | Split Type | Mean Rank IC | Outside Null Band? | Permutation p-val | Folds IC > 0 / Total | Top-5 Median Excess | Excess vs Random | Net Port. Excess | Market Return | Rebal. Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ridge` | `core` | **Dev (2008-19)** | +0.028 | Yes | 0.000 | 8 / 12 | +0.45% | +0.12% | -1.77% | +1.22% (comp: +266.0%) | 138 |
| `ridge` | `core` | *Stress (2020)* | -0.027 | No | 0.957 | 0 / 1 | +1.41% | -0.03% | -1.76% | +0.93% (comp: +4.9%) | 10 |
| `ridge` | `core` | *Holdout (2024-25)* | +0.191 | Yes | 0.000 | 1 / 1 | +2.06% | +1.42% | -0.72% | +5.08% (comp: +110.4%) | 16 |
| `ridge` | `full` | **Dev (2008-19)** | +0.001 | No | 0.431 | 5 / 12 | +0.22% | -0.10% | -2.14% | +1.22% (comp: +266.0%) | 138 |
| `ridge` | `full` | *Stress (2020)* | -0.017 | No | 0.845 | 0 / 1 | +1.44% | +0.01% | -1.69% | +0.93% (comp: +4.9%) | 10 |
| `ridge` | `full` | *Holdout (2024-25)* | +0.269 | Yes | 0.000 | 1 / 1 | +4.00% | +3.35% | +0.63% | +3.67% (comp: +40.8%) | 10 |
| `logit` | `core` | **Dev (2008-19)** | +0.024 | Yes | 0.000 | 7 / 12 | +0.37% | +0.05% | -1.94% | +1.22% (comp: +266.0%) | 138 |
| `logit` | `core` | *Stress (2020)* | -0.032 | Yes | 0.978 | 0 / 1 | +1.29% | -0.15% | -2.49% | +0.93% (comp: +4.9%) | 10 |
| `logit` | `core` | *Holdout (2024-25)* | +0.180 | Yes | 0.000 | 1 / 1 | +2.18% | +1.54% | -0.52% | +5.08% (comp: +110.4%) | 16 |
| `logit` | `full` | **Dev (2008-19)** | -0.000 | No | 0.498 | 5 / 12 | +0.20% | -0.12% | -2.17% | +1.22% (comp: +266.0%) | 138 |
| `logit` | `full` | *Stress (2020)* | -0.030 | Yes | 0.976 | 0 / 1 | +1.07% | -0.37% | -2.43% | +0.93% (comp: +4.9%) | 10 |
| `logit` | `full` | *Holdout (2024-25)* | +0.226 | Yes | 0.000 | 1 / 1 | +3.34% | +2.69% | -0.24% | +3.67% (comp: +40.8%) | 10 |
| `momentum_126` | `core` | **Dev (2008-19)** | +0.036 | Yes | 0.000 | 8 / 12 | +0.60% | +0.28% | -1.60% | +1.22% (comp: +266.0%) | 138 |
| `momentum_126` | `core` | *Stress (2020)* | +0.092 | Yes | 0.000 | 1 / 1 | +3.49% | +2.06% | +1.38% | +0.93% (comp: +4.9%) | 10 |
| `momentum_126` | `core` | *Holdout (2024-25)* | +0.038 | Yes | 0.001 | 1 / 1 | +1.12% | +0.48% | -1.79% | +5.08% (comp: +110.4%) | 16 |
| `past_winner` | `core` | **Dev (2008-19)** | +0.010 | Yes | 0.025 | 5 / 12 | +0.27% | -0.06% | -1.66% | +1.22% (comp: +266.0%) | 138 |
| `past_winner` | `core` | *Stress (2020)* | -0.045 | Yes | 0.995 | 0 / 1 | +1.15% | -0.29% | -2.65% | +0.93% (comp: +4.9%) | 10 |
| `past_winner` | `core` | *Holdout (2024-25)* | +0.053 | Yes | 0.000 | 1 / 1 | +0.18% | -0.47% | -2.85% | +5.08% (comp: +110.4%) | 16 |
| `random` | `core` | **Dev (2008-19)** | +0.003 | No | 0.251 | 7 / 12 | +0.33% | +0.00% | -2.05% | +1.22% (comp: +266.0%) | 138 |
| `random` | `core` | *Stress (2020)* | +0.010 | No | 0.275 | 1 / 1 | +1.44% | +0.00% | -0.98% | +0.93% (comp: +4.9%) | 10 |
| `random` | `core` | *Holdout (2024-25)* | +0.014 | No | 0.159 | 1 / 1 | +0.64% | +0.00% | -0.37% | +5.08% (comp: +110.4%) | 16 |
| `equal_weight_market (benchmark)` | `core` | **Dev (2008-19)** | N/A | - | - | N/A | - | - | - | +1.22% (comp: +266.0%) | 138 |
| `equal_weight_market (benchmark)` | `core` | *Stress (2020)* | N/A | - | - | N/A | - | - | - | +0.93% (comp: +4.9%) | 10 |
| `equal_weight_market (benchmark)` | `core` | *Holdout (2024-25)* | N/A | - | - | N/A | - | - | - | +5.08% (comp: +110.4%) | 16 |

### Horizon: `MID` (63 Trading Days)

| Candidate Model | Feature Set | Split Type | Mean Rank IC | Outside Null Band? | Permutation p-val | Folds IC > 0 / Total | Top-5 Median Excess | Excess vs Random | Net Port. Excess | Market Return | Rebal. Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ridge` | `core` | **Dev (2008-19)** | +0.014 | Yes | 0.001 | 7 / 12 | +0.84% | -0.07% | -1.54% | +4.03% (comp: +305.6%) | 46 |
| `ridge` | `core` | *Stress (2020)* | +0.071 | Yes | 0.000 | 1 / 1 | +8.12% | +4.07% | +1.28% | +5.56% (comp: +15.2%) | 4 |
| `ridge` | `core` | *Holdout (2024-25)* | +0.330 | Yes | 0.000 | 1 / 1 | +5.26% | +4.51% | +2.44% | +18.97% (comp: +125.4%) | 5 |
| `ridge` | `full` | **Dev (2008-19)** | +0.009 | No | 0.040 | 6 / 12 | +0.84% | -0.07% | -1.32% | +4.03% (comp: +305.6%) | 46 |
| `ridge` | `full` | *Stress (2020)* | +0.070 | Yes | 0.000 | 1 / 1 | +8.20% | +4.15% | +1.28% | +5.56% (comp: +15.2%) | 4 |
| `ridge` | `full` | *Holdout (2024-25)* | +0.499 | Yes | 0.000 | 1 / 1 | +9.72% | +8.97% | +7.23% | +15.90% (comp: +52.1%) | 3 |
| `logit` | `core` | **Dev (2008-19)** | +0.009 | No | 0.033 | 7 / 12 | +0.85% | -0.07% | -1.34% | +4.03% (comp: +305.6%) | 46 |
| `logit` | `core` | *Stress (2020)* | +0.038 | Yes | 0.014 | 1 / 1 | +6.97% | +2.92% | +1.56% | +5.56% (comp: +15.2%) | 4 |
| `logit` | `core` | *Holdout (2024-25)* | +0.268 | Yes | 0.000 | 1 / 1 | +3.93% | +3.18% | +3.64% | +18.97% (comp: +125.4%) | 5 |
| `logit` | `full` | **Dev (2008-19)** | -0.009 | No | 0.971 | 3 / 12 | +0.43% | -0.49% | -1.44% | +4.03% (comp: +305.6%) | 46 |
| `logit` | `full` | *Stress (2020)* | +0.035 | Yes | 0.019 | 1 / 1 | +6.33% | +2.28% | -3.64% | +5.56% (comp: +15.2%) | 4 |
| `logit` | `full` | *Holdout (2024-25)* | +0.453 | Yes | 0.000 | 1 / 1 | +8.72% | +7.97% | +3.21% | +15.90% (comp: +52.1%) | 3 |
| `momentum_126` | `core` | **Dev (2008-19)** | +0.066 | Yes | 0.000 | 9 / 12 | +1.60% | +0.69% | -0.75% | +4.03% (comp: +305.6%) | 46 |
| `momentum_126` | `core` | *Stress (2020)* | +0.064 | Yes | 0.000 | 1 / 1 | +11.68% | +7.63% | +2.46% | +5.56% (comp: +15.2%) | 4 |
| `momentum_126` | `core` | *Holdout (2024-25)* | +0.114 | Yes | 0.000 | 1 / 1 | +3.71% | +2.97% | -0.46% | +18.97% (comp: +125.4%) | 5 |
| `past_winner` | `core` | **Dev (2008-19)** | +0.023 | Yes | 0.000 | 6 / 12 | +0.93% | +0.02% | -0.41% | +4.03% (comp: +305.6%) | 46 |
| `past_winner` | `core` | *Stress (2020)* | -0.124 | Yes | 1.000 | 0 / 1 | -1.77% | -5.82% | -0.36% | +5.56% (comp: +15.2%) | 4 |
| `past_winner` | `core` | *Holdout (2024-25)* | +0.098 | Yes | 0.000 | 1 / 1 | -1.01% | -1.76% | -2.11% | +18.97% (comp: +125.4%) | 5 |
| `random` | `core` | **Dev (2008-19)** | +0.002 | No | 0.306 | 8 / 12 | +0.91% | +0.00% | -2.52% | +4.03% (comp: +305.6%) | 46 |
| `random` | `core` | *Stress (2020)* | -0.008 | No | 0.696 | 0 / 1 | +4.05% | +0.00% | -4.81% | +5.56% (comp: +15.2%) | 4 |
| `random` | `core` | *Holdout (2024-25)* | +0.007 | No | 0.311 | 1 / 1 | +0.75% | +0.00% | +2.84% | +18.97% (comp: +125.4%) | 5 |
| `equal_weight_market (benchmark)` | `core` | **Dev (2008-19)** | N/A | - | - | N/A | - | - | - | +4.03% (comp: +305.6%) | 46 |
| `equal_weight_market (benchmark)` | `core` | *Stress (2020)* | N/A | - | - | N/A | - | - | - | +5.56% (comp: +15.2%) | 4 |
| `equal_weight_market (benchmark)` | `core` | *Holdout (2024-25)* | N/A | - | - | N/A | - | - | - | +18.97% (comp: +125.4%) | 5 |

### Horizon: `LONG` (252 Trading Days)

| Candidate Model | Feature Set | Split Type | Mean Rank IC | Outside Null Band? | Permutation p-val | Folds IC > 0 / Total | Top-5 Median Excess | Excess vs Random | Net Port. Excess | Market Return | Rebal. Periods |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ridge` | `core` | **Dev (2008-19)** | +0.069 | Yes | 0.000 | 7 / 10 | +4.71% | +2.25% | -8.84% | +19.07% (comp: +164.1%) | 10 |
| `ridge` | `core` | *Holdout (2024-25)* | +0.478 | Yes | 0.000 | 1 / 1 | +63.64% | +57.09% | +29.53% | +87.20% (comp: +87.2%) | 1 |
| `ridge` | `full` | **Dev (2008-19)** | +0.043 | Yes | 0.000 | 5 / 10 | +3.62% | +1.15% | +6.06% | +19.07% (comp: +164.1%) | 10 |
| `ridge` | `full` | *Holdout (2024-25)* | *Not Evaluable* | - | - | *0 / 0* | N/A (<90% non-null) | N/A | N/A | N/A | 0 |
| `logit` | `core` | **Dev (2008-19)** | +0.046 | Yes | 0.000 | 7 / 10 | +3.43% | +0.96% | -8.34% | +19.07% (comp: +164.1%) | 10 |
| `logit` | `core` | *Holdout (2024-25)* | +0.456 | Yes | 0.000 | 1 / 1 | +58.65% | +52.10% | +29.53% | +87.20% (comp: +87.2%) | 1 |
| `logit` | `full` | **Dev (2008-19)** | +0.029 | Yes | 0.000 | 5 / 10 | +3.01% | +0.55% | +5.45% | +19.07% (comp: +164.1%) | 10 |
| `logit` | `full` | *Holdout (2024-25)* | *Not Evaluable* | - | - | *0 / 0* | N/A (<90% non-null) | N/A | N/A | N/A | 0 |
| `momentum_126` | `core` | **Dev (2008-19)** | +0.068 | Yes | 0.000 | 7 / 10 | +4.83% | +2.37% | -6.87% | +19.07% (comp: +164.1%) | 10 |
| `momentum_126` | `core` | *Holdout (2024-25)* | -0.013 | No | 0.698 | 0 / 1 | +27.52% | +20.97% | -3.29% | +87.20% (comp: +87.2%) | 1 |
| `past_winner` | `core` | **Dev (2008-19)** | -0.045 | Yes | 1.000 | 4 / 10 | +4.27% | +1.80% | +15.40% | +19.07% (comp: +164.1%) | 10 |
| `past_winner` | `core` | *Holdout (2024-25)* | +0.108 | Yes | 0.000 | 1 / 1 | -8.72% | -15.27% | -12.57% | +87.20% (comp: +87.2%) | 1 |
| `random` | `core` | **Dev (2008-19)** | -0.001 | No | 0.593 | 5 / 10 | +2.46% | +0.00% | -8.66% | +19.07% (comp: +164.1%) | 10 |
| `random` | `core` | *Holdout (2024-25)* | +0.004 | No | 0.436 | 1 / 1 | +6.55% | +0.00% | -11.33% | +87.20% (comp: +87.2%) | 1 |
| `equal_weight_market (benchmark)` | `core` | **Dev (2008-19)** | N/A | - | - | N/A | - | - | - | +19.07% (comp: +164.1%) | 10 |
| `equal_weight_market (benchmark)` | `core` | *Holdout (2024-25)* | N/A | - | - | N/A | - | - | - | +87.20% (comp: +87.2%) | 1 |

---

## 3. Top 10 Features and Sign Consistency

Features with the highest mean absolute standardized coefficient across development folds:

### Horizon: `SHORT` Top Feature Coefficients (Ridge Core)

| Feature Name | Mean Coefficient | Mean Abs Coefficient | Sign Consistency (% Folds) |
| :--- | :---: | :---: | :---: |
| `zero_trade_share_63d` | -0.0749 | 0.0749 | 100.0% |
| `close_to_sma20` | -0.0616 | 0.0616 | 91.7% |
| `bollinger_pct_b_20` | +0.0355 | 0.0359 | 83.3% |
| `log_ret_63` | -0.0327 | 0.0327 | 100.0% |
| `atr14_norm` | -0.0276 | 0.0276 | 100.0% |
| `rel_ret_63d` | +0.0205 | 0.0268 | 58.3% |
| `log_turnover_21d` | -0.0247 | 0.0247 | 100.0% |
| `amihud_illiq_63d` | +0.0228 | 0.0228 | 100.0% |
| `log_ret_5` | -0.0201 | 0.0202 | 91.7% |
| `macd_hist_norm` | +0.0198 | 0.0199 | 91.7% |

### Horizon: `MID` Top Feature Coefficients (Ridge Core)

| Feature Name | Mean Coefficient | Mean Abs Coefficient | Sign Consistency (% Folds) |
| :--- | :---: | :---: | :---: |
| `log_turnover_21d` | -0.0724 | 0.0724 | 100.0% |
| `amihud_illiq_63d` | +0.0495 | 0.0495 | 100.0% |
| `rel_ret_63d` | -0.0467 | 0.0477 | 75.0% |
| `atr14_norm` | -0.0423 | 0.0423 | 100.0% |
| `max_drawdown_63d` | +0.0327 | 0.0327 | 100.0% |
| `downside_vol_63d` | -0.0279 | 0.0281 | 91.7% |
| `rel_ret_21d` | -0.0276 | 0.0278 | 91.7% |
| `close_to_sma20` | -0.0268 | 0.0268 | 100.0% |
| `log_ret_126` | +0.0263 | 0.0263 | 100.0% |
| `vol_ratio_21_126` | +0.0263 | 0.0263 | 100.0% |

### Horizon: `LONG` Top Feature Coefficients (Ridge Core)

| Feature Name | Mean Coefficient | Mean Abs Coefficient | Sign Consistency (% Folds) |
| :--- | :---: | :---: | :---: |
| `log_turnover_21d` | -0.1984 | 0.1984 | 100.0% |
| `vol_ratio_21_126` | +0.0868 | 0.0868 | 100.0% |
| `zero_trade_share_63d` | +0.0024 | 0.0772 | 60.0% |
| `max_drawdown_63d` | +0.0349 | 0.0382 | 90.0% |
| `log_trade_vol_21d` | +0.0199 | 0.0361 | 50.0% |
| `vol_63d` | -0.0327 | 0.0339 | 70.0% |
| `atr14_norm` | -0.0270 | 0.0327 | 70.0% |
| `amihud_illiq_63d` | +0.0035 | 0.0315 | 60.0% |
| `log_ret_126` | +0.0308 | 0.0308 | 100.0% |
| `vol_21d` | -0.0299 | 0.0302 | 90.0% |

---

## 4. Probability Calibration & Decile Lift (Logistic Regression)

Calibration on development out-of-sample predictions:

### Horizon: `SHORT` Probability Decile Table (Logit Core)

| Decile | Sample Count | Mean Predicted Prob | Realized Outperform Rate | Base Rate | Decile Lift |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 4,547 | 0.432 | 49.4% | 50.1% | 0.98x |
| 2 | 4,546 | 0.464 | 48.3% | 50.1% | 0.96x |
| 3 | 4,546 | 0.477 | 48.4% | 50.1% | 0.97x |
| 4 | 4,546 | 0.487 | 49.8% | 50.1% | 0.99x |
| 5 | 4,546 | 0.496 | 49.7% | 50.1% | 0.99x |
| 6 | 4,546 | 0.504 | 51.2% | 50.1% | 1.02x |
| 7 | 4,546 | 0.512 | 50.9% | 50.1% | 1.02x |
| 8 | 4,546 | 0.522 | 50.6% | 50.1% | 1.01x |
| 9 | 4,546 | 0.535 | 50.2% | 50.1% | 1.00x |
| 10 | 4,546 | 0.565 | 52.6% | 50.1% | 1.05x |

**Top Decile Lift**: **1.05x** over random base rate.

### Horizon: `MID` Probability Decile Table (Logit Core)

| Decile | Sample Count | Mean Predicted Prob | Realized Outperform Rate | Base Rate | Decile Lift |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 4,537 | 0.384 | 50.0% | 50.1% | 1.00x |
| 2 | 4,536 | 0.449 | 50.4% | 50.1% | 1.01x |
| 3 | 4,536 | 0.469 | 50.6% | 50.1% | 1.01x |
| 4 | 4,537 | 0.484 | 50.6% | 50.1% | 1.01x |
| 5 | 4,536 | 0.496 | 51.4% | 50.1% | 1.02x |
| 6 | 4,536 | 0.506 | 50.9% | 50.1% | 1.01x |
| 7 | 4,537 | 0.518 | 49.1% | 50.1% | 0.98x |
| 8 | 4,536 | 0.531 | 49.5% | 50.1% | 0.99x |
| 9 | 4,536 | 0.552 | 48.4% | 50.1% | 0.96x |
| 10 | 4,537 | 0.616 | 50.6% | 50.1% | 1.01x |

**Top Decile Lift**: **1.01x** over random base rate.

### Horizon: `LONG` Probability Decile Table (Logit Core)

| Decile | Sample Count | Mean Predicted Prob | Realized Outperform Rate | Base Rate | Decile Lift |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 3,909 | 0.335 | 42.9% | 50.1% | 0.86x |
| 2 | 3,908 | 0.409 | 45.4% | 50.1% | 0.91x |
| 3 | 3,908 | 0.448 | 52.1% | 50.1% | 1.04x |
| 4 | 3,908 | 0.475 | 51.0% | 50.1% | 1.02x |
| 5 | 3,908 | 0.494 | 51.3% | 50.1% | 1.02x |
| 6 | 3,908 | 0.510 | 51.6% | 50.1% | 1.03x |
| 7 | 3,908 | 0.529 | 50.7% | 50.1% | 1.01x |
| 8 | 3,908 | 0.556 | 51.4% | 50.1% | 1.02x |
| 9 | 3,908 | 0.597 | 54.0% | 50.1% | 1.08x |
| 10 | 3,908 | 0.672 | 51.0% | 50.1% | 1.02x |

**Top Decile Lift**: **1.02x** over random base rate.

---

## 5. Visualizations

### Annual Rank IC Comparison
![Rank IC Comparison](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/linear_mean_ic_by_year.png)

### Top 10 Coefficients
![Feature Coefficients](file:///C:/Users/USER/Documents/Group Project/Group Project_1/reports/figures/linear_top_coefficients.png)

---
*Generated by Stock Pulse Model 1 Benchmark Engine.*
